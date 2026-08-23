from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from nyssa_bench.core.episode import EpisodeResult, StepRecord
from nyssa_bench.core.registry import make_engine, make_policy
from nyssa_bench.core.suite import Suite
from nyssa_bench.core.task import REPO_ROOT
from nyssa_bench.datasets.export_json import export_json
from nyssa_bench.datasets.export_jsonl import export_jsonl
from nyssa_bench.datasets.export_metrics_csv import export_metrics_csv
from nyssa_bench.datasets.provenance import write_dataset_manifest
from nyssa_bench.datasets.recovery import write_recovery_dataset
from nyssa_bench.experts import ExpertProvider, make_expert_provider
from nyssa_bench.failures import (
    FailureEventLedger,
    compact_stressor_context,
    derive_failure_label,
    drain_component_failure_events,
    emit_info_failure_events,
    recovery_attempt_draft,
    stressor_condition_drafts,
    summarize_failure_ledgers,
    terminal_failure_draft,
    verifier_rejection_draft,
    write_failure_ledger_manifest,
)
from nyssa_bench.failures.detectors import (
    FailureDetectorManager,
    build_default_failure_detectors,
)
from nyssa_bench.metrics.failure_mapper import FailureMapper
from nyssa_bench.metrics.robustness import robustness_metrics
from nyssa_bench.metrics.safety import safety_metrics
from nyssa_bench.metrics.sim_to_real import score_summary
from nyssa_bench.metrics.success import aggregate_episodes
from nyssa_bench.policies.base import Policy, PolicyLike, load_policy_from_path
from nyssa_bench.randomization import (
    aggregate_stressor_support,
    summarize_stressor_support,
)
from nyssa_bench.replay.video import (
    write_episode_video,
    write_failure_clip,
    write_failure_gallery,
    write_replay_manifest,
)
from nyssa_bench.replay.viewer import replay_viewer_placeholder
from nyssa_bench.reports.html_report import Report
from nyssa_bench.stressors import (
    StressorConfig,
    StressorContext,
    StressorPipeline,
    StressorSpec,
    summarize_stressor_execution,
    write_stressor_manifest,
)
from nyssa_bench.metrics.run_claims import RunClaimValidator
from nyssa_bench.utils.reproducibility import (
    environment_metadata,
    git_info,
    make_run_id,
    package_versions,
    utc_now,
    write_json,
)

EPISODE_SEED_STRIDE = 1_000_000
EPISODE_SEED_FORMAT = "nyssa-episode-seed-v2"
DEFAULT_RECOVERY_ATTRIBUTION_HORIZON = 5
RECOVERY_OUTCOME_FORMAT = "nyssa-recovery-outcomes-v1"
RECOVERY_ATTRIBUTION_CRITERION = (
    "task_success_within_bounded_window_before_next_attempt"
)


@dataclass
class _RecoveryAttempt:
    attempt_id: int
    start_step: int
    horizon_steps: int
    applied: bool = False
    plan_length: int = 0
    outcome: str = "pending"
    outcome_step: int | None = None
    event_step_indices: list[int] = field(default_factory=list)
    window_step_indices: list[int] = field(default_factory=list)

    @property
    def attribution_steps(self) -> int:
        return max(1, self.horizon_steps, self.plan_length)

    @property
    def deadline_step(self) -> int:
        return self.start_step + self.attribution_steps - 1

    def resolve(self, outcome: str, step_index: int) -> None:
        if self.outcome == "pending":
            self.outcome = outcome
            self.outcome_step = step_index


class PolicyRunner:
    """Evaluation harness for NyssaBench suites."""

    def __init__(
        self,
        policy: str | PolicyLike,
        engine: str = "maniskill",
        episodes: int = 10,
        seed: int = 0,
        out: str | Path | None = None,
        max_steps: int | None = None,
        capture_replay: bool = True,
        expert_provider: str | Path | ExpertProvider | None = None,
        enable_recovery: bool = False,
        enable_verifier: bool = False,
        policy_action_horizon: int = 1,
        policy_execution_horizon: int = 1,
        recovery_attribution_horizon: int = DEFAULT_RECOVERY_ATTRIBUTION_HORIZON,
        stressor_config: StressorConfig | dict[str, Any] | str | Path | None = None,
    ) -> None:
        if int(episodes) <= 0:
            raise ValueError("episodes must be a positive integer")
        if int(episodes) > EPISODE_SEED_STRIDE:
            raise ValueError(f"episodes must not exceed {EPISODE_SEED_STRIDE} per task")
        if int(seed) < 0:
            raise ValueError("seed must be a non-negative integer")
        if int(recovery_attribution_horizon) <= 0:
            raise ValueError("recovery_attribution_horizon must be a positive integer")
        self.policy_ref = policy
        self.engine_name = engine
        self.episodes = episodes
        self.seed = seed
        self.out = Path(out) if out else None
        self.max_steps = max_steps
        self.capture_replay = capture_replay
        self.expert_provider_ref = expert_provider
        self.enable_recovery = enable_recovery
        self.enable_verifier = enable_verifier
        self.policy_action_horizon = max(1, int(policy_action_horizon))
        self.policy_execution_horizon = max(1, int(policy_execution_horizon))
        self.recovery_attribution_horizon = int(recovery_attribution_horizon)
        self.stressor_config = _coerce_stressor_config(stressor_config)
        self.episode_results: list[EpisodeResult] = []
        self.run_metadata: dict[str, Any] = {}
        self._failure_mapper = FailureMapper()

    def evaluate(self, suite: Suite) -> Report:
        policy = self._load_policy()
        engine = make_engine(self.engine_name)
        expert_provider = make_expert_provider(self.expert_provider_ref)
        results: list[EpisodeResult] = []
        started_at = utc_now()
        started_perf = time.perf_counter()

        try:
            for task in suite.tasks:
                engine.load_task(task)
                for episode_index in range(self.episodes):
                    episode_seed = _episode_seed(self.seed, episode_index)
                    if hasattr(policy, "reset"):
                        policy.reset(task=task, seed=episode_seed)
                    expert_provider.reset(task=task, seed=episode_seed, engine=engine)
                    results.append(
                        self._run_episode(
                            engine,
                            policy,
                            expert_provider,
                            task,
                            episode_index,
                            episode_seed,
                        )
                    )
        finally:
            engine.close()
            if hasattr(policy, "close"):
                policy.close()
            expert_provider.close()

        self.episode_results = results
        summary = aggregate_episodes(results)
        recovery_outcomes = {
            "format": RECOVERY_OUTCOME_FORMAT,
            "attribution_horizon_steps": self.recovery_attribution_horizon,
            "attribution_criterion": RECOVERY_ATTRIBUTION_CRITERION,
            "attempt_success_rate_denominator": "applied_recovery_attempts",
            "episode_success_rate_denominator": "episodes_with_applied_recovery",
        }
        summary["recovery_outcomes"] = recovery_outcomes
        wall_time_seconds = time.perf_counter() - started_perf
        summary["compute"] = {
            "wall_time_seconds": wall_time_seconds,
            "episodes_per_second": len(results) / wall_time_seconds
            if wall_time_seconds > 0
            else 0.0,
            "training_time_seconds": 0.0,
            "inference_only": True,
        }
        score = score_summary(summary)
        summary["prototype_reliability_score"] = score
        summary["score_kind"] = "prototype_reliability_heuristic"
        summary["sim_to_real_score"] = score
        summary["sim_to_real_score_deprecated"] = True
        declared_task_stressors = {
            task.task_id: summarize_stressor_support(
                task.randomization, self.engine_name
            )
            for task in suite.tasks
        }
        stressor_execution = summarize_stressor_execution(results)
        failure_event_summary = summarize_failure_ledgers(results)
        summary["failure_event_summary"] = failure_event_summary
        summary["stressor_execution"] = stressor_execution
        summary["stressor_support"] = _stressor_support_summary(
            stressor_execution,
            fallback=aggregate_stressor_support(declared_task_stressors),
        )
        summary["task_stressor_support"] = declared_task_stressors
        self.run_metadata = {
            "run_id": make_run_id(suite.suite_id, self._policy_name()),
            "suite_id": suite.suite_id,
            "task_ids": [task.task_id for task in suite.tasks],
            "policy_name": self._policy_name(),
            "engine_name": self.engine_name,
            "episodes_per_task": self.episodes,
            "seed": self.seed,
            "seed_protocol": {
                "format": EPISODE_SEED_FORMAT,
                "run_seed": self.seed,
                "episode_seed_stride": EPISODE_SEED_STRIDE,
                "formula": "run_seed * episode_seed_stride + episode_index",
                "shared_across_tasks": True,
            },
            "started_at": started_at,
            "finished_at": utc_now(),
            "wall_time_seconds": wall_time_seconds,
            "expert_provider": expert_provider.metadata(),
            "recovery_enabled": self.enable_recovery,
            "verifier_enabled": self.enable_verifier,
            "recovery_outcomes": recovery_outcomes,
            "policy_metadata": _policy_metadata(policy),
            "action_sequence": {
                "action_horizon": self.policy_action_horizon,
                "execution_horizon": self.policy_execution_horizon,
                "receding_horizon": self.policy_action_horizon > 1,
            },
            "stressor_config": self.stressor_config.to_dict()
            if self.stressor_config
            else None,
            "stressor_execution": stressor_execution,
            "failure_event_summary": failure_event_summary,
        }
        env_metadata = environment_metadata()
        versions = package_versions()
        git = git_info(REPO_ROOT)
        validation = RunClaimValidator().validate(
            suite=suite,
            engine_name=self.engine_name,
            episodes_per_task=self.episodes,
            episodes=results,
            out_dir=self.out,
            package_versions=versions,
            git_info=git,
            stressor_execution=stressor_execution,
        )
        summary["benchmark_tier"] = validation.benchmark_tier
        summary["public_claim"] = validation.public_claim
        summary["public_claim_validation"] = validation.to_dict()
        report = Report(
            suite_id=suite.suite_id,
            policy=self._policy_name(),
            engine=self.engine_name,
            summary=summary,
            run_dir=self.out,
        )
        if self.out:
            self._write_run_artifacts(
                suite, report, env_metadata=env_metadata, versions=versions, git=git
            )
        return report

    def _run_episode(
        self,
        engine: Any,
        policy: PolicyLike,
        expert_provider: ExpertProvider,
        task: Any,
        episode_index: int,
        seed: int,
    ) -> EpisodeResult:
        stressor_config = self._task_stressor_config(task)
        stressor_pipeline = StressorPipeline(
            stressor_config.stressors,
            context=StressorContext(
                engine_name=self.engine_name,
                task_id=task.task_id,
                observation_mode=_task_mode(task, "obs_mode", "observation"),
                action_mode=_task_mode(task, "control_mode", "action"),
            ),
            episode_seed=seed,
            condition_id=stressor_config.condition_id,
            unsupported_policy=stressor_config.unsupported_policy,
        )
        stressor_pipeline.before_reset(engine)
        observation, reset_info = engine.reset(seed=seed)
        observation = _restore_policy_initial_state(engine, policy, observation)
        stressor_pipeline.after_reset(engine, observation)
        observation = stressor_pipeline.transform_observation(
            observation, step_index=-1
        )
        active_stressors = compact_stressor_context(
            stressor_pipeline.application_context()
        )
        failure_ledger = FailureEventLedger(
            task_id=task.task_id,
            episode_index=episode_index,
            episode_seed=seed,
            engine_name=self.engine_name,
            stressor_context=active_stressors,
        )
        expert_provider_id = expert_provider.metadata().get("provider_id", "unknown")
        engine_event_emitter = failure_ledger.emitter(
            "simulator_state",
            engine.__class__.__name__,
            annotation_source="engine_adapter",
        )
        policy_event_emitter = failure_ledger.emitter(
            "policy_output",
            policy.__class__.__name__,
            annotation_source="policy_adapter",
        )
        verifier_event_emitter = failure_ledger.emitter(
            "verifier_output",
            expert_provider_id,
            annotation_source="expert_provider",
        )
        stressor_event_emitter = failure_ledger.emitter(
            "stressor",
            "stressor_pipeline",
            annotation_source="stressor_pipeline",
        )
        recovery_event_emitter = failure_ledger.emitter(
            "recovery",
            expert_provider_id,
            annotation_source="recovery_runner",
        )
        monitor_event_emitter = failure_ledger.emitter(
            "external_monitor",
            "failure_detector_manager",
            annotation_source="detector_pipeline",
        )
        detector_manager = FailureDetectorManager(
            detectors=build_default_failure_detectors()
        )
        detector_manager.reset(
            task=task,
            engine=engine,
            observation=observation,
            stressor_context=active_stressors,
        )
        stressor_event_emitter.emit_many(
            stressor_condition_drafts(active_stressors),
            default_step=0,
        )
        drain_component_failure_events(
            stressor_pipeline, stressor_event_emitter, default_step=0
        )
        emit_info_failure_events(reset_info, engine_event_emitter, default_step=0)
        drain_component_failure_events(engine, engine_event_emitter, default_step=0)
        steps: list[StepRecord] = []
        frames: list[Any] = []
        last_info: dict[str, Any] = {}
        last_reward = 0.0
        last_terminated = False
        last_truncated = False
        expert_intervention_count = 0
        recovery_attempts: list[_RecoveryAttempt] = []
        active_recovery_attempt: _RecoveryAttempt | None = None
        verifier_rejection_count = 0
        action_assessment_count = 0
        action_rejection_count = 0
        policy_action_chunk_count = 0
        policy_cached_action_count = 0
        recovery_plan_action_count = 0
        recovery_cached_action_count = 0
        pending_actions: list[Any] = []
        pending_action_source: str | None = None
        pending_recovery_attempt_id: int | None = None
        pending_recovery_action_index = 1
        step_limit = self.max_steps or getattr(engine, "max_steps", 1000)
        if self.out and self.capture_replay:
            frame = _safe_render(engine)
            if frame is not None:
                frames.append(frame)

        for step_index in range(step_limit):
            recovery_attempt_id: int | None = None
            recovery_plan_action_index: int | None = None
            verifier_failure_event_id: str | None = None
            if pending_actions:
                action = pending_actions.pop(0)
                action_source = pending_action_source or "pending"
                if action_source == "policy":
                    policy_cached_action_count += 1
                elif action_source == "recovery":
                    recovery_cached_action_count += 1
                    recovery_attempt_id = pending_recovery_attempt_id
                    recovery_plan_action_index = pending_recovery_action_index
                    pending_recovery_action_index += 1
                if not pending_actions:
                    pending_action_source = None
                    pending_recovery_attempt_id = None
                    pending_recovery_action_index = 1
                chunk_size = 0
            else:
                raw_action = policy.act(observation)
                drain_component_failure_events(
                    policy,
                    policy_event_emitter,
                    default_step=step_index,
                )
                action, pending_actions, chunk_size = _split_action_chunk(
                    raw_action,
                    action_horizon=self.policy_action_horizon,
                    execution_horizon=self.policy_execution_horizon,
                )
                action_source = "policy"
                pending_action_source = "policy" if pending_actions else None
                if chunk_size > 1:
                    policy_action_chunk_count += 1
            detector_manager.observe_before_action(
                step_index=step_index,
                observation=observation,
                action=action,
                task=task,
                engine=engine,
                stressor_context=active_stressors,
            )
            expert_info: dict[str, Any] = {
                "expert_provider": expert_provider_id,
                "expert_intervention": False,
                "recovery_attempted": False,
                "recovery_applied": False,
                "recovery_success": False,
                "recovery_attempt_id": recovery_attempt_id,
                "recovery_attribution_attempt_id": active_recovery_attempt.attempt_id
                if active_recovery_attempt is not None
                else None,
                "recovery_plan_action_index": recovery_plan_action_index,
                "recovery_outcome": None,
                "recovery_outcome_step": None,
                "recovery_plan_outcome": None,
                "recovery_plan_success": False,
                "recovery_attribution_start_step": None,
                "recovery_attribution_end_step": None,
                "recovery_attribution_horizon_steps": None,
                "recovery_attribution_criterion": None,
                "verifier_rejected": False,
                "action_assessed": False,
                "action_rejected": False,
                "policy_action_chunk_size": chunk_size,
                "policy_cached_action": chunk_size == 0 and action_source == "policy",
                "recovery_cached_action": action_source == "recovery",
                "action_source": action_source,
            }
            if (
                self.enable_verifier or self.enable_recovery
            ) and action_source != "recovery":
                score = expert_provider.score_action(
                    observation, action, task=task, engine=engine
                )
                score_payload = score.to_dict()
                action_assessment_count += 1
                expert_info["action_assessed"] = True
                expert_info["action_assessment"] = score_payload
                if self.enable_verifier:
                    expert_info["verifier"] = score_payload
                if not score.accepted:
                    action_rejection_count += 1
                    expert_info["action_rejected"] = True
                    if self.enable_verifier:
                        verifier_rejection_count += 1
                        expert_info["verifier_rejected"] = True
                    verifier_event = verifier_event_emitter.emit(
                        verifier_rejection_draft(
                            score_payload,
                            step_index=step_index,
                            recovery_enabled=self.enable_recovery,
                        )
                    )
                    verifier_failure_event_id = verifier_event.event_id
                drain_component_failure_events(
                    expert_provider,
                    verifier_event_emitter,
                    default_step=step_index,
                )
            if self.enable_recovery and expert_info["action_rejected"]:
                if active_recovery_attempt is not None:
                    active_recovery_attempt.resolve(
                        "superseded",
                        max(active_recovery_attempt.start_step, step_index - 1),
                    )
                    active_recovery_attempt = None
                recovery_attempt = _RecoveryAttempt(
                    attempt_id=len(recovery_attempts) + 1,
                    start_step=step_index,
                    horizon_steps=self.recovery_attribution_horizon,
                    event_step_indices=[step_index],
                )
                recovery_attempts.append(recovery_attempt)
                recovery_attempt_id = recovery_attempt.attempt_id
                expert_info["recovery_attempted"] = True
                expert_info["recovery_attempt_id"] = recovery_attempt_id
                expert_info["recovery_attribution_attempt_id"] = recovery_attempt_id
                recovery_plan = expert_provider.recover(
                    state=_safe_get_state(engine, observation=observation),
                    failure=expert_info.get("action_assessment", {}).get("reason"),
                    task=task,
                    engine=engine,
                )
                if recovery_plan:
                    recovery_plan = list(recovery_plan)
                    recovery_attempt.applied = True
                    recovery_attempt.plan_length = len(recovery_plan)
                    active_recovery_attempt = recovery_attempt
                    action = recovery_plan[0]
                    pending_actions = recovery_plan[1:]
                    pending_action_source = "recovery" if pending_actions else None
                    pending_recovery_attempt_id = (
                        recovery_attempt_id if pending_actions else None
                    )
                    pending_recovery_action_index = 1
                    recovery_plan_action_count += len(recovery_plan)
                    if not expert_info["expert_intervention"]:
                        expert_intervention_count += 1
                    expert_info["expert_intervention"] = True
                    expert_info["recovery_applied"] = True
                    expert_info["action_source"] = "recovery"
                    expert_info["recovery_plan_action_index"] = 0
                    expert_info["recovery_plan_length"] = len(recovery_plan)
                    expert_info["recovery_plan_pending_count"] = len(pending_actions)
                    recovery_details = getattr(
                        expert_provider, "last_recovery_details", None
                    )
                    if isinstance(recovery_details, dict):
                        expert_info["recovery_plan"] = recovery_details
                else:
                    recovery_attempt.resolve("not_applied", step_index)
                recovery_event_emitter.emit(
                    recovery_attempt_draft(
                        step_index=step_index,
                        attempt_id=recovery_attempt_id,
                        applied=recovery_attempt.applied,
                        plan_length=recovery_attempt.plan_length,
                        reason=expert_info.get("action_assessment", {}).get("reason"),
                        verifier_event_id=verifier_failure_event_id,
                    )
                )
                drain_component_failure_events(
                    expert_provider,
                    recovery_event_emitter,
                    default_step=step_index,
                )
            if (
                self.enable_verifier
                and expert_info["action_rejected"]
                and not expert_info["recovery_applied"]
            ):
                expert_action = expert_provider.act(
                    observation, task=task, engine=engine
                )
                if expert_action is not None:
                    action = expert_action
                    pending_actions = []
                    pending_action_source = None
                    pending_recovery_attempt_id = None
                    pending_recovery_action_index = 1
                    expert_intervention_count += 1
                    expert_info["expert_intervention"] = True
                    expert_info["action_source"] = "expert"
            drain_component_failure_events(
                expert_provider,
                verifier_event_emitter,
                default_step=step_index,
            )
            action_before_stressors = action
            stressor_pipeline.before_step(engine, step_index=step_index)
            action = stressor_pipeline.transform_action(
                action,
                observation=observation,
                step_index=step_index,
            )
            active_stressors = compact_stressor_context(
                stressor_pipeline.application_context()
            )
            failure_ledger.set_stressor_context(active_stressors)
            next_observation, reward, terminated, truncated, info = engine.step(action)
            last_reward = reward
            last_terminated = terminated
            last_truncated = truncated
            engine_info_events = emit_info_failure_events(
                info,
                engine_event_emitter,
                default_step=step_index,
            )
            if "failure_events" in info:
                info = {
                    **info,
                    "engine_failure_event_ids": [
                        event.event_id for event in engine_info_events
                    ],
                }
                info.pop("failure_events", None)
            drain_component_failure_events(
                engine,
                engine_event_emitter,
                default_step=step_index,
            )
            stressor_pipeline.after_step(engine, info, step_index=step_index)
            next_observation = stressor_pipeline.transform_observation(
                next_observation,
                step_index=step_index + 1,
            )
            active_stressors = compact_stressor_context(
                stressor_pipeline.application_context()
            )
            failure_ledger.set_stressor_context(active_stressors)
            drain_component_failure_events(
                stressor_pipeline,
                stressor_event_emitter,
                default_step=step_index,
            )
            monitor_drafts = detector_manager.observe_after_action(
                step_index=step_index,
                pre_observation=observation,
                post_observation=next_observation,
                action=action,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info,
                task=task,
                engine=engine,
                stressor_context=active_stressors,
            )
            monitor_drafts.extend(
                detector_manager.detect(
                    step_index=step_index,
                    observation=next_observation,
                    action=action,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    info=info,
                    task=task,
                    engine=engine,
                    stressor_context=active_stressors,
                )
            )
            if monitor_drafts:
                monitor_event_emitter.emit_many(
                    detector_manager.drain_draft_payloads(monitor_drafts),
                    default_step=step_index,
                )
            expert_info["action_before_stressors"] = action_before_stressors
            expert_info["stressor_action_modified"] = not _actions_equal(
                action_before_stressors, action
            )
            expert_info["stressor_condition_id"] = stressor_config.condition_id
            expert_info["stressor_applications"] = [
                application.to_dict() for application in stressor_pipeline.applications
            ]
            expert_info["stressor_state"] = stressor_pipeline.get_state()
            expert_info["failure_event_ids"] = [
                event.event_id
                for event in failure_ledger.events
                if event.onset_step == step_index
            ]
            info = {**info, **expert_info}
            if self.out and self.capture_replay:
                frame = _safe_render(engine)
                if frame is not None:
                    frames.append(frame)
            steps.append(
                StepRecord(
                    observation=observation,
                    action=action,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    info=info,
                )
            )
            if recovery_attempt_id is not None:
                attempt = recovery_attempts[recovery_attempt_id - 1]
                if step_index not in attempt.event_step_indices:
                    attempt.event_step_indices.append(step_index)
            if active_recovery_attempt is not None:
                active_recovery_attempt.window_step_indices.append(step_index)
                if bool(info.get("success", False)):
                    active_recovery_attempt.resolve("success", step_index)
                    active_recovery_attempt = None
                elif terminated:
                    active_recovery_attempt.resolve("episode_terminated", step_index)
                    active_recovery_attempt = None
                elif truncated:
                    active_recovery_attempt.resolve("episode_truncated", step_index)
                    active_recovery_attempt = None
                elif step_index >= active_recovery_attempt.deadline_step:
                    active_recovery_attempt.resolve("window_expired", step_index)
                    active_recovery_attempt = None
            observation = next_observation
            last_info = info
            if terminated or truncated:
                break

        if active_recovery_attempt is not None:
            active_recovery_attempt.resolve("episode_ended", max(0, len(steps) - 1))
        _annotate_recovery_outcomes(steps, recovery_attempts)
        success = bool(last_info.get("success", False))
        if not success:
            finalize_events = detector_manager.finalize(
                step_index=max(0, len(steps) - 1),
                final_observation=observation,
                reward=last_reward,
                terminated=last_terminated,
                truncated=last_truncated,
                success=success,
                info=last_info,
                task=task,
                engine=engine,
                stressor_context=active_stressors,
            )
            if finalize_events:
                emitted_final = monitor_event_emitter.emit_many(
                    detector_manager.drain_draft_payloads(finalize_events),
                    default_step=max(0, len(steps) - 1),
                )
                if steps and emitted_final:
                    steps[-1].info.setdefault("failure_event_ids", []).extend(
                        [event.event_id for event in emitted_final]
                    )
        classification = self._failure_mapper.classify(
            last_info,
            task_spec=task,
            step_count=len(steps),
            terminated=bool(steps[-1].terminated) if steps else False,
            truncated=bool(last_info.get("truncated", False))
            or (bool(steps[-1].truncated) if steps else False),
        )
        failure_label = None if success else classification.label
        if failure_label is not None:
            terminal_emitter = failure_ledger.emitter(
                "task_logic" if classification.source == "env" else "legacy_mapper",
                "environment_task"
                if classification.source == "env"
                else "FailureMapper",
                annotation_source="environment"
                if classification.source == "env"
                else "automatic_mapper",
            )
            terminal_event = terminal_emitter.emit(
                terminal_failure_draft(
                    label=failure_label,
                    label_source=classification.source,
                    reason=classification.reason,
                    info=last_info,
                    step_index=max(0, len(steps) - 1),
                )
            )
            if steps:
                steps[-1].info.setdefault("failure_event_ids", []).append(
                    terminal_event.event_id
                )
        failure_ledger_record = failure_ledger.snapshot()
        failure_label = derive_failure_label(
            failure_ledger_record,
            fallback=failure_label,
        )
        recovery_attempt_count = len(recovery_attempts)
        recovery_applied_count = sum(
            1 for attempt in recovery_attempts if attempt.applied
        )
        recovery_success_count = sum(
            1 for attempt in recovery_attempts if attempt.outcome == "success"
        )
        recovery_failure_count = recovery_applied_count - recovery_success_count
        recovery_not_applied_count = recovery_attempt_count - recovery_applied_count
        recovery_episode_attempt_count = int(recovery_attempt_count > 0)
        recovery_episode_applied_count = int(recovery_applied_count > 0)
        recovery_episode_success_count = int(recovery_success_count > 0)
        metrics = {
            "completion_time": float(last_info.get("completion_time", len(steps))),
            "path_efficiency": float(last_info.get("path_efficiency", 0.0)),
            "grasp_success_rate": 1.0
            if bool(last_info.get("grasp_success", False))
            else 0.0,
            "expert_intervention_count": float(expert_intervention_count),
            "expert_intervention_rate": float(expert_intervention_count / len(steps))
            if steps
            else 0.0,
            "recovery_attempt_count": float(recovery_attempt_count),
            "recovery_applied_count": float(recovery_applied_count),
            "recovery_success_count": float(recovery_success_count),
            "recovery_failure_count": float(recovery_failure_count),
            "recovery_not_applied_count": float(recovery_not_applied_count),
            "recovery_success_rate": float(
                recovery_success_count / recovery_applied_count
            )
            if recovery_applied_count
            else 0.0,
            "recovery_episode_attempt_count": float(recovery_episode_attempt_count),
            "recovery_episode_applied_count": float(recovery_episode_applied_count),
            "recovery_episode_success_count": float(recovery_episode_success_count),
            "recovery_episode_success_rate": float(
                recovery_episode_success_count / recovery_episode_applied_count
            )
            if recovery_episode_applied_count
            else 0.0,
            "verifier_rejection_count": float(verifier_rejection_count),
            "verifier_rejection_rate": float(verifier_rejection_count / len(steps))
            if steps
            else 0.0,
            "action_assessment_count": float(action_assessment_count),
            "action_assessment_rate": float(action_assessment_count / len(steps))
            if steps
            else 0.0,
            "action_rejection_count": float(action_rejection_count),
            "action_rejection_rate": float(
                action_rejection_count / action_assessment_count
            )
            if action_assessment_count
            else 0.0,
            "policy_action_chunk_count": float(policy_action_chunk_count),
            "policy_cached_action_count": float(policy_cached_action_count),
            "policy_cached_action_rate": float(policy_cached_action_count / len(steps))
            if steps
            else 0.0,
            "recovery_plan_action_count": float(recovery_plan_action_count),
            "recovery_cached_action_count": float(recovery_cached_action_count),
            "recovery_cached_action_rate": float(
                recovery_cached_action_count / len(steps)
            )
            if steps
            else 0.0,
            "drop_rate": 1.0 if failure_label == "object_slip" else 0.0,
            "stressor_applied_count": float(
                sum(
                    application.status == "applied"
                    for application in stressor_pipeline.applications
                )
            ),
            "stressor_unsupported_count": float(
                sum(
                    application.status == "unsupported"
                    for application in stressor_pipeline.applications
                )
            ),
            **safety_metrics({**last_info, "failure_label": failure_label}),
            **robustness_metrics({**last_info, "failure_label": failure_label}),
        }
        episode = EpisodeResult(
            task_id=task.task_id,
            episode_index=episode_index,
            seed=seed,
            success=success,
            failure_label=failure_label,
            metrics=metrics,
            failure_label_source=None if success else classification.source,
            steps=steps,
            stressor_context=stressor_pipeline.manifest(),
            failure_ledger=failure_ledger_record,
        )
        if self.out and self.capture_replay:
            episode.replay_path = write_episode_video(
                frames, self.out, task.task_id, episode_index
            )
            if episode.replay_path is None:
                raise RuntimeError(
                    "Replay capture was requested, but no video could be written. "
                    "Install and verify the simulator rendering stack, then rerun, "
                    "or pass --no-replay for non-public smoke runs."
                )
        return episode

    def _task_stressor_config(self, task: Any) -> StressorConfig:
        task_stressors = task.randomization.get("stressors", [])
        if not isinstance(task_stressors, list):
            raise ValueError(
                f"Task '{task.task_id}' randomization.stressors must be a list"
            )
        task_specs = tuple(
            StressorSpec.from_dict(dict(item)) for item in task_stressors
        )
        if self.stressor_config is None:
            condition_id = f"task:{task.task_id}" if task_specs else "clean"
            return StressorConfig(condition_id=condition_id, stressors=task_specs)
        return StressorConfig(
            condition_id=self.stressor_config.condition_id,
            stressors=(*task_specs, *self.stressor_config.stressors),
            unsupported_policy=self.stressor_config.unsupported_policy,
        )

    def _load_policy(self) -> PolicyLike:
        if isinstance(self.policy_ref, Policy):
            return self.policy_ref
        if not isinstance(self.policy_ref, str) and callable(
            getattr(self.policy_ref, "act", None)
        ):
            return self.policy_ref
        path = Path(str(self.policy_ref))
        if path.suffix == ".py" or path.exists():
            return load_policy_from_path(path)
        return make_policy(str(self.policy_ref))

    def _policy_name(self) -> str:
        if isinstance(self.policy_ref, str):
            return self.policy_ref
        return self.policy_ref.__class__.__name__

    def _write_run_artifacts(
        self,
        suite: Suite,
        report: Report,
        *,
        env_metadata: dict[str, Any],
        versions: dict[str, Any],
        git: dict[str, Any],
    ) -> None:
        assert self.out is not None
        self.out.mkdir(parents=True, exist_ok=True)
        config = {
            "run_id": self.run_metadata.get("run_id"),
            "suite": suite.to_dict(),
            "policy": self._policy_name(),
            "engine": self.engine_name,
            "episodes_per_task": self.episodes,
            "seed": self.seed,
            "seed_protocol": self.run_metadata.get("seed_protocol"),
            "expert_provider": self.run_metadata.get(
                "expert_provider", {"provider_id": "none"}
            ),
            "recovery_enabled": self.enable_recovery,
            "verifier_enabled": self.enable_verifier,
            "recovery_outcomes": self.run_metadata.get("recovery_outcomes"),
            "action_sequence": self.run_metadata.get(
                "action_sequence",
                {
                    "action_horizon": 1,
                    "execution_horizon": 1,
                    "receding_horizon": False,
                },
            ),
            "stressor_config": self.run_metadata.get("stressor_config"),
            "stressor_execution": self.run_metadata.get("stressor_execution"),
        }
        with (self.out / "run.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.run_metadata, handle, sort_keys=False)
        with (self.out / "config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        write_json(self.out / "environment.json", env_metadata)
        write_json(self.out / "package_versions.json", versions)
        write_json(self.out / "git_info.json", git)
        (self.out / "plots").mkdir(exist_ok=True)
        for episode in self.episode_results:
            write_failure_clip(self.out, episode)
        with (self.out / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(report.summary, handle, indent=2)
        export_metrics_csv(report.summary, self.out / "metrics.csv")
        serialized_episodes = [episode.to_dict() for episode in self.episode_results]
        export_json(serialized_episodes, self.out / "episodes.json")
        export_jsonl(serialized_episodes, self.out / "episodes.jsonl")
        write_replay_manifest(self.episode_results, self.out)
        write_stressor_manifest(
            self.episode_results,
            self.out,
            configured=self.stressor_config.to_dict() if self.stressor_config else None,
        )
        write_failure_ledger_manifest(self.episode_results, self.out)
        write_failure_gallery(self.episode_results, self.out)
        write_recovery_dataset(self.episode_results, self.out)
        write_dataset_manifest(
            out_dir=self.out,
            suite=suite,
            run_metadata=self.run_metadata,
            artifact_names=[
                "episodes.json",
                "episodes.jsonl",
                "metrics.json",
                "metrics.csv",
                "replay_manifest.json",
                "stressor_manifest.json",
                "failure_ledger.json",
                "failure_gallery.html",
                "recovery_dataset/episodes.jsonl",
            ],
        )
        replay_viewer_placeholder(self.out)
        report.save(self.out / "report.html")


def _annotate_recovery_outcomes(
    steps: list[StepRecord], attempts: list[_RecoveryAttempt]
) -> None:
    for attempt in attempts:
        relevant_steps = sorted(
            set(attempt.window_step_indices + attempt.event_step_indices)
        )
        for step_index in relevant_steps:
            if step_index < 0 or step_index >= len(steps):
                continue
            info = steps[step_index].info
            info.update(
                {
                    "recovery_attribution_attempt_id": attempt.attempt_id,
                    "recovery_outcome": attempt.outcome,
                    "recovery_outcome_step": attempt.outcome_step,
                    "recovery_attribution_start_step": attempt.start_step,
                    "recovery_attribution_end_step": attempt.deadline_step,
                    "recovery_attribution_horizon_steps": attempt.attribution_steps,
                    "recovery_attribution_criterion": RECOVERY_ATTRIBUTION_CRITERION,
                    "recovery_success": attempt.outcome == "success",
                }
            )
        for step_index in attempt.event_step_indices:
            if step_index < 0 or step_index >= len(steps):
                continue
            info = steps[step_index].info
            info["recovery_attempt_id"] = attempt.attempt_id
            info["recovery_plan_outcome"] = attempt.outcome if attempt.applied else None
            info["recovery_plan_success"] = (
                attempt.applied and attempt.outcome == "success"
            )


def _episode_seed(run_seed: int, episode_index: int) -> int:
    if run_seed < 0 or episode_index < 0:
        raise ValueError("run_seed and episode_index must be non-negative")
    if episode_index >= EPISODE_SEED_STRIDE:
        raise ValueError(f"episode_index must be smaller than {EPISODE_SEED_STRIDE}")
    return run_seed * EPISODE_SEED_STRIDE + episode_index


def _coerce_stressor_config(
    value: StressorConfig | dict[str, Any] | str | Path | None,
) -> StressorConfig | None:
    if value is None or isinstance(value, StressorConfig):
        return value
    if isinstance(value, dict):
        return StressorConfig.from_dict(value)
    return StressorConfig.load(value)


def _task_mode(task: Any, success_key: str, contract_name: str) -> str | None:
    value = task.success.get(success_key)
    if value is None:
        contract = getattr(task, contract_name, {})
        if isinstance(contract, dict):
            value = contract.get("mode") or contract.get("type")
    return str(value) if value is not None else None


def _actions_equal(left: Any, right: Any) -> bool:
    try:
        import numpy as np

        return bool(np.array_equal(np.asarray(left), np.asarray(right)))
    except (TypeError, ValueError):
        return left == right


def _stressor_support_summary(
    execution: dict[str, Any], *, fallback: dict[str, Any]
) -> dict[str, Any]:
    if not execution.get("requested_stressors"):
        return fallback
    supported_by_task = {
        task_id: list(values.get("applied_stressors", []))
        for task_id, values in execution.get("by_task", {}).items()
        if values.get("applied_stressors")
    }
    unsupported_by_task = {
        task_id: list(values.get("unsupported_stressors", []))
        for task_id, values in execution.get("by_task", {}).items()
        if values.get("unsupported_stressors")
    }
    return {
        "supported_by_task": supported_by_task,
        "unsupported_by_task": unsupported_by_task,
        "unsupported_stressors": list(execution.get("unsupported_stressors", [])),
    }


def _safe_render(engine: Any) -> Any:
    try:
        return engine.render()
    except Exception:
        return None


def _safe_get_state(
    engine: Any, *, observation: dict[str, Any] | None = None
) -> dict[str, Any]:
    try:
        state = engine.get_state()
    except Exception:
        state = {}
    state_dict = state if isinstance(state, dict) else {"state": state}
    if observation is not None:
        state_dict = {**state_dict, "observation": observation}
    return state_dict


def _restore_policy_initial_state(
    engine: Any, policy: PolicyLike, observation: dict[str, Any]
) -> dict[str, Any]:
    initial_state = getattr(policy, "initial_state", None)
    if initial_state is None:
        return observation
    try:
        state = initial_state(observation)
    except TypeError:
        state = initial_state()
    if state is None:
        return observation
    set_state = getattr(engine, "set_state", None)
    if set_state is None:
        raise RuntimeError(
            "Policy requested an initial simulator state, but the selected engine cannot restore state."
        )
    restored_observation = set_state(state)
    return restored_observation if restored_observation is not None else observation


def _policy_metadata(policy: Any) -> dict[str, Any]:
    metadata = getattr(policy, "metadata", None)
    if callable(metadata):
        value = metadata()
        if isinstance(value, dict):
            return value
    return {"policy_class": policy.__class__.__name__}


def _split_action_chunk(
    action: Any,
    *,
    action_horizon: int,
    execution_horizon: int,
) -> tuple[Any, list[Any], int]:
    if action_horizon <= 1:
        return action, [], 1

    sequence = _as_action_sequence(action)
    if not sequence:
        return action, [], 1

    limited = sequence[: max(1, min(action_horizon, execution_horizon, len(sequence)))]
    return limited[0], list(limited[1:]), len(limited)


def _as_action_sequence(action: Any) -> list[Any] | None:
    if hasattr(action, "detach"):
        action = action.detach()
    if hasattr(action, "cpu"):
        action = action.cpu()
    if hasattr(action, "numpy"):
        action = action.numpy()
    try:
        import numpy as np

        array = np.asarray(action)
        if array.ndim >= 2:
            return [array[index] for index in range(array.shape[0])]
    except Exception:
        pass
    if isinstance(action, (list, tuple)) and action:
        first = action[0]
        if isinstance(first, (list, tuple, dict)) or hasattr(first, "tolist"):
            return list(action)
    return None
