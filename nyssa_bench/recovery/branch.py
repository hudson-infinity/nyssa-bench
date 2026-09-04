from __future__ import annotations

import copy
import hashlib
import math
from typing import Any, Sequence

from nyssa_bench.metrics.safety import safety_metrics
from nyssa_bench.recovery.protocol import (
    BranchOutcome,
    BranchKind,
    BranchPoint,
    BranchStep,
    CounterfactualRecoveryRecord,
)
from nyssa_bench.recovery.state import (
    BranchSnapshot,
    reseed_branch_streams,
    state_sha256,
)


class CounterfactualBranchRunner:
    def __init__(
        self,
        *,
        repeats: int,
        horizon_steps: int,
        include_oracle: bool = False,
    ) -> None:
        if repeats <= 0:
            raise ValueError("counterfactual repeats must be positive")
        if horizon_steps <= 0:
            raise ValueError("counterfactual horizon_steps must be positive")
        self.repeats = int(repeats)
        self.horizon_steps = int(horizon_steps)
        self.include_oracle = bool(include_oracle)

    def evaluate_recovery(
        self,
        *,
        engine: Any,
        policy: Any,
        expert: Any,
        stressors: Any,
        task: Any,
        observation: dict[str, Any],
        task_id: str,
        episode_index: int,
        episode_seed: int,
        step_index: int,
        recovery_attempt_id: int,
        continuation_actions: Sequence[Any],
        recovery_actions: Sequence[Any],
        trigger_reason: str | None,
        trigger_event_id: str | None,
    ) -> CounterfactualRecoveryRecord:
        branch_point_id = (
            f"{task_id}:episode-{episode_index}:step-{step_index}:"
            f"recovery-{recovery_attempt_id}"
        )
        baseline = BranchSnapshot.capture(
            observation=observation,
            engine=engine,
            policy=policy,
            expert=expert,
            stressors=stressors,
            require_expert=self.include_oracle,
        )
        if baseline.restoration_grade == "unsupported":
            return CounterfactualRecoveryRecord(
                branch_point=BranchPoint(
                    branch_point_id=branch_point_id,
                    task_id=task_id,
                    episode_index=episode_index,
                    episode_seed=episode_seed,
                    step_index=step_index,
                    recovery_attempt_id=recovery_attempt_id,
                    requested_repeats=self.repeats,
                    requested_branches=self._requested_branches,
                    trigger_kind="recovery_decision",
                    trigger_reason=trigger_reason,
                    trigger_event_id=trigger_event_id,
                    snapshot_sha256=None,
                    restoration_grade="unsupported",
                    restore_capabilities=baseline.capabilities,
                    matched_randomness=False,
                    repeat_seed_strategy="not_executed",
                    reseeded_components=(),
                    strongest_causal_claim_eligible=False,
                    unsupported_reason=baseline.unsupported_reason,
                )
            )

        outcomes: list[BranchOutcome] = []
        reseeded_components: set[str] = set()
        try:
            for repeat_index in range(self.repeats):
                baseline.restore(
                    engine=engine,
                    policy=policy,
                    expert=expert,
                    stressors=stressors,
                )
                branch_seed = _branch_seed(
                    episode_seed=episode_seed,
                    step_index=step_index,
                    repeat_index=repeat_index,
                )
                reseeded_components.update(
                    reseed_branch_streams(
                        seed=branch_seed,
                        engine=engine,
                        policy=policy,
                        expert=expert,
                        stressors=stressors,
                        include_expert=self.include_oracle,
                    )
                )
                trial = BranchSnapshot.capture(
                    observation=observation,
                    engine=engine,
                    policy=policy,
                    expert=expert,
                    stressors=stressors,
                    require_expert=self.include_oracle,
                )
                if trial.restoration_grade == "unsupported":
                    raise RuntimeError(
                        "branch state became non-restorable after repeat seeding: "
                        f"{trial.unsupported_reason}"
                    )
                for branch_kind, initial_actions in (
                    ("continue", continuation_actions),
                    ("recovery", recovery_actions),
                ):
                    trial.restore(
                        engine=engine,
                        policy=policy,
                        expert=expert,
                        stressors=stressors,
                    )
                    outcomes.append(
                        self._execute_branch(
                            branch_point_id=branch_point_id,
                            branch_kind=branch_kind,
                            repeat_index=repeat_index,
                            branch_seed=branch_seed,
                            matched_rng_sha256=trial.randomness_sha256,
                            initial_actions=initial_actions,
                            start_step_index=step_index,
                            observation=trial.observation,
                            engine=engine,
                            policy=policy,
                            expert=expert,
                            stressors=stressors,
                            task=task,
                        )
                    )
                    _discard_branch_events(engine, policy, expert, stressors)
                if self.include_oracle:
                    trial.restore(
                        engine=engine,
                        policy=policy,
                        expert=expert,
                        stressors=stressors,
                    )
                    outcomes.append(
                        self._execute_branch(
                            branch_point_id=branch_point_id,
                            branch_kind="oracle",
                            repeat_index=repeat_index,
                            branch_seed=branch_seed,
                            matched_rng_sha256=trial.randomness_sha256,
                            initial_actions=(),
                            start_step_index=step_index,
                            observation=trial.observation,
                            engine=engine,
                            policy=policy,
                            expert=expert,
                            stressors=stressors,
                            task=task,
                        )
                    )
                    _discard_branch_events(engine, policy, expert, stressors)
        finally:
            baseline.restore(
                engine=engine,
                policy=policy,
                expert=expert,
                stressors=stressors,
            )

        core_outcomes = [
            outcome
            for outcome in outcomes
            if outcome.branch_kind in {"continue", "recovery"}
        ]
        exact_outcomes = len(core_outcomes) == 2 * self.repeats and all(
            outcome.status == "completed" for outcome in core_outcomes
        )
        strongest_eligible = bool(
            baseline.restoration_grade == "exact"
            and baseline.matched_randomness
            and exact_outcomes
        )
        return CounterfactualRecoveryRecord(
            branch_point=BranchPoint(
                branch_point_id=branch_point_id,
                task_id=task_id,
                episode_index=episode_index,
                episode_seed=episode_seed,
                step_index=step_index,
                recovery_attempt_id=recovery_attempt_id,
                requested_repeats=self.repeats,
                requested_branches=self._requested_branches,
                trigger_kind="recovery_decision",
                trigger_reason=trigger_reason,
                trigger_event_id=trigger_event_id,
                snapshot_sha256=baseline.snapshot_sha256,
                restoration_grade=baseline.restoration_grade,
                restore_capabilities=baseline.capabilities,
                matched_randomness=baseline.matched_randomness,
                repeat_seed_strategy="sha256_episode_step_repeat_v1",
                reseeded_components=tuple(sorted(reseeded_components)),
                strongest_causal_claim_eligible=strongest_eligible,
                unsupported_reason=None,
            ),
            outcomes=tuple(outcomes),
        )

    @property
    def _requested_branches(self) -> tuple[BranchKind, ...]:
        return (
            ("continue", "recovery", "oracle")
            if self.include_oracle
            else ("continue", "recovery")
        )

    def _execute_branch(
        self,
        *,
        branch_point_id: str,
        branch_kind: str,
        repeat_index: int,
        branch_seed: int,
        matched_rng_sha256: str,
        initial_actions: Sequence[Any],
        start_step_index: int,
        observation: dict[str, Any],
        engine: Any,
        policy: Any,
        expert: Any,
        stressors: Any,
        task: Any,
    ) -> BranchOutcome:
        queued_actions = copy.deepcopy(list(initial_actions))
        initial_action_count = len(queued_actions)
        branch_steps: list[BranchStep] = []
        total_reward = 0.0
        terminated = False
        truncated = False
        success = False
        terminal_reason = "horizon_exhausted"
        try:
            for offset in range(self.horizon_steps):
                absolute_step = start_step_index + offset
                if queued_actions:
                    action = queued_actions.pop(0)
                elif branch_kind == "oracle":
                    action = expert.act(observation, task=task, engine=engine)
                else:
                    action = policy.act(observation)
                if action is None:
                    raise RuntimeError(
                        f"{branch_kind} branch produced no action at offset {offset}"
                    )
                stressors.before_step(engine, step_index=absolute_step)
                executed_action = stressors.transform_action(
                    action,
                    observation=observation,
                    step_index=absolute_step,
                )
                next_observation, reward, terminated, truncated, info = engine.step(
                    executed_action
                )
                info = dict(info)
                stressors.after_step(engine, info, step_index=absolute_step)
                next_observation = stressors.transform_observation(
                    next_observation,
                    step_index=absolute_step + 1,
                )
                reward = float(reward)
                if not math.isfinite(reward):
                    raise ValueError("branch reward must be finite")
                total_reward += reward
                success = bool(info.get("success", False))
                safety = safety_metrics(info)
                branch_steps.append(
                    BranchStep(
                        offset=offset,
                        action=executed_action,
                        reward=reward,
                        terminated=bool(terminated),
                        truncated=bool(truncated),
                        success=success,
                        safety_violation=bool(safety.get("safety_violation_rate", 0.0)),
                        damage_event_count=_non_negative_float(
                            info.get("damage_event_count", 0.0)
                        ),
                    )
                )
                observation = next_observation
                if success:
                    terminal_reason = "task_success"
                    break
                if terminated:
                    terminal_reason = "terminated"
                    break
                if truncated:
                    terminal_reason = "truncated"
                    break
        except Exception as exc:
            return BranchOutcome(
                branch_point_id=branch_point_id,
                branch_kind=branch_kind,  # type: ignore[arg-type]
                repeat_index=repeat_index,
                branch_seed=branch_seed,
                status="error",
                success=False,
                terminated=bool(terminated),
                truncated=bool(truncated),
                total_reward=total_reward,
                terminal_reason="execution_error",
                initial_action_count=initial_action_count,
                trajectory_sha256=None,
                matched_rng_sha256=matched_rng_sha256,
                steps=tuple(branch_steps),
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
        trajectory_payload = [step.to_dict() for step in branch_steps]
        return BranchOutcome(
            branch_point_id=branch_point_id,
            branch_kind=branch_kind,  # type: ignore[arg-type]
            repeat_index=repeat_index,
            branch_seed=branch_seed,
            status="completed",
            success=success,
            terminated=bool(terminated),
            truncated=bool(truncated),
            total_reward=total_reward,
            terminal_reason=terminal_reason,
            initial_action_count=initial_action_count,
            trajectory_sha256=state_sha256(trajectory_payload),
            matched_rng_sha256=matched_rng_sha256,
            steps=tuple(branch_steps),
        )


def _branch_seed(*, episode_seed: int, step_index: int, repeat_index: int) -> int:
    payload = (
        f"nyssa-counterfactual-repeat-seed-v1:{episode_seed}:"
        f"{step_index}:{repeat_index}"
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def _discard_branch_events(*components: Any) -> None:
    for component in components:
        drain = getattr(component, "drain_failure_events", None)
        if callable(drain):
            drain()


def _non_negative_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, result) if math.isfinite(result) else 0.0
