from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nyssa_bench.core.episode import EpisodeResult
from nyssa_bench.core.suite import Suite
from nyssa_bench.core.task import TaskSpec
from nyssa_bench.metrics.vector import sim_real_metrics_are_supported
from nyssa_bench.validity.protocol import BenchmarkValidityReport


PUBLIC_CLAIM_ENGINES = {"maniskill", "mujoco"}
EXPERIMENTAL_ENGINES = {"genesis", "robocasa"}
MIN_PUBLIC_EPISODES_PER_TASK = 100


@dataclass(frozen=True)
class RunClaimValidation:
    public_claim: bool
    benchmark_tier: str
    status: str
    checks: dict[str, bool]
    failures: list[str]
    warnings: list[str]
    benchmark_validity: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "public_claim": self.public_claim,
            "benchmark_tier": self.benchmark_tier,
            "checks": self.checks,
            "failures": self.failures,
            "warnings": self.warnings,
            "benchmark_validity": self.benchmark_validity,
        }


class RunClaimValidator:
    """Conservative gate for publishing benchmark claims."""

    def validate(
        self,
        *,
        suite: Suite,
        engine_name: str,
        episodes_per_task: int,
        episodes: list[EpisodeResult],
        out_dir: str | Path | None,
        package_versions: dict[str, Any] | None = None,
        git_info: dict[str, Any] | None = None,
        stressor_execution: dict[str, Any] | None = None,
        metric_vector: dict[str, Any] | None = None,
        scenario_validation: dict[str, Any] | None = None,
        real_evidence_validation: dict[str, Any] | None = None,
        benchmark_validity: BenchmarkValidityReport | dict[str, Any] | None = None,
    ) -> RunClaimValidation:
        benchmark_validity_payload = _benchmark_validity_payload(benchmark_validity)
        checks = {
            "supported_real_simulator_backend": engine_name in PUBLIC_CLAIM_ENGINES,
            "non_experimental_backend": engine_name not in EXPERIMENTAL_ENGINES,
            "explicit_task_mappings": all(_has_explicit_mapping(task, engine_name) for task in suite.tasks),
            "success_predicates_mapped": all(_has_success_predicate(task) for task in suite.tasks),
            "minimum_episodes_per_task": episodes_per_task >= MIN_PUBLIC_EPISODES_PER_TASK,
            "complete_task_episode_matrix": _has_complete_task_episode_matrix(
                suite.tasks, episodes_per_task, episodes
            ),
            "unique_episode_seeds_per_task": _has_unique_episode_seeds_per_task(episodes),
            "paired_task_seed_matrix": _has_paired_task_seed_matrix(suite.tasks, episodes),
            "episode_evidence": _has_episode_evidence(episodes),
            "replay_video_evidence": _has_replay_video_evidence(episodes, out_dir),
            "diagnosed_failure_labels": _has_diagnosed_failures(episodes),
            "package_versions_present": bool(package_versions),
            "engine_package_version_present": _has_engine_package_version(engine_name, package_versions),
            "git_info_present": bool(git_info),
            "git_commit_present": bool((git_info or {}).get("commit")),
            "git_worktree_clean": (git_info or {}).get("dirty") is False,
            "artifact_directory_present": out_dir is not None,
            "stressor_requests_resolved": not bool(
                (stressor_execution or {}).get("unsupported_stressors")
            ),
            "sim_real_metrics_have_hardware_calibration": sim_real_metrics_are_supported(
                metric_vector
            ),
            "scenario_package_valid": scenario_validation is None
            or bool(scenario_validation.get("valid", False)),
            "scenario_execution_ready": scenario_validation is None
            or bool(scenario_validation.get("execution_ready", False)),
            "scenario_claim_ready": scenario_validation is None
            or bool(scenario_validation.get("claim_ready", False)),
            "real_evidence_claim_ready": real_evidence_validation is None
            or bool(real_evidence_validation.get("claim_ready", False)),
            "benchmark_validity_present": benchmark_validity_payload is not None,
            "benchmark_validity_claim_ready": bool(
                (benchmark_validity_payload or {}).get("claim_ready", False)
            ),
            "benchmark_validity_matches_suite": bool(
                benchmark_validity_payload
                and benchmark_validity_payload.get("benchmark_id") == suite.suite_id
            ),
            "benchmark_validity_identity_present": bool(
                benchmark_validity_payload
                and benchmark_validity_payload.get("spec_sha256")
                and benchmark_validity_payload.get("report_sha256")
            ),
        }
        failures = [name for name, passed in checks.items() if not passed]
        warnings = _warnings(suite.tasks, engine_name)
        public_claim = not failures
        if public_claim:
            tier = "real"
            status = "validated"
        elif engine_name in EXPERIMENTAL_ENGINES:
            tier = "experimental_contract_only"
            status = "not_public"
        elif (benchmark_validity_payload or {}).get("status") == "downgraded":
            tier = "benchmark_validity_downgraded"
            status = "not_public"
        else:
            tier = "prototype"
            status = "not_public"
        return RunClaimValidation(
            public_claim=public_claim,
            benchmark_tier=tier,
            status=status,
            checks=checks,
            failures=failures,
            warnings=warnings,
            benchmark_validity=benchmark_validity_payload,
        )


def _benchmark_validity_payload(
    value: BenchmarkValidityReport | dict[str, Any] | None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, BenchmarkValidityReport):
        return value.to_dict()
    if not isinstance(value, dict):
        raise TypeError("benchmark_validity must be a report or report mapping")
    return BenchmarkValidityReport.from_dict(value).to_dict()


def _has_explicit_mapping(task: TaskSpec, engine_name: str) -> bool:
    engine_env_ids = task.success.get("engine_env_ids", {})
    engine_factory = task.success.get("engine_factory", {})
    return (
        isinstance(engine_env_ids, dict)
        and bool(engine_env_ids.get(engine_name))
        or isinstance(engine_factory, dict)
        and bool(engine_factory.get(engine_name))
    )


def _has_success_predicate(task: TaskSpec) -> bool:
    success = task.success
    predicate_keys = {
        "success_info_keys",
        "success_metric",
        "reward_threshold",
        "return_threshold",
        "min_success_steps",
        "object_lifted",
        "object_inside",
        "object_on_top",
        "ee_at_target",
    }
    return any(key in success for key in predicate_keys)


def _has_episode_evidence(episodes: list[EpisodeResult]) -> bool:
    return bool(episodes) and all(episode.steps for episode in episodes)


def _has_complete_task_episode_matrix(
    tasks: tuple[TaskSpec, ...],
    episodes_per_task: int,
    episodes: list[EpisodeResult],
) -> bool:
    expected = {task.task_id for task in tasks}
    observed = {episode.task_id for episode in episodes}
    if observed != expected:
        return False
    return all(sum(episode.task_id == task_id for episode in episodes) == episodes_per_task for task_id in expected)


def _has_unique_episode_seeds_per_task(episodes: list[EpisodeResult]) -> bool:
    task_ids = {episode.task_id for episode in episodes}
    return all(
        len({episode.seed for episode in episodes if episode.task_id == task_id})
        == sum(episode.task_id == task_id for episode in episodes)
        for task_id in task_ids
    )


def _has_paired_task_seed_matrix(tasks: tuple[TaskSpec, ...], episodes: list[EpisodeResult]) -> bool:
    seed_sets = [
        {episode.seed for episode in episodes if episode.task_id == task.task_id}
        for task in tasks
    ]
    return bool(seed_sets) and bool(seed_sets[0]) and all(seeds == seed_sets[0] for seeds in seed_sets[1:])


def _has_replay_video_evidence(episodes: list[EpisodeResult], out_dir: str | Path | None) -> bool:
    if not episodes or out_dir is None:
        return False
    root = Path(out_dir)
    return all(
        bool(episode.replay_path)
        and Path(str(episode.replay_path)).suffix.lower() == ".mp4"
        and (root / str(episode.replay_path)).is_file()
        for episode in episodes
    )


def _has_engine_package_version(engine_name: str, package_versions: dict[str, Any] | None) -> bool:
    package_name = {"maniskill": "mani-skill", "mujoco": "mujoco"}.get(engine_name)
    if package_name is None:
        return False
    return bool((package_versions or {}).get(package_name))


def _has_diagnosed_failures(episodes: list[EpisodeResult]) -> bool:
    failures = [episode for episode in episodes if not episode.success]
    return all(
        bool(episode.failure_label)
        and episode.failure_label != "unknown_failure"
        and episode.failure_label_source in {"env", "mapper"}
        for episode in failures
    )


def _warnings(tasks: tuple[TaskSpec, ...], engine_name: str) -> list[str]:
    warnings: list[str] = []
    for task in tasks:
        if task.engine != engine_name:
            warnings.append(f"task {task.task_id} declares engine {task.engine}, run used {engine_name}")
    return warnings
