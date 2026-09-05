from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import yaml

from nyssa_bench.metrics.vector import metric_measurement, validate_metric_vector
from nyssa_bench.stress_search.protocol import (
    StressObservation,
    StressProposal,
    StressSearchSpace,
)


def observation_from_run(
    proposal: StressProposal,
    run_dir: str | Path,
    *,
    search_space: StressSearchSpace,
    success_threshold: float,
) -> StressObservation:
    if not math.isfinite(success_threshold) or not 0.0 <= success_threshold <= 1.0:
        raise ValueError("success_threshold must be finite and within [0, 1]")
    run_dir = Path(run_dir)
    try:
        metrics = _load_json(run_dir / "metrics.json")
        episodes = _load_json(run_dir / "episodes.json")
        stressors = _load_json(run_dir / "stressor_manifest.json")
        run = _load_yaml(run_dir / "run.yaml")
    except ValueError as exc:
        return _non_policy(proposal, "invalid", str(exc))
    identity_errors = []
    if run.get("engine_name") != search_space.engine_name:
        identity_errors.append("engine_name")
    if run.get("task_ids") != [search_space.task_id]:
        identity_errors.append("task_ids")
    if _integer(run.get("seed")) != proposal.discovery_seed:
        identity_errors.append("run_seed")
    expected_config = search_space.stressor_config(proposal).to_dict()
    if stressors.get("configured") != expected_config:
        identity_errors.append("stressor_config")
    if identity_errors:
        return _non_policy(
            proposal,
            "invalid",
            "run does not match proposal identity: " + ", ".join(identity_errors),
        )
    if not isinstance(episodes, list):
        return _non_policy(proposal, "invalid", "episodes.json is not a list")
    if any(not isinstance(episode, dict) for episode in episodes):
        return _non_policy(proposal, "invalid", "episodes.json contains non-object records")
    unsupported = stressors.get("summary", {}).get("unsupported_stressors", [])
    if unsupported:
        return _non_policy(
            proposal,
            "unsupported",
            "stressor application was unsupported: " + ", ".join(map(str, unsupported)),
        )
    metric_vector = metrics.get("metric_vector")
    if not isinstance(metric_vector, dict):
        return _non_policy(proposal, "invalid", "metrics.json has no metric vector")
    try:
        validate_metric_vector(metric_vector)
    except ValueError as exc:
        return _non_policy(proposal, "invalid", f"invalid metric vector: {exc}")
    if not episodes:
        return _non_policy(proposal, "invalid", "run contains no episodes")
    if all(_is_censored(episode) for episode in episodes if isinstance(episode, dict)):
        return _non_policy(proposal, "censored", "all run episodes are censored")
    success_rate = _finite(metrics.get("success_rate"))
    if success_rate is None or not 0.0 <= success_rate <= 1.0:
        return _non_policy(proposal, "invalid", "run success_rate is invalid")
    success = success_rate >= success_threshold
    failure_events = tuple(_failure_events(episodes))
    if not success and not failure_events:
        return _non_policy(
            proposal,
            "invalid",
            "failed search outcome has no temporal FailureEvent evidence",
        )
    latency = metric_measurement(metric_vector, "mean_inference_latency_ms")
    recovery = metric_measurement(metric_vector, "counterfactual_recovery_gain")
    return StressObservation(
        proposal_id=proposal.proposal_id,
        status="success" if success else "policy_failure",
        success=success,
        metric_vector=metric_vector,
        failure_events=failure_events,
        safety_events=tuple(_safety_events(episodes)),
        latency_ms=_available_value(latency),
        recovery_gain=_available_value(recovery),
        reason=f"aggregate success_rate={success_rate:.6g}, threshold={success_threshold:.6g}",
        provenance={
            "source": "nyssa_result_pack",
            "source_id": str(run.get("run_id") or run_dir.resolve()),
            "run_dir": run_dir.as_posix(),
            "engine_name": run.get("engine_name"),
            "task_ids": run.get("task_ids"),
            "run_seed": run.get("seed"),
        },
        application_evidence={
            "configured": stressors.get("configured"),
            "summary": stressors.get("summary"),
            "episodes": stressors.get("episodes", []),
        },
    )


def write_stress_observations(
    observations: list[StressObservation] | tuple[StressObservation, ...],
    path: str | Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"observations": [item.to_dict() for item in observations]},
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _non_policy(
    proposal: StressProposal, status: str, reason: str
) -> StressObservation:
    return StressObservation(
        proposal_id=proposal.proposal_id,
        status=status,  # type: ignore[arg-type]
        success=None,
        metric_vector=None,
        reason=reason,
    )


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load stress-search run artifact: {path}") from exc


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot load stress-search run artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"stress-search run metadata must be a mapping: {path}")
    return value


def _failure_events(episodes: list[Any]):
    for episode in episodes:
        if not isinstance(episode, dict):
            continue
        ledger = episode.get("failure_ledger")
        events = ledger.get("events", []) if isinstance(ledger, dict) else []
        for event in events:
            if isinstance(event, dict):
                yield event


def _safety_events(episodes: list[Any]):
    for episode in episodes:
        if not isinstance(episode, dict):
            continue
        for step_index, step in enumerate(episode.get("steps", [])):
            info = step.get("info", {}) if isinstance(step, dict) else {}
            if isinstance(info, dict) and (
                info.get("safety_violation") or info.get("damage_event_count", 0)
            ):
                yield {
                    "episode_index": episode.get("episode_index"),
                    "step_index": step_index,
                    "safety_violation": bool(info.get("safety_violation", False)),
                    "damage_event_count": float(info.get("damage_event_count", 0.0)),
                }


def _is_censored(episode: dict[str, Any]) -> bool:
    steps = episode.get("steps", [])
    if not steps or episode.get("success"):
        return False
    last = steps[-1]
    return isinstance(last, dict) and bool(last.get("truncated"))


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result


def _available_value(measurement: dict[str, Any]) -> float | None:
    return (
        _finite(measurement.get("value"))
        if measurement.get("status") == "available"
        else None
    )
