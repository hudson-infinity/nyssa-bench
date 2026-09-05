from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from nyssa_bench.core.episode import EpisodeResult, StepRecord
from nyssa_bench.failures.protocol import FailureLedgerRecord
from nyssa_bench.monitors.protocol import MonitorPredictionRecord
from nyssa_bench.metrics.vector import validate_metric_vector
from nyssa_bench.recovery.protocol import CounterfactualRecoveryRecord
from nyssa_bench.reports.comparison import (
    comparison_contract_hash,
    load_comparison_contract,
)
from nyssa_bench.regression.protocol import (
    REQUIRED_RUN_ARTIFACTS,
    PolicyCheckpointIdentity,
    RunArtifactReference,
)
from nyssa_bench.validity import BenchmarkValidityReport


@dataclass(frozen=True)
class RunEvidence:
    root: Path
    metadata: dict[str, Any]
    manifest: dict[str, Any]
    summary: dict[str, Any]
    episodes: tuple[EpisodeResult, ...]
    artifacts_sha256: dict[str, str]

    @property
    def metric_vector(self) -> dict[str, Any]:
        value = self.summary.get("metric_vector")
        return dict(value) if isinstance(value, Mapping) else {}


def fingerprint_run(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    metadata = _load_mapping(root / "run.yaml")
    policy_metadata = metadata.get("policy_metadata")
    if not isinstance(policy_metadata, Mapping):
        raise ValueError("regression run is missing policy_metadata")
    policy = PolicyCheckpointIdentity(
        policy_name=str(metadata.get("policy_name", "")),
        checkpoint_id=str(policy_metadata.get("checkpoint_id", "")),
        checkpoint_sha256=str(policy_metadata.get("checkpoint_sha256", "")),
        preprocessing_sha256=str(policy_metadata.get("preprocessing_sha256", "")),
    )
    core_hashes = {
        name: file_sha256(root / name)
        for name in sorted(
            {"run.yaml", "dataset_manifest.json", "metrics.json", "episodes.json"}
        )
    }
    reference = RunArtifactReference(
        run_dir=root.as_posix(),
        run_id=str(metadata.get("run_id", "")),
        artifact_binding="pinned",
        artifacts_sha256=core_hashes,
    )
    run = load_run_evidence(reference, policy, spec_root=root.parent)
    artifact_hashes = dict(core_hashes)
    for episode in run.episodes:
        if not episode.replay_path:
            continue
        replay = Path(episode.replay_path)
        path = _resolve_replay_path(root, episode.replay_path)
        if not path.is_file() or path.suffix.lower() != ".mp4":
            raise ValueError(f"regression replay evidence is unavailable: {path}")
        artifact_hashes[replay.as_posix()] = file_sha256(path)
    reference = RunArtifactReference(
        run_dir=root.as_posix(),
        run_id=reference.run_id,
        artifact_binding="pinned",
        artifacts_sha256=artifact_hashes,
    )
    contract = load_comparison_contract(root)
    return {
        "format": "nyssa-regression-run-fingerprint-v1",
        "policy_identity": policy.to_dict(),
        "run_reference": reference.to_dict(),
        "comparison_contract": contract,
        "comparison_contract_sha256": comparison_contract_hash(contract),
        "episode_keys": [
            {
                "task_id": episode.task_id,
                "seed": episode.seed,
                "episode_index": episode.episode_index,
            }
            for episode in sorted(
                run.episodes,
                key=lambda item: (item.task_id, item.seed, item.episode_index),
            )
        ],
        "stressor_config": metadata.get("stressor_config"),
        "run_validity": run.summary.get("public_claim_validation"),
        "benchmark_validity": run.summary.get("benchmark_validity"),
    }


def load_run_evidence(
    reference: RunArtifactReference,
    policy: PolicyCheckpointIdentity,
    *,
    spec_root: Path,
) -> RunEvidence:
    root = _resolve_reference_path(reference.run_dir, spec_root)
    if not root.is_dir():
        raise ValueError(f"regression run directory not found: {root}")
    observed_hashes = {}
    artifacts_to_hash = set(REQUIRED_RUN_ARTIFACTS) | set(
        reference.artifacts_sha256
    )
    for artifact in sorted(artifacts_to_hash):
        path = (root / artifact).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as exc:
            raise ValueError(
                f"regression artifact path escapes its result pack: {artifact}"
            ) from exc
        if not path.is_file():
            raise ValueError(f"regression run artifact not found: {path}")
        observed = file_sha256(path)
        observed_hashes[artifact] = observed
        expected_sha256 = reference.artifacts_sha256.get(artifact)
        if expected_sha256 is not None and observed != expected_sha256:
            raise ValueError(
                f"regression run artifact hash mismatch for {artifact}: "
                f"expected {expected_sha256}, observed {observed}"
            )

    metadata = _load_mapping(root / "run.yaml")
    manifest = _load_mapping(root / "dataset_manifest.json")
    summary = _load_mapping(root / "metrics.json")
    metric_vector = summary.get("metric_vector")
    if not isinstance(metric_vector, Mapping):
        raise ValueError("regression run is missing its metric vector")
    validate_metric_vector(metric_vector)
    if metadata.get("run_id") != reference.run_id:
        raise ValueError(
            f"regression run_id mismatch: expected {reference.run_id}, "
            f"observed {metadata.get('run_id')}"
        )
    _validate_policy_identity(metadata, policy)
    episodes = _load_episodes(root / "episodes.json")
    if not episodes:
        raise ValueError(f"regression run has no episodes: {root}")
    if reference.artifact_binding == "observe_and_record":
        for episode in episodes:
            if not episode.replay_path:
                continue
            replay = _resolve_replay_path(root, episode.replay_path)
            if not replay.is_file() or replay.suffix.lower() != ".mp4":
                raise ValueError(f"regression replay evidence is unavailable: {replay}")
            observed_hashes[Path(episode.replay_path).as_posix()] = file_sha256(
                replay
            )
    return RunEvidence(
        root=root,
        metadata=metadata,
        manifest=manifest,
        summary=summary,
        episodes=episodes,
        artifacts_sha256=observed_hashes,
    )


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replay_evidence_available(run: RunEvidence, episode: EpisodeResult) -> bool:
    if not episode.replay_path:
        return False
    path = Path(episode.replay_path)
    if path.is_absolute():
        return False
    candidate = run.root / path
    try:
        candidate.resolve().relative_to(run.root.resolve())
    except ValueError:
        return False
    artifact_key = path.as_posix()
    return bool(
        artifact_key in run.artifacts_sha256
        and candidate.is_file()
        and candidate.suffix.lower() == ".mp4"
    )


def detector_evidence_available(episode: EpisodeResult) -> bool:
    context = episode.failure_detector_context
    detectors = context.get("detectors") if isinstance(context, Mapping) else None
    if not isinstance(detectors, list) or not detectors:
        return False
    supported = []
    for detector in detectors:
        if not isinstance(detector, Mapping):
            return False
        support = detector.get("support")
        contract = detector.get("contract")
        if (
            not isinstance(support, Mapping)
            or not isinstance(contract, Mapping)
            or contract.get("format") != "nyssa-failure-detector-v1"
            or contract.get("protocol_version") != 1
            or not contract.get("detector_id")
            or not contract.get("detector_version")
        ):
            return False
        supported.append(support.get("status") == "supported")
    return all(supported)


def native_failure_ledger_available(episode: EpisodeResult) -> bool:
    return episode.failure_ledger is not None


def benchmark_validity_available(run: RunEvidence) -> bool:
    value = run.summary.get("benchmark_validity")
    if not isinstance(value, Mapping):
        return False
    try:
        report = BenchmarkValidityReport.from_dict(value)
    except (TypeError, ValueError):
        return False
    return report.status == "validated"


def run_validity_available(run: RunEvidence) -> bool:
    value = run.summary.get("public_claim_validation")
    if not isinstance(value, Mapping):
        return False
    checks = value.get("checks")
    failures = value.get("failures")
    return bool(
        value.get("status") == "validated"
        and value.get("public_claim") is True
        and isinstance(checks, Mapping)
        and checks
        and all(result is True for result in checks.values())
        and failures == []
    )


def _resolve_reference_path(value: str, spec_root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (spec_root / path).resolve()


def _resolve_replay_path(root: Path, value: str) -> Path:
    replay = Path(value)
    if replay.is_absolute():
        raise ValueError("regression replay paths must be relative to the result pack")
    path = (root / replay).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("regression replay path escapes its result pack") from exc
    return path


def _validate_policy_identity(
    metadata: Mapping[str, Any], expected: PolicyCheckpointIdentity
) -> None:
    policy_metadata = metadata.get("policy_metadata")
    if not isinstance(policy_metadata, Mapping):
        raise ValueError("regression run is missing policy_metadata")
    observed = {
        "policy_name": metadata.get("policy_name"),
        "checkpoint_id": policy_metadata.get("checkpoint_id"),
        "checkpoint_sha256": policy_metadata.get("checkpoint_sha256"),
        "preprocessing_sha256": policy_metadata.get("preprocessing_sha256"),
    }
    expected_values = {
        "policy_name": expected.policy_name,
        "checkpoint_id": expected.checkpoint_id,
        "checkpoint_sha256": expected.checkpoint_sha256,
        "preprocessing_sha256": expected.preprocessing_sha256,
    }
    mismatches = [
        key for key in expected_values if observed.get(key) != expected_values[key]
    ]
    if mismatches:
        details = ", ".join(
            f"{key}={observed.get(key)!r} expected {expected_values[key]!r}"
            for key in mismatches
        )
        raise ValueError(f"regression policy identity mismatch: {details}")


def _load_episodes(path: Path) -> tuple[EpisodeResult, ...]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid regression episodes artifact: {path}") from exc
    if not isinstance(raw, list):
        raise ValueError("regression episodes artifact must contain a list")
    episodes = tuple(
        _episode_from_dict(item)
        for item in raw
        if isinstance(item, Mapping)
    )
    if len(episodes) != len(raw):
        raise ValueError("regression episodes must be mappings")
    keys = [
        (episode.task_id, episode.seed, episode.episode_index)
        for episode in episodes
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("regression episodes contain duplicate identities")
    return episodes


def _episode_from_dict(item: Mapping[str, Any]) -> EpisodeResult:
    required = {"task_id", "episode_index", "seed", "success", "metrics", "steps"}
    missing = sorted(required - set(item))
    if missing:
        raise ValueError("regression episode is missing fields: " + ", ".join(missing))
    if not isinstance(item.get("success"), bool):
        raise ValueError("regression episode success must be a boolean")
    metrics_raw = item.get("metrics")
    steps_raw = item.get("steps")
    if not isinstance(metrics_raw, Mapping):
        raise ValueError("regression episode metrics must be a mapping")
    if not isinstance(steps_raw, list) or not all(
        isinstance(step, Mapping) for step in steps_raw
    ):
        raise ValueError("regression episode steps must be a list of mappings")
    metrics = {}
    for key, value in metrics_raw.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"regression episode metric '{key}' must be numeric")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"regression episode metric '{key}' must be finite")
        metrics[str(key)] = numeric
    steps = [_step_from_dict(step) for step in steps_raw]
    ledger_raw = item.get("failure_ledger")
    ledger = (
        FailureLedgerRecord.from_dict(dict(ledger_raw))
        if isinstance(ledger_raw, Mapping)
        else None
    )
    counterfactual_raw = item.get("counterfactual_recovery", [])
    monitor_raw = item.get("failure_monitor_records", [])
    if not isinstance(counterfactual_raw, list) or not all(
        isinstance(value, Mapping) for value in counterfactual_raw
    ):
        raise ValueError("counterfactual recovery records must be mappings")
    if not isinstance(monitor_raw, list) or not all(
        isinstance(value, Mapping) for value in monitor_raw
    ):
        raise ValueError("failure monitor records must be mappings")
    stressor_context = item.get("stressor_context", {})
    detector_context = item.get("failure_detector_context", {})
    monitor_context = item.get("failure_monitor_context", {})
    if not all(
        isinstance(value, Mapping)
        for value in (stressor_context, detector_context, monitor_context)
    ):
        raise ValueError("episode evidence contexts must be mappings")
    failure_label = item.get("failure_label")
    if failure_label is not None and not isinstance(failure_label, str):
        raise ValueError("failure_label must be a string or null")
    return EpisodeResult(
        task_id=str(item["task_id"]),
        episode_index=_integer(item["episode_index"], "episode_index"),
        seed=_integer(item["seed"], "episode seed"),
        success=bool(item["success"]),
        failure_label=failure_label,
        metrics=metrics,
        failure_label_source=str(item["failure_label_source"])
        if item.get("failure_label_source") is not None
        else None,
        steps=steps,
        replay_path=str(item["replay_path"])
        if item.get("replay_path") is not None
        else None,
        failure_clip_path=str(item["failure_clip_path"])
        if item.get("failure_clip_path") is not None
        else None,
        stressor_context=dict(stressor_context),
        failure_detector_context=dict(detector_context),
        failure_ledger=ledger,
        failure_monitor_context=dict(monitor_context),
        failure_monitor_records=[
            MonitorPredictionRecord.from_dict(value) for value in monitor_raw
        ],
        counterfactual_recovery=[
            CounterfactualRecoveryRecord.from_dict(dict(value))
            for value in counterfactual_raw
        ],
    )


def _step_from_dict(value: Mapping[str, Any]) -> StepRecord:
    required = {
        "observation",
        "action",
        "reward",
        "terminated",
        "truncated",
        "info",
    }
    missing = sorted(required - set(value))
    if missing:
        raise ValueError("regression step is missing fields: " + ", ".join(missing))
    observation = value.get("observation")
    info = value.get("info")
    if not isinstance(observation, Mapping) or not isinstance(info, Mapping):
        raise ValueError("regression step observation and info must be mappings")
    if not isinstance(value.get("terminated"), bool) or not isinstance(
        value.get("truncated"), bool
    ):
        raise ValueError("regression step termination flags must be booleans")
    reward = value.get("reward")
    if isinstance(reward, bool) or not isinstance(reward, (int, float)):
        raise ValueError("regression step reward must be numeric")
    reward_value = float(reward)
    if not math.isfinite(reward_value):
        raise ValueError("regression step reward must be finite")
    return StepRecord(
        observation=dict(observation),
        action=value.get("action"),
        reward=reward_value,
        terminated=bool(value["terminated"]),
        truncated=bool(value["truncated"]),
        info=dict(info),
    )


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
        value = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid regression run artifact: {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"regression run artifact must contain a mapping: {path}")
    return dict(value)


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{label} must be an integer")
    return result
