from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from nyssa_bench.policy_tracks.protocol import (
    ComputeContract,
    PolicyTrack,
    PolicyTrackRegistry,
)
from nyssa_bench.reference_benchmark import (
    ArtifactReference,
    load_reference_benchmark,
)
from nyssa_bench.regression.evidence import (
    RunEvidence,
    benchmark_validity_available,
    load_run_evidence,
    native_failure_ledger_available,
    replay_evidence_available,
    run_validity_available,
)
from nyssa_bench.regression.protocol import (
    PolicyCheckpointIdentity,
    RunArtifactReference,
)


POLICY_TRACK_REPORT_FORMAT = "nyssa-policy-track-report-v1"


def evaluate_policy_tracks(
    registry: PolicyTrackRegistry, *, root: str | Path
) -> dict[str, Any]:
    base = Path(root).resolve()
    checks = [_reference_check(registry, base)]
    track_reports = []
    for track in registry.tracks:
        report = _evaluate_track(
            track, base, minimum_episodes_per_task=registry.minimum_episodes_per_task
        )
        track_reports.append(report)
        checks.extend(report["checks"])
    statuses = Counter(check["status"] for check in checks)
    learned = [
        report
        for report in track_reports
        if report["role"] in {"learned", "vla"} and report["validated"]
    ]
    families = {report["policy_family"] for report in learned}
    oracle = any(
        report["role"] == "oracle_control" and report["validated"]
        for report in track_reports
    )
    release_ready = bool(
        registry.status == "release"
        and statuses["failed"] == 0
        and statuses["missing"] == 0
        and oracle
        and len(families) >= registry.required_learned_policy_families
    )
    return {
        "format": POLICY_TRACK_REPORT_FORMAT,
        "registry_id": registry.registry_id,
        "registry_version": registry.registry_version,
        "registry_sha256": registry.sha256,
        "declared_status": registry.status,
        "status": (
            "release_ready"
            if release_ready
            else "failed"
            if statuses["failed"]
            else "evidence_missing"
        ),
        "release_ready": release_ready,
        "validated_oracle": oracle,
        "validated_learned_policy_families": sorted(families),
        "tracks": track_reports,
        "checks": checks,
        "status_counts": {
            status: statuses.get(status, 0)
            for status in ("passed", "failed", "missing", "not_applicable")
        },
        "claim_boundary": (
            "Adapter or conformance availability is not a validated policy track. "
            "Promotion requires checkpoint-bound training provenance and paired, "
            "replay-backed clean and shifted result packs."
        ),
    }


def _reference_check(registry: PolicyTrackRegistry, root: Path) -> dict[str, Any]:
    reference = registry.reference_benchmark
    error = _verify(reference, root)
    if error:
        return _check("reference_benchmark", "failed", error, reference)
    try:
        spec = load_reference_benchmark(_resolve(root, reference.path))
        if spec.status != "release":
            return _check(
                "reference_benchmark",
                "missing",
                "reference benchmark is still a candidate",
                reference,
            )
        if not set(registry.benchmark_task_subset) <= {
            task.contract.task_id for task in spec.tasks
        }:
            raise ValueError("policy task subset is outside the reference benchmark")
        split_ids = {track.evaluation_split_id for track in registry.tracks}
        split_hashes = {track.evaluation_split_sha256 for track in registry.tracks}
        split = next((item for item in spec.splits if item.split_id in split_ids), None)
        if split is None or len(split_ids) != 1 or len(split_hashes) != 1:
            raise ValueError("policy evaluation split is not uniquely defined")
        observed_hash = _model_sha256(split.model_dump(mode="json"))
        if split_hashes != {observed_hash}:
            raise ValueError("policy evaluation split hash differs from the benchmark")
    except (OSError, TypeError, ValueError) as exc:
        return _check("reference_benchmark", "failed", str(exc), reference)
    return _check("reference_benchmark", "passed", None, reference)


def _evaluate_track(
    track: PolicyTrack, root: Path, *, minimum_episodes_per_task: int
) -> dict[str, Any]:
    prefix = f"track:{track.track_id}"
    checks = [
        _artifact_check(f"{prefix}:setup", track.setup_document, root),
        _identity_artifact_check(
            f"{prefix}:checkpoint",
            track.checkpoint_artifact,
            track.contract.checkpoint_sha256,
            root,
        ),
        _identity_artifact_check(
            f"{prefix}:preprocessing",
            track.preprocessing_artifact,
            track.contract.preprocessing_sha256,
            root,
        ),
    ]
    if track.role in {"learned", "vla"}:
        checks.append(_training_check(track, root))
    else:
        checks.append(
            _check(
                f"{prefix}:training_provenance",
                "not_applicable",
                "control track has no learned training phase",
            )
        )
    checks.append(_conformance_check(track, root))
    checks.append(_result_check(track, root, minimum_episodes_per_task))
    validated = bool(
        track.status == "validated"
        and all(item["status"] in {"passed", "not_applicable"} for item in checks)
    )
    if not track.required_for_release and track.status == "integration_only":
        for check in checks:
            if check["status"] == "missing":
                check["status"] = "not_applicable"
                check["reason"] = "optional candidate track is not release-blocking"
    return {
        "track_id": track.track_id,
        "role": track.role,
        "declared_status": track.status,
        "required_for_release": track.required_for_release,
        "policy_id": track.contract.policy_id,
        "policy_family": track.contract.policy_family,
        "validated": validated,
        "checks": checks,
    }


def _artifact_check(
    check_id: str, reference: ArtifactReference | None, root: Path
) -> dict[str, Any]:
    if reference is None:
        return _check(check_id, "missing", "artifact reference is absent")
    error = _verify(reference, root)
    return _check(check_id, "failed" if error else "passed", error, reference)


def _identity_artifact_check(
    check_id: str,
    reference: ArtifactReference | None,
    expected_sha256: str,
    root: Path,
) -> dict[str, Any]:
    result = _artifact_check(check_id, reference, root)
    if result["status"] != "passed" or reference is None:
        return result
    if reference.sha256 != expected_sha256:
        return _check(
            check_id,
            "failed",
            "artifact hash differs from the NEP policy contract",
            reference,
        )
    return result


def _training_check(track: PolicyTrack, root: Path) -> dict[str, Any]:
    check_id = f"track:{track.track_id}:training_provenance"
    reference = track.training_provenance
    if reference is None:
        return _check(check_id, "missing", "training provenance is absent")
    error = _verify(reference, root)
    if error:
        return _check(check_id, "failed", error, reference)
    try:
        payload = _load_json(_resolve(root, reference.path))
        if payload.get("format") != "nyssa-policy-training-provenance-v1":
            raise ValueError("unsupported training provenance format")
        if (
            payload.get("policy_id") != track.contract.policy_id
            or payload.get("checkpoint_sha256") != track.contract.checkpoint_sha256
            or payload.get("preprocessing_sha256")
            != track.contract.preprocessing_sha256
        ):
            raise ValueError("training and policy contract identities differ")
        sources = payload.get("training_data")
        if not isinstance(sources, list) or not sources:
            raise ValueError("training provenance lacks source datasets")
        declared = {
            (item.dataset_id, item.dataset_version, item.sha256): set(item.split_ids)
            for item in track.contract.training_data
        }
        observed = {}
        training_episode_seeds = set()
        training_assets = set()
        for source in sources:
            if not isinstance(source, Mapping):
                raise ValueError("training source must be a mapping")
            key = (
                source.get("dataset_id"),
                source.get("dataset_version"),
                source.get("sha256"),
            )
            split_ids = _string_set(source.get("split_ids"), "training split IDs")
            observed[key] = split_ids
            training_episode_seeds.update(
                _integer_set(source.get("episode_seeds"), "training episode seeds")
            )
            training_assets.update(
                _string_set(source.get("asset_ids"), "training asset IDs")
            )
            _string_set(source.get("task_ids"), "training task IDs")
            _string_set(source.get("demonstration_ids"), "training demonstration IDs")
        if any(
            track.evaluation_split_id in split_ids for split_ids in observed.values()
        ):
            raise ValueError("evaluation split occurs in training provenance")
        if training_assets.intersection(track.evaluation_asset_ids):
            raise ValueError("training and evaluation assets overlap")
        if observed != declared:
            raise ValueError("training sources differ from the NEP policy contract")
        action_transform = payload.get("action_transform")
        if not (
            isinstance(action_transform, Mapping)
            and action_transform.get("representation")
            == track.contract.action_representation
            and action_transform.get("lower_bounds")
            == list(track.contract.action_lower_bounds)
            and action_transform.get("upper_bounds")
            == list(track.contract.action_upper_bounds)
        ):
            raise ValueError("training provenance lacks an action transform")
        temporal = payload.get("temporal_context")
        if not (
            isinstance(temporal, Mapping)
            and temporal.get("prediction_horizon") == track.contract.prediction_horizon
            and temporal.get("execution_horizon") == track.contract.execution_horizon
        ):
            raise ValueError(
                "training temporal context differs from the policy contract"
            )
        if not isinstance(payload.get("compute"), Mapping):
            raise ValueError("training compute provenance is absent")
        ComputeContract.model_validate(payload["compute"])
        payload["_training_episode_seeds"] = sorted(training_episode_seeds)
    except (OSError, TypeError, ValueError) as exc:
        return _check(check_id, "failed", str(exc), reference)
    return _check(
        check_id,
        "passed",
        None,
        {
            "reference": reference.model_dump(mode="json"),
            "training_episode_seeds": payload["_training_episode_seeds"],
        },
    )


def _conformance_check(track: PolicyTrack, root: Path) -> dict[str, Any]:
    check_id = f"track:{track.track_id}:conformance"
    if not track.conformance_reports:
        return _check(check_id, "missing", "per-task conformance reports are absent")
    task_ids = set()
    errors = []
    for reference in track.conformance_reports:
        error = _verify(reference, root)
        if error:
            errors.append(error)
            continue
        try:
            payload = _load_json(_resolve(root, reference.path))
            if not (
                payload.get("format") == "nyssa-policy-conformance-report-v1"
                and payload.get("status") == "conformant"
                and payload.get("conformant") is True
                and payload.get("smoke", {}).get("status") == "passed"
                and payload.get("policy_contract")
                == track.contract.model_dump(mode="json")
            ):
                raise ValueError("policy conformance report did not pass its contract")
            task_ids.add(payload.get("task_id"))
        except (OSError, TypeError, ValueError) as exc:
            errors.append(str(exc))
    if task_ids != set(track.evaluation_task_ids):
        errors.append("conformance reports do not cover every evaluation task")
    return _check(
        check_id,
        "failed" if errors else "passed",
        "; ".join(errors) if errors else None,
        [item.model_dump(mode="json") for item in track.conformance_reports],
    )


def _result_check(
    track: PolicyTrack, root: Path, minimum_episodes_per_task: int
) -> dict[str, Any]:
    check_id = f"track:{track.track_id}:paired_results"
    if not track.clean_run_fingerprints or not track.shifted_run_fingerprints:
        return _check(check_id, "missing", "paired clean and shifted runs are absent")
    try:
        clean = [
            _load_run(reference, track, root)
            for reference in track.clean_run_fingerprints
        ]
        shifted = [
            _load_run(reference, track, root)
            for reference in track.shifted_run_fingerprints
        ]
        clean_by_seed = {_run_seed(run): run for run in clean}
        shifted_by_seed = {_run_seed(run): run for run in shifted}
        expected = set(track.evaluation_seeds)
        if set(clean_by_seed) != expected or set(shifted_by_seed) != expected:
            raise ValueError("run seeds do not match the prespecified evaluation seeds")
        training_seeds = _training_episode_seeds(track, root)
        for seed in sorted(expected):
            _validate_run_pair(
                clean_by_seed[seed],
                shifted_by_seed[seed],
                track,
                training_seeds,
                minimum_episodes_per_task,
            )
    except (OSError, TypeError, ValueError) as exc:
        return _check(check_id, "failed", str(exc))
    return _check(
        check_id,
        "passed",
        None,
        {
            "clean": [
                item.model_dump(mode="json") for item in track.clean_run_fingerprints
            ],
            "shifted": [
                item.model_dump(mode="json") for item in track.shifted_run_fingerprints
            ],
        },
    )


def _load_run(
    fingerprint: ArtifactReference, track: PolicyTrack, root: Path
) -> RunEvidence:
    error = _verify(fingerprint, root)
    if error:
        raise ValueError(error)
    path = _resolve(root, fingerprint.path)
    payload = _load_json(path)
    if payload.get("format") != "nyssa-regression-run-fingerprint-v1":
        raise ValueError("unsupported run fingerprint format")
    identity_data = payload.get("policy_identity")
    reference_data = payload.get("run_reference")
    if not isinstance(identity_data, Mapping) or not isinstance(
        reference_data, Mapping
    ):
        raise ValueError("run fingerprint lacks policy or artifact identity")
    identity = PolicyCheckpointIdentity.from_dict(identity_data)
    if (
        identity.policy_name != track.contract.policy_id
        or identity.checkpoint_id != track.contract.checkpoint_id
        or identity.checkpoint_sha256 != track.contract.checkpoint_sha256
        or identity.preprocessing_sha256 != track.contract.preprocessing_sha256
    ):
        raise ValueError("run fingerprint and policy contract identities differ")
    reference = RunArtifactReference.from_dict(reference_data)
    return load_run_evidence(reference, identity, spec_root=path.parent)


def _validate_run_pair(
    clean: RunEvidence,
    shifted: RunEvidence,
    track: PolicyTrack,
    training_episode_seeds: set[int],
    minimum_episodes_per_task: int,
) -> None:
    for run in (clean, shifted):
        if not run_validity_available(run) or not benchmark_validity_available(run):
            raise ValueError(
                "policy run did not pass RunValidity and BenchmarkValidity"
            )
        if {episode.task_id for episode in run.episodes} != set(
            track.evaluation_task_ids
        ):
            raise ValueError("policy run task coverage differs from its track")
        if any(episode.seed in training_episode_seeds for episode in run.episodes):
            raise ValueError("training episode seed leaked into evaluation")
        if not all(
            native_failure_ledger_available(episode) for episode in run.episodes
        ):
            raise ValueError("policy run lacks temporal failure ledgers")
        if not all(replay_evidence_available(run, episode) for episode in run.episodes):
            raise ValueError("policy run lacks content-pinned replay evidence")
        counts = Counter(episode.task_id for episode in run.episodes)
        if any(
            counts[task_id] < minimum_episodes_per_task
            for task_id in track.evaluation_task_ids
        ):
            raise ValueError("policy run is below the per-task episode requirement")
        _validate_success_uncertainty(run)
    clean_keys = {
        (episode.task_id, episode.seed, episode.episode_index)
        for episode in clean.episodes
    }
    shifted_keys = {
        (episode.task_id, episode.seed, episode.episode_index)
        for episode in shifted.episodes
    }
    if clean_keys != shifted_keys:
        raise ValueError("clean and shifted episode keys are not completely paired")
    if clean.metadata.get("stressor_config"):
        raise ValueError("clean policy run unexpectedly contains stressors")
    stressor = shifted.metadata.get("stressor_config")
    if not isinstance(stressor, Mapping) or stressor.get("condition_id") != (
        track.stressor_condition_id
    ):
        raise ValueError("shifted run does not match the track stressor condition")


def _validate_success_uncertainty(run: RunEvidence) -> None:
    values = run.metric_vector.get("values")
    if not isinstance(values, Mapping):
        raise ValueError("policy run lacks metric-vector values")
    available = [
        values.get("clean_success_rate"),
        values.get("shifted_success_rate"),
    ]
    measurements = [
        value
        for value in available
        if isinstance(value, Mapping) and value.get("status") == "available"
    ]
    if len(measurements) != 1:
        raise ValueError(
            "run must expose exactly one clean or shifted success estimate"
        )
    interval = measurements[0].get("ci95")
    if not (
        isinstance(interval, list)
        and len(interval) == 2
        and all(isinstance(value, (int, float)) for value in interval)
    ):
        raise ValueError("policy success estimate lacks a confidence interval")


def _training_episode_seeds(track: PolicyTrack, root: Path) -> set[int]:
    if track.training_provenance is None:
        return set()
    payload = _load_json(_resolve(root, track.training_provenance.path))
    seeds = set()
    for source in payload.get("training_data", []):
        if isinstance(source, Mapping):
            seeds.update(_integer_set(source.get("episode_seeds"), "training seeds"))
    return seeds


def _run_seed(run: RunEvidence) -> int:
    value = run.metadata.get("seed")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("run metadata lacks a valid run seed")
    return value


def _verify(reference: ArtifactReference, root: Path) -> str | None:
    try:
        path = _resolve(root, reference.path)
        if _sha256_file(path) != reference.sha256:
            return "artifact SHA-256 mismatch"
    except (OSError, ValueError) as exc:
        return f"artifact is unavailable: {exc}"
    return None


def _resolve(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"policy track path escapes root: {value}") from exc
    return path


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant
        )
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid policy-track JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError("policy-track artifact must contain a JSON object")
    return dict(value)


def _string_set(value: Any, label: str) -> set[str]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item.strip() for item in value)
    ):
        raise ValueError(f"{label} must be a non-empty string list")
    if len(value) != len(set(value)):
        raise ValueError(f"{label} must be unique")
    return set(value)


def _integer_set(value: Any, label: str) -> set[int]:
    if not isinstance(value, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) and item >= 0
        for item in value
    ):
        raise ValueError(f"{label} must be a non-negative integer list")
    if len(value) != len(set(value)):
        raise ValueError(f"{label} must be unique")
    return set(value)


def _check(
    check_id: str, status: str, reason: str | None, evidence: Any = None
) -> dict[str, Any]:
    if hasattr(evidence, "model_dump"):
        evidence = evidence.model_dump(mode="json")
    if isinstance(evidence, tuple):
        evidence = [
            item.model_dump(mode="json") if hasattr(item, "model_dump") else item
            for item in evidence
        ]
    return {
        "check_id": check_id,
        "status": status,
        "reason": reason,
        "evidence": evidence,
    }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")
