from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from nyssa_bench.credibility.protocol import (
    CredibilityEvidence,
    CredibilitySpec,
    EvidenceCategory,
    EvidenceReference,
)
from nyssa_bench.metrics.vector import validate_metric_vector
from nyssa_bench.nep import PolicyContract
from nyssa_bench.simreal import SimRealStudySpec
from nyssa_bench.validity import BenchmarkValidityReport


CORE_BENCHMARK_AUDITS = {
    "shortcut_solvability",
    "train_evaluation_leakage",
    "language_observation_ablations",
    "statistical_precision",
    "paired_design",
    "rank_stability",
    "hidden_test_integrity",
}


@dataclass(frozen=True)
class LoadedEvidence:
    reference: EvidenceReference
    record: CredibilityEvidence
    facts: dict[str, Any]

    def citation(self) -> dict[str, Any]:
        return {
            "evidence_id": self.reference.evidence_id,
            "category": self.reference.category,
            "path": self.reference.path,
            "sha256": self.reference.sha256,
            "artifacts": [
                item.model_dump(mode="json") for item in self.record.artifacts
            ],
        }


def load_evidence(
    spec: CredibilitySpec, root: Path
) -> tuple[list[LoadedEvidence], list[dict[str, Any]]]:
    records: list[LoadedEvidence] = []
    errors: list[dict[str, Any]] = []
    for reference in spec.evidence:
        try:
            path = resolve_within(root, reference.path)
            if sha256_file(path) != reference.sha256:
                raise ValueError("evidence record hash mismatch")
            raw_record = _load_json(path, "evidence record")
            record = CredibilityEvidence.model_validate(raw_record)
            if (
                record.evidence_id != reference.evidence_id
                or record.category != reference.category
            ):
                raise ValueError("evidence record identity or category mismatch")
            artifact_payloads: list[Mapping[str, Any]] = []
            for artifact in record.artifacts:
                artifact_path = resolve_within(path.parent, artifact.path)
                if sha256_file(artifact_path) != artifact.sha256:
                    raise ValueError(f"artifact hash mismatch: {artifact.path}")
                artifact_payloads.append(_load_json(artifact_path, artifact.path))
            facts = _validate_category(record, artifact_payloads)
            records.append(LoadedEvidence(reference, record, facts))
        except (OSError, TypeError, ValueError) as exc:
            errors.append(
                {
                    "evidence_id": reference.evidence_id,
                    "category": reference.category,
                    "path": reference.path,
                    "sha256": reference.sha256,
                    "message": str(exc),
                }
            )
    return records, errors


def matching(
    evidence: Sequence[LoadedEvidence], category: EvidenceCategory
) -> list[LoadedEvidence]:
    return [item for item in evidence if item.record.category == category]


def resolve_within(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"credibility path escapes its root: {value}") from exc
    return path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_category(
    record: CredibilityEvidence, payloads: list[Mapping[str, Any]]
) -> dict[str, Any]:
    if record.category in {"reference_benchmark", "learned_policy_track"}:
        _validated_run(payloads)
        validity = _validated_benchmark_report(
            payloads, required_audits=CORE_BENCHMARK_AUDITS
        )
        if record.category == "reference_benchmark":
            manifest = _one_format(payloads, "nyssa-reference-benchmark-report-v1")
            if (
                manifest.get("benchmark_id") != validity.benchmark_id
                or manifest.get("benchmark_version") != validity.benchmark_version
                or manifest.get("status") != "release_ready"
                or manifest.get("release_ready") is not True
                or not 12 <= int(manifest.get("task_count", 0) or 0) <= 20
                or not isinstance(manifest.get("spec_sha256"), str)
                or len(manifest["spec_sha256"]) != 64
            ):
                raise ValueError(
                    "reference audit is not release-ready or differs from BenchmarkValidity"
                )
            return {
                "benchmark_id": manifest["benchmark_id"],
                "benchmark_version": manifest["benchmark_version"],
                "task_count": manifest["task_count"],
                "oracle_control_policy_ids": list(
                    manifest.get("oracle_control_policy_ids", [])
                ),
            }
        raw = _one_format(payloads, "nyssa-nep-policy-contract-v0.1")
        policy = PolicyContract.model_validate(raw)
        if policy.policy_family.lower() in {
            "integration_control",
            "oracle",
            "random",
            "scripted",
        }:
            raise ValueError("learned policy evidence identifies a control policy")
        if (
            record.metadata.get("benchmark_id") != validity.benchmark_id
            or record.metadata.get("benchmark_version") != validity.benchmark_version
            or record.metadata.get("policy_id") != policy.policy_id
        ):
            raise ValueError(
                "learned policy, result, and BenchmarkValidity identities differ"
            )
        return {
            "policy_id": policy.policy_id,
            "policy_family": policy.policy_family,
        }
    if record.category == "paired_clean_shifted":
        sweep = _one_format(payloads, "nyssa-robustness-sweep-v1")
        points = _mapping_list(sweep.get("points"))
        severities = [point.get("severity") for point in points]
        coverage = _positive_int(sweep.get("paired_episode_coverage"))
        if (
            not severities
            or len(severities) != len(set(severities))
            or 0.0 not in severities
            or not any(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and value > 0
                for value in severities
            )
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0.0 <= float(value) <= 1.0
                for value in severities
            )
            or coverage is None
            or any(point.get("episodes") != coverage for point in points)
        ):
            raise ValueError("robustness sweep lacks paired clean and shifted evidence")
        report = _validated_benchmark_report(
            payloads,
            required_audits={"statistical_precision", "paired_design"},
        )
        return {
            "adequate_power": True,
            "paired_episode_coverage": coverage,
            "validity_report_sha256": report.to_dict()["report_sha256"],
        }
    if record.category == "benchmark_validity":
        report = _validated_benchmark_report(
            payloads, required_audits=CORE_BENCHMARK_AUDITS
        )
        return {
            "benchmark_id": report.benchmark_id,
            "report_sha256": report.to_dict()["report_sha256"],
        }
    if record.category == "simulator_ci":
        return _validate_simulator_ci(record, payloads)
    if record.category == "hardware_calibration":
        return _validate_hardware(record, payloads)
    if record.category == "sim_real_predictive_result":
        return _validate_predictive(payloads)
    raise ValueError(f"unsupported evidence category: {record.category}")


def _validate_simulator_ci(
    record: CredibilityEvidence, payloads: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    engine = record.metadata.get("engine")
    if engine not in {"mujoco", "maniskill"}:
        raise ValueError("simulator CI metadata requires a supported engine")
    smoke = _one_format(payloads, "nyssa-simulator-ci-smoke-v1")
    required = {
        "run.yaml",
        "metrics.json",
        "episodes.json",
        "dataset_manifest.json",
        "stressor_manifest.json",
        "failure_ledger.json",
        "nep_manifest.json",
    }
    if not (
        smoke.get("status") == "passed"
        and smoke.get("engine") == engine
        and _positive_int(smoke.get("episodes")) is not None
        and isinstance(smoke.get("state_restore_capability"), Mapping)
        and smoke["state_restore_capability"].get("supported") is True
        and isinstance(smoke.get("stressor_id"), str)
        and smoke.get("stressor_id")
        and isinstance(smoke.get("required_artifacts"), list)
        and required <= set(smoke["required_artifacts"])
        and isinstance(smoke.get("package_versions"), Mapping)
        and smoke.get("package_versions")
        and isinstance(smoke.get("restore_checks"), list)
        and len(smoke["restore_checks"]) >= 2
        and all(
            isinstance(item, Mapping) and item.get("action_within_bounds") is True
            for item in smoke["restore_checks"]
        )
        and isinstance(smoke.get("episode_seeds"), list)
        and len(smoke["episode_seeds"]) == len(set(smoke["episode_seeds"]))
    ):
        raise ValueError(
            "simulator CI smoke is incomplete or does not match its engine"
        )
    episodes = _positive_int(smoke["episodes"])
    assert episodes is not None
    if len(smoke["episode_seeds"]) != episodes:
        raise ValueError("simulator CI episode seed count does not match episodes")
    replay_count = _nonnegative_int(smoke.get("replay_count"))
    if engine == "maniskill" and not (
        smoke.get("replay_requested") is True
        and replay_count is not None
        and replay_count >= episodes
    ):
        raise ValueError("ManiSkill CI evidence requires per-episode replay output")
    return {"engine": engine, "episodes": episodes}


def _validate_hardware(
    record: CredibilityEvidence, payloads: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    validation = _one_format(payloads, "nyssa-real-evidence-validation-v1")
    if not (
        validation.get("valid") is True
        and validation.get("evidence_ready") is True
        and validation.get("calibration_ready") is True
        and validation.get("governance_ready") is True
        and validation.get("comparison_ready") is True
        and validation.get("claim_ready") is True
        and validation.get("issues") == []
        and isinstance(validation.get("package_identity"), str)
        and validation.get("package_identity")
        and record.metadata.get("prespecified") is True
    ):
        raise ValueError("hardware evidence is not claim-ready and prespecified")
    return {
        "package_identity": validation.get("package_identity"),
        "prespecified": True,
    }


def _validate_predictive(
    payloads: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    raw_spec = _one_format(payloads, "nyssa-sim-real-study-v1")
    study_spec = SimRealStudySpec.model_validate(raw_spec)
    study = _one_format(payloads, "nyssa-sim-real-study-report-v1")
    if not (
        study.get("status") == "complete"
        and study.get("study_id") == study_spec.study_id
        and study.get("study_version") == study_spec.study_version
        and study.get("study_sha256") == study_spec.sha256
        and "incremental_predictive_value" in study_spec.primary_metrics
        and set(study_spec.holdout_shift_ids)
        and study.get("pair_count") == sum(pair.included for pair in study_spec.pairs)
    ):
        raise ValueError(
            "sim-real report does not match its prespecified held-out study"
        )
    metrics = study.get("metrics")
    incremental = (
        metrics.get("incremental_predictive_value")
        if isinstance(metrics, Mapping)
        else None
    )
    if not isinstance(incremental, Mapping) or incremental.get("status") != "available":
        raise ValueError("sim-real study lacks held-out incremental analysis")
    interval = incremental.get("incremental_brier_improvement_ci95")
    if not _finite_interval(interval):
        raise ValueError("incremental analysis lacks a finite confidence interval")
    train_pairs = _positive_int(incremental.get("train_pairs"))
    holdout_pairs = _positive_int(incremental.get("holdout_pairs"))
    holdout_ids = incremental.get("holdout_shift_ids")
    if not (
        train_pairs is not None
        and train_pairs >= 3
        and holdout_pairs is not None
        and holdout_pairs >= 2
        and isinstance(holdout_ids, list)
        and set(holdout_ids) == set(study_spec.holdout_shift_ids)
    ):
        raise ValueError("incremental analysis has incomplete held-out coverage")
    _validated_benchmark_report(
        payloads,
        required_audits={
            "statistical_precision",
            "sim_real_predictive_validity",
        },
    )
    assert isinstance(interval, list)
    return {
        "adequate_power": True,
        "held_out": True,
        "outcome": "positive" if float(interval[0]) > 0.0 else "negative",
        "incremental_brier_improvement_ci95": interval,
    }


def _validated_run(payloads: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    run = _one_format(payloads, "nyssa-run-metrics-v2")
    validation = run.get("public_claim_validation")
    if not (
        isinstance(validation, Mapping)
        and validation.get("status") == "validated"
        and validation.get("public_claim") is True
        and validation.get("failures") == []
    ):
        raise ValueError("result evidence did not pass RunValidity")
    vector = run.get("metric_vector")
    if not isinstance(vector, Mapping):
        raise ValueError("result evidence lacks a metric vector")
    validate_metric_vector(vector)
    return run


def _validated_benchmark_report(
    payloads: Sequence[Mapping[str, Any]],
    *,
    required_audits: set[str],
) -> BenchmarkValidityReport:
    raw = _one_format(payloads, "nyssa-benchmark-validity-report-v1")
    report = BenchmarkValidityReport.from_dict(raw)
    if not report.claim_ready:
        raise ValueError("BenchmarkValidity report is not claim-ready")
    passed = {item.audit_id for item in report.audits if item.status == "passed"}
    missing = sorted(required_audits - passed)
    if missing:
        raise ValueError(
            "BenchmarkValidity report lacks passing required audits: "
            + ", ".join(missing)
        )
    return report


def _one_format(
    payloads: Sequence[Mapping[str, Any]], format_id: str
) -> Mapping[str, Any]:
    matches = [payload for payload in payloads if payload.get("format") == format_id]
    if len(matches) != 1:
        raise ValueError(f"evidence requires exactly one {format_id} artifact")
    return matches[0]


def _mapping_list(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or not all(
        isinstance(item, Mapping) for item in value
    ):
        raise ValueError("artifact field must be a list of mappings")
    return value


def _finite_interval(value: Any) -> bool:
    return bool(
        isinstance(value, list)
        and len(value) == 2
        and all(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(float(item))
            for item in value
        )
        and float(value[0]) <= float(value[1])
    )


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid JSON {label}: {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain a JSON object")
    return dict(value)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")
