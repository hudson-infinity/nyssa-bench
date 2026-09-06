from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from nyssa_bench.hardware_study.protocol import HardwareCalibrationStudy
from nyssa_bench.policy_tracks import load_policy_track_registry
from nyssa_bench.real_evidence import RealEvidencePackage, RealEvidenceValidator
from nyssa_bench.reference_benchmark import ArtifactReference, load_reference_benchmark
from nyssa_bench.simreal import load_sim_real_study
from nyssa_bench.validity import BenchmarkValidityReport


HARDWARE_STUDY_REPORT_FORMAT = "nyssa-hardware-calibration-report-v1"


def evaluate_hardware_study(
    study: HardwareCalibrationStudy, *, root: str | Path
) -> dict[str, Any]:
    base = Path(root).resolve()
    checks = [
        _dependency_check(study, base),
        _preregistration_check(study, base),
    ]
    packages, package_check = _real_evidence_check(study, base)
    checks.append(package_check)
    checks.append(_sim_real_check(study, base, packages))
    checks.append(_validity_check(study, base))
    statuses = Counter(item["status"] for item in checks)
    complete = bool(
        study.status == "complete"
        and statuses["failed"] == 0
        and statuses["missing"] == 0
    )
    return {
        "format": HARDWARE_STUDY_REPORT_FORMAT,
        "study_id": study.study_id,
        "study_version": study.study_version,
        "design_sha256": study.design_sha256,
        "declared_status": study.status,
        "status": (
            "claim_ready"
            if complete
            else "failed"
            if statuses["failed"]
            else "evidence_missing"
        ),
        "claim_ready": complete,
        "condition_count": len(study.conditions),
        "planned_trial_count": sum(item.trial_count for item in study.conditions),
        "planned_recovery_trial_count": sum(
            item.recovery_trial_count for item in study.conditions
        ),
        "planned_evidence_package_count": sum(
            item.trial_count + item.recovery_trial_count for item in study.conditions
        ),
        "checks": checks,
        "status_counts": {
            status: statuses.get(status, 0)
            for status in ("passed", "failed", "missing", "not_applicable")
        },
        "claim_boundary": (
            "A draft or preregistered design is not hardware evidence. Predictive "
            "claims require complete real packages, paired analysis, and validity."
        ),
    }


def _dependency_check(study: HardwareCalibrationStudy, root: Path) -> dict[str, Any]:
    references = (study.reference_benchmark, study.policy_track_registry)
    for reference in references:
        error = _verify(reference, root)
        if error:
            return _check("released_dependencies", "failed", error, references)
    try:
        benchmark = load_reference_benchmark(
            _resolve(root, study.reference_benchmark.path)
        )
        tracks = load_policy_track_registry(
            _resolve(root, study.policy_track_registry.path)
        )
        task_contracts = {
            task.contract.task_id: task.contract.model_dump(mode="json")
            for task in benchmark.tasks
        }
        policy_contracts = {
            track.contract.policy_id: track.contract.model_dump(mode="json")
            for track in tracks.tracks
        }
        if any(
            task_contracts.get(task.task_id) != task.model_dump(mode="json")
            for task in study.tasks
        ):
            raise ValueError(
                "hardware task contracts differ from the reference benchmark"
            )
        if any(
            policy_contracts.get(policy.policy_id) != policy.model_dump(mode="json")
            for policy in study.policies
        ):
            raise ValueError("hardware policy contracts differ from the track registry")
        if benchmark.status != "release" or tracks.status != "release":
            return _check(
                "released_dependencies",
                "missing",
                "reference benchmark or policy-track registry is still a candidate",
                references,
            )
    except (OSError, TypeError, ValueError) as exc:
        return _check("released_dependencies", "failed", str(exc), references)
    return _check("released_dependencies", "passed", None, references)


def _preregistration_check(
    study: HardwareCalibrationStudy, root: Path
) -> dict[str, Any]:
    reference = study.evidence.preregistration_receipt
    if reference is None:
        return _check(
            "preregistration",
            "missing",
            "immutable third-party preregistration receipt is absent",
        )
    error = _verify(reference, root)
    if error:
        return _check("preregistration", "failed", error, reference)
    try:
        payload = _load_json(_resolve(root, reference.path))
        if payload.get("format") != "nyssa-preregistration-receipt-v1":
            raise ValueError("unsupported preregistration receipt format")
        if payload.get("study_id") != study.study_id:
            raise ValueError("preregistration study identity differs")
        if payload.get("design_sha256") != study.design_sha256:
            raise ValueError("preregistration design hash differs")
        registered = _datetime(payload.get("registered_at"))
        if registered < study.protocol_authored_at:
            raise ValueError("preregistration predates protocol authorship")
        if registered >= study.first_trial_not_before:
            raise ValueError("preregistration does not precede first permitted trial")
        if (
            not isinstance(payload.get("registry_uri"), str)
            or "://" not in payload["registry_uri"]
        ):
            raise ValueError("preregistration receipt lacks a registry URI")
    except (OSError, TypeError, ValueError) as exc:
        return _check("preregistration", "failed", str(exc), reference)
    return _check("preregistration", "passed", None, reference)


def _real_evidence_check(
    study: HardwareCalibrationStudy, root: Path
) -> tuple[list[RealEvidencePackage], dict[str, Any]]:
    references = study.evidence.real_evidence_packages
    if not references:
        return [], _check(
            "real_evidence",
            "missing",
            "no real-robot trial packages are attached",
        )
    packages = []
    errors = []
    observed: Counter[str] = Counter()
    observed_arms: Counter[tuple[str, str]] = Counter()
    trial_ids = set()
    condition_by_id = {
        condition.condition_id: condition for condition in study.conditions
    }
    for reference in references:
        error = _verify(reference, root)
        if error:
            errors.append(error)
            continue
        try:
            package = RealEvidencePackage.load(_resolve(root, reference.path))
            validation = RealEvidenceValidator().validate(
                package, require_artifacts=True
            )
            if not validation.claim_ready:
                raise ValueError("real evidence package is not claim-ready")
            condition_id = package.metadata.get("hardware_condition_id")
            condition = condition_by_id.get(str(condition_id))
            if condition is None:
                raise ValueError(
                    "real package references an unknown hardware condition"
                )
            identity = package.real_episode.identity
            outcome = package.real_episode.outcome
            if (
                identity.policy_id != condition.policy_id
                or outcome.task_id != condition.task_id
            ):
                raise ValueError(
                    "real package task or policy differs from its condition"
                )
            trial_key = (condition.condition_id, identity.trial_id)
            if trial_key in trial_ids:
                raise ValueError("duplicate hardware condition/trial identity")
            trial_ids.add(trial_key)
            observed[condition.condition_id] += 1
            arm = package.metadata.get("recovery_arm", "standard")
            if arm not in {"standard", "continue", "recovery"}:
                raise ValueError("real package has an invalid recovery arm")
            if condition.recovery_design == "disabled" and arm != "standard":
                raise ValueError("non-recovery condition contains a recovery arm")
            if condition.recovery_design == "matched_trials" and arm == "standard":
                raise ValueError("matched recovery condition lacks branch assignment")
            observed_arms[(condition.condition_id, str(arm))] += 1
            packages.append(package)
        except (OSError, TypeError, ValueError) as exc:
            errors.append(str(exc))
    expected = {
        condition.condition_id: condition.trial_count + condition.recovery_trial_count
        for condition in study.conditions
    }
    if dict(observed) != expected:
        errors.append(
            f"trial coverage differs; expected={expected}, observed={dict(observed)}"
        )
    for condition in study.conditions:
        expected_arms = (
            {"standard": condition.trial_count}
            if condition.recovery_design == "disabled"
            else {
                "continue": condition.trial_count,
                "recovery": condition.recovery_trial_count,
            }
        )
        actual_arms = {
            arm: observed_arms[(condition.condition_id, arm)] for arm in expected_arms
        }
        if actual_arms != expected_arms:
            errors.append(
                f"recovery-arm coverage differs for {condition.condition_id}; "
                f"expected={expected_arms}, observed={actual_arms}"
            )
    return packages, _check(
        "real_evidence",
        "failed" if errors else "passed",
        "; ".join(errors) if errors else None,
        references,
    )


def _sim_real_check(
    study: HardwareCalibrationStudy,
    root: Path,
    packages: list[RealEvidencePackage],
) -> dict[str, Any]:
    spec_ref = study.evidence.sim_real_study_spec
    report_ref = study.evidence.sim_real_study_report
    if spec_ref is None or report_ref is None:
        return _check(
            "sim_real_analysis",
            "missing",
            "paired sim-real study spec or report is absent",
        )
    for reference in (spec_ref, report_ref):
        error = _verify(reference, root)
        if error:
            return _check("sim_real_analysis", "failed", error, (spec_ref, report_ref))
    try:
        spec = load_sim_real_study(_resolve(root, spec_ref.path))
        report = _load_json(_resolve(root, report_ref.path))
        if not (
            report.get("format") == "nyssa-sim-real-study-report-v1"
            and report.get("status") == "complete"
            and report.get("study_id") == spec.study_id
            and report.get("study_sha256") == spec.sha256
        ):
            raise ValueError("sim-real report does not match its frozen study")
        if not set(study.analysis.primary_metrics) <= set(spec.primary_metrics):
            raise ValueError("sim-real study omits preregistered primary analyses")
        package_ids = {package.identity for package in packages}
        pair_ids = {pair.real.package_identity for pair in spec.pairs if pair.included}
        if pair_ids != package_ids:
            raise ValueError("sim-real pairs do not cover the real evidence packages")
        if "recovery_effect" in study.analysis.primary_metrics:
            recovery = report.get("metrics", {}).get("recovery_effect")
            if (
                not isinstance(recovery, Mapping)
                or recovery.get("status") != "available"
            ):
                raise ValueError("matched recovery analysis is unavailable")
    except (OSError, TypeError, ValueError) as exc:
        return _check("sim_real_analysis", "failed", str(exc), (spec_ref, report_ref))
    return _check("sim_real_analysis", "passed", None, (spec_ref, report_ref))


def _validity_check(study: HardwareCalibrationStudy, root: Path) -> dict[str, Any]:
    reference = study.evidence.benchmark_validity_report
    if reference is None:
        return _check(
            "benchmark_validity",
            "missing",
            "hardware BenchmarkValidity report is absent",
        )
    error = _verify(reference, root)
    if error:
        return _check("benchmark_validity", "failed", error, reference)
    try:
        report = BenchmarkValidityReport.from_dict(
            _load_json(_resolve(root, reference.path))
        )
        passed = {audit.audit_id for audit in report.audits if audit.status == "passed"}
        required = {"statistical_precision", "sim_real_predictive_validity"}
        if not report.claim_ready or not required <= passed:
            raise ValueError(
                "hardware validity report lacks predictive and power audits"
            )
    except (OSError, TypeError, ValueError) as exc:
        return _check("benchmark_validity", "failed", str(exc), reference)
    return _check("benchmark_validity", "passed", None, reference)


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
        raise ValueError(f"hardware study path escapes root: {value}") from exc
    return path


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant
        )
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid hardware-study JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError("hardware-study artifact must contain a JSON object")
    return dict(value)


def _datetime(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("receipt timestamp must be a string")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("receipt timestamp must include a timezone")
    return result


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


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")
