from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import yaml

from nyssa_bench.core.task import TaskSpec
from nyssa_bench.nep import PolicyContract
from nyssa_bench.reference_benchmark.protocol import (
    ArtifactReference,
    ReferenceBenchmarkSpec,
    ReferenceTask,
)
from nyssa_bench.validity import BenchmarkValidityReport


REFERENCE_REPORT_FORMAT = "nyssa-reference-benchmark-report-v1"
REQUIRED_VALIDITY_AUDITS = {
    "shortcut_solvability",
    "train_evaluation_leakage",
    "language_observation_ablations",
    "statistical_precision",
    "paired_design",
    "rank_stability",
    "hidden_test_integrity",
}


def evaluate_reference_benchmark(
    spec: ReferenceBenchmarkSpec, *, root: str | Path
) -> dict[str, Any]:
    base = Path(root).resolve()
    checks = []
    for task in spec.tasks:
        checks.append(_task_check(task, base))
        checks.append(
            _check(
                f"task:{task.contract.task_id}:asset_provenance",
                "passed" if task.asset_provenance_status == "verified" else "missing",
                None
                if task.asset_provenance_status == "verified"
                else "asset identities, hashes, and licenses are not verified",
            )
        )
        checks.append(
            _check(
                f"task:{task.contract.task_id}:success_predicate",
                "passed" if task.success_predicate_status == "verified" else "missing",
                None
                if task.success_predicate_status == "verified"
                else "simulator success semantics are not execution-verified",
            )
        )
        checks.append(_solvability_check(task, spec, base))
    for split in spec.splits:
        for dimension in split.dimensions:
            check_id = f"split:{split.split_id}:{dimension.dimension}"
            if dimension.status == "pending":
                checks.append(_check(check_id, "missing", "commitment is pending"))
                continue
            if split.contamination_status == "unknown":
                checks.append(
                    _check(check_id, "missing", "contamination audit is pending")
                )
                continue
            if split.contamination_status == "contaminated":
                checks.append(_check(check_id, "failed", "split is contaminated"))
                continue
            if dimension.public_artifact is not None:
                error = _verify_reference(dimension.public_artifact, base)
                checks.append(
                    _check(
                        check_id,
                        "failed" if error else "passed",
                        error,
                        dimension.public_artifact,
                    )
                )
            else:
                checks.append(
                    _check(
                        check_id,
                        "passed",
                        "protected content commitment only",
                        {
                            "content_sha256": dimension.content_sha256,
                            "item_count": dimension.item_count,
                        },
                    )
                )
    checks.extend(_learned_policy_checks(spec, base))
    statuses = Counter(check["status"] for check in checks)
    evidence_complete = statuses["failed"] == 0 and statuses["missing"] == 0
    return {
        "format": REFERENCE_REPORT_FORMAT,
        "benchmark_id": spec.benchmark_id,
        "benchmark_version": spec.benchmark_version,
        "spec_sha256": spec.sha256,
        "declared_status": spec.status,
        "status": (
            "release_ready"
            if spec.status == "release" and evidence_complete
            else "candidate_complete"
            if evidence_complete
            else "failed"
            if statuses["failed"]
            else "evidence_missing"
        ),
        "release_ready": spec.status == "release" and evidence_complete,
        "task_count": len(spec.tasks),
        "mechanism_coverage": sorted(
            {mechanism for task in spec.tasks for mechanism in task.mechanisms}
        ),
        "oracle_control_policy_ids": _oracle_policy_ids(spec, base),
        "checks": checks,
        "status_counts": {
            status: statuses.get(status, 0)
            for status in ("passed", "failed", "missing", "not_applicable")
        },
        "claim_boundary": (
            "A candidate manifest or passing structural check is not validated "
            "reference-benchmark evidence. Release readiness requires protected "
            "commitments, oracle result packs, and distinct learned policy tracks."
        ),
    }


def _task_check(task: ReferenceTask, root: Path) -> dict[str, Any]:
    check_id = f"task:{task.contract.task_id}:contract"
    error = _verify_reference(task.task_spec, root)
    if error:
        return _check(check_id, "failed", error, task.task_spec)
    path = _resolve(root, task.task_spec.path)
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, Mapping):
            raise ValueError("task YAML must contain a mapping")
        runtime = TaskSpec.from_dict(dict(payload), source_path=path)
        contract = task.contract
        expected_engine = (
            contract.engine_ids[0] if len(contract.engine_ids) == 1 else None
        )
        observed_horizon = runtime.success.get("max_steps")
        observed_modality = runtime.success.get("obs_mode")
        if runtime.task_id != contract.task_id:
            raise ValueError("TaskSpec and TaskContract IDs differ")
        if expected_engine is not None and runtime.engine != expected_engine:
            raise ValueError("TaskSpec and TaskContract engines differ")
        if runtime.robot != contract.robot_id or runtime.scene != contract.scene_id:
            raise ValueError("TaskSpec and TaskContract robot or scene differ")
        if observed_horizon != contract.horizon_steps:
            raise ValueError("TaskSpec and TaskContract horizons differ")
        if observed_modality not in contract.observation_modalities:
            raise ValueError(
                "TaskSpec observation mode is not declared by its contract"
            )
        if runtime.success.get("control_mode") != contract.action_representation:
            raise ValueError("TaskSpec and TaskContract action representations differ")
        if not runtime.success.get("engine_env_ids"):
            raise ValueError("reference TaskSpec lacks an explicit simulator mapping")
        env_ids = runtime.success["engine_env_ids"]
        if not isinstance(env_ids, Mapping) or env_ids.get(runtime.engine) != (
            contract.success_predicate.get("engine_env_id")
        ):
            raise ValueError("TaskSpec and TaskContract simulator mappings differ")
        if not runtime.failure_labels:
            raise ValueError("reference TaskSpec lacks failure labels")
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        return _check(check_id, "failed", str(exc), task.task_spec)
    return _check(
        check_id,
        "passed",
        None,
        {
            "task_spec": task.task_spec.model_dump(mode="json"),
            "task_contract_sha256": _contract_sha256(task),
        },
    )


def _solvability_check(
    task: ReferenceTask, spec: ReferenceBenchmarkSpec, root: Path
) -> dict[str, Any]:
    check_id = f"task:{task.contract.task_id}:solvability"
    reference = task.solvability_evidence
    if reference is None:
        return _check(check_id, "missing", "oracle solvability evidence is absent")
    error = _verify_reference(reference, root)
    if error:
        return _check(check_id, "failed", error, reference)
    try:
        payload = _load_json(_resolve(root, reference.path))
        if payload.get("format") != "nyssa-reference-solvability-v1":
            raise ValueError("unsupported solvability evidence format")
        if payload.get("task_id") != task.contract.task_id:
            raise ValueError("solvability evidence task identity differs")
        if payload.get("task_contract_sha256") != _contract_sha256(task):
            raise ValueError("solvability evidence uses a different task contract")
        if not isinstance(payload.get("oracle_policy_id"), str) or not payload.get(
            "oracle_policy_id"
        ):
            raise ValueError("solvability evidence lacks an oracle identity")
        episodes = _positive_int(payload.get("episodes"))
        success_rate = _rate(payload.get("success_rate"))
        if (
            episodes is None
            or episodes < spec.experimental_design.minimum_episodes_per_condition
        ):
            raise ValueError("solvability evidence is underpowered")
        if (
            success_rate is None
            or success_rate < spec.experimental_design.minimum_oracle_success_rate
        ):
            raise ValueError(
                "oracle did not meet the prespecified solvability threshold"
            )
        _validated_result(payload)
    except (OSError, TypeError, ValueError) as exc:
        return _check(check_id, "failed", str(exc), reference)
    return _check(check_id, "passed", None, reference)


def _learned_policy_checks(
    spec: ReferenceBenchmarkSpec, root: Path
) -> list[dict[str, Any]]:
    required = spec.experimental_design.required_learned_policy_families
    if not spec.learned_policy_evidence:
        return [
            _check(
                "learned_policy_families",
                "missing",
                f"requires {required} distinct learned policy families",
            )
        ]
    families = set()
    evidence = []
    errors = []
    for reference in spec.learned_policy_evidence:
        error = _verify_reference(reference, root)
        if error:
            errors.append(error)
            continue
        try:
            payload = _load_json(_resolve(root, reference.path))
            if payload.get("format") != "nyssa-reference-learned-policy-v1":
                raise ValueError("unsupported learned-policy evidence format")
            contract_data = payload.get("policy_contract")
            if not isinstance(contract_data, Mapping):
                raise ValueError("learned evidence lacks a policy contract")
            contract = PolicyContract.model_validate(contract_data)
            if contract.policy_family in {
                "random",
                "scripted",
                "oracle",
                "integration_control",
            }:
                raise ValueError(
                    "control policy cannot satisfy learned-policy coverage"
                )
            if (
                payload.get("policy_id") != contract.policy_id
                or payload.get("checkpoint_sha256") != contract.checkpoint_sha256
            ):
                raise ValueError(
                    "learned result and policy checkpoint identities differ"
                )
            if set(payload.get("task_ids", [])) != {
                task.contract.task_id for task in spec.tasks
            }:
                raise ValueError("learned-policy evidence does not cover every task")
            _validated_result(payload)
            families.add(contract.policy_family)
            evidence.append(reference.model_dump(mode="json"))
        except (OSError, TypeError, ValueError) as exc:
            errors.append(str(exc))
    if errors:
        return [
            _check("learned_policy_families", "failed", "; ".join(errors), evidence)
        ]
    if len(families) < required:
        return [
            _check(
                "learned_policy_families",
                "missing",
                f"found {len(families)} of {required} distinct families",
                evidence,
            )
        ]
    return [_check("learned_policy_families", "passed", None, evidence)]


def _oracle_policy_ids(spec: ReferenceBenchmarkSpec, root: Path) -> list[str]:
    policy_ids = set()
    for task in spec.tasks:
        reference = task.solvability_evidence
        if reference is None or _verify_reference(reference, root):
            continue
        try:
            payload = _load_json(_resolve(root, reference.path))
        except (OSError, ValueError):
            continue
        policy_id = payload.get("oracle_policy_id")
        if isinstance(policy_id, str) and policy_id.strip():
            policy_ids.add(policy_id)
    return sorted(policy_ids)


def _validated_result(payload: Mapping[str, Any]) -> None:
    run = payload.get("run_validity")
    benchmark = payload.get("benchmark_validity")
    if not (
        isinstance(run, Mapping)
        and run.get("status") == "validated"
        and run.get("public_claim") is True
        and run.get("failures") == []
    ):
        raise ValueError("result did not pass RunValidity")
    if not isinstance(benchmark, Mapping):
        raise ValueError("result lacks BenchmarkValidity")
    report = BenchmarkValidityReport.from_dict(benchmark)
    if not report.claim_ready:
        raise ValueError("result did not pass BenchmarkValidity")
    passed = {audit.audit_id for audit in report.audits if audit.status == "passed"}
    missing = sorted(REQUIRED_VALIDITY_AUDITS - passed)
    if missing:
        raise ValueError(
            "BenchmarkValidity lacks required passing audits: " + ", ".join(missing)
        )


def _verify_reference(reference: ArtifactReference, root: Path) -> str | None:
    try:
        path = _resolve(root, reference.path)
        if _sha256_file(path) != reference.sha256:
            return "artifact SHA-256 mismatch"
    except OSError as exc:
        return f"artifact is unavailable: {exc}"
    return None


def _resolve(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"reference path escapes root: {value}") from exc
    return path


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant
        )
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid JSON evidence: {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError("JSON evidence must contain an object")
    return dict(value)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _contract_sha256(task: ReferenceTask) -> str:
    encoded = json.dumps(
        task.contract.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _positive_int(value: Any) -> int | None:
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and value > 0
        else None
    )


def _rate(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if 0.0 <= result <= 1.0 else None


def _check(
    check_id: str,
    status: str,
    reason: str | None,
    evidence: Any = None,
) -> dict[str, Any]:
    if hasattr(evidence, "model_dump"):
        evidence = evidence.model_dump(mode="json")
    return {
        "check_id": check_id,
        "status": status,
        "reason": reason,
        "evidence": evidence,
    }
