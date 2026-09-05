from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from nyssa_bench.claims import load_claim_evidence, validate_claim_evidence
from nyssa_bench.credibility.evidence import (
    LoadedEvidence,
    load_evidence,
    matching,
    resolve_within,
    sha256_file,
)
from nyssa_bench.credibility.gates import GATE_DEFINITIONS, GATES_BY_ID
from nyssa_bench.credibility.protocol import (
    CheckStatus,
    CredibilitySpec,
    EvidenceCategory,
    GateDefinition,
)


CREDIBILITY_REPORT_FORMAT = "nyssa-phase1-credibility-report-v1"


def evaluate_credibility(
    spec: CredibilitySpec,
    *,
    spec_root: str | Path,
    source_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(spec_root).resolve()
    repository = Path(source_root).resolve() if source_root is not None else root
    matrix_path = resolve_within(root, spec.claim_matrix_path)
    matrix_errors: list[str] = []
    try:
        matrix_hash: str | None = sha256_file(matrix_path)
    except OSError as exc:
        matrix_hash = None
        matrix_errors.append(f"claim evidence matrix is unavailable: {exc}")
    matrix: dict[str, Any] | None = None
    claim_report: dict[str, Any] | None = None
    if matrix_hash is not None and matrix_hash != spec.claim_matrix_sha256:
        matrix_errors.append("claim evidence matrix hash mismatch")
    elif matrix_hash is not None:
        try:
            matrix = load_claim_evidence(matrix_path)
            claim_report = validate_claim_evidence(matrix, repo_root=repository)
        except ValueError as exc:
            matrix_errors.append(str(exc))

    evidence, evidence_errors = load_evidence(spec, root)
    gate_a = _gate_a(matrix, matrix_errors, spec)
    gate_b = _gate_b(evidence, evidence_errors, gate_a)
    gate_c = _gate_c(evidence, evidence_errors, gate_b)
    gates = {"A": gate_a, "B": gate_b, "C": gate_c}
    highest = (
        "predictive_validity"
        if gate_c["status"] == "passed"
        else "reference_benchmark_evidence"
        if gate_b["status"] == "passed"
        else "measurement_core"
        if gate_a["status"] == "passed"
        else "none"
    )
    wording = _select_public_wording(matrix, claim_report, gate_c)
    statuses = Counter(
        check["status"] for gate in gates.values() for check in gate["checks"]
    )
    return {
        "format": CREDIBILITY_REPORT_FORMAT,
        "program_id": spec.program_id,
        "program_version": spec.program_version,
        "claim_matrix_sha256": matrix_hash,
        "claim_evidence_report": claim_report,
        "gate_definitions": [item.model_dump(mode="json") for item in GATE_DEFINITIONS],
        "issue_dependencies": sorted(
            {
                issue_id
                for gate in GATE_DEFINITIONS
                for check in gate.required_checks
                for issue_id in check.issue_ids
            }
        ),
        "gates": gates,
        "check_status_counts": {
            status: statuses.get(status, 0)
            for status in ("passed", "failed", "missing", "not_applicable")
        },
        "highest_completed_gate": highest,
        "phase1_complete": gate_c["status"] == "passed",
        "public_wording": wording,
        "public_wording_claim_id": wording["claim_id"],
        "scope_exclusions": [
            "generated worlds",
            "real-to-sim reconstruction",
            "interpretability method development",
            "policy learning algorithms",
            "hosted product infrastructure",
        ],
        "scope_statement": (
            "Completing NyssaBench Phase 1 does not complete the broader Hudson Labs roadmap."
        ),
        "evidence_records": [
            {"reference": item.citation(), "facts": item.facts} for item in evidence
        ],
        "evidence_errors": evidence_errors,
    }


def _gate_a(
    matrix: Mapping[str, Any] | None,
    errors: list[str],
    spec: CredibilitySpec,
) -> dict[str, Any]:
    claim_reference = [
        {"path": spec.claim_matrix_path, "sha256": spec.claim_matrix_sha256}
    ]
    checks = [
        _check(
            "claim_matrix_integrity",
            "failed" if errors else "passed",
            claim_reference,
            "; ".join(errors) if errors else None,
        )
    ]
    claims = (
        {
            str(item.get("claim_id")): item
            for item in matrix.get("claims", [])
            if isinstance(item, Mapping)
        }
        if matrix
        else {}
    )
    for definition in GATES_BY_ID["A"].required_checks[1:]:
        claim = claims.get(definition.check_id)
        source = claim.get("source_paths", []) if claim else []
        tests = claim.get("test_paths", []) if claim else []
        passed = bool(
            claim
            and claim.get("status") == "implemented"
            and claim.get("evidence_tier")
            in {"source_verified", "result_validated", "predictive_validated"}
            and source
            and tests
            and all(not str(path).endswith(".md") for path in (*source, *tests))
        )
        checks.append(
            _check(
                definition.check_id,
                "passed" if passed else "missing",
                claim_reference if passed else [],
                None if passed else "implemented source and test evidence is missing",
            )
        )
    return _gate(GATES_BY_ID["A"], checks)


def _gate_b(
    evidence: list[LoadedEvidence],
    errors: list[dict[str, Any]],
    gate_a: Mapping[str, Any],
) -> dict[str, Any]:
    reference = matching(evidence, "reference_benchmark")
    learned = matching(evidence, "learned_policy_track")
    paired = matching(evidence, "paired_clean_shifted")
    validity = matching(evidence, "benchmark_validity")
    simulators = matching(evidence, "simulator_ci")
    families = {str(item.facts["policy_family"]) for item in learned}
    checks = [
        _dependency_check("gate_a_dependency", gate_a, "Gate A"),
        _evidence_check("reference_benchmark", reference, errors),
        _evidence_check(
            "oracle_controls",
            [item for item in reference if item.facts.get("oracle_control_policy_ids")],
            errors,
            category="reference_benchmark",
        ),
        _evidence_check(
            "two_learned_policy_families",
            learned if len(families) >= 2 else [],
            errors,
            category="learned_policy_track",
            reason=f"found {len(families)} distinct validated learned policy families",
        ),
        _evidence_check("paired_clean_shifted", paired, errors),
        _evidence_check(
            "adequate_power",
            [item for item in paired if item.facts.get("adequate_power") is True],
            errors,
            category="paired_clean_shifted",
        ),
        _evidence_check("benchmark_validity", validity, errors),
        _evidence_check(
            "mujoco_ci",
            [item for item in simulators if item.facts.get("engine") == "mujoco"],
            errors,
            category="simulator_ci",
        ),
        _evidence_check(
            "maniskill_ci",
            [item for item in simulators if item.facts.get("engine") == "maniskill"],
            errors,
            category="simulator_ci",
        ),
    ]
    return _gate(GATES_BY_ID["B"], checks)


def _gate_c(
    evidence: list[LoadedEvidence],
    errors: list[dict[str, Any]],
    gate_b: Mapping[str, Any],
) -> dict[str, Any]:
    hardware = matching(evidence, "hardware_calibration")
    predictive = matching(evidence, "sim_real_predictive_result")
    positive = [item for item in predictive if item.facts.get("outcome") == "positive"]
    negative = [item for item in predictive if item.facts.get("outcome") == "negative"]
    positive_check, negative_check = _predictive_outcome_checks(
        positive, negative, errors
    )
    checks = [
        _dependency_check("gate_b_dependency", gate_b, "Gate B"),
        _evidence_check("prespecified_hardware_calibration", hardware, errors),
        _evidence_check("held_out_incremental_analysis", predictive, errors),
        _evidence_check(
            "predictive_power",
            [item for item in predictive if item.facts.get("adequate_power") is True],
            errors,
            category="sim_real_predictive_result",
        ),
        positive_check,
        negative_check,
    ]
    return _gate(GATES_BY_ID["C"], checks)


def _predictive_outcome_checks(
    positive: list[LoadedEvidence],
    negative: list[LoadedEvidence],
    errors: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    category: EvidenceCategory = "sim_real_predictive_result"
    if positive:
        return (
            _evidence_check(
                "positive_incremental_result", positive, errors, category=category
            ),
            _check(
                "well_powered_negative_result",
                "not_applicable",
                [],
                "the validated result is positive",
            ),
        )
    if negative:
        return (
            _check(
                "positive_incremental_result",
                "not_applicable",
                [],
                "the validated result is negative",
            ),
            _evidence_check(
                "well_powered_negative_result", negative, errors, category=category
            ),
        )
    return (
        _evidence_check("positive_incremental_result", [], errors, category=category),
        _evidence_check("well_powered_negative_result", [], errors, category=category),
    )


def _dependency_check(
    check_id: str, dependency: Mapping[str, Any], name: str
) -> dict[str, Any]:
    status = dependency.get("status")
    if status == "passed":
        return _check(check_id, "passed", [])
    dependency_status: CheckStatus = "failed" if status == "failed" else "missing"
    return _check(
        check_id,
        dependency_status,
        [],
        f"{name} has not passed",
    )


def _evidence_check(
    check_id: str,
    matches: Sequence[LoadedEvidence],
    errors: Sequence[Mapping[str, Any]],
    *,
    category: EvidenceCategory | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    expected = category or check_id
    relevant_errors = [item for item in errors if item.get("category") == expected]
    if relevant_errors:
        return _check(
            check_id,
            "failed",
            relevant_errors,
            "referenced evidence is invalid",
        )
    if matches:
        return _check(check_id, "passed", [item.citation() for item in matches])
    return _check(
        check_id,
        "missing",
        [],
        reason or f"missing validated {expected} evidence",
    )


def _gate(definition: GateDefinition, checks: list[dict[str, Any]]) -> dict[str, Any]:
    expected = [item.check_id for item in definition.required_checks]
    observed = [item["check_id"] for item in checks]
    if observed != expected:
        raise ValueError(f"Gate {definition.gate_id} checks do not match its contract")
    status: CheckStatus = (
        "failed"
        if any(item["status"] == "failed" for item in checks)
        else "missing"
        if any(item["status"] == "missing" for item in checks)
        else "passed"
    )
    counts = Counter(item["status"] for item in checks)
    return {
        "gate_id": definition.gate_id,
        "name": definition.name,
        "status": status,
        "checks": checks,
        "status_counts": {
            value: counts.get(value, 0)
            for value in ("passed", "failed", "missing", "not_applicable")
        },
    }


def _check(
    check_id: str,
    status: CheckStatus,
    evidence_references: Any,
    reason: str | None = None,
) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "status": status,
        "evidence_references": evidence_references,
        "reason": reason,
    }


def _select_public_wording(
    matrix: Mapping[str, Any] | None,
    claim_report: Mapping[str, Any] | None,
    gate_c: Mapping[str, Any],
) -> dict[str, Any]:
    if matrix is None or claim_report is None:
        return {
            "claim_id": None,
            "wording": None,
            "basis": "No wording is authorized because the claim matrix is invalid.",
        }
    claims = (
        {
            str(item.get("claim_id")): item
            for item in matrix.get("claims", [])
            if isinstance(item, Mapping)
        }
        if matrix
        else {}
    )
    promotion_ready = bool(claim_report.get("promotion_ready"))
    if gate_c.get("status") == "passed" and promotion_ready:
        claim_id = str(claim_report.get("promotion_claim_id"))
        basis = "Gate C passed and the claim matrix authorizes promotion."
    else:
        claim_id = str(claim_report.get("current_public_claim_id"))
        basis = "Stronger wording remains unauthorized by the claim matrix."
    claim = claims.get(claim_id, {})
    return {
        "claim_id": claim_id,
        "wording": claim.get("wording"),
        "basis": basis,
    }
