from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from nyssa_bench.validity.protocol import (
    AuditResult,
    BenchmarkValidityReport,
    BenchmarkValiditySpec,
)


SHORTCUT_AUDIT = "shortcut_solvability"
LEAKAGE_AUDIT = "train_evaluation_leakage"
ABLATION_AUDIT = "language_observation_ablations"
STATISTICS_AUDIT = "statistical_precision"
PAIRING_AUDIT = "paired_design"
RANK_STABILITY_AUDIT = "rank_stability"
HIDDEN_TEST_AUDIT = "hidden_test_integrity"
SIM_REAL_AUDIT = "sim_real_predictive_validity"

DEFAULT_REQUIRED_AUDITS = (
    SHORTCUT_AUDIT,
    LEAKAGE_AUDIT,
    ABLATION_AUDIT,
    STATISTICS_AUDIT,
    PAIRING_AUDIT,
    RANK_STABILITY_AUDIT,
    HIDDEN_TEST_AUDIT,
    SIM_REAL_AUDIT,
)


class BenchmarkValidityEvaluator:
    def __init__(self) -> None:
        self._auditors: dict[str, Callable[[Mapping[str, Any]], AuditResult]] = {
            SHORTCUT_AUDIT: audit_shortcut_solvability,
            LEAKAGE_AUDIT: audit_train_evaluation_leakage,
            ABLATION_AUDIT: audit_language_observation_ablations,
            STATISTICS_AUDIT: audit_statistical_precision,
            PAIRING_AUDIT: audit_paired_design,
            RANK_STABILITY_AUDIT: audit_rank_stability,
            HIDDEN_TEST_AUDIT: audit_hidden_test_integrity,
            SIM_REAL_AUDIT: audit_sim_real_predictive_validity,
        }

    def evaluate(self, spec: BenchmarkValiditySpec) -> BenchmarkValidityReport:
        audits = []
        effective_required = list(spec.required_audits)
        if spec.claim_tier in {"public_simulation", "sim_real_predictive"}:
            effective_required.extend(
                audit_id
                for audit_id in DEFAULT_REQUIRED_AUDITS
                if audit_id not in effective_required
            )
        for audit_id in effective_required:
            auditor = self._auditors.get(audit_id)
            inputs = spec.audit_inputs.get(audit_id)
            if auditor is None:
                audits.append(_missing(audit_id, "unknown", {}, "No auditor is registered."))
            elif inputs is None:
                audits.append(
                    _missing(
                        audit_id,
                        _category(audit_id),
                        {},
                        "Add the required audit_inputs entry and rerun validation.",
                    )
                )
            elif not isinstance(inputs, Mapping):
                audits.append(
                    _missing(
                        audit_id,
                        _category(audit_id),
                        {},
                        "Audit inputs must be a mapping.",
                    )
                )
            else:
                if (
                    audit_id == SIM_REAL_AUDIT
                    and spec.claim_tier == "sim_real_predictive"
                    and inputs.get("required") is not True
                ):
                    audits.append(
                        _missing(
                            audit_id,
                            _category(audit_id),
                            dict(inputs),
                            "Set required: true and attach the predictive hardware study.",
                        )
                    )
                    continue
                try:
                    audits.append(auditor(inputs))
                except (TypeError, ValueError, OverflowError) as exc:
                    audits.append(
                        _missing(
                            audit_id,
                            _category(audit_id),
                            dict(inputs),
                            "Correct the malformed audit input and rerun validation.",
                            evidence={
                                "input_error": type(exc).__name__,
                                "message": str(exc),
                            },
                        )
                    )
        return BenchmarkValidityReport(
            benchmark_id=spec.benchmark_id,
            benchmark_version=spec.benchmark_version,
            claim_tier=spec.claim_tier,
            spec_sha256=spec.sha256,
            audits=tuple(audits),
            metadata={
                "required_audits": effective_required,
                "spec_required_audits": list(spec.required_audits),
            },
        )


def audit_shortcut_solvability(inputs: Mapping[str, Any]) -> AuditResult:
    baselines = _mapping_list(inputs.get("baselines"))
    threshold = _finite(inputs.get("max_trivial_success_rate", 0.05))
    trivial = [item for item in baselines if item.get("kind") == "trivial"]
    if threshold is None or not 0.0 <= threshold <= 1.0 or not trivial:
        return _missing(
            SHORTCUT_AUDIT,
            "construct_validity",
            dict(inputs),
            "Provide a finite threshold and at least one trivial baseline result.",
        )
    invalid = []
    for item in trivial:
        rate = _finite(item.get("success_rate"))
        episodes = _positive_int(item.get("episodes"))
        if rate is None or not 0.0 <= rate <= 1.0 or episodes is None:
            invalid.append(str(item.get("policy_id", "unknown")))
    if invalid:
        return _missing(
            SHORTCUT_AUDIT,
            "construct_validity",
            dict(inputs),
            "Record valid success rates and episode denominators for every trivial baseline.",
            evidence={"invalid_baselines": invalid},
        )
    violations = [
        {
            "policy_id": item.get("policy_id"),
            "success_rate": float(item["success_rate"]),
        }
        for item in trivial
        if float(item["success_rate"]) > threshold
    ]
    return _result(
        SHORTCUT_AUDIT,
        "construct_validity",
        not violations,
        dict(inputs),
        {
            "threshold": threshold,
            "evaluated_baselines": len(trivial),
            "violations": violations,
        },
        "Investigate task shortcuts and tighten success predicates or held-out conditions.",
        "No trivial baseline exceeded the shortcut threshold."
        if not violations
        else "One or more trivial baselines solve the claimed task.",
    )


def audit_train_evaluation_leakage(inputs: Mapping[str, Any]) -> AuditResult:
    training = inputs.get("training")
    evaluation = inputs.get("evaluation")
    dimensions = ("seeds", "assets", "tasks", "demonstrations", "language")
    if not isinstance(training, Mapping) or not isinstance(evaluation, Mapping):
        return _missing(
            LEAKAGE_AUDIT,
            "data_validity",
            dict(inputs),
            "Provide training and evaluation identity mappings.",
        )
    allowed = set(_string_list(inputs.get("allowed_overlap_dimensions", [])))
    unknown_allowed = sorted(allowed - set(dimensions))
    missing_dimensions = [
        name for name in dimensions if name not in training or name not in evaluation
    ]
    invalid_dimensions = [
        name
        for name in dimensions
        if name in training
        and name in evaluation
        and (
            not isinstance(training.get(name), (list, tuple))
            or not isinstance(evaluation.get(name), (list, tuple))
        )
    ]
    if missing_dimensions or invalid_dimensions or unknown_allowed:
        return _missing(
            LEAKAGE_AUDIT,
            "data_validity",
            dict(inputs),
            "Declare every identity dimension, using an empty list when it is not used.",
            evidence={
                "missing_dimensions": missing_dimensions,
                "invalid_dimensions": invalid_dimensions,
                "unknown_allowed_dimensions": unknown_allowed,
            },
        )
    overlaps = {}
    for dimension in dimensions:
        train_values = set(_string_list(training.get(dimension)))
        eval_values = set(_string_list(evaluation.get(dimension)))
        overlap = sorted(train_values & eval_values)
        if overlap and dimension not in allowed:
            overlaps[dimension] = overlap
    return _result(
        LEAKAGE_AUDIT,
        "data_validity",
        not overlaps,
        dict(inputs),
        {"overlaps": overlaps, "allowed_overlap_dimensions": sorted(allowed)},
        "Remove overlapping identities or explicitly justify a non-held-out claim tier.",
        "No undeclared train/evaluation identity overlap was found."
        if not overlaps
        else "Training and evaluation identities overlap.",
    )


def audit_language_observation_ablations(inputs: Mapping[str, Any]) -> AuditResult:
    full = _finite(inputs.get("full_success_rate"))
    language = _finite(inputs.get("language_ablated_success_rate"))
    observation = _finite(inputs.get("observation_ablated_success_rate"))
    retained_limit = _finite(inputs.get("max_retained_fraction", 0.8))
    values = (full, language, observation, retained_limit)
    if any(value is None or not 0.0 <= value <= 1.0 for value in values):
        return _missing(
            ABLATION_AUDIT,
            "construct_validity",
            dict(inputs),
            "Provide finite full, language-ablated, and observation-ablated success rates.",
        )
    assert full is not None and language is not None and observation is not None
    assert retained_limit is not None
    if full <= 0.0:
        return _result(
            ABLATION_AUDIT,
            "construct_validity",
            False,
            dict(inputs),
            {"reason": "full policy has zero success"},
            "Establish non-zero clean solvability before interpreting shortcut ablations.",
            "The reference policy does not solve the unablated benchmark.",
        )
    retained = {
        "language": language / full,
        "observation": observation / full,
    }
    violations = {key: value for key, value in retained.items() if value > retained_limit}
    return _result(
        ABLATION_AUDIT,
        "construct_validity",
        not violations,
        dict(inputs),
        {"retained_fraction": retained, "violations": violations},
        "Inspect leakage or task shortcuts that permit high ablated performance.",
        "Ablated policies remain below the retained-performance threshold."
        if not violations
        else "Ablated inputs retain too much task performance.",
    )


def audit_statistical_precision(inputs: Mapping[str, Any]) -> AuditResult:
    estimates = _mapping_list(inputs.get("estimates"))
    minimum = _positive_int(inputs.get("min_sample_size"))
    max_width = _finite(inputs.get("max_ci95_width"))
    if not estimates or minimum is None or max_width is None or max_width <= 0:
        return _missing(
            STATISTICS_AUDIT,
            "statistical_validity",
            dict(inputs),
            "Provide estimates, a positive minimum sample size, and maximum CI width.",
        )
    failures = []
    for estimate in estimates:
        sample = _positive_int(estimate.get("sample_size"))
        interval = _interval(estimate.get("ci95"))
        reasons = []
        if sample is None or sample < minimum:
            reasons.append("underpowered")
        if interval is None or interval[1] - interval[0] > max_width:
            reasons.append("interval_too_wide_or_missing")
        if reasons:
            failures.append({"metric_id": estimate.get("metric_id"), "reasons": reasons})
    return _result(
        STATISTICS_AUDIT,
        "statistical_validity",
        not failures,
        dict(inputs),
        {"failed_estimates": failures, "estimate_count": len(estimates)},
        "Increase paired trials or revise the prespecified precision requirement.",
        "All estimates meet sample-size and interval-width requirements."
        if not failures
        else "One or more estimates are underpowered or imprecise.",
    )


def audit_paired_design(inputs: Mapping[str, Any]) -> AuditResult:
    comparison = inputs.get("comparison")
    coverage = inputs.get("coverage")
    if not isinstance(comparison, Mapping) or not isinstance(coverage, Mapping):
        return _missing(
            PAIRING_AUDIT,
            "comparison_validity",
            dict(inputs),
            "Attach comparison compatibility and pairwise coverage evidence.",
        )
    failures = []
    if comparison.get("comparable") is not True or comparison.get("mismatches"):
        failures.append("comparison_contract_mismatch")
    if not _sha256(comparison.get("comparison_contract_sha256")):
        failures.append("comparison_contract_identity_missing")
    if coverage.get("complete") is not True:
        failures.append("incomplete_pair_coverage")
    joint_coverage = _finite(coverage.get("joint_coverage"))
    if joint_coverage is None or not math.isclose(joint_coverage, 1.0):
        failures.append("joint_coverage_not_complete")
    counts = {
        key: _nonnegative_int(coverage.get(key, 0))
        for key in (
            "unmatched_a_count",
            "unmatched_b_count",
            "duplicate_a_count",
            "duplicate_b_count",
        )
    }
    if any(value is None for value in counts.values()):
        return _missing(
            PAIRING_AUDIT,
            "comparison_validity",
            dict(inputs),
            "Pairing counts must be non-negative integers.",
            evidence={"invalid_counts": counts},
        )
    if counts["unmatched_a_count"] or counts["unmatched_b_count"]:
        failures.append("unmatched_episode_keys")
    if counts["duplicate_a_count"] or counts["duplicate_b_count"]:
        failures.append("duplicate_episode_keys")
    return _result(
        PAIRING_AUDIT,
        "comparison_validity",
        not failures,
        dict(inputs),
        {"failures": failures, "joint_coverage": joint_coverage},
        "Regenerate comparison-compatible runs with complete duplicate-free pairing.",
        "Comparison contracts and episode pairing are complete."
        if not failures
        else "Comparison compatibility or paired coverage failed.",
    )


def audit_rank_stability(inputs: Mapping[str, Any]) -> AuditResult:
    rankings = inputs.get("rankings")
    ranking_ids = _string_list(inputs.get("ranking_ids"))
    threshold = _finite(inputs.get("min_pairwise_agreement", 0.8))
    if (
        not isinstance(rankings, list)
        or len(rankings) < 2
        or threshold is None
        or not 0.0 <= threshold <= 1.0
        or len(ranking_ids) != len(rankings or [])
        or len(set(ranking_ids)) != len(ranking_ids)
    ):
        return _missing(
            RANK_STABILITY_AUDIT,
            "statistical_validity",
            dict(inputs),
            "Provide at least two complete rankings and an agreement threshold.",
        )
    normalized = [_string_list(ranking) for ranking in rankings]
    policy_set = set(normalized[0]) if normalized else set()
    if not policy_set or any(set(ranking) != policy_set or len(ranking) != len(policy_set) for ranking in normalized):
        return _missing(
            RANK_STABILITY_AUDIT,
            "statistical_validity",
            dict(inputs),
            "Every ranking must contain each policy exactly once.",
        )
    agreements = []
    reference = _pair_order(normalized[0])
    for ranking in normalized[1:]:
        observed = _pair_order(ranking)
        agreements.append(sum(reference[pair] == observed[pair] for pair in reference) / len(reference) if reference else 1.0)
    minimum_observed = min(agreements)
    return _result(
        RANK_STABILITY_AUDIT,
        "statistical_validity",
        minimum_observed >= threshold,
        dict(inputs),
        {
            "reference_ranking": ranking_ids[0],
            "agreements": [
                {"ranking_id": ranking_id, "agreement": agreement}
                for ranking_id, agreement in zip(ranking_ids[1:], agreements)
            ],
            "minimum_observed": minimum_observed,
        },
        "Increase seeds/tasks or report rankings by condition instead of one unstable aggregate.",
        "Policy ordering is stable across declared aggregations."
        if minimum_observed >= threshold
        else "Policy ordering is unstable across declared aggregations.",
        failure_impact="downgrade",
    )


def audit_hidden_test_integrity(inputs: Mapping[str, Any]) -> AuditResult:
    splits = _mapping_list(inputs.get("splits"))
    hidden = [split for split in splits if split.get("partition") == "hidden_test"]
    if not hidden:
        return _missing(
            HIDDEN_TEST_AUDIT,
            "data_validity",
            dict(inputs),
            "Declare at least one hidden_test split commitment.",
        )
    failures = []
    hashes: dict[str, list[dict[str, Any]]] = {}
    for split in splits:
        content_hash = split.get("content_sha256")
        if isinstance(content_hash, str):
            hashes.setdefault(content_hash, []).append(split)
    for split in hidden:
        reasons = []
        if split.get("protected") is not True:
            reasons.append("not_protected")
        if split.get("contents_published") is not False:
            reasons.append("contents_disclosed_or_unknown")
        if split.get("contamination_status") != "clean":
            reasons.append("contamination_not_clean")
        if not _sha256(split.get("content_sha256")):
            reasons.append("invalid_commitment")
        if not split.get("evaluator_id") or split.get("evaluator_id") == split.get("producer_id"):
            reasons.append("evaluator_not_independent")
        collisions = [
            item.get("split_id")
            for item in hashes.get(str(split.get("content_sha256")), [])
            if item.get("split_id") != split.get("split_id")
        ]
        if collisions:
            reasons.append("content_commitment_collision")
        if reasons:
            failures.append(
                {
                    "split_id": split.get("split_id"),
                    "reasons": reasons,
                    "colliding_split_ids": collisions,
                }
            )
    return _result(
        HIDDEN_TEST_AUDIT,
        "data_validity",
        not failures,
        dict(inputs),
        {"hidden_split_count": len(hidden), "failures": failures},
        "Protect hidden contents, publish only commitments, and use an independent evaluator.",
        "Hidden-test commitments and evaluator separation are valid."
        if not failures
        else "Hidden-test integrity requirements failed.",
    )


def audit_sim_real_predictive_validity(inputs: Mapping[str, Any]) -> AuditResult:
    required = inputs.get("required") is True
    if inputs.get("hardware_available") is not True:
        if required:
            return _missing(
                SIM_REAL_AUDIT,
                "predictive_validity",
                dict(inputs),
                "Attach a validated hardware calibration study for this claim tier.",
            )
        return AuditResult(
            audit_id=SIM_REAL_AUDIT,
            category="predictive_validity",
            status="not_applicable",
            severity="info",
            inputs=dict(inputs),
            evidence={"hardware_available": False},
            remediation="Run the preregistered hardware track before making sim-real claims.",
            claim_impact="none",
            summary="No sim-real claim was requested and no hardware evidence is attached.",
        )
    study = inputs.get("study")
    metrics = study.get("metrics") if isinstance(study, Mapping) else None
    required_metrics = (
        "rank_correlation",
        "failure_distribution_similarity",
        "incremental_predictive_value",
    )
    metric_issues = []
    if not isinstance(metrics, Mapping):
        metric_issues.append("metrics_missing")
    else:
        for metric_id in required_metrics:
            metric = metrics.get(metric_id)
            if not _valid_predictive_metric(metric, metric_id=metric_id):
                metric_issues.append(metric_id)
    valid = (
        isinstance(study, Mapping)
        and study.get("validated") is True
        and bool(study.get("study_id"))
        and _sha256(study.get("contract_sha256"))
        and not metric_issues
    )
    return _result(
        SIM_REAL_AUDIT,
        "predictive_validity",
        bool(valid),
        dict(inputs),
        {
            "study_id": study.get("study_id") if isinstance(study, Mapping) else None,
            "metric_issues": metric_issues,
            "required_metrics": list(required_metrics),
        },
        "Validate the paired sim-real study and attach its immutable contract hash.",
        "Validated sim-real evidence is attached."
        if valid
        else "Hardware evidence is present but not claim-ready.",
    )


def _result(
    audit_id: str,
    category: str,
    passed: bool,
    inputs: dict[str, Any],
    evidence: dict[str, Any],
    remediation: str,
    summary: str,
    *,
    failure_impact: str = "block",
) -> AuditResult:
    return AuditResult(
        audit_id=audit_id,
        category=category,
        status="passed" if passed else "failed",
        severity="info" if passed else "error" if failure_impact == "downgrade" else "blocking",
        inputs=inputs,
        evidence=evidence,
        remediation=remediation,
        claim_impact="none" if passed else failure_impact,  # type: ignore[arg-type]
        summary=summary,
    )


def _missing(audit_id: str, category: str, inputs: dict[str, Any], remediation: str, *, evidence: dict[str, Any] | None = None) -> AuditResult:
    return AuditResult(
        audit_id=audit_id,
        category=category,
        status="missing",
        severity="blocking",
        inputs=inputs,
        evidence=evidence or {},
        remediation=remediation,
        claim_impact="block",
        summary="Required benchmark-validity evidence is missing or invalid.",
    )


def _category(audit_id: str) -> str:
    if audit_id in {LEAKAGE_AUDIT, HIDDEN_TEST_AUDIT}:
        return "data_validity"
    if audit_id in {STATISTICS_AUDIT, RANK_STABILITY_AUDIT}:
        return "statistical_validity"
    if audit_id == PAIRING_AUDIT:
        return "comparison_validity"
    if audit_id == SIM_REAL_AUDIT:
        return "predictive_validity"
    return "construct_validity"


def _mapping_list(value: Any) -> list[dict[str, Any]]:
    return [dict(item) for item in value] if isinstance(value, list) and all(isinstance(item, Mapping) for item in value) else []


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, (list, tuple)) else []


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _positive_int(value: Any) -> int | None:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 and not isinstance(value, bool) else None


def _nonnegative_int(value: Any) -> int | None:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 and not isinstance(value, bool) else None


def _interval(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    low, high = _finite(value[0]), _finite(value[1])
    return (low, high) if low is not None and high is not None and low <= high else None


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _pair_order(ranking: Sequence[str]) -> dict[tuple[str, str], bool]:
    positions = {policy: index for index, policy in enumerate(ranking)}
    policies = sorted(positions)
    return {(left, right): positions[left] < positions[right] for index, left in enumerate(policies) for right in policies[index + 1 :]}


def _valid_predictive_metric(value: Any, *, metric_id: str) -> bool:
    if not isinstance(value, Mapping):
        return False
    estimate = _finite(value.get("value"))
    interval = _interval(value.get("ci95"))
    sample_size = _positive_int(value.get("sample_size"))
    if estimate is None or interval is None or sample_size is None or sample_size < 2:
        return False
    if metric_id == "failure_distribution_similarity" and not 0.0 <= estimate <= 1.0:
        return False
    if metric_id == "rank_correlation" and not -1.0 <= estimate <= 1.0:
        return False
    if metric_id == "incremental_predictive_value" and value.get("held_out") is not True:
        return False
    return interval[0] <= estimate <= interval[1]
