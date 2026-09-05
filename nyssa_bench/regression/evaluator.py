from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from nyssa_bench.arena.pairwise_runner import compare_episode_pairs
from nyssa_bench.core.episode import EpisodeResult
from nyssa_bench.reports.comparison import (
    IncompatibleRunsError,
    compare_runs,
    comparison_contract_hash,
    load_comparison_contract,
)
from nyssa_bench.regression.evidence import (
    RunEvidence,
    benchmark_validity_available,
    detector_evidence_available,
    file_sha256,
    load_run_evidence,
    native_failure_ledger_available,
    replay_evidence_available,
    run_validity_available,
)
from nyssa_bench.regression.protocol import (
    ConfirmedBoundaryReference,
    RegressionCellSpec,
    RegressionEpisodeKey,
    RegressionRule,
    RegressionStudySpec,
    RunArtifactReference,
)
from nyssa_bench.stress_search import load_stress_search_study


REGRESSION_REPORT_FORMAT = "nyssa-policy-regression-report-v1"
REGRESSION_DECISION_EXIT_CODES = {
    "pass": 0,
    "fail": 1,
    "inconclusive": 2,
    "invalid": 3,
}


@dataclass(frozen=True)
class _CellEvidence:
    spec: RegressionCellSpec
    baseline: RunEvidence
    candidate: RunEvidence
    paired: Any
    report: dict[str, Any]


class RegressionStudyEvaluator:
    def __init__(self, spec: RegressionStudySpec, *, spec_root: str | Path) -> None:
        self.spec = spec
        self.spec_root = Path(spec_root).resolve()

    def evaluate(self) -> dict[str, Any]:
        cells: dict[str, _CellEvidence] = {}
        cell_reports = []
        for cell in self.spec.cells:
            try:
                evidence = self._evaluate_cell(cell)
            except Exception as exc:
                cell_reports.append(
                    {
                        "cell_id": cell.cell_id,
                        "status": "invalid",
                        "condition_kind": cell.condition_kind,
                        "reason": str(exc),
                        "evidence_checks": [],
                    }
                )
                continue
            cells[cell.cell_id] = evidence
            cell_reports.append(evidence.report)

        rule_reports = [self._evaluate_rule(rule, cells) for rule in self.spec.rules]
        decision = _study_decision(cell_reports, rule_reports)
        report = {
            "format": REGRESSION_REPORT_FORMAT,
            "schema_version": 1,
            "study_id": self.spec.study_id,
            "study_version": self.spec.study_version,
            "prespecified_at": self.spec.prespecified_at,
            "spec_sha256": self.spec.sha256,
            "decision": decision,
            "exit_code": REGRESSION_DECISION_EXIT_CODES[decision],
            "decision_semantics": {
                "pass": "all prespecified rules passed with required evidence",
                "fail": "at least one prespecified performance or safety rule failed",
                "inconclusive": "no rule failed, but evidence or uncertainty was insufficient",
                "invalid": "a run identity, artifact, condition, or comparison contract was invalid",
            },
            "summary": {
                "cell_status_counts": _status_counts(cell_reports),
                "rule_status_counts": _status_counts(rule_reports),
                "cells": len(cell_reports),
                "rules": len(rule_reports),
            },
            "policies": {
                "baseline": self.spec.baseline_policy.to_dict(),
                "candidate": self.spec.candidate_policy.to_dict(),
            },
            "evidence_requirements": self.spec.evidence_requirements.to_dict(),
            "study_metadata": self.spec.metadata,
            "cells": cell_reports,
            "rules": rule_reports,
            "interpretation": (
                "This is a policy-checkpoint release decision, not a public benchmark "
                "ranking, deployment-safety certification, or universal reliability score."
            ),
        }
        report["report_sha256"] = _sha256(report)
        return report

    def _evaluate_cell(self, cell: RegressionCellSpec) -> _CellEvidence:
        baseline = load_run_evidence(
            cell.baseline_run,
            self.spec.baseline_policy,
            spec_root=self.spec_root,
        )
        candidate = load_run_evidence(
            cell.candidate_run,
            self.spec.candidate_policy,
            spec_root=self.spec_root,
        )
        _validate_prespecification(self.spec.prespecified_at, candidate)
        try:
            comparison = compare_runs([baseline.root, candidate.root])
        except IncompatibleRunsError as exc:
            fields = ", ".join(sorted({item["field"] for item in exc.mismatches}))
            raise ValueError(
                f"baseline and candidate packs are comparison-incompatible: {fields}"
            ) from exc
        if not comparison.get("comparable"):
            raise ValueError("baseline and candidate packs are not strictly comparable")
        observed_contract = comparison_contract_hash(
            load_comparison_contract(baseline.root)
        )
        if observed_contract != cell.comparison_contract_sha256:
            raise ValueError(
                "comparison contract hash mismatch: "
                f"expected {cell.comparison_contract_sha256}, observed {observed_contract}"
            )
        _validate_condition(cell, baseline, candidate)
        _validate_episode_keys(cell, baseline, candidate)
        boundary_checks = [
            self._validate_boundary(reference, cell, baseline, candidate)
            for reference in cell.boundary_references
        ]
        paired = compare_episode_pairs(
            list(candidate.episodes),
            list(baseline.episodes),
            policy_a_label="candidate",
            policy_b_label="baseline",
        )
        checks = self._evidence_checks(cell, baseline, candidate, paired)
        checks.extend(boundary_checks)
        evidence_ready = all(check["status"] == "passed" for check in checks)
        report = {
            "cell_id": cell.cell_id,
            "status": "ready" if evidence_ready else "inconclusive",
            "condition_kind": cell.condition_kind,
            "condition_id": cell.condition_id,
            "severity_levels": cell.severity_levels,
            "comparison_contract_sha256": observed_contract,
            "baseline_run_id": cell.baseline_run.run_id,
            "candidate_run_id": cell.candidate_run.run_id,
            "baseline_run_reference": cell.baseline_run.to_dict(),
            "candidate_run_reference": cell.candidate_run.to_dict(),
            "baseline_evaluated_reference": _evaluated_reference(
                cell.baseline_run, baseline
            ),
            "candidate_evaluated_reference": _evaluated_reference(
                cell.candidate_run, candidate
            ),
            "pinned_episode_count": len(cell.episode_keys),
            "pairing": paired.coverage.to_dict(),
            "pairwise_comparison_contract": paired.comparison_contract,
            "pairwise_comparison_contract_sha256": (
                paired.comparison_contract_sha256
            ),
            "paired_outcomes": [outcome.to_dict() for outcome in paired.outcomes],
            "paired_metrics": paired.paired_metrics,
            "evidence_checks": checks,
            "boundary_references": boundary_checks,
        }
        return _CellEvidence(cell, baseline, candidate, paired, report)

    def _validate_boundary(
        self,
        reference: ConfirmedBoundaryReference,
        cell: RegressionCellSpec,
        baseline: RunEvidence,
        candidate: RunEvidence,
    ) -> dict[str, Any]:
        path = Path(reference.study_path)
        path = path.resolve() if path.is_absolute() else (self.spec_root / path).resolve()
        if not path.is_file():
            raise ValueError(f"confirmed boundary artifact not found: {path}")
        observed_hash = file_sha256(path)
        if observed_hash != reference.artifact_sha256:
            raise ValueError(
                "confirmed boundary artifact hash mismatch: "
                f"expected {reference.artifact_sha256}, observed {observed_hash}"
            )
        study = load_stress_search_study(path)
        conditions = study.summary().get("confirmation", {}).get("conditions", [])
        match = next(
            (
                condition
                for condition in conditions
                if isinstance(condition, Mapping)
                and _canonical_json(condition.get("point"))
                == _canonical_json(reference.point)
            ),
            None,
        )
        if match is None or not bool(match.get("confirmed_boundary")):
            raise ValueError(
                "referenced stress-search point is not a confirmed held-out boundary"
            )
        stressor_specs = study.spec.search_space.stressor_specs(
            reference.point, seed=0
        )
        expected_specs: list[dict[str, Any]] = [
            {
                "stressor_id": spec.stressor_id,
                "severity": spec.severity,
                "parameters": spec.parameters,
            }
            for spec in stressor_specs
        ]
        expected_severities = {
            spec.stressor_id: spec.severity for spec in stressor_specs
        }
        if expected_severities != cell.severity_levels:
            raise ValueError(
                "confirmed boundary point does not match the cell severity levels"
            )
        for label, run in (("baseline", baseline), ("candidate", candidate)):
            if _run_stressor_semantics(run) != expected_specs:
                raise ValueError(
                    f"{label} stressor configuration does not match the confirmed "
                    "boundary point"
                )
        return {
            "check_id": "confirmed_boundary_provenance",
            "status": "passed",
            "study_path": reference.study_path,
            "artifact_sha256": observed_hash,
            "study_sha256": study.to_dict()["study_sha256"],
            "point": reference.point,
            "confirmation": dict(match),
            "stressor_specs": expected_specs,
            "reason": None,
        }

    def _evidence_checks(
        self,
        cell: RegressionCellSpec,
        baseline: RunEvidence,
        candidate: RunEvidence,
        paired: Any,
    ) -> list[dict[str, Any]]:
        requirements = self.spec.evidence_requirements
        checks = []
        coverage = float(paired.coverage.joint_coverage)
        checks.append(
            _check(
                "paired_episode_coverage",
                coverage >= requirements.minimum_pair_coverage,
                observed=coverage,
                required=requirements.minimum_pair_coverage,
                missing=False,
            )
        )
        for label, run in (("baseline", baseline), ("candidate", candidate)):
            checks.extend(
                _run_evidence_checks(label, run, requirements.required_metric_vector)
            )
            if requirements.require_failure_ledger:
                available = sum(
                    native_failure_ledger_available(episode)
                    for episode in run.episodes
                )
                checks.append(
                    _coverage_check(
                        f"{label}_failure_ledger",
                        available,
                        len(run.episodes),
                    )
                )
            if requirements.require_detector_evidence:
                available = sum(
                    detector_evidence_available(episode) for episode in run.episodes
                )
                checks.append(
                    _coverage_check(
                        f"{label}_failure_detector_evidence",
                        available,
                        len(run.episodes),
                    )
                )
            if requirements.require_replays:
                available = sum(
                    replay_evidence_available(run, episode) for episode in run.episodes
                )
                checks.append(
                    _coverage_check(
                        f"{label}_replay_evidence",
                        available,
                        len(run.episodes),
                    )
                )
            if requirements.require_run_validity:
                checks.append(
                    _check(
                        f"{label}_run_validity",
                        run_validity_available(run),
                        observed=(run.summary.get("public_claim_validation") or {}).get(
                            "status"
                        ),
                        required="validated",
                        missing=not isinstance(
                            run.summary.get("public_claim_validation"), Mapping
                        ),
                    )
                )
            if requirements.require_benchmark_validity:
                checks.append(
                    _check(
                        f"{label}_benchmark_validity",
                        benchmark_validity_available(run),
                        observed=(run.summary.get("benchmark_validity") or {}).get(
                            "status"
                        ),
                        required="validated",
                        missing=not isinstance(
                            run.summary.get("benchmark_validity"), Mapping
                        ),
                    )
                )
        return checks

    def _evaluate_rule(
        self, rule: RegressionRule, cells: Mapping[str, _CellEvidence]
    ) -> dict[str, Any]:
        missing_cells = [cell_id for cell_id in rule.cell_ids if cell_id not in cells]
        if missing_cells:
            return _rule_result(
                rule,
                "invalid",
                f"invalid or missing cells: {', '.join(missing_cells)}",
            )
        selected = [cells[cell_id] for cell_id in rule.cell_ids]
        unready = [
            cell.report["cell_id"]
            for cell in selected
            if cell.report["status"] != "ready"
        ]
        if unready:
            return _rule_result(
                rule,
                "inconclusive",
                "required evidence is incomplete for cells: " + ", ".join(unready),
            )
        measurement = _rule_measurement(rule, selected, self.spec.sha256)
        if measurement["status"] != "available":
            return _rule_result(
                rule,
                "inconclusive",
                str(measurement.get("reason") or "required metric is unavailable"),
                measurement,
            )
        independent_units = int(
            measurement["paired_difference"].get(
                "independent_units",
                measurement["paired_difference"]["sample_size"],
            )
        )
        if independent_units < rule.minimum_pairs:
            return _rule_result(
                rule,
                "inconclusive",
                f"requires {rule.minimum_pairs} independent pairs, observed "
                f"{independent_units}",
                measurement,
            )
        if rule.kind == "safety_block":
            return _evaluate_safety_rule(rule, measurement)
        return _evaluate_non_inferiority_rule(rule, measurement)


def _rule_measurement(
    rule: RegressionRule,
    cells: Sequence[_CellEvidence],
    study_sha256: str,
) -> dict[str, Any]:
    if rule.source == "metric_vector":
        return _metric_vector_measurement(
            rule, cells, seed=_seed(study_sha256, rule.rule_id)
        )
    paired_values: list[tuple[float, float, tuple[str, int, int]]] = []
    missing = 0
    binary = rule.source in {"paired_success", "failure_category_rate"}
    for cell in cells:
        baseline_by_key = {_episode_key(item): item for item in cell.baseline.episodes}
        candidate_by_key = {_episode_key(item): item for item in cell.candidate.episodes}
        for key in sorted(set(baseline_by_key) & set(candidate_by_key)):
            baseline_value = _episode_measurement(baseline_by_key[key], rule)
            candidate_value = _episode_measurement(candidate_by_key[key], rule)
            if baseline_value is None or candidate_value is None:
                missing += 1
                continue
            paired_values.append(
                (
                    baseline_value,
                    candidate_value,
                    (cell.spec.cell_id, key[1], key[2]),
                )
            )
    if not paired_values:
        return {
            "status": "unavailable",
            "reason": "no matched pairs have the required metric evidence",
            "paired_difference": {"sample_size": 0},
        }
    baseline_values = [value[0] for value in paired_values]
    candidate_values = [value[1] for value in paired_values]
    binary = binary or all(
        value in {0.0, 1.0} for value in (*baseline_values, *candidate_values)
    )
    oriented = [
        candidate - baseline
        if rule.direction == "higher"
        else baseline - candidate
        for baseline, candidate, _ in paired_values
    ]
    clusters: dict[tuple[str, int, int], list[float]] = {}
    for value, (_, _, cluster_id) in zip(oriented, paired_values):
        clusters.setdefault(cluster_id, []).append(value)
    seed = _seed(study_sha256, rule.rule_id)
    return {
        "status": "available",
        "source": rule.source,
        "metric_id": rule.metric_id,
        "direction": rule.direction,
        "baseline": _sample_summary(baseline_values, binary=binary),
        "candidate": _sample_summary(candidate_values, binary=binary),
        "paired_difference": {
            "value": mean(oriented),
            "ci95": _cluster_bootstrap_ci(list(clusters.values()), seed=seed),
            "sample_size": len(oriented),
            "independent_units": len(clusters),
            "resampling_unit": "cell_id_episode_seed_episode_index",
            "missing_pairs": missing,
            "semantics": "candidate_minus_baseline"
            if rule.direction == "higher"
            else "baseline_minus_candidate",
        },
        "reason": None,
    }


def _metric_vector_measurement(
    rule: RegressionRule, cells: Sequence[_CellEvidence], *, seed: int
) -> dict[str, Any]:
    if len(cells) > 1:
        pairs = []
        for cell in cells:
            baseline = _vector_measurement(cell.baseline, rule.metric_id)
            candidate = _vector_measurement(cell.candidate, rule.metric_id)
            if baseline is not None and candidate is not None:
                pairs.append((baseline["value"], candidate["value"]))
        if not pairs:
            return {
                "status": "unavailable",
                "reason": "no run-pair cells have the required metric-vector evidence",
                "paired_difference": {"sample_size": 0},
            }
        baseline_values = [float(item[0]) for item in pairs]
        candidate_values = [float(item[1]) for item in pairs]
        oriented = [
            candidate - baseline
            if rule.direction == "higher"
            else baseline - candidate
            for baseline, candidate in zip(baseline_values, candidate_values)
        ]
        return {
            "status": "available",
            "source": rule.source,
            "metric_id": rule.metric_id,
            "direction": rule.direction,
            "baseline": _sample_summary(baseline_values, binary=False),
            "candidate": _sample_summary(candidate_values, binary=False),
            "paired_difference": {
                "value": mean(oriented),
                "ci95": _cluster_bootstrap_ci(
                    [[value] for value in oriented], seed=seed
                ),
                "sample_size": len(oriented),
                "independent_units": len(oriented),
                "resampling_unit": "run_pair_cell",
                "missing_pairs": len(cells) - len(oriented),
                "semantics": "candidate_minus_baseline"
                if rule.direction == "higher"
                else "baseline_minus_candidate",
                "interval_method": "paired_cell_bootstrap",
            },
            "reason": None,
        }
    cell = cells[0]
    baseline = _vector_measurement(cell.baseline, rule.metric_id)
    candidate = _vector_measurement(cell.candidate, rule.metric_id)
    if baseline is None or candidate is None:
        return {
            "status": "unavailable",
            "reason": "baseline or candidate metric-vector evidence is unavailable",
            "paired_difference": {"sample_size": 0},
        }
    baseline_ci = baseline["ci95"]
    candidate_ci = candidate["ci95"]
    if baseline_ci is None or candidate_ci is None:
        return {
            "status": "unavailable",
            "reason": "metric-vector comparison requires uncertainty intervals",
            "paired_difference": {"sample_size": 0},
        }
    if rule.direction == "higher":
        value = candidate["value"] - baseline["value"]
        interval = [
            candidate_ci[0] - baseline_ci[1],
            candidate_ci[1] - baseline_ci[0],
        ]
        semantics = "candidate_minus_baseline"
    else:
        value = baseline["value"] - candidate["value"]
        interval = [
            baseline_ci[0] - candidate_ci[1],
            baseline_ci[1] - candidate_ci[0],
        ]
        semantics = "baseline_minus_candidate"
    sample_size = min(baseline["sample_size"], candidate["sample_size"])
    return {
        "status": "available",
        "source": rule.source,
        "metric_id": rule.metric_id,
        "direction": rule.direction,
        "baseline": baseline,
        "candidate": candidate,
        "paired_difference": {
            "value": value,
            "ci95": interval,
            "sample_size": sample_size,
            "independent_units": sample_size,
            "resampling_unit": "metric_declared_sample",
            "missing_pairs": 0,
            "semantics": semantics,
            "interval_method": "conservative_difference_of_run_intervals",
        },
        "reason": None,
    }


def _evaluate_non_inferiority_rule(
    rule: RegressionRule, measurement: dict[str, Any]
) -> dict[str, Any]:
    interval = measurement["paired_difference"].get("ci95")
    if not isinstance(interval, list) or len(interval) != 2:
        return _rule_result(
            rule,
            "inconclusive",
            "non-inferiority requires a 95% uncertainty interval",
            measurement,
        )
    boundary = -rule.non_inferiority_margin
    if float(interval[0]) >= boundary:
        return _rule_result(
            rule,
            "passed",
            "the full 95% interval is within the prespecified non-inferiority margin",
            measurement,
        )
    if float(interval[1]) < boundary:
        return _rule_result(
            rule,
            "failed",
            "the full 95% interval is below the prespecified non-inferiority margin",
            measurement,
        )
    return _rule_result(
        rule,
        "inconclusive",
        "the 95% interval crosses the prespecified non-inferiority margin",
        measurement,
    )


def _evaluate_safety_rule(
    rule: RegressionRule, measurement: dict[str, Any]
) -> dict[str, Any]:
    candidate = measurement["candidate"]
    interval = candidate.get("ci95")
    assert rule.candidate_limit is not None
    if float(candidate["value"]) > rule.candidate_limit:
        return _rule_result(
            rule,
            "failed",
            "candidate point estimate exceeds the prespecified blocking safety limit",
            measurement,
        )
    if not isinstance(interval, list) or len(interval) != 2:
        return _rule_result(
            rule,
            "inconclusive",
            "safety clearance requires a 95% candidate interval",
            measurement,
        )
    if float(interval[1]) <= rule.candidate_limit:
        return _rule_result(
            rule,
            "passed",
            "the candidate 95% upper bound is within the safety limit",
            measurement,
        )
    return _rule_result(
        rule,
        "inconclusive",
        "the candidate interval does not clear the prespecified safety limit",
        measurement,
    )


def _episode_measurement(
    episode: EpisodeResult, rule: RegressionRule
) -> float | None:
    if rule.source == "paired_success":
        return float(episode.success)
    if rule.source == "episode_metric":
        value = episode.metrics.get(rule.metric_id)
        return float(value) if value is not None and math.isfinite(value) else None
    events = _outcome_events(episode)
    if rule.source == "failure_category_rate":
        return float(any(event.category == rule.metric_id for event in events))
    if rule.source == "failure_onset_steps":
        return float(min(event.onset_step for event in events)) if events else None
    if rule.source == "failure_duration_steps":
        if not events:
            return None
        first = min(events, key=lambda event: (event.onset_step, event.event_id))
        end = first.end_step if first.end_step is not None else first.onset_step
        return float(end - first.onset_step + 1)
    return None


def _outcome_events(episode: EpisodeResult) -> list[Any]:
    if episode.failure_ledger is None:
        return []
    return [
        event
        for event in episode.failure_ledger.events
        if event.role in {"symptom", "mechanism", "consequence"}
    ]


def _vector_measurement(
    run: RunEvidence, metric_id: str
) -> dict[str, Any] | None:
    values = run.metric_vector.get("values")
    if not isinstance(values, Mapping):
        return None
    measurement = values.get(metric_id)
    if not isinstance(measurement, Mapping) or measurement.get("status") != "available":
        return None
    value = _finite_float(measurement.get("value"))
    interval = measurement.get("ci95")
    ci = (
        [_finite_float(interval[0]), _finite_float(interval[1])]
        if isinstance(interval, (list, tuple)) and len(interval) == 2
        else None
    )
    if value is None or (ci is not None and any(item is None for item in ci)):
        return None
    normalized_ci = None
    if ci is not None:
        low, high = ci
        assert low is not None and high is not None
        normalized_ci = [low, high]
    return {
        "value": value,
        "ci95": normalized_ci,
        "sample_size": int(measurement.get("sample_size", 0) or 0),
        "status": "available",
        "source": "metric_vector",
    }


def _validate_condition(
    cell: RegressionCellSpec, baseline: RunEvidence, candidate: RunEvidence
) -> None:
    for label, run in (("baseline", baseline), ("candidate", candidate)):
        condition_id, severities = _run_condition(run)
        if condition_id != cell.condition_id:
            raise ValueError(
                f"{label} condition_id mismatch: expected {cell.condition_id}, "
                f"observed {condition_id}"
            )
        if severities != cell.severity_levels:
            raise ValueError(
                f"{label} severity levels mismatch: expected {cell.severity_levels}, "
                f"observed {severities}"
            )


def _validate_prespecification(
    prespecified_at: str, candidate: RunEvidence
) -> None:
    started_at = candidate.metadata.get("started_at")
    if not isinstance(started_at, str):
        raise ValueError("candidate run is missing started_at provenance")
    try:
        prespecified = datetime.fromisoformat(prespecified_at.replace("Z", "+00:00"))
        started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("regression timestamps must use ISO-8601") from exc
    if started.tzinfo is None:
        raise ValueError("candidate started_at must include a timezone")
    if prespecified > started:
        raise ValueError(
            "regression study was prespecified after the candidate run started"
        )


def _run_condition(run: RunEvidence) -> tuple[str, dict[str, float]]:
    config = run.metadata.get("stressor_config")
    if not isinstance(config, Mapping):
        return "clean", {}
    condition_id = str(config.get("condition_id", "clean"))
    raw = config.get("stressors", [])
    if not isinstance(raw, list):
        raise ValueError("run stressor_config.stressors must be a list")
    severities = {}
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError("run stressor specifications must be mappings")
        stressor_id = str(item.get("stressor_id", ""))
        if not stressor_id or stressor_id in severities:
            raise ValueError("run stressor IDs must be non-empty and unique")
        value = _finite_float(item.get("severity"))
        if value is None or value < 0.0:
            raise ValueError("run stressor severity must be finite and non-negative")
        severities[stressor_id] = value
    return condition_id, severities


def _run_stressor_semantics(run: RunEvidence) -> list[dict[str, Any]]:
    config = run.metadata.get("stressor_config")
    if not isinstance(config, Mapping):
        return []
    raw = config.get("stressors", [])
    if not isinstance(raw, list):
        raise ValueError("run stressor_config.stressors must be a list")
    result = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError("run stressor specifications must be mappings")
        parameters = item.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise ValueError("run stressor parameters must be mappings")
        result.append(
            {
                "stressor_id": str(item.get("stressor_id", "")),
                "severity": float(item.get("severity", float("nan"))),
                "parameters": dict(parameters),
            }
        )
    return result


def _validate_episode_keys(
    cell: RegressionCellSpec, baseline: RunEvidence, candidate: RunEvidence
) -> None:
    expected = set(cell.episode_keys)
    for label, run in (("baseline", baseline), ("candidate", candidate)):
        observed = {
            RegressionEpisodeKey(item.task_id, item.seed, item.episode_index)
            for item in run.episodes
        }
        if observed != expected:
            missing = sorted(expected - observed)
            unexpected = sorted(observed - expected)
            raise ValueError(
                f"{label} episode identity mismatch; missing={len(missing)}, "
                f"unexpected={len(unexpected)}"
            )


def _run_evidence_checks(
    label: str, run: RunEvidence, required_metrics: Sequence[str]
) -> list[dict[str, Any]]:
    checks = []
    values = run.metric_vector.get("values")
    for metric_id in required_metrics:
        measurement = values.get(metric_id) if isinstance(values, Mapping) else None
        available = (
            isinstance(measurement, Mapping)
            and measurement.get("status") == "available"
            and _finite_float(measurement.get("value")) is not None
        )
        checks.append(
            _check(
                f"{label}_metric_vector:{metric_id}",
                available,
                observed=measurement.get("status")
                if isinstance(measurement, Mapping)
                else None,
                required="available",
                missing=measurement is None,
            )
        )
    return checks


def _evaluated_reference(
    reference: RunArtifactReference, evidence: RunEvidence
) -> dict[str, Any]:
    return RunArtifactReference(
        run_dir=reference.run_dir,
        run_id=reference.run_id,
        artifact_binding="pinned",
        artifacts_sha256=evidence.artifacts_sha256,
    ).to_dict()


def _coverage_check(check_id: str, available: int, total: int) -> dict[str, Any]:
    return _check(
        check_id,
        available == total and total > 0,
        observed={"available": available, "total": total},
        required={"available": total, "total": total},
        missing=available == 0,
    )


def _check(
    check_id: str,
    passed: bool,
    *,
    observed: Any,
    required: Any,
    missing: bool,
) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "status": "passed" if passed else "missing" if missing else "failed",
        "observed": observed,
        "required": required,
        "reason": None
        if passed
        else "required evidence is missing"
        if missing
        else "observed evidence does not satisfy the prespecified requirement",
    }


def _sample_summary(values: Sequence[float], *, binary: bool) -> dict[str, Any]:
    interval = _wilson(sum(value > 0.5 for value in values), len(values)) if binary else _mean_ci(values)
    return {
        "value": mean(values),
        "ci95": interval,
        "sample_size": len(values),
        "interval_method": "wilson_binomial" if binary else "normal_mean",
    }


def _mean_ci(values: Sequence[float]) -> list[float] | None:
    if len(values) < 2:
        return None
    center = mean(values)
    variance = sum((value - center) ** 2 for value in values) / (len(values) - 1)
    margin = 1.959963984540054 * math.sqrt(variance / len(values))
    return [center - margin, center + margin]


def _cluster_bootstrap_ci(
    groups: Sequence[Sequence[float]], *, seed: int
) -> list[float] | None:
    if len(groups) < 2:
        return None
    rng = random.Random(seed)
    estimates = []
    for _ in range(2000):
        sampled = [groups[rng.randrange(len(groups))] for _ in groups]
        estimates.append(mean(value for group in sampled for value in group))
    estimates.sort()
    return [
        estimates[round(0.025 * (len(estimates) - 1))],
        estimates[round(0.975 * (len(estimates) - 1))],
    ]


def _wilson(successes: int, total: int) -> list[float] | None:
    if total <= 0:
        return None
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z**2 / total
    center = (proportion + z**2 / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            (proportion * (1.0 - proportion) + z**2 / (4.0 * total)) / total
        )
        / denominator
    )
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _rule_result(
    rule: RegressionRule,
    status: str,
    reason: str,
    measurement: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "rule_id": rule.rule_id,
        "status": status,
        "kind": rule.kind,
        "source": rule.source,
        "metric_id": rule.metric_id,
        "cell_ids": list(rule.cell_ids),
        "direction": rule.direction,
        "non_inferiority_margin": rule.non_inferiority_margin,
        "candidate_limit": rule.candidate_limit,
        "minimum_pairs": rule.minimum_pairs,
        "measurement": dict(measurement) if measurement is not None else None,
        "reason": reason,
    }


def _study_decision(
    cells: Sequence[Mapping[str, Any]], rules: Sequence[Mapping[str, Any]]
) -> str:
    if any(item.get("status") == "invalid" for item in (*cells, *rules)):
        return "invalid"
    if any(item.get("status") == "failed" for item in rules):
        return "fail"
    if any(item.get("status") != "passed" for item in rules):
        return "inconclusive"
    return "pass"


def _status_counts(values: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        status = str(value.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def _episode_key(episode: EpisodeResult) -> tuple[str, int, int]:
    return episode.task_id, episode.seed, episode.episode_index


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _seed(study_sha256: str, rule_id: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{study_sha256}:{rule_id}".encode()).digest()[:8], "big"
    )


def _sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
