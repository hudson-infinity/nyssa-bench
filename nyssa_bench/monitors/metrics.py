from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from statistics import mean
from typing import Any, Mapping, Sequence

from nyssa_bench.monitors.protocol import (
    FailureMonitorContract,
    MonitorPredictionRecord,
    contract_sha256,
)


MONITOR_METRICS_FORMAT = "nyssa-failure-monitor-metrics-v1"
MONITOR_COMPARISON_FORMAT = "nyssa-failure-monitor-comparison-v1"


def summarize_monitor_records(
    records: Sequence[MonitorPredictionRecord],
    contracts: Mapping[str, FailureMonitorContract],
) -> dict[str, Any]:
    grouped: dict[str, list[MonitorPredictionRecord]] = defaultdict(list)
    for record in records:
        contract = contracts.get(record.prediction.monitor_id)
        if contract is None:
            raise ValueError(
                f"monitor record has no contract: {record.prediction.monitor_id}"
            )
        if record.prediction.contract_sha256 != contract_sha256(contract):
            raise ValueError("monitor record contract hash mismatch")
        grouped[record.prediction.monitor_id].append(record)
    return {
        "format": MONITOR_METRICS_FORMAT,
        "monitors": {
            monitor_id: _monitor_summary(grouped.get(monitor_id, []), contract)
            for monitor_id, contract in sorted(contracts.items())
        },
        "prediction_count": len(records),
        "interpretation": "prediction quality is reported separately from intervention recovery effects",
    }


def compare_monitor_records(
    records: Sequence[MonitorPredictionRecord],
    contracts: Mapping[str, FailureMonitorContract],
    monitor_a: str,
    monitor_b: str,
) -> dict[str, Any]:
    if monitor_a == monitor_b:
        raise ValueError("paired monitor comparison requires distinct monitor IDs")
    if monitor_a not in contracts or monitor_b not in contracts:
        raise ValueError("paired monitor comparison is missing a monitor contract")
    maps = {}
    for monitor_id in (monitor_a, monitor_b):
        mapping = {}
        for record in records:
            if record.prediction.monitor_id != monitor_id:
                continue
            key = _record_key(record)
            if key in mapping:
                raise ValueError(f"duplicate monitor prediction identity: {monitor_id}:{key}")
            mapping[key] = record
        maps[monitor_id] = mapping
    if set(maps[monitor_a]) != set(maps[monitor_b]) or not maps[monitor_a]:
        raise ValueError("paired monitors require identical prediction identities")
    deltas = []
    episode_deltas: dict[
        tuple[str, int, int], list[float]
    ] = defaultdict(list)
    discordance = {"a_correct_b_wrong": 0, "b_correct_a_wrong": 0}
    for key in sorted(maps[monitor_a]):
        a = maps[monitor_a][key]
        b = maps[monitor_b][key]
        if a.outcome.to_dict() != b.outcome.to_dict():
            raise ValueError(f"paired monitor labels differ at {key}")
        if a.outcome.status != "observed":
            continue
        target = bool(a.outcome.failure_within_horizon)
        a_error = (a.prediction.risk - float(target)) ** 2
        b_error = (b.prediction.risk - float(target)) ** 2
        delta = a_error - b_error
        deltas.append(delta)
        episode_deltas[key[:3]].append(delta)
        a_correct = (a.prediction.risk >= contracts[monitor_a].alert_threshold) == target
        b_correct = (b.prediction.risk >= contracts[monitor_b].alert_threshold) == target
        if a_correct and not b_correct:
            discordance["a_correct_b_wrong"] += 1
        elif b_correct and not a_correct:
            discordance["b_correct_a_wrong"] += 1
    return {
        "format": MONITOR_COMPARISON_FORMAT,
        "monitor_a": monitor_a,
        "monitor_b": monitor_b,
        "monitor_a_claim_tier": contracts[monitor_a].observability_tier,
        "monitor_b_claim_tier": contracts[monitor_b].observability_tier,
        "matched_predictions": len(maps[monitor_a]),
        "observed_pairs": len(deltas),
        "brier_delta_a_minus_b": mean(deltas) if deltas else None,
        "brier_delta_ci95": _cluster_bootstrap_ci(
            list(episode_deltas.values()), seed=0
        )
        if deltas
        else None,
        "discordance": discordance,
        "recovery_effects_included": False,
    }


def _monitor_summary(
    records: Sequence[MonitorPredictionRecord], contract: FailureMonitorContract
) -> dict[str, Any]:
    observed = [record for record in records if record.outcome.status == "observed"]
    censored = sum(record.outcome.status == "censored" for record in records)
    invalid = sum(record.outcome.status == "invalid" for record in records)
    risks = [record.prediction.risk for record in observed]
    targets = [bool(record.outcome.failure_within_horizon) for record in observed]
    alerts = [risk >= contract.alert_threshold for risk in risks]
    tp = sum(alert and target for alert, target in zip(alerts, targets))
    fp = sum(alert and not target for alert, target in zip(alerts, targets))
    tn = sum(not alert and not target for alert, target in zip(alerts, targets))
    fn = sum(not alert and target for alert, target in zip(alerts, targets))
    ece, calibration_bins = _ece(risks, targets, contract.calibration_bins)
    brier = mean((risk - float(target)) ** 2 for risk, target in zip(risks, targets)) if risks else None
    lead_times = _lead_times(observed, contract.alert_threshold)
    category_correct, category_total = _categorical_accuracy(
        observed, "failure_category"
    )
    mechanism_correct, mechanism_total = _categorical_accuracy(
        observed, "failure_mechanism"
    )
    recovery_correct, recovery_total = _recovery_accuracy(observed)
    time_errors = [
        abs(
            float(record.prediction.expected_time_to_failure)
            - float(record.outcome.failure_onset_step - record.prediction.environment_step)
        )
        for record in observed
        if record.outcome.failure_within_horizon
        and record.outcome.failure_onset_step is not None
        and record.prediction.expected_time_to_failure is not None
    ]
    latencies = [record.prediction.latency_ms for record in records]
    recommendations = [
        record for record in records if record.prediction.intervention_recommended
    ]
    linked = sum(record.intervention_branch_point_id is not None for record in recommendations)
    return {
        "contract_sha256": contract_sha256(contract),
        "observability_tier": contract.observability_tier,
        "deployment_claim_eligible": (
            contract.observability_tier == "deployable_monitor"
        ),
        "claim_scope": "deployment_monitor"
        if contract.observability_tier == "deployable_monitor"
        else "privileged_diagnostic_only",
        "prediction_horizon_steps": contract.prediction_horizon_steps,
        "alert_threshold": contract.alert_threshold,
        "predictions": len(records),
        "observed_labels": len(observed),
        "censored_labels": censored,
        "invalid_labels": invalid,
        "calibration": {
            "ece": ece,
            "ece_ci95": _bootstrap_ece_ci(
                observed, contract.calibration_bins, contract.monitor_id
            )
            if observed
            else None,
            "brier_score": brier,
            "bins": calibration_bins,
        },
        "classification": {
            "true_positive": tp,
            "false_positive": fp,
            "true_negative": tn,
            "false_negative": fn,
            "precision": _ratio(tp, tp + fp),
            "recall": _ratio(tp, tp + fn),
            "false_alarm_rate": _ratio(fp, fp + tn),
            "missed_failure_rate": _ratio(fn, fn + tp),
        },
        "risk_coverage": _risk_coverage(risks, targets, alerts),
        "lead_time": {
            "episodes_with_advance_alert": len(lead_times),
            "mean_steps": mean(lead_times) if lead_times else None,
            "values": lead_times,
        },
        "category_accuracy": _ratio(category_correct, category_total),
        "category_predictions": category_total,
        "mechanism_accuracy": _ratio(mechanism_correct, mechanism_total),
        "mechanism_predictions": mechanism_total,
        "expected_time_to_failure_mae": mean(time_errors) if time_errors else None,
        "expected_time_predictions": len(time_errors),
        "recovery_eligibility_accuracy": _ratio(recovery_correct, recovery_total),
        "recovery_eligibility_predictions": recovery_total,
        "latency": {
            "mean_ms": mean(latencies) if latencies else None,
            "p95_ms": _percentile(latencies, 0.95) if latencies else None,
            "total_ms": sum(latencies),
        },
        "compute": {
            "declared": contract.declared_compute,
            "observed": [record.prediction.compute for record in records],
            "observed_numeric": _numeric_compute_summary(records),
        },
        "intervention_recommendations": {
            "count": len(recommendations),
            "counterfactual_branch_links": linked,
            "link_rate": _ratio(linked, len(recommendations)),
            "prediction_metrics_exclude_recovery_outcomes": True,
        },
    }


def _ece(
    risks: Sequence[float], targets: Sequence[bool], bins: int
) -> tuple[float | None, list[dict[str, Any]]]:
    if not risks:
        return None, []
    details = []
    weighted = 0.0
    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        selected = [
            position
            for position, risk in enumerate(risks)
            if low <= risk < high or (index == bins - 1 and risk == 1.0)
        ]
        if not selected:
            continue
        confidence = mean(risks[position] for position in selected)
        frequency = mean(float(targets[position]) for position in selected)
        gap = abs(confidence - frequency)
        weighted += len(selected) / len(risks) * gap
        details.append(
            {
                "lower": low,
                "upper": high,
                "count": len(selected),
                "mean_risk": confidence,
                "failure_frequency": frequency,
                "absolute_gap": gap,
            }
        )
    return weighted, details


def _bootstrap_ece_ci(
    records: Sequence[MonitorPredictionRecord], bins: int, monitor_id: str
) -> list[float] | None:
    episodes: dict[tuple[str, int, int], list[MonitorPredictionRecord]] = defaultdict(list)
    for record in records:
        episodes[
            (
                record.prediction.task_id,
                record.prediction.episode_seed,
                record.prediction.episode_index,
            )
        ].append(record)
    groups = list(episodes.values())
    if len(groups) == 1:
        return None
    seed = int.from_bytes(hashlib.sha256(monitor_id.encode()).digest()[:8], "big")
    rng = random.Random(seed)
    estimates = []
    for _ in range(1000):
        sampled = [groups[rng.randrange(len(groups))] for _ in groups]
        flat = [record for group in sampled for record in group]
        value, _ = _ece(
            [item.prediction.risk for item in flat],
            [bool(item.outcome.failure_within_horizon) for item in flat],
            bins,
        )
        estimates.append(float(value or 0.0))
    estimates.sort()
    return [
        estimates[round(0.025 * (len(estimates) - 1))],
        estimates[round(0.975 * (len(estimates) - 1))],
    ]


def _risk_coverage(
    risks: Sequence[float], targets: Sequence[bool], alerts: Sequence[bool]
) -> list[dict[str, Any]]:
    if not risks:
        return []
    order = sorted(
        range(len(risks)),
        key=lambda index: (-abs(risks[index] - 0.5), index),
    )
    points = []
    for fraction in (0.1, 0.25, 0.5, 0.75, 1.0):
        count = max(1, math.ceil(len(order) * fraction))
        selected = order[:count]
        errors = sum(alerts[index] != targets[index] for index in selected)
        points.append(
            {
                "coverage": count / len(order),
                "selective_error_rate": errors / count,
                "count": count,
            }
        )
    unique = {}
    for point in points:
        unique[point["count"]] = point
    return [unique[key] for key in sorted(unique)]


def _lead_times(
    records: Sequence[MonitorPredictionRecord], threshold: float
) -> list[int]:
    episodes: dict[tuple[str, int, int], list[MonitorPredictionRecord]] = defaultdict(list)
    for record in records:
        episodes[
            (
                record.prediction.task_id,
                record.prediction.episode_seed,
                record.prediction.episode_index,
            )
        ].append(record)
    values = []
    for episode_records in episodes.values():
        positives = [
            record
            for record in episode_records
            if record.outcome.failure_within_horizon
            and record.outcome.failure_onset_step is not None
        ]
        alerts = []
        for record in positives:
            onset_step = record.outcome.failure_onset_step
            assert onset_step is not None
            if (
                record.prediction.risk >= threshold
                and record.prediction.environment_step <= onset_step
            ):
                alerts.append((record, onset_step))
        if alerts:
            first_alert = min(
                record.prediction.environment_step for record, _ in alerts
            )
            onset = min(onset_step for _, onset_step in alerts)
            values.append(onset - first_alert)
    return values


def _categorical_accuracy(
    records: Sequence[MonitorPredictionRecord], field: str
) -> tuple[int, int]:
    correct = total = 0
    for record in records:
        if not record.outcome.failure_within_horizon:
            continue
        predicted = getattr(record.prediction, field)
        actual = getattr(record.outcome, field)
        if predicted is None or actual is None:
            continue
        total += 1
        correct += predicted == actual
    return correct, total


def _recovery_accuracy(
    records: Sequence[MonitorPredictionRecord],
) -> tuple[int, int]:
    correct = total = 0
    for record in records:
        if record.outcome.recovery_eligible is None:
            continue
        prediction = record.prediction.recovery_eligibility
        if prediction == "unknown":
            continue
        total += 1
        correct += (prediction == "eligible") == record.outcome.recovery_eligible
    return correct, total


def _record_key(record: MonitorPredictionRecord) -> tuple[str, int, int, int]:
    prediction = record.prediction
    return (
        prediction.task_id,
        prediction.episode_seed,
        prediction.episode_index,
        prediction.environment_step,
    )


def _numeric_compute_summary(
    records: Sequence[MonitorPredictionRecord],
) -> dict[str, dict[str, float | int]]:
    values: dict[str, list[float]] = defaultdict(list)
    for record in records:
        for key, value in record.prediction.compute.items():
            if (
                not isinstance(value, bool)
                and isinstance(value, (int, float))
                and math.isfinite(float(value))
            ):
                values[key].append(float(value))
    return {
        key: {
            "count": len(items),
            "mean": mean(items),
            "total": sum(items),
            "p95": _percentile(items, 0.95),
        }
        for key, items in sorted(values.items())
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * quantile)]


def _cluster_bootstrap_ci(
    groups: Sequence[Sequence[float]], *, seed: int
) -> list[float] | None:
    if len(groups) == 1:
        return None
    rng = random.Random(seed)
    estimates = []
    for _ in range(1000):
        sampled = [groups[rng.randrange(len(groups))] for _ in groups]
        estimates.append(mean(value for group in sampled for value in group))
    estimates.sort()
    return [
        estimates[round(0.025 * (len(estimates) - 1))],
        estimates[round(0.975 * (len(estimates) - 1))],
    ]
