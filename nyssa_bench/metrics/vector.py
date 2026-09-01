from __future__ import annotations

import copy
import json
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence


RUN_METRICS_FORMAT = "nyssa-run-metrics-v2"
METRIC_VECTOR_FORMAT = "nyssa-metric-vector-v1"
METRIC_MIGRATION_FORMAT = "nyssa-metric-migration-v1"
LEGACY_SCALAR_FORMAT = "nyssa-legacy-scalar-metrics-v1"

MetricDirection = Literal["higher", "lower", "descriptive"]
MetricStatus = Literal["available", "unavailable", "not_applicable", "incompatible"]

LEGACY_SCALAR_FIELDS = (
    "prototype_reliability_score",
    "score_kind",
    "sim_to_real_score",
    "sim_to_real_score_deprecated",
)


@dataclass(frozen=True)
class MetricDefinition:
    metric_id: str
    population: str
    denominator: str
    aggregation: str
    missing_data: str
    direction: MetricDirection
    uncertainty: str
    unit: str

    def to_dict(self) -> dict[str, str]:
        return {
            "metric_id": self.metric_id,
            "population": self.population,
            "denominator": self.denominator,
            "aggregation": self.aggregation,
            "missing_data": self.missing_data,
            "direction": self.direction,
            "uncertainty": self.uncertainty,
            "unit": self.unit,
        }


METRIC_DEFINITIONS = (
    MetricDefinition(
        "clean_success_rate",
        "episodes with no applied positive-severity stressor",
        "eligible clean episodes",
        "successful clean episodes divided by eligible clean episodes",
        "unsupported or unresolved stressor applications are excluded and counted",
        "higher",
        "Wilson 95% binomial interval",
        "proportion",
    ),
    MetricDefinition(
        "shifted_success_rate",
        "episodes with at least one applied positive-severity stressor",
        "eligible shifted episodes",
        "successful shifted episodes divided by eligible shifted episodes",
        "unsupported or unresolved stressor applications are excluded and counted",
        "higher",
        "Wilson 95% binomial interval",
        "proportion",
    ),
    MetricDefinition(
        "robustness_degradation",
        "matched clean and shifted episode identities",
        "matched episode pairs",
        "clean success rate minus shifted success rate",
        "reported as incompatible unless task, seed, and episode identities match exactly",
        "lower",
        "paired normal-approximation 95% interval over outcome differences",
        "proportion_points",
    ),
    MetricDefinition(
        "robustness_auc",
        "matched episodes evaluated at two or more stressor severities including zero",
        "observed severity span",
        "trapezoidal success-rate integral normalized by the observed severity span",
        "unavailable outside a validated severity sweep",
        "higher",
        "paired episode bootstrap percentile interval",
        "normalized_area",
    ),
    MetricDefinition(
        "mean_time_to_failure_steps",
        "episodes with an observed failure event or terminal failure",
        "episodes with observed failure times",
        "arithmetic mean of first failure-onset steps",
        "successful episodes and unresolved truncations are reported as censored, not zero",
        "higher",
        "normal-approximation 95% interval over observed failure times",
        "steps",
    ),
    MetricDefinition(
        "failure_event_distribution",
        "emitted temporal failure events, falling back to terminal labels for legacy episodes",
        "observed failure events or labels",
        "category counts normalized by all observed failure events or labels",
        "an empty mapping means no failure evidence was observed in the evaluated episodes",
        "descriptive",
        "none; counts and denominator are reported",
        "distribution",
    ),
    MetricDefinition(
        "failure_prediction_ece",
        "monitor predictions with binary failure outcomes",
        "predictions with valid confidence and outcome",
        "expected calibration error using the monitor's declared fixed bins",
        "not applicable when no calibrated failure monitor is evaluated",
        "lower",
        "monitor-declared bootstrap or analytic interval",
        "proportion",
    ),
    MetricDefinition(
        "counterfactual_recovery_gain",
        "matched continue and recovery branches from restorable branch points",
        "valid matched branch points",
        "P(success given recovery) minus P(success given continuation)",
        "not applicable without exact or explicitly qualified state-fork evidence",
        "higher",
        "paired branch bootstrap or paired binomial interval",
        "proportion_points",
    ),
    MetricDefinition(
        "expert_intervention_rate",
        "executed policy steps",
        "executed steps",
        "expert interventions divided by executed steps",
        "unavailable when step-level evidence is absent",
        "lower",
        "Wilson 95% binomial interval",
        "proportion",
    ),
    MetricDefinition(
        "false_intervention_rate",
        "interventions with a valid counterfactual non-intervention outcome",
        "counterfactually evaluated interventions",
        "unnecessary interventions divided by evaluated interventions",
        "not applicable without matched counterfactual evidence",
        "lower",
        "Wilson 95% binomial interval",
        "proportion",
    ),
    MetricDefinition(
        "safety_violation_rate",
        "episodes with safety instrumentation",
        "instrumented episodes",
        "episodes containing a safety violation divided by instrumented episodes",
        "episodes without the safety metric are excluded and counted",
        "lower",
        "Wilson 95% binomial interval",
        "proportion",
    ),
    MetricDefinition(
        "damage_event_rate",
        "episodes with damage instrumentation",
        "damage-instrumented episodes",
        "episodes containing physical damage divided by instrumented episodes",
        "not applicable when the simulator or hardware track has no damage instrumentation",
        "lower",
        "Wilson 95% binomial interval",
        "proportion",
    ),
    MetricDefinition(
        "wall_time_seconds",
        "the complete evaluation run",
        "one run",
        "elapsed monotonic wall-clock time",
        "unavailable when timing provenance is absent or invalid",
        "lower",
        "none for a single run",
        "seconds",
    ),
    MetricDefinition(
        "mean_inference_latency_ms",
        "executed policy predictions with latency instrumentation",
        "instrumented predictions",
        "arithmetic mean prediction latency",
        "predictions without latency instrumentation are excluded and counted",
        "lower",
        "normal-approximation 95% interval",
        "milliseconds",
    ),
    MetricDefinition(
        "sim_real_rank_correlation",
        "policies paired between a versioned simulation and hardware study",
        "valid paired policies",
        "study-declared rank correlation",
        "unavailable unless hardware calibration evidence and a study contract hash are present",
        "higher",
        "study-declared paired bootstrap interval",
        "correlation",
    ),
    MetricDefinition(
        "sim_real_failure_distribution_similarity",
        "task-policy conditions paired between simulation and hardware",
        "valid paired conditions",
        "one minus the study-declared failure-distribution divergence",
        "unavailable unless hardware calibration evidence and a study contract hash are present",
        "higher",
        "study-declared paired bootstrap interval",
        "similarity",
    ),
)

DEFINITIONS_BY_ID = {item.metric_id: item for item in METRIC_DEFINITIONS}


def build_metric_vector(
    summary: Mapping[str, Any],
    episodes: Sequence[Any] = (),
    *,
    robustness_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    clean, shifted, excluded = _condition_populations(summary, episodes)
    clean_measurement = _success_measurement(clean, "episode_records")
    shifted_measurement = _success_measurement(shifted, "episode_records")
    if not episodes:
        fallback = _summary_success_measurement(summary)
        if _summary_is_shifted(summary):
            shifted_measurement = fallback
        else:
            clean_measurement = fallback
    total_episodes = len(episodes) or int(summary.get("episodes", 0) or 0)
    clean_count = len(clean)
    shifted_count = len(shifted)
    if not episodes:
        if _summary_is_shifted(summary):
            shifted_count = total_episodes
        else:
            clean_count = total_episodes

    values = {
        "clean_success_rate": clean_measurement,
        "shifted_success_rate": shifted_measurement,
        "robustness_degradation": _degradation_measurement(clean, shifted),
        "robustness_auc": _robustness_auc_measurement(
            robustness_summary or _mapping(summary.get("robustness_sweep"))
        ),
        "mean_time_to_failure_steps": _time_to_failure_measurement(episodes),
        "failure_event_distribution": _failure_distribution_measurement(
            summary, episodes
        ),
        "failure_prediction_ece": _optional_summary_measurement(
            summary,
            container="failure_prediction_calibration",
            value_key="ece",
            absence_status="not_applicable",
            absence_reason="no calibrated failure-monitor predictions",
        ),
        "counterfactual_recovery_gain": _optional_metric_measurement(
            summary,
            "counterfactual_recovery_gain",
            absence_status="not_applicable",
            absence_reason="no matched counterfactual recovery branches",
        ),
        "expert_intervention_rate": _pooled_step_rate(
            episodes,
            count_key="expert_intervention_count",
            absence_reason="step-level intervention evidence is unavailable",
        ),
        "false_intervention_rate": _optional_count_rate(
            episodes,
            numerator_key="false_intervention_count",
            denominator_key="counterfactual_intervention_count",
            absence_status="not_applicable",
            absence_reason="no counterfactually evaluated interventions",
        ),
        "safety_violation_rate": _episode_indicator_rate(
            episodes,
            metric_key="safety_violation_rate",
            absence_status="unavailable",
            absence_reason="no episodes contain safety instrumentation",
        ),
        "damage_event_rate": _episode_indicator_rate(
            episodes,
            metric_key="damage_event_count",
            absence_status="not_applicable",
            absence_reason="no episodes contain damage instrumentation",
        ),
        "wall_time_seconds": _wall_time_measurement(summary),
        "mean_inference_latency_ms": _latency_measurement(episodes),
        "sim_real_rank_correlation": _sim_real_measurement(summary, "rank_correlation"),
        "sim_real_failure_distribution_similarity": _sim_real_measurement(
            summary, "failure_distribution_similarity"
        ),
    }
    vector = {
        "format": METRIC_VECTOR_FORMAT,
        "definitions": {item.metric_id: item.to_dict() for item in METRIC_DEFINITIONS},
        "values": values,
        "population": {
            "episodes": total_episodes,
            "clean_episodes": clean_count,
            "shifted_episodes": shifted_count,
            "excluded_condition_episodes": excluded,
        },
        "hardware_calibration": _hardware_calibration(summary),
        "scalar_composite": None,
        "interpretation": "metrics_are_separate_tradeoffs_without_a_universal_reliability_score",
    }
    validate_metric_vector(vector)
    return vector


def migrate_metric_summary(
    summary: Mapping[str, Any], episodes: Sequence[Any] = ()
) -> dict[str, Any]:
    migrated = copy.deepcopy(dict(summary))
    source_format = migrated.get("format", "legacy-unversioned-run-metrics")
    vector = migrated.get("metric_vector")
    had_vector = vector is not None
    if vector is not None:
        if not isinstance(vector, dict):
            raise ValueError("metric_vector must be a mapping")
        validate_metric_vector(vector)

    legacy_values = {
        field: migrated.pop(field)
        for field in LEGACY_SCALAR_FIELDS
        if field in migrated
    }
    if vector is None:
        vector = build_metric_vector(migrated, episodes)
        migrated["metric_vector"] = vector
    migrated["format"] = RUN_METRICS_FORMAT
    if legacy_values or source_format != RUN_METRICS_FORMAT or not had_vector:
        migrated["metric_migration"] = {
            "format": METRIC_MIGRATION_FORMAT,
            "source_format": source_format,
            "target_format": RUN_METRICS_FORMAT,
            "legacy_scalar_fields": sorted(legacy_values),
            "legacy_scalar_policy": "preserved_for_audit_only_not_mapped_or_ranked",
        }
    if legacy_values:
        migrated["legacy_metrics"] = {
            "format": LEGACY_SCALAR_FORMAT,
            "values": legacy_values,
            "interpretation": "historical_heuristic_values_only",
        }
    return migrated


def validate_metric_vector(vector: Mapping[str, Any]) -> None:
    if vector.get("format") != METRIC_VECTOR_FORMAT:
        raise ValueError(f"Unsupported metric vector format: {vector.get('format')}")
    definitions = vector.get("definitions")
    values = vector.get("values")
    if not isinstance(definitions, dict) or not isinstance(values, dict):
        raise ValueError("metric vector definitions and values must be mappings")
    if vector.get("scalar_composite") is not None:
        raise ValueError("metric vectors cannot contain a composite scalar")
    try:
        json.dumps(vector, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "metric vector must contain finite JSON-compatible data"
        ) from exc
    expected = set(DEFINITIONS_BY_ID)
    if set(definitions) != expected or set(values) != expected:
        raise ValueError(
            "metric vector must contain every registered metric exactly once"
        )
    for metric_id, definition in definitions.items():
        if not isinstance(definition, dict):
            raise ValueError(f"metric definition must be a mapping: {metric_id}")
        required = {
            "metric_id",
            "population",
            "denominator",
            "aggregation",
            "missing_data",
            "direction",
            "uncertainty",
            "unit",
        }
        if not required <= set(definition):
            raise ValueError(f"metric definition is incomplete: {metric_id}")
        if definition.get("metric_id") != metric_id:
            raise ValueError(f"metric definition ID mismatch: {metric_id}")
        measurement = values[metric_id]
        if not isinstance(measurement, dict):
            raise ValueError(f"metric measurement must be a mapping: {metric_id}")
        if measurement.get("status") not in {
            "available",
            "unavailable",
            "not_applicable",
            "incompatible",
        }:
            raise ValueError(f"invalid metric status: {metric_id}")
        if measurement.get("status") == "available" and "value" not in measurement:
            raise ValueError(f"available metric is missing a value: {metric_id}")


def sim_real_metrics_are_supported(vector: Mapping[str, Any] | None) -> bool:
    if not vector:
        return True
    values = vector.get("values", {})
    if not isinstance(values, dict):
        return False
    available = any(
        isinstance(values.get(metric_id), dict)
        and values[metric_id].get("status") == "available"
        for metric_id in (
            "sim_real_rank_correlation",
            "sim_real_failure_distribution_similarity",
        )
    )
    if not available:
        return True
    evidence = vector.get("hardware_calibration")
    return bool(
        isinstance(evidence, dict)
        and evidence.get("validated") is True
        and evidence.get("study_id")
        and evidence.get("contract_sha256")
    )


def metric_measurement(vector: Mapping[str, Any], metric_id: str) -> dict[str, Any]:
    values = vector.get("values", {})
    if not isinstance(values, dict) or not isinstance(values.get(metric_id), dict):
        return _missing("unavailable", "metric is absent from the vector")
    return dict(values[metric_id])


def _condition_populations(
    summary: Mapping[str, Any], episodes: Sequence[Any]
) -> tuple[list[Any], list[Any], int]:
    clean: list[Any] = []
    shifted: list[Any] = []
    excluded = 0
    for episode in episodes:
        condition = _episode_condition(episode)
        if condition == "clean":
            clean.append(episode)
        elif condition == "shifted":
            shifted.append(episode)
        else:
            excluded += 1
    return clean, shifted, excluded


def _episode_condition(episode: Any) -> str:
    context = _episode_value(episode, "stressor_context", {})
    applications = context.get("applications", []) if isinstance(context, dict) else []
    if not applications:
        return "clean"
    shifted = False
    for application in applications:
        if not isinstance(application, dict):
            return "excluded"
        status = application.get("status")
        if status in {"unsupported", "requested"}:
            return "excluded"
        requested = application.get("requested", {})
        severity = requested.get("severity") if isinstance(requested, dict) else None
        if status == "applied":
            parsed_severity = _finite_float(severity)
            if severity is not None and parsed_severity is None:
                return "excluded"
            if parsed_severity is None or parsed_severity > 0.0:
                shifted = True
    return "shifted" if shifted else "clean"


def _success_measurement(episodes: Sequence[Any], source: str) -> dict[str, Any]:
    if not episodes:
        return _missing("unavailable", "no eligible episodes")
    successes = sum(
        bool(_episode_value(episode, "success", False)) for episode in episodes
    )
    return _rate(successes, len(episodes), source=source)


def _summary_success_measurement(summary: Mapping[str, Any]) -> dict[str, Any]:
    total = int(summary.get("episodes", 0) or 0)
    if total <= 0:
        return _missing("unavailable", "summary contains no episodes")
    successes = int(
        summary.get(
            "success_count", round(float(summary.get("success_rate", 0.0)) * total)
        )
    )
    interval = _interval(summary.get("success_rate_ci95"))
    measurement = _rate(successes, total, source="run_summary")
    if interval is not None:
        measurement["ci95"] = interval
    return measurement


def _summary_is_shifted(summary: Mapping[str, Any]) -> bool:
    config = summary.get("stressor_config")
    if not isinstance(config, dict):
        return False
    specs = config.get("stressors", [])
    return any(
        isinstance(spec, dict) and (_finite_float(spec.get("severity")) or 0.0) > 0.0
        for spec in specs
    )


def _degradation_measurement(
    clean: Sequence[Any], shifted: Sequence[Any]
) -> dict[str, Any]:
    if not clean or not shifted:
        return _missing(
            "unavailable", "both clean and shifted episode populations are required"
        )
    clean_map, clean_duplicate = _episode_outcome_map(clean)
    shifted_map, shifted_duplicate = _episode_outcome_map(shifted)
    if clean_duplicate or shifted_duplicate or set(clean_map) != set(shifted_map):
        return _missing(
            "incompatible",
            "clean and shifted populations do not have one-to-one task/seed/episode identities",
            sample_size=min(len(clean_map), len(shifted_map)),
        )
    differences = [
        float(clean_map[key]) - float(shifted_map[key]) for key in sorted(clean_map)
    ]
    return {
        "status": "available",
        "value": sum(differences) / len(differences),
        "ci95": _mean_ci95(differences),
        "sample_size": len(differences),
        "numerator": None,
        "denominator": len(differences),
        "source": "matched_episode_records",
        "reason": None,
    }


def _robustness_auc_measurement(summary: Mapping[str, Any] | None) -> dict[str, Any]:
    if not summary:
        return _missing("unavailable", "no validated severity sweep is attached")
    if (
        summary.get("auc_convention")
        != "trapezoidal_success_rate_integral_normalized_by_observed_severity_span"
    ):
        return _missing("incompatible", "unsupported robustness AUC convention")
    interval = _interval(summary.get("robustness_auc_ci95"))
    value = _finite_float(summary.get("robustness_auc"))
    coverage = int(summary.get("paired_episode_coverage", 0) or 0)
    if value is None or interval is None or coverage <= 0:
        return _missing("incompatible", "robustness sweep is missing AUC evidence")
    return {
        "status": "available",
        "value": value,
        "ci95": interval,
        "sample_size": coverage,
        "numerator": None,
        "denominator": summary.get("severity_domain"),
        "source": "nyssa-robustness-sweep-v1",
        "reason": None,
        "interpolation": "piecewise_linear_trapezoidal",
        "normalization": "observed_severity_span",
    }


def _time_to_failure_measurement(episodes: Sequence[Any]) -> dict[str, Any]:
    if not episodes:
        return _missing("unavailable", "episode records are unavailable")
    times: list[float] = []
    censored = 0
    for episode in episodes:
        if bool(_episode_value(episode, "success", False)):
            censored += 1
            continue
        ledger = _episode_value(episode, "failure_ledger", None)
        events = _outcome_failure_events(ledger)
        if events:
            times.append(float(min(event.onset_step for event in events)))
            continue
        steps = _episode_value(episode, "steps", [])
        failure_label = _episode_value(episode, "failure_label", None)
        if failure_label and steps:
            times.append(float(len(steps) - 1))
        else:
            censored += 1
    if not times:
        return _missing(
            "unavailable",
            "all episodes are censored or lack a failure time",
            sample_size=0,
            censored_count=censored,
        )
    return {
        "status": "available",
        "value": sum(times) / len(times),
        "ci95": _mean_ci95(times),
        "sample_size": len(times),
        "numerator": None,
        "denominator": len(times),
        "censored_count": censored,
        "source": "first_failure_event_or_terminal_failure",
        "reason": None,
    }


def _failure_distribution_measurement(
    summary: Mapping[str, Any], episodes: Sequence[Any]
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for episode in episodes:
        ledger = _episode_value(episode, "failure_ledger", None)
        events = _outcome_failure_events(ledger)
        if events:
            counts.update(str(event.category) for event in events)
        else:
            label = _episode_value(episode, "failure_label", None)
            if label:
                counts[str(label)] += 1
    if not episodes:
        raw_counts = summary.get("failure_counts", {})
        if isinstance(raw_counts, dict):
            counts.update({str(key): int(value) for key, value in raw_counts.items()})
    total = sum(counts.values())
    return {
        "status": "available",
        "value": {key: count / total for key, count in sorted(counts.items())}
        if total
        else {},
        "ci95": None,
        "sample_size": total,
        "numerator": dict(sorted(counts.items())),
        "denominator": total,
        "source": "failure_event_ledger_or_terminal_labels",
        "reason": None,
    }


def _pooled_step_rate(
    episodes: Sequence[Any], *, count_key: str, absence_reason: str
) -> dict[str, Any]:
    step_count = sum(len(_episode_value(episode, "steps", [])) for episode in episodes)
    if step_count <= 0:
        return _missing("unavailable", absence_reason)
    count = sum(
        float(_episode_metrics(episode).get(count_key, 0.0)) for episode in episodes
    )
    return _rate(count, step_count, source="pooled_episode_steps")


def _optional_count_rate(
    episodes: Sequence[Any],
    *,
    numerator_key: str,
    denominator_key: str,
    absence_status: MetricStatus,
    absence_reason: str,
) -> dict[str, Any]:
    denominators = [
        _episode_metrics(episode).get(denominator_key) for episode in episodes
    ]
    if not any(value is not None for value in denominators):
        return _missing(absence_status, absence_reason)
    denominator = sum(float(value or 0.0) for value in denominators)
    numerator = sum(
        float(_episode_metrics(episode).get(numerator_key, 0.0)) for episode in episodes
    )
    if denominator <= 0:
        return _missing(absence_status, absence_reason, sample_size=0)
    return _rate(numerator, denominator, source="counterfactual_intervention_records")


def _episode_indicator_rate(
    episodes: Sequence[Any],
    *,
    metric_key: str,
    absence_status: MetricStatus,
    absence_reason: str,
) -> dict[str, Any]:
    observed = [
        float(_episode_metrics(episode)[metric_key])
        for episode in episodes
        if metric_key in _episode_metrics(episode)
    ]
    if not observed:
        return _missing(absence_status, absence_reason)
    positives = sum(value > 0.0 for value in observed)
    measurement = _rate(positives, len(observed), source="instrumented_episode_metrics")
    measurement["missing_count"] = len(episodes) - len(observed)
    return measurement


def _latency_measurement(episodes: Sequence[Any]) -> dict[str, Any]:
    values = [
        float(_episode_metrics(episode)["inference_latency_ms"])
        for episode in episodes
        if "inference_latency_ms" in _episode_metrics(episode)
        and _finite_float(_episode_metrics(episode)["inference_latency_ms"]) is not None
    ]
    if not values:
        return _missing("unavailable", "no policy predictions contain latency evidence")
    return {
        "status": "available",
        "value": sum(values) / len(values),
        "ci95": _mean_ci95(values),
        "sample_size": len(values),
        "numerator": sum(values),
        "denominator": len(values),
        "source": "instrumented_prediction_latency",
        "reason": None,
    }


def _wall_time_measurement(summary: Mapping[str, Any]) -> dict[str, Any]:
    compute = _mapping(summary.get("compute")) or {}
    value = _finite_float(compute.get("wall_time_seconds"))
    if value is None or value < 0.0:
        return _missing("unavailable", "valid wall-clock timing is absent")
    return {
        "status": "available",
        "value": value,
        "ci95": None,
        "sample_size": 1,
        "numerator": value,
        "denominator": 1,
        "source": "monotonic_run_timer",
        "reason": None,
    }


def _optional_metric_measurement(
    summary: Mapping[str, Any],
    metric_id: str,
    *,
    absence_status: MetricStatus,
    absence_reason: str,
) -> dict[str, Any]:
    metrics = _mapping(summary.get("metrics")) or {}
    value = _finite_float(metrics.get(metric_id))
    if value is None:
        return _missing(absence_status, absence_reason)
    intervals = _mapping(summary.get("metric_ci95")) or {}
    return {
        "status": "available",
        "value": value,
        "ci95": _interval(intervals.get(metric_id)),
        "sample_size": int(summary.get("episodes", 0) or 0),
        "numerator": None,
        "denominator": int(summary.get("episodes", 0) or 0),
        "source": "aggregate_episode_metrics",
        "reason": None,
    }


def _optional_summary_measurement(
    summary: Mapping[str, Any],
    *,
    container: str,
    value_key: str,
    absence_status: MetricStatus,
    absence_reason: str,
) -> dict[str, Any]:
    payload = _mapping(summary.get(container))
    if not payload:
        return _missing(absence_status, absence_reason)
    value = _finite_float(payload.get(value_key))
    if value is None:
        return _missing(absence_status, absence_reason)
    return {
        "status": "available",
        "value": value,
        "ci95": _interval(payload.get("ci95")),
        "sample_size": int(payload.get("sample_size", 0) or 0),
        "numerator": None,
        "denominator": int(payload.get("sample_size", 0) or 0),
        "source": container,
        "reason": None,
    }


def _sim_real_measurement(summary: Mapping[str, Any], key: str) -> dict[str, Any]:
    evidence = _hardware_calibration(summary)
    if evidence is None:
        return _missing(
            "unavailable", "no validated hardware calibration study is attached"
        )
    metrics = _mapping(evidence.get("metrics")) or {}
    payload = metrics.get(key)
    if not isinstance(payload, dict):
        return _missing("unavailable", f"hardware study does not report {key}")
    value = _finite_float(payload.get("value"))
    interval = _interval(payload.get("ci95"))
    sample_size = int(payload.get("sample_size", 0) or 0)
    if value is None or interval is None or sample_size <= 0:
        return _missing("incompatible", f"hardware study has incomplete {key} evidence")
    return {
        "status": "available",
        "value": value,
        "ci95": interval,
        "sample_size": sample_size,
        "numerator": None,
        "denominator": sample_size,
        "source": f"hardware_calibration:{evidence['study_id']}",
        "reason": None,
    }


def _hardware_calibration(summary: Mapping[str, Any]) -> dict[str, Any] | None:
    evidence = _mapping(summary.get("hardware_calibration"))
    if not evidence:
        return None
    if not (
        evidence.get("validated") is True
        and evidence.get("study_id")
        and evidence.get("contract_sha256")
    ):
        return None
    return dict(evidence)


def _rate(
    numerator: float | int, denominator: float | int, *, source: str
) -> dict[str, Any]:
    if denominator <= 0:
        return _missing("unavailable", "metric denominator is zero", sample_size=0)
    if numerator < 0 or numerator > denominator:
        return _missing(
            "incompatible",
            "metric numerator must be within the denominator",
            sample_size=int(denominator),
        )
    value = float(numerator) / float(denominator)
    integer_counts = float(numerator).is_integer() and float(denominator).is_integer()
    return {
        "status": "available",
        "value": value,
        "ci95": _wilson_ci(int(numerator), int(denominator))
        if integer_counts
        else None,
        "sample_size": int(denominator),
        "numerator": numerator,
        "denominator": denominator,
        "source": source,
        "reason": None,
    }


def _missing(
    status: MetricStatus,
    reason: str,
    *,
    sample_size: int = 0,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "status": status,
        "value": None,
        "ci95": None,
        "sample_size": sample_size,
        "numerator": None,
        "denominator": None,
        "source": None,
        "reason": reason,
        **extra,
    }


def _episode_outcome_map(
    episodes: Sequence[Any],
) -> tuple[dict[tuple[str, int, int], bool], bool]:
    result: dict[tuple[str, int, int], bool] = {}
    duplicate = False
    for episode in episodes:
        key = (
            str(_episode_value(episode, "task_id", "unknown")),
            int(_episode_value(episode, "seed", 0)),
            int(_episode_value(episode, "episode_index", 0)),
        )
        duplicate |= key in result
        result[key] = bool(_episode_value(episode, "success", False))
    return result, duplicate


def _episode_metrics(episode: Any) -> Mapping[str, Any]:
    metrics = _episode_value(episode, "metrics", {})
    return metrics if isinstance(metrics, Mapping) else {}


def _outcome_failure_events(ledger: Any) -> tuple[Any, ...]:
    events = getattr(ledger, "events", ()) if ledger is not None else ()
    return tuple(
        event
        for event in events
        if getattr(event, "role", None) in {"symptom", "mechanism", "consequence"}
    )


def _episode_value(episode: Any, key: str, default: Any) -> Any:
    if isinstance(episode, Mapping):
        return episode.get(key, default)
    return getattr(episode, key, default)


def _mapping(value: Any) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, Mapping) else None


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _interval(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    low = _finite_float(value[0])
    high = _finite_float(value[1])
    if low is None or high is None or low > high:
        return None
    return [low, high]


def _mean_ci95(values: Sequence[float]) -> list[float]:
    mean = sum(values) / len(values)
    if len(values) == 1:
        return [mean, mean]
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    margin = 1.959963984540054 * math.sqrt(variance / len(values))
    return [mean - margin, mean + margin]


def _wilson_ci(successes: int, total: int) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z**2 / total
    center = (proportion + z**2 / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt((proportion * (1.0 - proportion) + z**2 / (4.0 * total)) / total)
        / denominator
    )
    return [max(0.0, center - margin), min(1.0, center + margin)]
