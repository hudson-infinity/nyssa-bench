from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from typing import Any, Mapping, Sequence

from nyssa_bench.core.episode import EpisodeResult
from nyssa_bench.recovery.metrics import summarize_counterfactual_recovery


PAIRWISE_COMPARISON_CONTRACT_FORMAT = "nyssa-pairwise-comparison-contract-v1"
PAIRED_EVIDENCE_FORMAT = "nyssa-paired-episode-evidence-v1"
PAIRED_METRICS_FORMAT = "nyssa-paired-metrics-v1"

NUMERIC_PAIR_METRICS = (
    "collision_count",
    "safety_violation_rate",
    "damage_event_count",
    "expert_intervention_rate",
    "false_intervention_rate",
    "harmful_intervention_rate",
    "counterfactual_recovery_gain",
    "counterfactual_branch_coverage",
    "mean_intervention_cost_steps",
)


def build_comparison_contract(
    policy_a_label: str,
    policy_b_label: str,
    supplied: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if supplied is None:
        contract: dict[str, Any] = {
            "format": PAIRWISE_COMPARISON_CONTRACT_FORMAT,
            "policy_a_label": policy_a_label,
            "policy_b_label": policy_b_label,
            "episode_key": ["task_id", "seed", "episode_index"],
            "condition_fields": [
                "initial_observation_sha256",
                "stressor_execution",
                "failure_detector_contracts",
            ],
            "task_contracts": {},
            "success_contracts": {},
            "missing_contract_semantics": "reported_not_inferred",
        }
    else:
        contract = _json_mapping(supplied, "comparison contract")
        if contract.get("format") != PAIRWISE_COMPARISON_CONTRACT_FORMAT:
            raise ValueError(
                "Unsupported pairwise comparison contract format: "
                f"{contract.get('format')}"
            )
        for key, expected in (
            ("policy_a_label", policy_a_label),
            ("policy_b_label", policy_b_label),
        ):
            if contract.get(key) != expected:
                raise ValueError(f"comparison contract {key} does not match input")
    _canonical_json(contract)
    return contract


def comparison_contract_sha256(contract: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(contract).encode()).hexdigest()


def pair_condition(episode: EpisodeResult) -> dict[str, Any]:
    return {
        "task_id": episode.task_id,
        "seed": episode.seed,
        "episode_index": episode.episode_index,
        "initial_observation_sha256": _initial_observation_sha256(episode),
        "stressor_execution": _stressor_condition(episode.stressor_context),
        "failure_detector_contracts": _detector_condition(
            episode.failure_detector_context
        ),
    }


def condition_sha256(condition: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(condition).encode()).hexdigest()


def paired_episode_evidence(
    episode_a: EpisodeResult,
    episode_b: EpisodeResult,
    *,
    condition_compatible: bool,
) -> dict[str, Any]:
    profile_a = _episode_profile(episode_a)
    profile_b = _episode_profile(episode_b)
    metrics = {
        metric_id: _metric_delta(episode_a, episode_b, metric_id)
        for metric_id in NUMERIC_PAIR_METRICS
    }
    metrics["success"] = {
        "status": "available" if condition_compatible else "incompatible",
        "policy_a": int(episode_a.success),
        "policy_b": int(episode_b.success),
        "delta_a_minus_b": int(episode_a.success) - int(episode_b.success)
        if condition_compatible
        else None,
        "reason": None if condition_compatible else "paired conditions differ",
    }
    if not condition_compatible:
        for metric_id, value in metrics.items():
            if metric_id == "success":
                continue
            value.update(
                {
                    "status": "incompatible",
                    "delta_a_minus_b": None,
                    "reason": "paired conditions differ",
                }
            )
    return {
        "format": PAIRED_EVIDENCE_FORMAT,
        "condition_compatible": condition_compatible,
        "policy_a": profile_a,
        "policy_b": profile_b,
        "failure_deltas": {
            "categories": _counter_delta(
                profile_a["failure_events"]["category_counts"],
                profile_b["failure_events"]["category_counts"],
            ),
            "roles": _counter_delta(
                profile_a["failure_events"]["role_counts"],
                profile_b["failure_events"]["role_counts"],
            ),
            "mechanisms": _counter_delta(
                profile_a["failure_events"]["mechanism_counts"],
                profile_b["failure_events"]["mechanism_counts"],
            ),
            "evidence_visibility": _counter_delta(
                profile_a["failure_events"]["evidence_visibility"],
                profile_b["failure_events"]["evidence_visibility"],
            ),
        },
        "time_to_failure": _time_delta(
            profile_a["time_to_failure"], profile_b["time_to_failure"]
        ),
        "metrics": metrics,
    }


def aggregate_paired_evidence(outcomes: Sequence[Any]) -> dict[str, Any]:
    compatible = [item for item in outcomes if item.condition_compatible]
    success_deltas = [
        float(item.evidence["metrics"]["success"]["delta_a_minus_b"])
        for item in compatible
    ]
    discordance = Counter(
        (
            "a_only"
            if item.policy_a_success and not item.policy_b_success
            else "b_only"
            if item.policy_b_success and not item.policy_a_success
            else "both_succeeded"
            if item.policy_a_success
            else "both_failed"
        )
        for item in compatible
    )
    metric_summaries = {
        metric_id: _aggregate_metric_delta(compatible, metric_id)
        for metric_id in NUMERIC_PAIR_METRICS
    }
    time_deltas = [
        float(item.evidence["time_to_failure"]["delta_a_minus_b"])
        for item in compatible
        if item.evidence["time_to_failure"]["status"] == "available"
    ]
    time_missing = len(compatible) - len(time_deltas)
    category_delta: Counter[str] = Counter()
    role_delta: Counter[str] = Counter()
    mechanism_delta: Counter[str] = Counter()
    for item in compatible:
        category_delta.update(item.evidence["failure_deltas"]["categories"])
        role_delta.update(item.evidence["failure_deltas"]["roles"])
        mechanism_delta.update(item.evidence["failure_deltas"]["mechanisms"])
    ledger_pairs = sum(
        bool(item.evidence["policy_a"]["failure_events"]["ledger_available"])
        and bool(item.evidence["policy_b"]["failure_events"]["ledger_available"])
        for item in compatible
    )
    initial_identity_pairs = sum(
        item.policy_a_condition.get("initial_observation_sha256") is not None
        and item.policy_b_condition.get("initial_observation_sha256") is not None
        for item in compatible
    )
    stressor_contract_pairs = sum(
        _condition_section_available(
            item.policy_a_condition.get("stressor_execution")
        )
        and _condition_section_available(
            item.policy_b_condition.get("stressor_execution")
        )
        for item in compatible
    )
    detector_contract_pairs = sum(
        _condition_section_available(
            item.policy_a_condition.get("failure_detector_contracts")
        )
        and _condition_section_available(
            item.policy_b_condition.get("failure_detector_contracts")
        )
        for item in compatible
    )
    detector_evidence_pairs = sum(
        bool(item.evidence["policy_a"]["failure_detectors"]["available"])
        and bool(item.evidence["policy_b"]["failure_detectors"]["available"])
        for item in compatible
    )
    success_summary = _mean_summary(success_deltas, missing=len(outcomes) - len(compatible))
    success_summary.update(
        {
            "interpretation": "policy_a_success_minus_policy_b_success",
            "discordance": dict(sorted(discordance.items())),
        }
    )
    return {
        "format": PAIRED_METRICS_FORMAT,
        "total_pairs": len(outcomes),
        "condition_compatible_pairs": len(compatible),
        "condition_incompatible_pairs": len(outcomes) - len(compatible),
        "success_difference": success_summary,
        "time_to_failure_difference": _mean_summary(
            time_deltas, missing=time_missing + len(outcomes) - len(compatible)
        ),
        "numeric_deltas": metric_summaries,
        "failure_event_deltas": {
            "categories": _drop_zeroes(category_delta),
            "roles": _drop_zeroes(role_delta),
            "mechanisms": _drop_zeroes(mechanism_delta),
            "semantics": "policy_a_event_count_minus_policy_b_event_count",
        },
        "evidence_coverage": {
            "failure_ledger_pairs": ledger_pairs,
            "failure_ledger_rate": ledger_pairs / len(compatible)
            if compatible
            else 0.0,
            "time_to_failure_observed_pairs": len(time_deltas),
            "time_to_failure_censored_or_missing_pairs": time_missing,
            "initial_observation_identity_pairs": initial_identity_pairs,
            "stressor_contract_pairs": stressor_contract_pairs,
            "failure_detector_contract_pairs": detector_contract_pairs,
            "failure_detector_evidence_pairs": detector_evidence_pairs,
            "by_metric": {
                metric_id: {
                    "available_pairs": value["sample_size"],
                    "missing_pairs": value["missing_count"],
                }
                for metric_id, value in metric_summaries.items()
            },
        },
    }


def _episode_profile(episode: EpisodeResult) -> dict[str, Any]:
    events = _failure_events(episode)
    categories = Counter(str(event.get("category", "unknown")) for event in events)
    roles = Counter(str(event.get("role", "unknown")) for event in events)
    mechanisms = Counter(
        str(event.get("subtype", "unknown"))
        for event in events
        if event.get("role") == "mechanism"
    )
    visibility: Counter[str] = Counter()
    for event in events:
        evidence = event.get("evidence", {})
        if not isinstance(evidence, dict):
            continue
        for key in ("policy_observable", "privileged", "external"):
            values = evidence.get(key, [])
            if isinstance(values, list):
                visibility[key] += len(values)
    return {
        "success": episode.success,
        "failure_label": episode.failure_label,
        "steps": len(episode.steps),
        "failure_detectors": _detector_evidence_profile(
            episode.failure_detector_context
        ),
        "failure_events": {
            "ledger_available": episode.failure_ledger is not None,
            "event_count": len(events),
            "category_counts": dict(sorted(categories.items())),
            "role_counts": dict(sorted(roles.items())),
            "mechanism_counts": dict(sorted(mechanisms.items())),
            "evidence_visibility": dict(sorted(visibility.items())),
            "events": events,
        },
        "time_to_failure": _time_to_failure(episode, events),
    }


def _failure_events(episode: EpisodeResult) -> list[dict[str, Any]]:
    ledger = episode.failure_ledger
    if ledger is None:
        return []
    if isinstance(ledger, Mapping):
        raw = ledger.get("events", [])
        return [dict(item) for item in raw if isinstance(item, Mapping)]
    return [event.to_dict() for event in ledger.events]


def _time_to_failure(
    episode: EpisodeResult, events: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    outcome_events = [
        event
        for event in events
        if event.get("role") in {"symptom", "mechanism", "consequence"}
    ]
    if outcome_events:
        return {
            "status": "observed_failure",
            "step": min(int(event.get("onset_step", 0)) for event in outcome_events),
            "censored": False,
        }
    if episode.success:
        return {
            "status": "right_censored_success",
            "step": len(episode.steps),
            "censored": True,
        }
    if episode.steps and episode.steps[-1].truncated:
        return {
            "status": "right_censored_truncation",
            "step": len(episode.steps),
            "censored": True,
        }
    return {"status": "unavailable", "step": None, "censored": None}


def _time_delta(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
    available = a.get("status") == "observed_failure" and b.get("status") == "observed_failure"
    return {
        "status": "available" if available else "censored_or_missing",
        "policy_a": dict(a),
        "policy_b": dict(b),
        "delta_a_minus_b": int(a["step"]) - int(b["step"])
        if available
        else None,
        "reason": None
        if available
        else "both policies require observed failure onsets",
    }


def _metric_delta(
    episode_a: EpisodeResult, episode_b: EpisodeResult, metric_id: str
) -> dict[str, Any]:
    a = _finite_metric(episode_a.metrics.get(metric_id))
    b = _finite_metric(episode_b.metrics.get(metric_id))
    tier_a = None
    tier_b = None
    if metric_id in {
        "counterfactual_recovery_gain",
        "counterfactual_branch_coverage",
        "false_intervention_rate",
        "harmful_intervention_rate",
        "mean_intervention_cost_steps",
    }:
        tier_a = _counterfactual_tier(episode_a)
        tier_b = _counterfactual_tier(episode_b)
        if tier_a in {None, "not_applicable", "unsupported"}:
            a = None
        if tier_b in {None, "not_applicable", "unsupported"}:
            b = None
    available = a is not None and b is not None
    missing = [
        label
        for label, value in (("policy_a", a), ("policy_b", b))
        if value is None
    ]
    return {
        "status": "available" if available else "unavailable",
        "policy_a": a,
        "policy_b": b,
        "policy_a_evidence_tier": tier_a,
        "policy_b_evidence_tier": tier_b,
        "delta_a_minus_b": a - b if a is not None and b is not None else None,
        "reason": None if available else f"missing metric: {', '.join(missing)}",
    }


def _counterfactual_tier(episode: EpisodeResult) -> str | None:
    if not episode.counterfactual_recovery:
        return None
    summary = summarize_counterfactual_recovery([episode])
    return str(summary.get("claim_tier"))


def _aggregate_metric_delta(outcomes: Sequence[Any], metric_id: str) -> dict[str, Any]:
    values = [
        float(item.evidence["metrics"][metric_id]["delta_a_minus_b"])
        for item in outcomes
        if item.evidence["metrics"][metric_id]["status"] == "available"
    ]
    return _mean_summary(values, missing=len(outcomes) - len(values))


def _mean_summary(values: Sequence[float], *, missing: int) -> dict[str, Any]:
    if not values:
        return {
            "status": "unavailable",
            "value": None,
            "ci95": None,
            "sample_size": 0,
            "missing_count": missing,
        }
    mean = sum(values) / len(values)
    if len(values) == 1:
        interval = [mean, mean]
    else:
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        margin = 1.959963984540054 * math.sqrt(variance / len(values))
        interval = [mean - margin, mean + margin]
    return {
        "status": "available",
        "value": mean,
        "ci95": interval,
        "sample_size": len(values),
        "missing_count": missing,
    }


def _initial_observation_sha256(episode: EpisodeResult) -> str | None:
    if not episode.steps:
        return None
    try:
        payload = _canonical_json(episode.steps[0].observation)
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(payload.encode()).hexdigest()


def _stressor_condition(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    applications = value.get("applications", [])
    return {
        "format": value.get("format"),
        "condition_id": value.get("condition_id"),
        "composition_order": value.get("composition_order", []),
        "episode_seed": value.get("episode_seed"),
        "applications": [
            {
                key: application.get(key)
                for key in (
                    "stressor_id",
                    "category",
                    "composition_index",
                    "application_points",
                    "requested",
                    "seed",
                    "status",
                    "applied_parameters",
                    "reason",
                    "backend_evidence",
                )
            }
            for application in applications
            if isinstance(application, Mapping)
        ],
    }


def _detector_condition(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    detectors = value.get("detectors", [])
    return {
        "format": value.get("format"),
        "engine_name": value.get("engine_name"),
        "task_id": value.get("task_id"),
        "detectors": [
            {"contract": item.get("contract")}
            for item in detectors
            if isinstance(item, Mapping)
        ],
    }


def _detector_evidence_profile(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {"available": False, "detectors": []}
    raw = value.get("detectors", [])
    detectors = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, Mapping):
            continue
        contract = item.get("contract")
        support = item.get("support")
        detectors.append(
            {
                "detector_id": contract.get("detector_id")
                if isinstance(contract, Mapping)
                else None,
                "support_status": support.get("status")
                if isinstance(support, Mapping)
                else None,
                "emitted_event_count": int(item.get("emitted_event_count", 0) or 0),
            }
        )
    return {"available": True, "detectors": detectors}


def _condition_section_available(value: Any) -> bool:
    return isinstance(value, Mapping) and any(
        item not in (None, "", [], {}) for item in value.values()
    )


def _counter_delta(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, int]:
    keys = set(a) | set(b)
    return {
        str(key): int(a.get(key, 0)) - int(b.get(key, 0))
        for key in sorted(keys, key=str)
        if int(a.get(key, 0)) != int(b.get(key, 0))
    }


def _drop_zeroes(values: Mapping[str, int]) -> dict[str, int]:
    return {key: value for key, value in sorted(values.items()) if value}


def _finite_metric(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _json_mapping(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    try:
        normalized = json.loads(_canonical_json(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain finite JSON data") from exc
    if not isinstance(normalized, dict):
        raise ValueError(f"{label} must be a mapping")
    return normalized


def _jsonable(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported comparison-contract value: {type(value).__name__}")
