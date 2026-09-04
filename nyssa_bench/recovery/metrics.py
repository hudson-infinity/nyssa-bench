from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from nyssa_bench.recovery.protocol import (
    COUNTERFACTUAL_RECOVERY_FORMAT,
    BranchOutcome,
    CounterfactualRecoveryRecord,
)


@dataclass(frozen=True)
class _MatchedPair:
    branch_point_id: str
    exact: bool
    continue_outcome: BranchOutcome
    recovery_outcome: BranchOutcome

    @property
    def recovery_delta(self) -> float:
        return float(self.recovery_outcome.success) - float(
            self.continue_outcome.success
        )


def summarize_counterfactual_recovery(episodes: Sequence[Any]) -> dict[str, Any]:
    records = list(_records(episodes))
    matched_pairs, incomplete_pair_count = _matched_pairs(records)
    branch_point_count = len(records)
    eligible_branch_points = max(
        branch_point_count,
        round(
            sum(
                float(
                    _episode_metrics(episode).get(
                        "counterfactual_eligible_branch_point_count", 0.0
                    )
                )
                for episode in episodes
            )
        ),
    )
    supported_points = sum(
        record.branch_point.restoration_grade != "unsupported" for record in records
    )
    exact_points = sum(
        record.branch_point.strongest_causal_claim_eligible for record in records
    )
    matched_point_ids = {pair.branch_point_id for pair in matched_pairs}
    exact_pairs = sum(pair.exact for pair in matched_pairs)
    qualified_pairs = len(matched_pairs) - exact_pairs
    recovery_successes = sum(pair.recovery_outcome.success for pair in matched_pairs)
    continue_successes = sum(pair.continue_outcome.success for pair in matched_pairs)
    false_interventions = sum(pair.continue_outcome.success for pair in matched_pairs)
    harmful_interventions = sum(
        pair.continue_outcome.success and not pair.recovery_outcome.success
        for pair in matched_pairs
    )
    helpful_interventions = sum(
        pair.recovery_outcome.success and not pair.continue_outcome.success
        for pair in matched_pairs
    )
    safety_harms = sum(
        pair.recovery_outcome.safety_event_count
        > pair.continue_outcome.safety_event_count
        or pair.recovery_outcome.damage_event_count
        > pair.continue_outcome.damage_event_count
        for pair in matched_pairs
    )
    gains_by_point = _point_means(matched_pairs, "recovery_delta")
    step_costs = [
        len(pair.recovery_outcome.steps) - len(pair.continue_outcome.steps)
        for pair in matched_pairs
    ]
    reward_deltas = [
        pair.recovery_outcome.total_reward - pair.continue_outcome.total_reward
        for pair in matched_pairs
    ]
    plan_action_counts = [
        pair.recovery_outcome.initial_action_count for pair in matched_pairs
    ]
    matched_pair_count = len(matched_pairs)
    metrics: dict[str, float] = {
        "counterfactual_branch_point_count": float(branch_point_count),
        "counterfactual_eligible_branch_point_count": float(eligible_branch_points),
        "counterfactual_supported_branch_point_count": float(supported_points),
        "counterfactual_exact_branch_point_count": float(exact_points),
        "counterfactual_matched_pair_count": float(matched_pair_count),
        "counterfactual_exact_pair_count": float(exact_pairs),
        "counterfactual_qualified_pair_count": float(qualified_pairs),
        "counterfactual_incomplete_pair_count": float(incomplete_pair_count),
        "counterfactual_intervention_count": float(matched_pair_count),
        "counterfactual_continue_success_count": float(continue_successes),
        "counterfactual_recovery_success_count": float(recovery_successes),
        "false_intervention_count": float(false_interventions),
        "harmful_intervention_count": float(harmful_interventions),
        "helpful_intervention_count": float(helpful_interventions),
        "counterfactual_safety_harm_count": float(safety_harms),
        "counterfactual_branch_coverage": (
            len(matched_point_ids) / eligible_branch_points
            if eligible_branch_points
            else 0.0
        ),
    }
    metric_ci95: dict[str, list[float]] = {}
    if matched_pair_count:
        recovery_gain = _mean(list(gains_by_point.values()))
        metrics.update(
            {
                "counterfactual_recovery_gain": recovery_gain,
                "counterfactual_continue_success_rate": continue_successes
                / matched_pair_count,
                "counterfactual_recovery_success_rate": recovery_successes
                / matched_pair_count,
                "false_intervention_rate": false_interventions / matched_pair_count,
                "harmful_intervention_rate": harmful_interventions / matched_pair_count,
                "helpful_intervention_rate": helpful_interventions / matched_pair_count,
                "counterfactual_safety_harm_rate": safety_harms / matched_pair_count,
                "mean_intervention_cost_steps": _mean(step_costs),
                "mean_intervention_plan_actions": _mean(plan_action_counts),
                "mean_recovery_reward_delta": _mean(reward_deltas),
            }
        )
        metric_ci95["counterfactual_recovery_gain"] = _cluster_bootstrap_ci(
            list(gains_by_point.values())
        )
        for metric_id, numerator in (
            ("counterfactual_continue_success_rate", continue_successes),
            ("counterfactual_recovery_success_rate", recovery_successes),
            ("false_intervention_rate", false_interventions),
            ("harmful_intervention_rate", harmful_interventions),
            ("helpful_intervention_rate", helpful_interventions),
            ("counterfactual_safety_harm_rate", safety_harms),
        ):
            metric_ci95[metric_id] = _wilson_ci(numerator, matched_pair_count)
        metric_ci95["mean_intervention_cost_steps"] = _mean_ci95(step_costs)
        metric_ci95["mean_intervention_plan_actions"] = _mean_ci95(
            plan_action_counts
        )
        metric_ci95["mean_recovery_reward_delta"] = _mean_ci95(reward_deltas)

    error_outcomes = sum(
        outcome.status == "error" for record in records for outcome in record.outcomes
    )
    restoration_grades: dict[str, int] = {}
    for record in records:
        grade = record.branch_point.restoration_grade
        restoration_grades[grade] = restoration_grades.get(grade, 0) + 1
    if not records:
        status = "not_observed"
        claim_tier = "not_applicable"
    elif not matched_pairs:
        status = "insufficient_evidence"
        claim_tier = "unsupported"
    elif (
        exact_pairs == matched_pair_count
        and len(matched_point_ids) == eligible_branch_points
    ):
        status = "available"
        claim_tier = "exact_counterfactual"
    elif exact_pairs == matched_pair_count:
        status = "available"
        claim_tier = "exact_counterfactual_partial_coverage"
    else:
        status = "available"
        claim_tier = "qualified_counterfactual"
    return {
        "format": COUNTERFACTUAL_RECOVERY_FORMAT,
        "status": status,
        "claim_tier": claim_tier,
        "branch_points": branch_point_count,
        "eligible_branch_points": eligible_branch_points,
        "supported_branch_points": supported_points,
        "exact_branch_points": exact_points,
        "matched_branch_points": len(matched_point_ids),
        "matched_pairs": matched_pair_count,
        "exact_pairs": exact_pairs,
        "qualified_pairs": qualified_pairs,
        "incomplete_pairs": incomplete_pair_count,
        "error_outcomes": error_outcomes,
        "coverage": {
            "numerator": len(matched_point_ids),
            "denominator": eligible_branch_points,
            "rate": len(matched_point_ids) / eligible_branch_points
            if eligible_branch_points
            else 0.0,
        },
        "restoration_grades": dict(sorted(restoration_grades.items())),
        "metrics": metrics,
        "metric_ci95": metric_ci95,
        "recovery_gain": {
            "value": metrics.get("counterfactual_recovery_gain"),
            "ci95": metric_ci95.get("counterfactual_recovery_gain"),
            "matched_pairs": matched_pair_count,
            "matched_branch_points": len(matched_point_ids),
            "estimator": "mean_within_branch_point_then_cluster_bootstrap",
            "bootstrap_samples": 2000 if gains_by_point else 0,
            "bootstrap_seed": 0,
        },
        "interventions": {
            "evaluated": matched_pair_count,
            "helpful": helpful_interventions,
            "false": false_interventions,
            "harmful": harmful_interventions,
            "safety_harmful": safety_harms,
            "mean_cost_steps": metrics.get("mean_intervention_cost_steps"),
            "mean_plan_actions": metrics.get("mean_intervention_plan_actions"),
            "mean_reward_delta": metrics.get("mean_recovery_reward_delta"),
        },
    }


def _records(episodes: Sequence[Any]) -> Iterable[CounterfactualRecoveryRecord]:
    for episode in episodes:
        values = (
            episode.get("counterfactual_recovery", [])
            if isinstance(episode, dict)
            else getattr(episode, "counterfactual_recovery", [])
        )
        for value in values or []:
            if isinstance(value, CounterfactualRecoveryRecord):
                yield value
            elif isinstance(value, dict):
                yield CounterfactualRecoveryRecord.from_dict(value)
            else:
                raise ValueError(
                    "counterfactual recovery records must be objects or protocol records"
                )


def _episode_metrics(episode: Any) -> dict[str, Any]:
    values = (
        episode.get("metrics", {})
        if isinstance(episode, dict)
        else getattr(episode, "metrics", {})
    )
    return dict(values) if isinstance(values, dict) else {}


def _matched_pairs(
    records: Sequence[CounterfactualRecoveryRecord],
) -> tuple[list[_MatchedPair], int]:
    pairs: list[_MatchedPair] = []
    incomplete = 0
    for record in records:
        by_repeat: dict[int, dict[str, BranchOutcome]] = {}
        for outcome in record.outcomes:
            by_repeat.setdefault(outcome.repeat_index, {})[outcome.branch_kind] = (
                outcome
            )
        for repeat_index in range(record.branch_point.requested_repeats):
            outcomes = by_repeat.get(repeat_index, {})
            continuation = outcomes.get("continue")
            recovery = outcomes.get("recovery")
            if (
                continuation is None
                or recovery is None
                or continuation.status != "completed"
                or recovery.status != "completed"
                or continuation.matched_rng_sha256 != recovery.matched_rng_sha256
            ):
                incomplete += 1
                continue
            pairs.append(
                _MatchedPair(
                    branch_point_id=record.branch_point.branch_point_id,
                    exact=record.branch_point.strongest_causal_claim_eligible,
                    continue_outcome=continuation,
                    recovery_outcome=recovery,
                )
            )
    return pairs, incomplete


def _point_means(pairs: Sequence[_MatchedPair], attribute: str) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for pair in pairs:
        values.setdefault(pair.branch_point_id, []).append(
            float(getattr(pair, attribute))
        )
    return {key: _mean(items) for key, items in values.items()}


def _cluster_bootstrap_ci(
    values: Sequence[float], *, samples: int = 2000, seed: int = 0
) -> list[float]:
    if not values:
        return [0.0, 0.0]
    if len(values) == 1:
        return [float(values[0]), float(values[0])]
    rng = random.Random(seed)
    estimates = sorted(
        _mean([values[rng.randrange(len(values))] for _ in values])
        for _ in range(samples)
    )
    return [
        estimates[_percentile_index(len(estimates), 0.025)],
        estimates[_percentile_index(len(estimates), 0.975)],
    ]


def _percentile_index(length: int, quantile: float) -> int:
    return min(length - 1, max(0, round((length - 1) * quantile)))


def _mean(values: Sequence[float | int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _mean_ci95(values: Sequence[float | int]) -> list[float]:
    if not values:
        return [0.0, 0.0]
    mean = _mean(values)
    if len(values) == 1:
        return [mean, mean]
    variance = sum((float(value) - mean) ** 2 for value in values) / (len(values) - 1)
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
