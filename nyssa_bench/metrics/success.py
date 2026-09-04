from __future__ import annotations

from collections import Counter, defaultdict
from math import isfinite, sqrt

from nyssa_bench.core.episode import EpisodeResult


RECOVERY_COUNT_METRICS = {
    "recovery_attempt_count",
    "recovery_applied_count",
    "recovery_success_count",
    "recovery_failure_count",
    "recovery_not_applied_count",
    "recovery_episode_attempt_count",
    "recovery_episode_applied_count",
    "recovery_episode_success_count",
    "counterfactual_eligible_branch_point_count",
    "counterfactual_skipped_branch_point_count",
    "counterfactual_branch_point_count",
    "counterfactual_supported_branch_point_count",
    "counterfactual_exact_branch_point_count",
    "counterfactual_matched_pair_count",
    "counterfactual_exact_pair_count",
    "counterfactual_qualified_pair_count",
    "counterfactual_incomplete_pair_count",
    "counterfactual_intervention_count",
    "counterfactual_continue_success_count",
    "counterfactual_recovery_success_count",
    "false_intervention_count",
    "harmful_intervention_count",
    "helpful_intervention_count",
    "counterfactual_safety_harm_count",
}
RECOVERY_RATE_METRICS = {
    "recovery_success_rate": ("recovery_success_count", "recovery_applied_count"),
    "recovery_episode_success_rate": (
        "recovery_episode_success_count",
        "recovery_episode_applied_count",
    ),
}


def aggregate_episodes(episodes: list[EpisodeResult]) -> dict[str, object]:
    if not episodes:
        return {
            "episodes": 0,
            "success_count": 0,
            "success_rate": 0.0,
            "success_rate_ci95": [0.0, 0.0],
            "failure_counts": {},
            "per_task": {},
            "per_seed": {},
            "metrics": {},
            "metric_ci95": {},
        }

    task_groups: dict[str, list[EpisodeResult]] = defaultdict(list)
    seed_groups: dict[int, list[EpisodeResult]] = defaultdict(list)
    for episode in episodes:
        task_groups[episode.task_id].append(episode)
        seed_groups[episode.seed].append(episode)

    return {
        **_episode_group_summary(episodes),
        "per_task": {
            task_id: _episode_group_summary(task_groups[task_id])
            for task_id in sorted(task_groups)
        },
        "per_seed": {
            str(seed): _episode_group_summary(seed_groups[seed])
            for seed in sorted(seed_groups)
        },
    }


def _per_task_summary(episodes: list[EpisodeResult]) -> dict[str, object]:
    groups: dict[str, list[EpisodeResult]] = defaultdict(list)
    for episode in episodes:
        groups[episode.task_id].append(episode)
    return {
        task_id: _episode_group_summary(groups[task_id]) for task_id in sorted(groups)
    }


def _per_seed_summary(episodes: list[EpisodeResult]) -> dict[str, object]:
    groups: dict[int, list[EpisodeResult]] = defaultdict(list)
    for episode in episodes:
        groups[episode.seed].append(episode)
    return {str(seed): _episode_group_summary(groups[seed]) for seed in sorted(groups)}


def _episode_group_summary(episodes: list[EpisodeResult]) -> dict[str, object]:
    failure_counts = Counter(
        episode.failure_label for episode in episodes if episode.failure_label
    )
    success_count = sum(episode.success for episode in episodes)
    aggregate_metrics, metric_ci95 = _aggregate_metrics(episodes)
    return {
        "episodes": len(episodes),
        "success_count": success_count,
        "success_rate": success_count / len(episodes),
        "success_rate_ci95": _wilson_ci(success_count, len(episodes)),
        "failure_counts": dict(failure_counts),
        "primary_failure_mode": failure_counts.most_common(1)[0][0]
        if failure_counts
        else None,
        "metrics": aggregate_metrics,
        "metric_ci95": metric_ci95,
    }


def _aggregate_metrics(
    episodes: list[EpisodeResult],
) -> tuple[dict[str, float], dict[str, list[float]]]:
    metric_sums: dict[str, float] = defaultdict(float)
    metric_square_sums: dict[str, float] = defaultdict(float)
    for episode in episodes:
        for key, raw_value in episode.metrics.items():
            value = float(raw_value)
            metric_sums[key] += value
            metric_square_sums[key] += value * value

    metric_keys = sorted(metric_sums)
    episode_count = len(episodes)
    metrics = {key: metric_sums[key] / episode_count for key in metric_keys}
    metric_ci95 = {
        key: _mean_ci95_from_moments(
            total=metric_sums[key],
            square_total=metric_square_sums[key],
            count=episode_count,
        )
        for key in metric_keys
    }

    for key in RECOVERY_COUNT_METRICS.intersection(metric_keys):
        total = metric_sums[key]
        metrics[key] = total
        metric_ci95[key] = [total, total]

    for rate_key, (success_key, denominator_key) in RECOVERY_RATE_METRICS.items():
        if rate_key not in metric_keys:
            continue
        successes = metric_sums.get(success_key, 0.0)
        denominator = metric_sums.get(denominator_key, 0.0)
        metrics[rate_key] = successes / denominator if denominator else 0.0
        metric_ci95[rate_key] = _wilson_ci(round(successes), round(denominator))
    return metrics, metric_ci95


def _mean_ci95_from_moments(
    *, total: float, square_total: float, count: int
) -> list[float]:
    mean = total / count
    if not isfinite(total) or not isfinite(square_total):
        return [float("nan"), float("nan")]
    if count == 1:
        return [mean, mean]
    centered_sum = max(0.0, square_total - count * mean * mean)
    variance = centered_sum / (count - 1)
    margin = 1.959963984540054 * sqrt(variance / count)
    return [mean - margin, mean + margin]


def _wilson_ci(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    proportion = successes / total
    denominator = 1.0 + z**2 / total
    center = (proportion + z**2 / (2.0 * total)) / denominator
    margin = (
        z
        * sqrt((proportion * (1.0 - proportion) + z**2 / (4.0 * total)) / total)
        / denominator
    )
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _mean_ci95(values: list[float]) -> list[float]:
    if not values:
        return [0.0, 0.0]
    mean = sum(values) / len(values)
    if len(values) == 1:
        return [mean, mean]
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    margin = 1.959963984540054 * sqrt(variance / len(values))
    return [mean - margin, mean + margin]
