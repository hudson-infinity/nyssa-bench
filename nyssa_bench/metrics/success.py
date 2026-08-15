from __future__ import annotations

from collections import Counter
from math import sqrt

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
}
RECOVERY_RATE_METRICS = {
    "recovery_success_rate": ("recovery_success_count", "recovery_applied_count"),
    "recovery_episode_success_rate": ("recovery_episode_success_count", "recovery_episode_applied_count"),
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

    failure_counts = Counter(ep.failure_label for ep in episodes if ep.failure_label)
    success_count = sum(1 for ep in episodes if ep.success)
    aggregate_metrics, metric_ci95 = _aggregate_metrics(episodes)

    return {
        "episodes": len(episodes),
        "success_count": success_count,
        "success_rate": success_count / len(episodes),
        "success_rate_ci95": _wilson_ci(success_count, len(episodes)),
        "failure_counts": dict(failure_counts),
        "primary_failure_mode": failure_counts.most_common(1)[0][0] if failure_counts else None,
        "metrics": aggregate_metrics,
        "metric_ci95": metric_ci95,
        "per_task": _per_task_summary(episodes),
        "per_seed": _per_seed_summary(episodes),
    }


def _per_task_summary(episodes: list[EpisodeResult]) -> dict[str, object]:
    task_ids = sorted({ep.task_id for ep in episodes})
    summaries: dict[str, object] = {}
    for task_id in task_ids:
        task_episodes = [ep for ep in episodes if ep.task_id == task_id]
        task_failures = Counter(ep.failure_label for ep in task_episodes if ep.failure_label)
        task_success_count = sum(1 for ep in task_episodes if ep.success)
        aggregate_metrics, metric_ci95 = _aggregate_metrics(task_episodes)
        summaries[task_id] = {
            "episodes": len(task_episodes),
            "success_count": task_success_count,
            "success_rate": task_success_count / len(task_episodes),
            "success_rate_ci95": _wilson_ci(task_success_count, len(task_episodes)),
            "failure_counts": dict(task_failures),
            "primary_failure_mode": task_failures.most_common(1)[0][0] if task_failures else None,
            "metrics": aggregate_metrics,
            "metric_ci95": metric_ci95,
        }
    return summaries


def _per_seed_summary(episodes: list[EpisodeResult]) -> dict[str, object]:
    seeds = sorted({ep.seed for ep in episodes})
    summaries: dict[str, object] = {}
    for seed in seeds:
        seed_episodes = [ep for ep in episodes if ep.seed == seed]
        seed_failures = Counter(ep.failure_label for ep in seed_episodes if ep.failure_label)
        seed_success_count = sum(1 for ep in seed_episodes if ep.success)
        aggregate_metrics, metric_ci95 = _aggregate_metrics(seed_episodes)
        summaries[str(seed)] = {
            "episodes": len(seed_episodes),
            "success_count": seed_success_count,
            "success_rate": seed_success_count / len(seed_episodes),
            "success_rate_ci95": _wilson_ci(seed_success_count, len(seed_episodes)),
            "failure_counts": dict(seed_failures),
            "primary_failure_mode": seed_failures.most_common(1)[0][0] if seed_failures else None,
            "metrics": aggregate_metrics,
            "metric_ci95": metric_ci95,
        }
    return summaries


def _aggregate_metrics(episodes: list[EpisodeResult]) -> tuple[dict[str, float], dict[str, list[float]]]:
    metric_keys = sorted({key for episode in episodes for key in episode.metrics})
    metrics = {
        key: sum(float(episode.metrics.get(key, 0.0)) for episode in episodes) / len(episodes)
        for key in metric_keys
    }
    metric_ci95 = {
        key: _mean_ci95([float(episode.metrics.get(key, 0.0)) for episode in episodes])
        for key in metric_keys
    }

    for key in RECOVERY_COUNT_METRICS.intersection(metric_keys):
        total = sum(float(episode.metrics.get(key, 0.0)) for episode in episodes)
        metrics[key] = total
        metric_ci95[key] = [total, total]

    for rate_key, (success_key, denominator_key) in RECOVERY_RATE_METRICS.items():
        if rate_key not in metric_keys:
            continue
        successes = sum(float(episode.metrics.get(success_key, 0.0)) for episode in episodes)
        denominator = sum(float(episode.metrics.get(denominator_key, 0.0)) for episode in episodes)
        metrics[rate_key] = successes / denominator if denominator else 0.0
        metric_ci95[rate_key] = _wilson_ci(round(successes), round(denominator))
    return metrics, metric_ci95


def _wilson_ci(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    proportion = successes / total
    denominator = 1.0 + z**2 / total
    center = (proportion + z**2 / (2.0 * total)) / denominator
    margin = z * sqrt((proportion * (1.0 - proportion) + z**2 / (4.0 * total)) / total) / denominator
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
