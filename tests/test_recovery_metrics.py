from nyssa_bench.core.episode import EpisodeResult
from nyssa_bench.metrics.success import aggregate_episodes
from nyssa_bench.reports.result_pack import _policy_metric_rates, _policy_rows


def _episode(
    episode_index: int,
    *,
    attempts: int,
    applied: int,
    successes: int,
    episode_success: int,
) -> EpisodeResult:
    return EpisodeResult(
        task_id="unit_task",
        episode_index=episode_index,
        seed=episode_index,
        success=bool(episode_success),
        failure_label=None if episode_success else "missed_target",
        metrics={
            "recovery_attempt_count": float(attempts),
            "recovery_applied_count": float(applied),
            "recovery_success_count": float(successes),
            "recovery_failure_count": float(applied - successes),
            "recovery_not_applied_count": float(attempts - applied),
            "recovery_success_rate": successes / applied if applied else 0.0,
            "recovery_episode_attempt_count": float(attempts > 0),
            "recovery_episode_applied_count": float(applied > 0),
            "recovery_episode_success_count": float(successes > 0),
            "recovery_episode_success_rate": float(successes > 0) if applied else 0.0,
        },
    )


def test_recovery_rates_aggregate_counts_before_division():
    summary = aggregate_episodes(
        [
            _episode(0, attempts=1, applied=1, successes=1, episode_success=1),
            _episode(1, attempts=9, applied=9, successes=0, episode_success=0),
        ]
    )

    metrics = summary["metrics"]
    assert metrics["recovery_attempt_count"] == 10.0
    assert metrics["recovery_applied_count"] == 10.0
    assert metrics["recovery_success_count"] == 1.0
    assert metrics["recovery_failure_count"] == 9.0
    assert metrics["recovery_success_rate"] == 0.1
    assert metrics["recovery_episode_applied_count"] == 2.0
    assert metrics["recovery_episode_success_count"] == 1.0
    assert metrics["recovery_episode_success_rate"] == 0.5


def test_result_pack_aggregates_recovery_rate_from_cross_run_counts():
    summaries = [
        {
            "policy": "unit_policy",
            "episodes": 1,
            "metrics": {
                "recovery_applied_count": 1.0,
                "recovery_success_count": 1.0,
                "recovery_success_rate": 1.0,
            },
        },
        {
            "policy": "unit_policy",
            "episodes": 1,
            "metrics": {
                "recovery_applied_count": 9.0,
                "recovery_success_count": 0.0,
                "recovery_success_rate": 0.0,
            },
        },
    ]

    rates = _policy_metric_rates(summaries, "recovery_success_rate")

    assert rates == {"unit_policy": 0.1}
    assert "0.1000 (1/10)" in _policy_rows(summaries)
