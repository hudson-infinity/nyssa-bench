from nyssa_bench.core.episode import EpisodeResult
from nyssa_bench.research import (
    StressCandidate,
    failure_episode_indices,
    paired_success_correlation,
    rank_stress_candidates,
)


def test_paired_success_correlation():
    assert paired_success_correlation([0.1, 0.5, 0.9], [0.2, 0.6, 1.0]) is not None
    assert paired_success_correlation([0.1], [0.2]) is None
    assert paired_success_correlation([0.5, 0.5], [0.1, 0.9]) is None


def test_rank_stress_candidates_orders_low_success_first():
    candidates = [
        StressCandidate("lighting", "bright", 0.8, 100),
        StressCandidate("camera", "tilted", 0.2, 50),
        StressCandidate("camera", "wide", 0.2, 100),
    ]

    ranked = rank_stress_candidates(candidates)

    assert [item.value for item in ranked] == ["wide", "tilted", "bright"]


def test_failure_episode_indices():
    episodes = [
        _episode(0, success=True),
        _episode(1, success=False, failure_label="timeout"),
        _episode(2, success=False, failure_label="bad_grasp"),
    ]

    assert failure_episode_indices(episodes) == [1, 2]
    assert failure_episode_indices(episodes, failure_label="timeout") == [1]


def _episode(index: int, *, success: bool, failure_label: str | None = None):
    return EpisodeResult(
        task_id="pick",
        episode_index=index,
        seed=index,
        success=success,
        failure_label=failure_label,
        failure_label_source="mapper" if failure_label else None,
        metrics={},
    )
