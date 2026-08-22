from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StressCandidate:
    stressor: str
    value: str
    success_rate: float
    episodes: int


def rank_stress_candidates(candidates: list[StressCandidate]) -> list[StressCandidate]:
    """Rank stress settings from most failure-inducing to least."""

    return sorted(candidates, key=lambda item: (item.success_rate, -item.episodes, item.stressor, item.value))
