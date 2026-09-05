from __future__ import annotations

from dataclasses import dataclass

from nyssa_bench.stress_search import (
    BoundaryStressSampler,
    LatinHypercubeStressSampler,
    RandomStressSampler,
    StressObservation,
    StressSearchSpace,
    StressSearchStudy,
    StressSearchStudySpec,
    compare_stress_search_studies,
    make_stress_sampler,
)

__all__ = [
    "BoundaryStressSampler",
    "LatinHypercubeStressSampler",
    "RandomStressSampler",
    "StressCandidate",
    "StressObservation",
    "StressSearchSpace",
    "StressSearchStudy",
    "StressSearchStudySpec",
    "compare_stress_search_studies",
    "make_stress_sampler",
    "rank_stress_candidates",
]


@dataclass(frozen=True)
class StressCandidate:
    stressor: str
    value: str
    success_rate: float
    episodes: int


def rank_stress_candidates(candidates: list[StressCandidate]) -> list[StressCandidate]:
    """Rank stress settings from most failure-inducing to least."""

    return sorted(candidates, key=lambda item: (item.success_rate, -item.episodes, item.stressor, item.value))
