"""Pairwise evaluation helpers for future arena-style comparisons."""

from nyssa_bench.arena.arena_report import (
    save_arena_report,
    save_pairwise_coverage,
    save_pairwise_results,
    save_pairwise_summary,
    save_preference_table,
)
from nyssa_bench.arena.pairwise_runner import (
    DuplicateEpisodeKey,
    DuplicateEpisodeKeysError,
    EpisodeKey,
    IncompletePairwiseComparisonError,
    PairwiseComparisonError,
    PairwiseCoverage,
    PairwiseOutcome,
    PairwiseSummary,
    assess_episode_pairing,
    compare_episode_pairs,
)
from nyssa_bench.arena.preference_schema import PreferenceRecord

__all__ = [
    "DuplicateEpisodeKey",
    "DuplicateEpisodeKeysError",
    "EpisodeKey",
    "IncompletePairwiseComparisonError",
    "PairwiseComparisonError",
    "PairwiseCoverage",
    "PairwiseOutcome",
    "PairwiseSummary",
    "PreferenceRecord",
    "assess_episode_pairing",
    "compare_episode_pairs",
    "save_arena_report",
    "save_pairwise_coverage",
    "save_pairwise_results",
    "save_pairwise_summary",
    "save_preference_table",
]
