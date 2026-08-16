from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from nyssa_bench.core.episode import EpisodeResult


@dataclass(frozen=True, order=True)
class EpisodeKey:
    task_id: str
    seed: int
    episode_index: int

    def to_dict(self) -> dict[str, str | int]:
        return {
            "task_id": self.task_id,
            "seed": self.seed,
            "episode_index": self.episode_index,
        }


@dataclass(frozen=True)
class DuplicateEpisodeKey:
    key: EpisodeKey
    count: int

    def to_dict(self) -> dict[str, Any]:
        return {**self.key.to_dict(), "count": self.count}


@dataclass(frozen=True)
class PairwiseCoverage:
    policy_a_label: str
    policy_b_label: str
    policy_a_requested_count: int
    policy_b_requested_count: int
    policy_a_unique_count: int
    policy_b_unique_count: int
    matched_keys: tuple[EpisodeKey, ...]
    unmatched_a_keys: tuple[EpisodeKey, ...]
    unmatched_b_keys: tuple[EpisodeKey, ...]
    duplicate_a_keys: tuple[DuplicateEpisodeKey, ...]
    duplicate_b_keys: tuple[DuplicateEpisodeKey, ...]

    @property
    def matched_count(self) -> int:
        return len(self.matched_keys)

    @property
    def unmatched_a_count(self) -> int:
        return len(self.unmatched_a_keys)

    @property
    def unmatched_b_count(self) -> int:
        return len(self.unmatched_b_keys)

    @property
    def duplicate_a_count(self) -> int:
        return len(self.duplicate_a_keys)

    @property
    def duplicate_b_count(self) -> int:
        return len(self.duplicate_b_keys)

    @property
    def policy_a_coverage(self) -> float:
        return _ratio(self.matched_count, self.policy_a_unique_count)

    @property
    def policy_b_coverage(self) -> float:
        return _ratio(self.matched_count, self.policy_b_unique_count)

    @property
    def joint_coverage(self) -> float:
        union_count = (
            self.policy_a_unique_count + self.policy_b_unique_count - self.matched_count
        )
        return _ratio(self.matched_count, union_count)

    @property
    def complete(self) -> bool:
        return (
            self.matched_count > 0
            and self.unmatched_a_count == 0
            and self.unmatched_b_count == 0
            and self.duplicate_a_count == 0
            and self.duplicate_b_count == 0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_a_label": self.policy_a_label,
            "policy_b_label": self.policy_b_label,
            "policy_a_requested_count": self.policy_a_requested_count,
            "policy_b_requested_count": self.policy_b_requested_count,
            "policy_a_unique_count": self.policy_a_unique_count,
            "policy_b_unique_count": self.policy_b_unique_count,
            "matched_count": self.matched_count,
            "unmatched_a_count": self.unmatched_a_count,
            "unmatched_b_count": self.unmatched_b_count,
            "duplicate_a_count": self.duplicate_a_count,
            "duplicate_b_count": self.duplicate_b_count,
            "policy_a_coverage": self.policy_a_coverage,
            "policy_b_coverage": self.policy_b_coverage,
            "joint_coverage": self.joint_coverage,
            "complete": self.complete,
            "matched_keys": [key.to_dict() for key in self.matched_keys],
            "unmatched_a_keys": [key.to_dict() for key in self.unmatched_a_keys],
            "unmatched_b_keys": [key.to_dict() for key in self.unmatched_b_keys],
            "duplicate_a_keys": [item.to_dict() for item in self.duplicate_a_keys],
            "duplicate_b_keys": [item.to_dict() for item in self.duplicate_b_keys],
        }


class PairwiseComparisonError(ValueError):
    def __init__(self, message: str, coverage: PairwiseCoverage) -> None:
        super().__init__(message)
        self.coverage = coverage


class DuplicateEpisodeKeysError(PairwiseComparisonError):
    pass


class IncompletePairwiseComparisonError(PairwiseComparisonError):
    pass


@dataclass(frozen=True)
class PairwiseOutcome:
    task_id: str
    seed: int
    episode_index: int
    winner: str
    policy_a_success: bool
    policy_b_success: bool
    policy_a_failure: str | None
    policy_b_failure: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "seed": self.seed,
            "episode_index": self.episode_index,
            "winner": self.winner,
            "policy_a_success": self.policy_a_success,
            "policy_b_success": self.policy_b_success,
            "policy_a_failure": self.policy_a_failure,
            "policy_b_failure": self.policy_b_failure,
        }


@dataclass(frozen=True)
class PairwiseSummary:
    outcomes: tuple[PairwiseOutcome, ...]
    wins: dict[str, int]
    failure_deltas: dict[str, int]
    coverage: PairwiseCoverage
    comparison_mode: str
    pairing_claim_eligible: bool
    caveats: tuple[str, ...]

    @property
    def total_pairs(self) -> int:
        return len(self.outcomes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "nyssa-pairwise-summary-v1",
            "comparison_mode": self.comparison_mode,
            "pairing_claim_eligible": self.pairing_claim_eligible,
            "caveats": list(self.caveats),
            "total_pairs": self.total_pairs,
            "wins": self.wins,
            "failure_deltas": self.failure_deltas,
            "coverage": self.coverage.to_dict(),
            "outcomes": [outcome.to_dict() for outcome in self.outcomes],
        }


def assess_episode_pairing(
    policy_a: list[EpisodeResult],
    policy_b: list[EpisodeResult],
    *,
    policy_a_label: str = "policy_a",
    policy_b_label: str = "policy_b",
) -> PairwiseCoverage:
    """Inspect episode identity coverage without running a comparison."""

    a_counts = Counter(_key(episode) for episode in policy_a)
    b_counts = Counter(_key(episode) for episode in policy_b)
    a_keys = set(a_counts)
    b_keys = set(b_counts)

    return PairwiseCoverage(
        policy_a_label=policy_a_label,
        policy_b_label=policy_b_label,
        policy_a_requested_count=len(policy_a),
        policy_b_requested_count=len(policy_b),
        policy_a_unique_count=len(a_keys),
        policy_b_unique_count=len(b_keys),
        matched_keys=tuple(sorted(a_keys & b_keys)),
        unmatched_a_keys=tuple(sorted(a_keys - b_keys)),
        unmatched_b_keys=tuple(sorted(b_keys - a_keys)),
        duplicate_a_keys=_duplicates(a_counts),
        duplicate_b_keys=_duplicates(b_counts),
    )


def compare_episode_pairs(
    policy_a: list[EpisodeResult],
    policy_b: list[EpisodeResult],
    *,
    policy_a_label: str = "policy_a",
    policy_b_label: str = "policy_b",
    allow_partial: bool = False,
) -> PairwiseSummary:
    """Compare episodes matched by task, seed, and episode index.

    Complete, duplicate-free matching is required by default. Exploratory
    comparisons may explicitly set ``allow_partial=True``; their summaries are
    visibly caveated and ineligible for benchmark claims. Duplicate identities
    are always rejected because selecting one would make the result ambiguous.
    """

    coverage = assess_episode_pairing(
        policy_a,
        policy_b,
        policy_a_label=policy_a_label,
        policy_b_label=policy_b_label,
    )
    if coverage.duplicate_a_keys or coverage.duplicate_b_keys:
        raise DuplicateEpisodeKeysError(_duplicate_error_message(coverage), coverage)
    if not coverage.complete and not allow_partial:
        raise IncompletePairwiseComparisonError(
            _incomplete_error_message(coverage), coverage
        )

    a_by_key = {_key(episode): episode for episode in policy_a}
    b_by_key = {_key(episode): episode for episode in policy_b}
    outcomes: list[PairwiseOutcome] = []
    wins: Counter[str] = Counter()
    failure_deltas: Counter[str] = Counter()

    for key in coverage.matched_keys:
        episode_a = a_by_key[key]
        episode_b = b_by_key[key]
        winner = _winner(episode_a, episode_b, policy_a_label, policy_b_label)
        wins[winner] += 1
        _count_failure_delta(
            failure_deltas, episode_a, episode_b, policy_a_label, policy_b_label
        )
        outcomes.append(
            PairwiseOutcome(
                task_id=episode_a.task_id,
                seed=episode_a.seed,
                episode_index=episode_a.episode_index,
                winner=winner,
                policy_a_success=episode_a.success,
                policy_b_success=episode_b.success,
                policy_a_failure=episode_a.failure_label,
                policy_b_failure=episode_b.failure_label,
            )
        )

    comparison_mode = "complete" if coverage.complete else "partial_exploratory"
    caveats = () if coverage.complete else _partial_caveats(coverage)
    return PairwiseSummary(
        outcomes=tuple(outcomes),
        wins=dict(wins),
        failure_deltas=dict(failure_deltas),
        coverage=coverage,
        comparison_mode=comparison_mode,
        pairing_claim_eligible=coverage.complete,
        caveats=caveats,
    )


def _key(episode: EpisodeResult) -> EpisodeKey:
    return EpisodeKey(episode.task_id, episode.seed, episode.episode_index)


def _duplicates(counts: Counter[EpisodeKey]) -> tuple[DuplicateEpisodeKey, ...]:
    return tuple(
        DuplicateEpisodeKey(key=key, count=count)
        for key, count in sorted(counts.items())
        if count > 1
    )


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _duplicate_error_message(coverage: PairwiseCoverage) -> str:
    details = []
    for label, duplicates in (
        (coverage.policy_a_label, coverage.duplicate_a_keys),
        (coverage.policy_b_label, coverage.duplicate_b_keys),
    ):
        if duplicates:
            keys = ", ".join(
                f"{_format_key(item.key)} x{item.count}" for item in duplicates
            )
            details.append(f"{label}: {keys}")
    return (
        "Duplicate episode keys make the pairwise comparison ambiguous: "
        + "; ".join(details)
    )


def _incomplete_error_message(coverage: PairwiseCoverage) -> str:
    return (
        "Pairwise comparison requires complete episode matching: "
        f"matched={coverage.matched_count}, "
        f"unmatched_{coverage.policy_a_label}={coverage.unmatched_a_count}, "
        f"unmatched_{coverage.policy_b_label}={coverage.unmatched_b_count}. "
        "Use allow_partial=True only for exploratory, non-claim comparisons."
    )


def _partial_caveats(coverage: PairwiseCoverage) -> tuple[str, ...]:
    caveats = [
        "Partial comparison: unmatched episodes were excluded; this output is exploratory and is not eligible for benchmark claims."
    ]
    if coverage.matched_count == 0:
        caveats.append(
            "No episode keys overlap, so this output contains no pairwise outcomes."
        )
    return tuple(caveats)


def _format_key(key: EpisodeKey) -> str:
    return f"({key.task_id!r}, {key.seed}, {key.episode_index})"


def _winner(
    episode_a: EpisodeResult,
    episode_b: EpisodeResult,
    policy_a_label: str,
    policy_b_label: str,
) -> str:
    if episode_a.success and not episode_b.success:
        return policy_a_label
    if episode_b.success and not episode_a.success:
        return policy_b_label
    if episode_a.success and episode_b.success:
        return "tie_success"
    return "tie_failure"


def _count_failure_delta(
    failure_deltas: Counter[str],
    episode_a: EpisodeResult,
    episode_b: EpisodeResult,
    policy_a_label: str,
    policy_b_label: str,
) -> None:
    if episode_a.failure_label and episode_a.failure_label != episode_b.failure_label:
        failure_deltas[f"{policy_a_label}:{episode_a.failure_label}"] += 1
    if episode_b.failure_label and episode_b.failure_label != episode_a.failure_label:
        failure_deltas[f"{policy_b_label}:{episode_b.failure_label}"] += 1
