from __future__ import annotations

import math
from collections import Counter
from typing import Mapping, Sequence


def pearson_correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    _paired_lengths(left, right)
    if len(left) < 2:
        return None
    left_mean = math.fsum(left) / len(left)
    right_mean = math.fsum(right) / len(right)
    numerator = math.fsum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right)
    )
    left_scale = math.fsum((value - left_mean) ** 2 for value in left)
    right_scale = math.fsum((value - right_mean) ** 2 for value in right)
    denominator = math.sqrt(left_scale * right_scale)
    return numerator / denominator if denominator > 0.0 else None


def spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    _paired_lengths(left, right)
    return pearson_correlation(_average_ranks(left), _average_ranks(right))


def kendall_tau_b(left: Sequence[float], right: Sequence[float]) -> float | None:
    _paired_lengths(left, right)
    if len(left) < 2:
        return None
    concordant = discordant = left_ties = right_ties = 0
    for index in range(len(left)):
        for other in range(index + 1, len(left)):
            left_sign = _sign(left[index] - left[other])
            right_sign = _sign(right[index] - right[other])
            if left_sign == 0 and right_sign == 0:
                continue
            if left_sign == 0:
                left_ties += 1
            elif right_sign == 0:
                right_ties += 1
            elif left_sign == right_sign:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + left_ties)
        * (concordant + discordant + right_ties)
    )
    return (concordant - discordant) / denominator if denominator > 0 else None


def mean_maximum_rank_violation(
    simulated: Sequence[float], real: Sequence[float]
) -> float | None:
    _paired_lengths(simulated, real)
    if len(simulated) < 2:
        return None
    violations = []
    for index in range(len(simulated)):
        maximum = 0.0
        for other in range(len(simulated)):
            sim_order = _sign(simulated[index] - simulated[other])
            real_order = _sign(real[index] - real[other])
            if sim_order != real_order:
                maximum = max(maximum, abs(real[index] - real[other]))
        violations.append(maximum)
    return math.fsum(violations) / len(violations)


def failure_distribution_similarity(
    simulated: Mapping[str, int | float], real: Mapping[str, int | float]
) -> float | None:
    keys = sorted(set(simulated) | set(real))
    sim_total = math.fsum(float(simulated.get(key, 0.0)) for key in keys)
    real_total = math.fsum(float(real.get(key, 0.0)) for key in keys)
    if sim_total <= 0.0 or real_total <= 0.0:
        return None
    p = [float(simulated.get(key, 0.0)) / sim_total for key in keys]
    q = [float(real.get(key, 0.0)) / real_total for key in keys]
    midpoint = [(left + right) / 2.0 for left, right in zip(p, q)]
    divergence = (_kl(p, midpoint) + _kl(q, midpoint)) / 2.0
    return max(0.0, min(1.0, 1.0 - divergence))


def category_counts(categories: Sequence[str | None]) -> dict[str, int]:
    return dict(sorted(Counter(value or "none" for value in categories).items()))


def _average_ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    position = 0
    while position < len(ordered):
        end = position + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[position]]:
            end += 1
        rank = (position + 1 + end) / 2.0
        for index in ordered[position:end]:
            ranks[index] = rank
        position = end
    return ranks


def _kl(left: Sequence[float], right: Sequence[float]) -> float:
    return math.fsum(
        value * math.log2(value / other)
        for value, other in zip(left, right)
        if value > 0.0
    )


def _paired_lengths(left: Sequence[float], right: Sequence[float]) -> None:
    if len(left) != len(right):
        raise ValueError("paired metric inputs must have equal length")
    if any(not math.isfinite(value) for value in (*left, *right)):
        raise ValueError("paired metric inputs must be finite")


def _sign(value: float) -> int:
    return 1 if value > 0 else -1 if value < 0 else 0
