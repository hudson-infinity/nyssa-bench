"""Legacy scalar helpers retained only for import compatibility.

NyssaBench does not call these functions when writing, comparing, or ranking
new result artifacts. Use :mod:`nyssa_bench.metrics.vector` for current output.
"""

from __future__ import annotations

import warnings
from typing import Any


def prototype_reliability_score(metrics: dict[str, float]) -> float:
    """Return the historical heuristic for legacy callers."""

    _warn_legacy("prototype_reliability_score")
    return _legacy_score(metrics)


def _legacy_score(metrics: dict[str, float]) -> float:
    success = metrics.get("success_rate", 0.0)
    safety = 1.0 - metrics.get("safety_violation_rate", 0.0)
    robustness = 1.0 - metrics.get("out_of_distribution_failure_rate", 0.0)
    return max(0.0, min(1.0, 0.5 * success + 0.25 * safety + 0.25 * robustness))


def sim_to_real_score(metrics: dict[str, float]) -> float:
    """Return the historical, misnamed scalar for legacy callers."""

    _warn_legacy("sim_to_real_score")
    return _legacy_score(metrics)


def score_summary(summary: dict[str, Any]) -> float:
    """Return the historical summary scalar for legacy callers."""

    _warn_legacy("score_summary")
    metric_means = dict(summary.get("metrics", {}))
    flat = {
        "success_rate": float(summary.get("success_rate", 0.0)),
        "safety_violation_rate": float(metric_means.get("safety_violation_rate", 0.0)),
        "out_of_distribution_failure_rate": float(
            metric_means.get("out_of_distribution_failure_rate", 0.0)
        ),
    }
    return _legacy_score(flat)


def _warn_legacy(name: str) -> None:
    warnings.warn(
        f"{name} is a legacy heuristic and is not emitted or ranked by current "
        "NyssaBench schemas; use nyssa_bench.metrics.vector instead",
        DeprecationWarning,
        stacklevel=2,
    )
