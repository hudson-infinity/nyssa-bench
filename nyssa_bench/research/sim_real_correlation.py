from __future__ import annotations

from math import sqrt
from pathlib import Path
from typing import Any

from nyssa_bench.real_evidence import (
    RealEvidencePackage,
    RealEvidenceValidationError,
    RealEvidenceValidator,
    comparison_pairs,
)


def load_sim_real_evidence_pairs(
    package_or_path: RealEvidencePackage | str | Path,
) -> list[dict[str, Any]]:
    """Load comparison-ready pairs from the versioned evidence contract."""

    package = (
        package_or_path
        if isinstance(package_or_path, RealEvidencePackage)
        else RealEvidencePackage.load(package_or_path)
    )
    validation = RealEvidenceValidator().validate(package)
    validation.raise_for_errors()
    if not validation.comparison_ready:
        raise RealEvidenceValidationError(
            "Real evidence package is not comparison-ready",
            report=validation,
        )
    return comparison_pairs(package)


def paired_success_correlation(
    sim_success: list[float], real_success: list[float]
) -> float | None:
    """Return Pearson correlation for paired simulator and real success rates."""

    if len(sim_success) != len(real_success) or len(sim_success) < 2:
        return None
    sim_mean = sum(sim_success) / len(sim_success)
    real_mean = sum(real_success) / len(real_success)
    numerator = sum(
        (sim - sim_mean) * (real - real_mean)
        for sim, real in zip(sim_success, real_success)
    )
    sim_var = sum((sim - sim_mean) ** 2 for sim in sim_success)
    real_var = sum((real - real_mean) ** 2 for real in real_success)
    denominator = sqrt(sim_var * real_var)
    if denominator == 0:
        return None
    return numerator / denominator
