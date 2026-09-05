from __future__ import annotations

from pathlib import Path
from typing import Any

from nyssa_bench.real_evidence import (
    RealEvidencePackage,
    RealEvidenceValidationError,
    RealEvidenceValidator,
    comparison_pairs,
)
from nyssa_bench.simreal.metrics import pearson_correlation


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

    if len(sim_success) != len(real_success):
        return None
    return pearson_correlation(sim_success, real_success)
