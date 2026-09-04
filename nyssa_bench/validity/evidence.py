from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def paired_design_audit_inputs(
    comparison: Mapping[str, Any], coverage: Any
) -> dict[str, Any]:
    """Adapt issue #10/#11 comparison evidence to the paired-design audit."""

    coverage_payload = (
        coverage.to_dict() if callable(getattr(coverage, "to_dict", None)) else coverage
    )
    if not isinstance(coverage_payload, Mapping):
        raise TypeError("pairwise coverage must be a mapping or expose to_dict()")
    contract_hash = comparison.get("comparison_contract_sha256")
    if not isinstance(contract_hash, str) or len(contract_hash) != 64:
        raise ValueError("comparison evidence requires comparison_contract_sha256")
    return {
        "comparison": {
            "comparable": comparison.get("comparable") is True,
            "mismatches": list(comparison.get("mismatches", [])),
            "comparison_contract_sha256": contract_hash,
            "comparison_mode": comparison.get("comparison_mode"),
        },
        "coverage": dict(coverage_payload),
    }
