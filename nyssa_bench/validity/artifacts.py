from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from nyssa_bench.validity.protocol import (
    BenchmarkValidityReport,
    BenchmarkValiditySpec,
)


def load_benchmark_validity_spec(path: str | Path) -> BenchmarkValiditySpec:
    path = Path(path)
    data = _load_mapping(path)
    return BenchmarkValiditySpec.from_dict(data)


def write_benchmark_validity_report(
    report: BenchmarkValidityReport, path: str | Path
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.to_dict(), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def load_benchmark_validity_report(path: str | Path) -> BenchmarkValidityReport:
    path = Path(path)
    return BenchmarkValidityReport.from_dict(_load_mapping(path))


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read benchmark-validity artifact: {path}") from exc
    try:
        data = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid benchmark-validity artifact: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError("benchmark-validity artifact must contain a mapping")
    return data
