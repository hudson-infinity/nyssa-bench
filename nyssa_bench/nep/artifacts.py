from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from nyssa_bench.nep.protocol import NEPManifest
from nyssa_bench.nep.migration import migrate_nep_data
from nyssa_bench.nep.validation import NEPValidationReport, validate_nep_manifest


def load_nep_data(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
        data = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid NEP artifact: {path}") from exc
    if not isinstance(data, Mapping):
        raise ValueError("NEP artifact must contain a mapping")
    return dict(data)


def load_nep_manifest(path: str | Path) -> NEPManifest:
    data, _ = migrate_nep_data(load_nep_data(path))
    report, manifest = validate_nep_manifest(data)
    if manifest is None:
        details = "; ".join(f"{item.path}: {item.message}" for item in report.issues)
        raise ValueError(f"NEP validation failed: {details}")
    return manifest


def write_nep_manifest(manifest: NEPManifest, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def write_nep_validation_report(
    report: NEPValidationReport, path: str | Path
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path
