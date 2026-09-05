from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from nyssa_bench.nep.protocol import (
    ClaimContract,
    FailureEvidenceContract,
    InterventionContract,
    NEPManifest,
    PolicyContract,
    StressorContract,
    TaskContract,
)


SCHEMA_MODELS: dict[str, type[BaseModel]] = {
    "task-contract.schema.json": TaskContract,
    "stressor-contract.schema.json": StressorContract,
    "policy-contract.schema.json": PolicyContract,
    "failure-evidence-contract.schema.json": FailureEvidenceContract,
    "intervention-contract.schema.json": InterventionContract,
    "claim-contract.schema.json": ClaimContract,
    "nep-manifest.schema.json": NEPManifest,
}


def generated_schemas() -> dict[str, dict[str, Any]]:
    return {
        name: model.model_json_schema()
        for name, model in sorted(SCHEMA_MODELS.items())
    }


def write_schemas(out_dir: str | Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, schema in generated_schemas().items():
        path = out_dir / name
        path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
        paths.append(path)
    return paths
