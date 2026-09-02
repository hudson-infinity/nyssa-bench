from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nyssa_bench.stressors import StressorConfig

from .protocol import SCENARIO_EXECUTION_FORMAT, ScenarioPackage
from .validation import ScenarioValidationReport


def scenario_execution_context(
    package: ScenarioPackage,
    validation: ScenarioValidationReport,
    stressor_config: StressorConfig,
) -> dict[str, Any]:
    resolved = set(validation.resolved_assets)
    return {
        "format": SCENARIO_EXECUTION_FORMAT,
        "scenario_identity": package.identity,
        "scenario_id": package.scenario_id,
        "scenario_version": package.scenario_version,
        "content_sha256": package.content_sha256,
        "protocol": package.protocol.to_dict(),
        "generator": package.generator.to_dict(),
        "engine": package.engine.to_dict(),
        "initial_state": package.initial_state.to_dict(),
        "assets": [
            {
                "asset_id": asset.asset_id,
                "sha256": asset.sha256,
                "license_id": asset.license_id,
                "provenance_uri": asset.provenance_uri,
                "redistribution": asset.redistribution,
                "required": asset.required,
                "resolved": asset.asset_id in resolved,
            }
            for asset in package.assets
        ],
        "split_lineage": [split.to_dict() for split in package.split_lineage],
        "evaluation": package.evaluation.to_dict(),
        "rare_event_provenance": package.rare_event_provenance,
        "stressor_config": stressor_config.to_dict(),
        "validation": validation.to_dict(),
    }


def write_scenario_execution(context: dict[str, Any], out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    path = out_dir / "scenario_execution.json"
    path.write_text(
        json.dumps(context, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path
