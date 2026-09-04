from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from nyssa_bench.recovery.metrics import summarize_counterfactual_recovery
from nyssa_bench.recovery.protocol import (
    COUNTERFACTUAL_RECOVERY_MANIFEST_FORMAT,
    CounterfactualRecoveryRecord,
)


def write_counterfactual_recovery_manifest(
    episodes: Sequence[Any],
    out_dir: str | Path,
    *,
    configuration: dict[str, Any],
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = [record for episode in episodes for record in _episode_records(episode)]
    summary = summarize_counterfactual_recovery(episodes)
    summary["requested"] = bool(configuration.get("enabled", False))
    payload = {
        "format": COUNTERFACTUAL_RECOVERY_MANIFEST_FORMAT,
        "configuration": configuration,
        "summary": summary,
        "branch_points": [record.to_dict() for record in records],
    }
    path = out_dir / "counterfactual_recovery.json"
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return path


def load_counterfactual_recovery_manifest(
    path: str | Path,
) -> tuple[dict[str, Any], tuple[CounterfactualRecoveryRecord, ...]]:
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Counterfactual recovery manifest is invalid JSON: {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("counterfactual recovery manifest must contain an object")
    if payload.get("format") != COUNTERFACTUAL_RECOVERY_MANIFEST_FORMAT:
        raise ValueError(
            "Unsupported counterfactual recovery manifest format: "
            f"{payload.get('format')}"
        )
    configuration = payload.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError("counterfactual recovery configuration must be a mapping")
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("counterfactual recovery summary must be a mapping")
    enabled = configuration.get("enabled")
    if not isinstance(enabled, bool):
        raise ValueError("counterfactual recovery configuration.enabled must be boolean")
    raw_records = payload.get("branch_points")
    if not isinstance(raw_records, list):
        raise ValueError("counterfactual recovery branch_points must be a list")
    records = tuple(_parse_record(item) for item in raw_records)
    identities = [record.branch_point.branch_point_id for record in records]
    if len(identities) != len(set(identities)):
        raise ValueError("counterfactual recovery branch-point IDs must be unique")
    try:
        eligible = int(summary["eligible_branch_points"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "counterfactual recovery summary requires eligible_branch_points"
        ) from exc
    recomputed = summarize_counterfactual_recovery(
        [
            {
                "counterfactual_recovery": list(records),
                "metrics": {
                    "counterfactual_eligible_branch_point_count": float(eligible)
                },
            }
        ]
    )
    recomputed["requested"] = enabled
    if summary != recomputed:
        raise ValueError(
            "counterfactual recovery summary does not match branch-point evidence"
        )
    return payload, records


def _episode_records(episode: Any) -> list[CounterfactualRecoveryRecord]:
    values = (
        episode.get("counterfactual_recovery", [])
        if isinstance(episode, dict)
        else getattr(episode, "counterfactual_recovery", [])
    )
    records: list[CounterfactualRecoveryRecord] = []
    for value in values or []:
        if isinstance(value, CounterfactualRecoveryRecord):
            records.append(value)
        elif isinstance(value, dict):
            records.append(CounterfactualRecoveryRecord.from_dict(value))
        else:
            raise ValueError(
                "counterfactual recovery records must be objects or protocol records"
            )
    return records


def _parse_record(value: Any) -> CounterfactualRecoveryRecord:
    if not isinstance(value, dict):
        raise ValueError("counterfactual recovery branch-point records must be mappings")
    return CounterfactualRecoveryRecord.from_dict(value)
