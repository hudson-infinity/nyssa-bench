from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from nyssa_bench.monitors.metrics import summarize_monitor_records
from nyssa_bench.monitors.protocol import (
    MONITOR_MANIFEST_FORMAT,
    FailureMonitorContract,
    MonitorPredictionRecord,
)


def write_monitor_manifest(
    records: Sequence[MonitorPredictionRecord],
    contracts: Mapping[str, FailureMonitorContract],
    support: Mapping[str, Any],
    path: str | Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": MONITOR_MANIFEST_FORMAT,
        "contracts": [contracts[key].to_dict() for key in sorted(contracts)],
        "support": {key: support[key] for key in sorted(support)},
        "records": [record.to_dict() for record in records],
        "summary": summarize_monitor_records(records, contracts),
    }
    payload["manifest_sha256"] = _sha256(payload)
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return path


def load_monitor_manifest(
    path: str | Path,
) -> tuple[
    dict[str, Any],
    dict[str, FailureMonitorContract],
    tuple[MonitorPredictionRecord, ...],
]:
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load failure monitor manifest: {path}") from exc
    if not isinstance(payload, dict) or payload.get("format") != MONITOR_MANIFEST_FORMAT:
        raise ValueError("unsupported failure monitor manifest")
    unknown = sorted(
        set(payload)
        - {
            "format",
            "contracts",
            "support",
            "records",
            "summary",
            "manifest_sha256",
        }
    )
    if unknown:
        raise ValueError(f"unknown monitor manifest fields: {', '.join(unknown)}")
    unhashed = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if payload.get("manifest_sha256") != _sha256(unhashed):
        raise ValueError("failure monitor manifest hash mismatch")
    contracts_raw = payload.get("contracts")
    records_raw = payload.get("records")
    support = payload.get("support")
    if not isinstance(contracts_raw, list) or not all(
        isinstance(item, Mapping) for item in contracts_raw
    ):
        raise ValueError("monitor contracts must be a list of mappings")
    if not isinstance(records_raw, list) or not all(
        isinstance(item, Mapping) for item in records_raw
    ):
        raise ValueError("monitor records must be a list of mappings")
    if not isinstance(support, Mapping):
        raise ValueError("monitor support must be a mapping")
    contracts_list = [FailureMonitorContract.from_dict(item) for item in contracts_raw]
    contracts = {contract.monitor_id: contract for contract in contracts_list}
    if len(contracts) != len(contracts_list):
        raise ValueError("monitor contract IDs must be unique")
    records = tuple(MonitorPredictionRecord.from_dict(item) for item in records_raw)
    expected_summary = summarize_monitor_records(records, contracts)
    if payload.get("summary") != expected_summary:
        raise ValueError("failure monitor summary does not match prediction records")
    if set(support) != set(contracts):
        raise ValueError("monitor support does not match contracts")
    return payload, contracts, records


def _sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
