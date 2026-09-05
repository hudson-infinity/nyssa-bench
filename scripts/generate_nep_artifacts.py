from __future__ import annotations

import json
from pathlib import Path

from nyssa_bench.nep.reference import reference_pipeline_manifest
from nyssa_bench.nep.schemas import write_schemas


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    write_schemas(ROOT / "schemas" / "nep" / "0.1.0")
    fixture_root = ROOT / "conformance" / "nep" / "0.1.0"
    valid_root = fixture_root / "valid"
    invalid_root = fixture_root / "invalid"
    valid_root.mkdir(parents=True, exist_ok=True)
    invalid_root.mkdir(parents=True, exist_ok=True)
    for engine in ("mujoco", "maniskill"):
        payload = reference_pipeline_manifest(engine).model_dump(mode="json")
        _write(valid_root / f"{engine}-pipeline.json", payload)
    invalid_hash = reference_pipeline_manifest("mujoco").model_dump(mode="json")
    invalid_hash["policy"]["checkpoint_id"] = "tampered-after-hash"
    _write(invalid_root / "content-hash-mismatch.json", invalid_hash)
    unknown = reference_pipeline_manifest("mujoco").model_dump(mode="json")
    unknown["failure_evidence"]["ledger_artifact_id"] = "missing-ledger"
    _write(invalid_root / "unknown-artifact-reference.json", unknown)
    return 0


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
