from __future__ import annotations

import argparse
import json
from pathlib import Path

from nyssa_bench.hardware_study.candidate import build_hardware_study_candidate


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "configs" / "hardware" / "nyssa_hardware_calibration_v0_1.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the hardware study draft.")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    payload = (
        json.dumps(
            build_hardware_study_candidate(ROOT).model_dump(mode="json"),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    encoded = payload.encode("utf-8")
    if OUTPUT.exists() and OUTPUT.read_bytes() == encoded:
        print(f"hardware study candidate is current: {OUTPUT}")
        return 0
    if args.check:
        print(f"hardware study candidate is stale or missing: {OUTPUT}")
        return 1
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_bytes(encoded)
    print(f"wrote hardware study candidate: {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
