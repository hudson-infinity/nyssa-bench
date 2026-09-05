from __future__ import annotations

import json
import argparse
from pathlib import Path

from nyssa_bench.reference_benchmark.candidate import build_reference_candidate


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "configs" / "reference" / "nyssa_reference_v0_1.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the reference candidate.")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    spec = build_reference_candidate(ROOT)
    payload = (
        json.dumps(
            spec.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    if OUTPUT.exists() and OUTPUT.read_text(encoding="utf-8") == payload:
        print(f"reference candidate is current: {OUTPUT}")
        return 0
    if args.check:
        print(f"reference candidate is stale or missing: {OUTPUT}")
        return 1
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(payload, encoding="utf-8")
    print(f"wrote reference candidate: {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
