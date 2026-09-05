from __future__ import annotations

import argparse
import json
from pathlib import Path

from nyssa_bench.policy_tracks.candidate import build_policy_track_candidate


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "configs" / "policy_tracks" / "nyssa_policy_tracks_v0_1.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the policy-track candidate.")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    payload = (
        json.dumps(
            build_policy_track_candidate(ROOT).model_dump(mode="json"),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    encoded = payload.encode("utf-8")
    if OUTPUT.exists() and OUTPUT.read_bytes() == encoded:
        print(f"policy-track candidate is current: {OUTPUT}")
        return 0
    if args.check:
        print(f"policy-track candidate is stale or missing: {OUTPUT}")
        return 1
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_bytes(encoded)
    print(f"wrote policy-track candidate: {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
