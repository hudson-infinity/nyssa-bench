from __future__ import annotations

import json
from pathlib import Path

from nyssa_bench.claims import load_claim_evidence, validate_claim_evidence


REPO_ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = REPO_ROOT / "claims" / "claim_evidence.json"


def main() -> int:
    try:
        matrix = load_claim_evidence(MATRIX_PATH)
        report = validate_claim_evidence(matrix, repo_root=REPO_ROOT)
    except ValueError as exc:
        print(f"claim evidence validation failed: {exc}")
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
