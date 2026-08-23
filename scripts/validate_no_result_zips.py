#!/usr/bin/env python
"""Pre-commit hook helper to avoid committing local benchmark result zips."""

from __future__ import annotations

import sys
from pathlib import Path


def _is_result_zip(path: str) -> bool:
    name = Path(path).name
    return name.endswith(".zip") and "_results_" in name


def main() -> int:
    files = [path for path in sys.argv[1:] if _is_result_zip(path)]
    if not files:
        return 0

    print("[pre-commit] blocked benchmark result zip artifacts:")
    for path in files:
        print(f"  - {path}")
    print("These files should be uploaded separately, not committed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
