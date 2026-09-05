from __future__ import annotations

from pathlib import Path


REQUIRED = [
    "README.md",
    "LICENSE",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    "SECURITY.md",
    "pyproject.toml",
    ".gitattributes",
    ".pre-commit-config.yaml",
    ".github/workflows/ci.yml",
    "docs/getting_started.md",
    "docs/benchmark_protocol.md",
    "docs/claim_evidence.md",
    "claims/claim_evidence.json",
    "scripts/validate_claim_evidence.py",
]


def main() -> int:
    missing = [path for path in REQUIRED if not Path(path).exists()]
    if missing:
        for path in missing:
            print(f"missing: {path}")
        return 1
    print("release checklist passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
