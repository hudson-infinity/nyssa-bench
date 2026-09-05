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
    ".github/actionlint.yaml",
    ".github/workflows/release.yml",
    ".github/workflows/installed-simulators.yml",
    ".github/workflows/container-ci.yml",
    "docs/getting_started.md",
    "docs/benchmark_protocol.md",
    "docs/claim_evidence.md",
    "docs/releasing.md",
    "docs/installed_artifact_validation.md",
    "docs/simulator_ci.md",
    "docs/nyssa_evaluation_protocol.md",
    "docs/external_policy_quickstart.md",
    "docs/sim_real_study.md",
    "docs/phase1_credibility_gate.md",
    "examples/policies/state_policy.py",
    "examples/policies/state_policy_contract.json",
    "schemas/nep/0.1.0/nep-manifest.schema.json",
    "conformance/nep/0.1.0/valid/mujoco-pipeline.json",
    "conformance/nep/0.1.0/valid/maniskill-pipeline.json",
    "scripts/generate_nep_artifacts.py",
    "claims/claim_evidence.json",
    "claims/phase1_credibility.json",
    "scripts/validate_claim_evidence.py",
    "scripts/validate_credibility.py",
    "scripts/validate_release_version.py",
    "scripts/validate_distributions.py",
    "nyssa_bench/container_smoke.py",
    "nyssa_bench/release_bundle.py",
    "tests/test_release_bundle.py",
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
