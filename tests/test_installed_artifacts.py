from __future__ import annotations

from pathlib import Path

import pytest

from nyssa_bench.package_resources import config_root, policy_example_root, resource_root
from nyssa_bench.packaging_smoke import run_packaging_smoke
from scripts.validate_distributions import validate_distributions


def test_resource_resolver_finds_source_bundles() -> None:
    assert (config_root("suites") / "tabletop_manipulation_v0.yaml").is_file()
    assert (resource_root("conformance") / "scenario" / "README.md").is_file()
    assert (
        resource_root("schemas") / "nep" / "0.1.0" / "nep-manifest.schema.json"
    ).is_file()
    assert (policy_example_root() / "state_policy.py").is_file()


@pytest.mark.parametrize("name", ["", ".", "..", "../configs", "a/b", "a\\b"])
def test_resource_resolver_rejects_non_segment_names(name: str) -> None:
    with pytest.raises(ValueError, match="one path segment"):
        resource_root(name)


def test_packaging_smoke_generates_complete_integration_pack(tmp_path: Path) -> None:
    report = run_packaging_smoke(tmp_path)

    assert report["format"] == "nyssa-installed-artifact-smoke-v1"
    assert report["success_rate"] == 1.0
    assert report["benchmark_tier"] == "prototype"
    assert report["public_claim"] is False
    assert report["suite_count"] >= 1
    assert report["stressor_count"] >= 1
    assert (tmp_path / "installed_artifact_smoke.json").is_file()
    for artifact in report["artifacts"]:
        assert (tmp_path / artifact).is_file()


def test_distribution_validator_requires_exactly_two_artifacts() -> None:
    with pytest.raises(ValueError, match="one wheel and one"):
        validate_distributions([])
