from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from nyssa_bench.cli import _parse_severity_overrides, main
from nyssa_bench.engines.base import NyssaEngine
from nyssa_bench.plugins import get_plugin_registry
from nyssa_bench.reports.comparison import load_comparison_contract
from nyssa_bench.scenarios import (
    SCENARIO_PACKAGE_FORMAT,
    ScenarioAsset,
    ScenarioPackage,
    ScenarioPackageValidator,
    ScenarioSplitLineage,
    ScenarioStressorAxis,
    ScenarioValidationError,
    scenario_conformance_fixture_path,
    scenario_execution_context,
)
from nyssa_bench.runner import PolicyRunner


FIXTURE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "conformance"
    / "scenario"
    / "v1"
    / "valid_seeded_mujoco"
)


def test_conformance_fixture_is_discoverable_from_the_installed_module():
    assert scenario_conformance_fixture_path() == FIXTURE_ROOT.resolve()
    with pytest.raises(FileNotFoundError):
        scenario_conformance_fixture_path("missing_fixture")
    with pytest.raises(ValueError, match="package-relative"):
        scenario_conformance_fixture_path("../outside")


def _rehash(package: ScenarioPackage) -> ScenarioPackage:
    provisional = replace(package, content_sha256="0" * 64)
    return replace(provisional, content_sha256=provisional.compute_content_sha256())


def test_static_external_scenario_fixture_is_content_addressed_and_executable():
    package = ScenarioPackage.load(FIXTURE_ROOT)
    report = ScenarioPackageValidator().validate(package)

    assert package.identity.startswith("conformance_inverted_pendulum_delay@1.0.0:")
    assert package.compute_content_sha256() == package.content_sha256
    assert report.valid is True
    assert report.execution_ready is True
    assert report.claim_ready is True
    assert report.resolved_assets == ("scene_descriptor",)
    assert report.stressor_contracts[0]["stressor_id"] == "action_delay"
    assert package.to_dict()["format"] == SCENARIO_PACKAGE_FORMAT


def test_scenario_parser_rejects_missing_generator_provenance_and_unknown_fields():
    payload = yaml.safe_load(
        (FIXTURE_ROOT / "scenario.yaml").read_text(encoding="utf-8")
    )
    payload["generator"].pop("revision")
    with pytest.raises(ValueError, match="generator.revision"):
        ScenarioPackage.from_dict(payload)

    payload = yaml.safe_load(
        (FIXTURE_ROOT / "scenario.yaml").read_text(encoding="utf-8")
    )
    payload["generator_algorithm"] = "must_not_bypass_generator_contract"
    with pytest.raises(ValueError, match="Unknown scenario package fields"):
        ScenarioPackage.from_dict(payload)


def test_validator_rejects_package_hash_and_engine_mismatch():
    package = ScenarioPackage.load(FIXTURE_ROOT)
    report = ScenarioPackageValidator().validate(
        replace(package, content_sha256="f" * 64),
        expected_engine="maniskill",
    )
    codes = {issue.code for issue in report.issues}

    assert "package_hash_mismatch" in codes
    assert "incompatible_engine" in codes
    with pytest.raises(ScenarioValidationError):
        report.raise_for_errors()


def test_schema_rejects_malformed_runtime_and_provenance_identifiers():
    package = ScenarioPackage.load(FIXTURE_ROOT)
    with pytest.raises(ValueError, match="version constraint"):
        replace(package.engine, version_spec="latest")
    with pytest.raises(ValueError, match="absolute URI"):
        replace(package.generator, repository_url="local/repository")
    with pytest.raises(ValueError, match="URI schemes"):
        replace(package.assets[0], provenance_uri="file:///private/asset")
    with pytest.raises(ValueError, match="SPDX ID"):
        replace(package.assets[0], license_id="unknown license")
    with pytest.raises(ValueError, match="package-relative path"):
        replace(package.assets[0], path="C:\\private\\asset.bin")
    with pytest.raises(ValueError, match="package-relative path"):
        replace(package.assets[0], path="../outside.bin")

    rare_event = dict(package.rare_event_provenance or {})
    rare_event["raw_search_states"] = ["must-not-enter-public-manifest"]
    with pytest.raises(ValueError, match="Unknown rare_event_provenance fields"):
        replace(package, rare_event_provenance=rare_event)


def test_validator_rejects_unresolved_redistributable_asset():
    package = ScenarioPackage.load(FIXTURE_ROOT)
    missing = replace(package.assets[0], path="assets/missing.json")
    package = _rehash(replace(package, assets=(missing,)))

    report = ScenarioPackageValidator().validate(package)

    assert report.valid is False
    assert any(issue.code == "asset_unresolved" for issue in report.issues)


def test_protected_asset_can_identify_metadata_without_being_execution_ready():
    package = ScenarioPackage.load(FIXTURE_ROOT)
    protected = ScenarioAsset(
        asset_id="protected_scene",
        sha256="a" * 64,
        license_id="LicenseRef-Protected-Evaluation",
        provenance_uri="https://example.invalid/protected-scene-provenance",
        redistribution="protected",
        external_locator="provider://protected-scene-v1",
        required=True,
    )
    package = _rehash(replace(package, assets=(protected,)))

    metadata_report = ScenarioPackageValidator().validate(
        package, require_execution_assets=False
    )
    execution_report = ScenarioPackageValidator().validate(
        package, require_execution_assets=True
    )

    assert metadata_report.valid is True
    assert metadata_report.execution_ready is False
    assert metadata_report.unresolved_protected_assets == ("protected_scene",)
    assert execution_report.valid is False
    assert any(
        issue.code == "protected_asset_unresolved" for issue in execution_report.issues
    )


def test_stressor_axes_compile_only_through_versioned_stressor_specs():
    package = ScenarioPackage.load(FIXTURE_ROOT)
    config = package.stressor_config(severities={"action_delay": 0.75}, seed=19)

    assert config.to_dict()["format"] == "nyssa-stressor-config-v1"
    assert config.stressors[0].to_dict()["format"] == "nyssa-stressor-spec-v1"
    assert config.stressors[0].severity == 0.75
    assert config.stressors[0].seed == 19
    with pytest.raises(ValueError, match="Unknown scenario stressor"):
        package.stressor_config(severities={"unregistered": 0.5})
    with pytest.raises(ValueError, match="outside scenario axis"):
        package.stressor_config(severities={"action_delay": 1.1})


def test_runner_rejects_scenario_context_mismatches_before_execution():
    package = ScenarioPackage.load(FIXTURE_ROOT)
    validation = ScenarioPackageValidator().validate(package)
    config = package.stressor_config(seed=package.initial_state.run_seed)
    context = scenario_execution_context(package, validation, config)

    PolicyRunner(
        policy="random",
        engine="mujoco",
        seed=package.initial_state.run_seed,
        stressor_config=config,
        scenario_context=context,
    )
    with pytest.raises(ValueError, match="run seed"):
        PolicyRunner(
            policy="random",
            engine="mujoco",
            seed=package.initial_state.run_seed + 1,
            stressor_config=config,
            scenario_context=context,
        )
    with pytest.raises(ValueError, match="stressor config"):
        PolicyRunner(
            policy="random",
            engine="mujoco",
            seed=package.initial_state.run_seed,
            scenario_context=context,
        )


def test_validator_rejects_unknown_stressor_and_malformed_composition():
    package = ScenarioPackage.load(FIXTURE_ROOT)
    unknown = ScenarioStressorAxis(
        stressor_id="external_magic_shift",
        severity_range=(0.0, 1.0),
        default_severity=0.5,
    )
    package = _rehash(replace(package, stressor_axes=(unknown,)))
    report = ScenarioPackageValidator().validate(package)

    assert any(issue.code == "unknown_stressor_contract" for issue in report.issues)

    package = ScenarioPackage.load(FIXTURE_ROOT)
    unresolved = replace(package.stressor_axes[0], composable_with=("missing_axis",))
    package = _rehash(replace(package, stressor_axes=(unresolved,)))
    report = ScenarioPackageValidator().validate(package)
    assert any(
        issue.code == "unresolved_stressor_composition" for issue in report.issues
    )

    package = ScenarioPackage.load(FIXTURE_ROOT)
    invalid_parameters = replace(
        package.stressor_axes[0], parameters={"max_delay_steps": "bad"}
    )
    package = _rehash(replace(package, stressor_axes=(invalid_parameters,)))
    report = ScenarioPackageValidator().validate(package)
    assert any(issue.code == "invalid_stressor_parameters" for issue in report.issues)


def test_validator_blocks_physical_parameter_bypass_and_missing_evaluation_split():
    package = ScenarioPackage.load(FIXTURE_ROOT)
    initial_state = replace(
        package.initial_state,
        physical_parameters={
            "source": "generator_direct_mutation",
            "mutation_policy": "direct_engine_write",
        },
    )
    train_only = (package.split_lineage[0],)
    package = _rehash(
        replace(
            package,
            initial_state=initial_state,
            split_lineage=train_only,
        )
    )
    report = ScenarioPackageValidator().validate(package)
    codes = {issue.code for issue in report.issues}

    assert "physical_parameter_bypass" in codes
    assert "evaluation_split_missing" in codes


def test_validator_rejects_bad_split_lineage_and_contamination():
    package = ScenarioPackage.load(FIXTURE_ROOT)
    hidden = package.split_lineage[1]
    bad_hidden = replace(
        hidden,
        parent_split_ids=("missing_parent",),
        contamination_status="unknown",
    )
    package = _rehash(
        replace(package, split_lineage=(package.split_lineage[0], bad_hidden))
    )
    report = ScenarioPackageValidator().validate(package)
    codes = {issue.code for issue in report.issues}

    assert "split_parent_unresolved" in codes
    assert "evaluation_split_contamination_unknown" in codes

    train = ScenarioSplitLineage(
        split_id="train",
        partition="train",
        content_sha256="b" * 64,
        parent_split_ids=("hidden",),
        member_count=1,
        protected=False,
        contamination_status="clean",
    )
    hidden = ScenarioSplitLineage(
        split_id="hidden",
        partition="hidden_test",
        content_sha256="c" * 64,
        parent_split_ids=("train",),
        member_count=1,
        protected=True,
        contamination_status="clean",
    )
    package = _rehash(replace(package, split_lineage=(train, hidden)))
    report = ScenarioPackageValidator().validate(package)
    assert any(issue.code == "split_lineage_cycle" for issue in report.issues)


def test_known_evaluation_overlap_is_executable_but_not_claim_ready():
    package = ScenarioPackage.load(FIXTURE_ROOT)
    hidden = replace(
        package.split_lineage[1],
        contamination_status="known_overlap",
        contamination_sources=("producer_train_v1",),
    )
    package = _rehash(
        replace(package, split_lineage=(package.split_lineage[0], hidden))
    )

    report = ScenarioPackageValidator().validate(package)

    assert report.valid is True
    assert report.execution_ready is True
    assert report.claim_ready is False
    assert any(
        issue.code == "evaluation_split_known_overlap" and issue.severity == "warning"
        for issue in report.issues
    )


@pytest.mark.parametrize(
    "values, expected",
    [
        (["action_delay=0.5"], {"action_delay": 0.5}),
        ([], {}),
    ],
)
def test_cli_severity_override_parser(values, expected):
    assert _parse_severity_overrides(values) == expected


@pytest.mark.parametrize(
    "values, message",
    [
        (["missing_separator"], "expected STRESSOR_ID=SEVERITY"),
        (["action_delay=bad"], "Invalid severity"),
        (["action_delay=0.2", "action_delay=0.3"], "Duplicate"),
    ],
)
def test_cli_severity_override_parser_rejects_malformed_values(values, message):
    with pytest.raises(ValueError, match=message):
        _parse_severity_overrides(values)


class _ScenarioUnitEngine(NyssaEngine):
    def __init__(self) -> None:
        self.steps = 0

    def load_task(self, task_spec) -> None:
        self.task_spec = task_spec

    def reset(self, seed: int | None = None):
        self.steps = 0
        return _observation(), {}

    def step(self, action: Any):
        self.steps += 1
        return (
            _observation(),
            1.0,
            True,
            False,
            {
                "success": True,
                "completion_time": float(self.steps),
            },
        )

    def render(self):
        return None

    def get_state(self):
        return {"steps": self.steps}

    def close(self) -> None:
        return None


def _observation() -> dict[str, Any]:
    return {
        "raw": [0.0, 0.0],
        "action_space": {
            "type": "box",
            "shape": [1],
            "low": [-1.0],
            "high": [1.0],
        },
    }


def test_run_scenario_executes_contract_and_writes_sanitized_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setitem(get_plugin_registry().engines, "mujoco", _ScenarioUnitEngine)
    out = tmp_path / "run"

    assert (
        main(
            [
                "run-scenario",
                str(FIXTURE_ROOT),
                "--episodes",
                "1",
                "--severity",
                "action_delay=0.5",
                "--out",
                str(out),
                "--no-replay",
            ]
        )
        == 0
    )

    scenario = json.loads((out / "scenario_execution.json").read_text(encoding="utf-8"))
    dataset = json.loads((out / "dataset_manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    report_html = (out / "report.html").read_text(encoding="utf-8")
    assert scenario["format"] == "nyssa-scenario-execution-v1"
    assert scenario["validation"]["execution_ready"] is True
    assert scenario["stressor_config"]["stressors"][0]["severity"] == 0.5
    assert "external_locator" not in json.dumps(scenario["assets"])
    assert "scenario_execution.json" in dataset["artifacts"]
    assert metrics["scenario"]["scenario_identity"] == scenario["scenario_identity"]
    assert "Scenario Package" in report_html
    assert scenario["scenario_identity"] in report_html
    assert (
        load_comparison_contract(out)["scenario_identity"]
        == scenario["scenario_identity"]
    )
    checks = metrics["public_claim_validation"]["checks"]
    assert checks["scenario_package_valid"] is True
    assert checks["scenario_execution_ready"] is True
    assert checks["scenario_claim_ready"] is True


def test_run_scenario_rejects_run_seed_override(tmp_path: Path):
    with pytest.raises(ValueError, match="identity-bearing scenario run seed"):
        main(
            [
                "run-scenario",
                str(FIXTURE_ROOT),
                "--episodes",
                "1",
                "--seed",
                "99",
                "--out",
                str(tmp_path / "run"),
                "--no-replay",
            ]
        )


def test_validate_command_accepts_scenario_directory(capsys):
    assert main(["validate", str(FIXTURE_ROOT)]) == 0
    assert f"valid: {FIXTURE_ROOT}" in capsys.readouterr().out
