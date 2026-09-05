from __future__ import annotations

import json
from pathlib import Path

import pytest

from nyssa_bench.cli import main
from nyssa_bench.nep import (
    ArtifactContract,
    AssetContract,
    ClaimContract,
    FailureEvidenceContract,
    InterventionContract,
    NEPManifest,
    PolicyContract,
    SplitLineageContract,
    StressorContract,
    StressorEntryContract,
    TaskContract,
    check_nep_compatibility,
    generated_schemas,
    load_nep_manifest,
    migrate_nep_data,
    validate_nep_manifest,
    write_nep_manifest,
)
from nyssa_bench.nep.artifacts import load_nep_data


ROOT = Path(__file__).resolve().parents[1]


def _hash(character: str) -> str:
    return character * 64


def _artifacts() -> tuple[ArtifactContract, ...]:
    ids = (
        "failure-ledger",
        "detector-contracts",
        "run-validity",
        "benchmark-validity",
        "stressor-backend",
        "counterfactual-branches",
        "real-evidence",
    )
    return tuple(
        ArtifactContract(
            artifact_id=artifact_id,
            media_type="application/json",
            sha256=_hash(hex(index + 1)[2:]),
            uri=f"nyssa-run://unit/{artifact_id}.json",
        )
        for index, artifact_id in enumerate(ids)
    )


def _task(*, engines: tuple[str, ...] = ("mujoco",)) -> TaskContract:
    return TaskContract(
        task_id="unit_task",
        task_version="1.0.0",
        engine_ids=engines,
        robot_id="unit_robot",
        scene_id="unit_scene",
        horizon_steps=100,
        observation_modalities=("state",),
        action_representation="normalized_joint_delta",
        success_predicate={"info_key": "success"},
        assets=(
            AssetContract(
                asset_id="unit_asset",
                asset_version="1.0.0",
                sha256=_hash("a"),
                license_id="Apache-2.0",
                split="public_test",
            ),
        ),
        split_lineage=SplitLineageContract(
            split_id="unit-public-test",
            partition="public_test",
            lineage_sha256=_hash("b"),
        ),
    )


def _policy() -> PolicyContract:
    return PolicyContract(
        policy_id="unit_policy",
        policy_version="1.0.0",
        policy_family="behavior_cloning",
        checkpoint_id="unit_checkpoint",
        checkpoint_sha256=_hash("c"),
        preprocessing_sha256=_hash("d"),
        observation_modalities=("state",),
        action_representation="normalized_joint_delta",
        action_dimension=2,
        action_lower_bounds=(-1.0, -1.0),
        action_upper_bounds=(1.0, 1.0),
        prediction_horizon=8,
        execution_horizon=4,
        state_semantics="resettable",
        deterministic_seeding=True,
    )


def _failure() -> FailureEvidenceContract:
    return FailureEvidenceContract(
        ledger_artifact_id="failure-ledger",
        detector_contract_artifact_id="detector-contracts",
        temporal_precision=("exact_step",),
        evidence_visibility=("policy_observable", "privileged"),
        causal_semantics="hypothesis_only",
    )


def _manifest(
    tier: str = "ood_robustness",
    *,
    engines: tuple[str, ...] = ("mujoco",),
) -> NEPManifest:
    stressors = (
        StressorEntryContract(
            stressor_id="action_gaussian_noise",
            stressor_version="1.0.0",
            category="action",
            severity=0.5,
            seed=7,
            application_points=("transform_action",),
            parameters={"max_std": 0.1},
            observable_by_policy=False,
            privileged=False,
            backend_confirmed=True,
            backend_evidence_artifact_id="stressor-backend",
        ),
    )
    intervention = InterventionContract(enabled=False)
    if tier == "clean_simulation":
        stressors = ()
    if tier == "recovery_effectiveness":
        intervention = InterventionContract(
            enabled=True,
            trigger_sources=("verifier_rejection",),
            intervention_types=("recovery_plan",),
            cost_metrics=("mean_intervention_cost_steps",),
            counterfactual_branch_artifact_id="counterfactual-branches",
            restoration_requirement="exact",
        )
    claim = ClaimContract(
        requested_tier=tier,  # type: ignore[arg-type]
        evidence_artifact_ids=("failure-ledger", "run-validity"),
        run_validity_artifact_id="run-validity",
        benchmark_validity_artifact_id=None
        if tier == "pipeline"
        else "benchmark-validity",
        real_evidence_artifact_id="real-evidence"
        if tier == "sim_real_predictive"
        else None,
    )
    return NEPManifest.create(
        evaluation_id=f"unit-{tier}",
        task=_task(engines=engines),
        stressor=StressorContract(
            condition_id="clean" if not stressors else "action-noise-s05",
            composition_semantics="ordered",
            stressors=stressors,
        ),
        policy=_policy(),
        failure_evidence=_failure(),
        intervention=intervention,
        claim=claim,
        artifacts=_artifacts(),
    )


@pytest.mark.parametrize(
    "tier",
    [
        "pipeline",
        "clean_simulation",
        "ood_robustness",
        "recovery_effectiveness",
        "sim_real_predictive",
    ],
)
def test_nep_manifest_round_trip_for_claim_tiers(tier: str, tmp_path: Path) -> None:
    manifest = _manifest(tier)
    path = write_nep_manifest(manifest, tmp_path / f"{tier}.json")

    loaded = load_nep_manifest(path)
    report, validated = validate_nep_manifest(loaded.model_dump(mode="json"))

    assert loaded == manifest
    assert validated == manifest
    assert report.valid is True
    assert report.claim_ready is True
    assert report.requested_claim_tier == tier


def test_cross_simulator_claim_requires_multiple_engines() -> None:
    with pytest.raises(ValueError, match="at least two engines"):
        _manifest("cross_simulator")

    assert _manifest(
        "cross_simulator", engines=("mujoco", "maniskill")
    ).claim.requested_tier == "cross_simulator"


def test_nep_hash_and_unknown_artifact_tampering_are_rejected() -> None:
    payload = _manifest().model_dump(mode="json")
    payload["policy"]["checkpoint_id"] = "tampered"
    report, manifest = validate_nep_manifest(payload)

    assert manifest is None
    assert report.valid is False
    assert any("content_sha256" in issue.message for issue in report.issues)

    payload = _manifest().model_dump(mode="json")
    payload["failure_evidence"]["ledger_artifact_id"] = "missing"
    payload["content_sha256"] = "0" * 64
    report, _ = validate_nep_manifest(payload)
    assert any("unknown artifacts" in issue.message for issue in report.issues)


def test_nep_zero_minor_compatibility_policy() -> None:
    assert check_nep_compatibility("0.1.0", "0.1.2").compatible is True
    assert check_nep_compatibility("0.2.0", "0.1.2").compatible is False
    assert check_nep_compatibility("0.1.3", "0.1.2").compatible is False


def test_draft_migration_is_explicit_and_content_addressed() -> None:
    payload = _manifest().model_dump(mode="json")
    draft = {
        key: value
        for key, value in payload.items()
        if key not in {"format", "nep_version", "content_sha256"}
    }
    draft["format"] = "nyssa-nep-draft-v0"

    migrated, record = migrate_nep_data(draft)
    report, manifest = validate_nep_manifest(migrated)

    assert record is not None
    assert record["semantics"] == "explicit_breaking_draft_to_0.1_migration"
    assert report.valid is True
    assert manifest is not None
    assert record["content_sha256"] == manifest.content_sha256


def test_nep_schemas_are_strict_and_cli_reports_errors(tmp_path: Path) -> None:
    schemas = generated_schemas()
    assert set(schemas) == {
        "claim-contract.schema.json",
        "failure-evidence-contract.schema.json",
        "intervention-contract.schema.json",
        "nep-manifest.schema.json",
        "policy-contract.schema.json",
        "stressor-contract.schema.json",
        "task-contract.schema.json",
    }
    assert schemas["task-contract.schema.json"]["additionalProperties"] is False

    payload = _manifest().model_dump(mode="json")
    payload["task"]["unknown"] = True
    manifest_path = tmp_path / "invalid.json"
    report_path = tmp_path / "report.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    assert (
        main(
            [
                "validate-nep",
                str(manifest_path),
                "--out",
                str(report_path),
            ]
        )
        == 3
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["valid"] is False
    assert any(issue["path"] == "task.unknown" for issue in report["issues"])


def test_committed_schemas_and_conformance_fixtures_match_runtime() -> None:
    schema_root = ROOT / "schemas" / "nep" / "0.1.0"
    for name, generated in generated_schemas().items():
        committed = json.loads((schema_root / name).read_text(encoding="utf-8"))
        assert committed == generated

    fixture_root = ROOT / "conformance" / "nep" / "0.1.0"
    for path in sorted((fixture_root / "valid").glob("*.json")):
        report, manifest = validate_nep_manifest(load_nep_data(path))
        assert report.valid is True, path
        assert manifest is not None
    for path in sorted((fixture_root / "invalid").glob("*.json")):
        report, manifest = validate_nep_manifest(load_nep_data(path))
        assert report.valid is False, path
        assert manifest is None
