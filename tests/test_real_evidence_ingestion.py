from __future__ import annotations

import json
from pathlib import Path

import pytest

from nyssa_bench.cli import main
from nyssa_bench.core.suite import Suite
from nyssa_bench.metrics.run_claims import RunClaimValidator
from nyssa_bench.real_evidence import (
    RealEvidencePackage,
    RealEvidenceValidationError,
    RealEvidenceValidator,
    comparison_pairs,
    real_evidence_conformance_fixture_path,
    sanitized_evidence_manifest,
    write_real_evidence_artifacts,
)
from nyssa_bench.research.sim_real_correlation import load_sim_real_evidence_pairs


FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "conformance"
    / "real_evidence"
    / "v1"
    / "valid_reconstructed_family"
)


def _rehash(package: RealEvidencePackage) -> RealEvidencePackage:
    provisional = package.model_copy(update={"content_sha256": "0" * 64})
    return provisional.model_copy(
        update={"content_sha256": provisional.compute_content_sha256()}
    )


def test_real_evidence_fixture_is_discoverable_from_installed_module():
    assert real_evidence_conformance_fixture_path() == FIXTURE.resolve()
    with pytest.raises(FileNotFoundError):
        real_evidence_conformance_fixture_path("missing")


def test_full_conformance_fixture_validates_and_preserves_provenance():
    package = RealEvidencePackage.load(FIXTURE)
    report = RealEvidenceValidator().validate(package)

    assert package.compute_content_sha256() == package.content_sha256
    assert report.valid is True
    assert report.evidence_ready is True
    assert report.calibration_ready is True
    assert report.comparison_ready is True
    assert report.governance_ready is True
    assert report.claim_ready is True
    assert report.real_ledger is not None
    assert report.real_ledger.events[0].provenance.source == "real_robot"
    assert (
        report.variant_ledgers["nominal_reconstruction"].events[0].provenance.source
        == "reconstructed_simulation"
    )
    assert len(comparison_pairs(package)) == 2
    assert len(load_sim_real_evidence_pairs(package)) == 2


def test_schema_rejects_unknown_fields_bad_actions_and_unsafe_paths():
    payload = json.loads(
        json.dumps(RealEvidencePackage.load(FIXTURE).model_dump(mode="json"))
    )
    payload["scene_reconstruction"] = {"must": "remain external"}
    with pytest.raises(ValueError, match="extra_forbidden"):
        RealEvidencePackage.model_validate(payload)

    package = RealEvidencePackage.load(FIXTURE)
    with pytest.raises(ValueError, match="match dimension"):
        type(package.real_episode.actions).model_validate(
            {
                **package.real_episode.actions.model_dump(mode="json"),
                "units": ["N*m", "rad"],
            }
        )
    with pytest.raises(ValueError, match="package-relative"):
        type(package.artifacts[0]).model_validate(
            {
                **package.artifacts[0].model_dump(mode="json"),
                "path": "../private.json",
            }
        )


def test_validator_rejects_clock_frame_action_and_failure_provenance_errors():
    package = RealEvidencePackage.load(FIXTURE)
    sensor = package.real_episode.sensors[0].model_copy(
        update={"clock_id": "missing_clock", "frame_id": "missing_frame"}
    )
    action = package.real_episode.actions.model_copy(
        update={"clock_id": "missing_clock", "lower_bounds": (0.3,)}
    )
    event = dict(package.real_episode.failure_events[0])
    event["provenance"] = {
        **event["provenance"],
        "source": "reconstructed_simulation",
    }
    episode = package.real_episode.model_copy(
        update={"sensors": (sensor,), "actions": action, "failure_events": (event,)}
    )
    package = _rehash(package.model_copy(update={"real_episode": episode}))

    report = RealEvidenceValidator().validate(package)
    codes = {item.code for item in report.issues}

    assert "stream_clock_unresolved" in codes
    assert "stream_frame_unresolved" in codes
    assert "action_clock_unresolved" in codes
    assert "action_out_of_bounds" in codes
    assert "failure_event_provenance_mismatch" in codes
    assert report.claim_ready is False


def test_missing_calibration_and_governance_fields_downgrade_claims():
    package = RealEvidencePackage.load(FIXTURE)
    calibrations = tuple(
        item
        for item in package.calibrations
        if item.calibration_type not in {"geometry", "dynamics"}
    )
    governance = package.governance.model_copy(
        update={
            "privacy_classification": "restricted",
            "redactions": (),
            "operator_ids_pseudonymous": False,
        }
    )
    package = _rehash(
        package.model_copy(
            update={"calibrations": calibrations, "governance": governance}
        )
    )

    report = RealEvidenceValidator().validate(package)

    assert (
        report.valid is False
    )  # variant parameter calibration references are unresolved
    assert report.calibration_ready is False
    assert report.governance_ready is False
    assert report.claim_ready is False
    codes = {item.code for item in report.issues}
    assert "calibration_missing" in codes
    assert "operator_identity_not_pseudonymous" in codes
    assert "privacy_redactions_missing" in codes


def test_incomplete_variant_mapping_is_not_comparison_ready():
    package = RealEvidencePackage.load(FIXTURE)
    mapping = package.mapping.model_copy(
        update={"variant_ids": ("nominal_reconstruction",)}
    )
    package = _rehash(package.model_copy(update={"mapping": mapping}))

    report = RealEvidenceValidator().validate(package)

    assert report.comparison_ready is False
    assert report.claim_ready is False
    assert any(
        item.code == "mapping_variant_family_incomplete" for item in report.issues
    )


def test_protected_artifact_metadata_mode_is_valid_but_not_evidence_ready():
    package = RealEvidencePackage.load(FIXTURE)
    protected = package.artifacts[0].model_copy(
        update={
            "access": "protected",
            "path": None,
            "external_locator": "provider://restricted-proprioception",
        }
    )
    package = _rehash(
        package.model_copy(update={"artifacts": (protected, *package.artifacts[1:])})
    )

    metadata_report = RealEvidenceValidator().validate(package, require_artifacts=False)
    execution_report = RealEvidenceValidator().validate(package, require_artifacts=True)

    assert metadata_report.valid is True
    assert metadata_report.evidence_ready is False
    assert metadata_report.claim_ready is False
    assert metadata_report.unresolved_protected_artifacts == ("proprioception",)
    assert execution_report.valid is False


def test_ingestion_writes_sanitized_manifest_ledgers_pairs_and_report(tmp_path: Path):
    package = RealEvidencePackage.load(FIXTURE)
    report = RealEvidenceValidator().validate(package)
    paths = write_real_evidence_artifacts(package, report, tmp_path)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    ledgers = json.loads(paths["ledgers"].read_text(encoding="utf-8"))
    pairs = json.loads(paths["pairs"].read_text(encoding="utf-8"))
    html = paths["report"].read_text(encoding="utf-8")

    identity = manifest["real_episode"]["identity"]
    assert "operator_id" not in identity
    assert identity["operator_id_included"] is False
    assert all("path" not in item for item in manifest["artifacts"])
    assert all("external_locator" not in item for item in manifest["artifacts"])
    assert (
        manifest["real_episode"]["failure_events"][0]["evidence_payloads_included"]
        is False
    )
    assert "evidence" not in manifest["real_episode"]["failure_events"][0]
    assert "real_ledger" not in manifest["validation"]
    assert ledgers["real"]["events"][0]["provenance"]["source"] == "real_robot"
    assert len(pairs["pairs"]) == 2
    assert "Calibration and uncertainty" in html
    assert "Real/sim mismatches" in html


def test_cli_validates_and_imports_full_fixture(tmp_path: Path, capsys):
    assert main(["validate-real-evidence", str(FIXTURE)]) == 0
    assert "claim_ready: True" in capsys.readouterr().out
    out = tmp_path / "imported"
    assert main(["import-real-evidence", str(FIXTURE), "--out", str(out)]) == 0
    assert (out / "real_evidence_manifest.json").is_file()
    assert (out / "real_evidence_ledgers.json").is_file()
    assert main(["validate", str(FIXTURE)]) == 0


def test_validation_error_exposes_machine_readable_report():
    package = RealEvidencePackage.load(FIXTURE).model_copy(
        update={"content_sha256": "f" * 64}
    )
    report = RealEvidenceValidator().validate(package)

    with pytest.raises(RealEvidenceValidationError) as exc_info:
        report.raise_for_errors()
    assert exc_info.value.report is report
    assert any(item.code == "package_hash_mismatch" for item in report.issues)


def test_run_claim_gate_rejects_unready_real_evidence():
    validation = RunClaimValidator().validate(
        suite=Suite.load("mujoco_control_v0"),
        engine_name="mujoco",
        episodes_per_task=0,
        episodes=[],
        out_dir=None,
        real_evidence_validation={"claim_ready": False},
    )

    assert validation.checks["real_evidence_claim_ready"] is False
    assert "real_evidence_claim_ready" in validation.failures


def test_sanitized_manifest_helper_matches_writer_policy():
    package = RealEvidencePackage.load(FIXTURE)
    report = RealEvidenceValidator().validate(package)
    manifest = sanitized_evidence_manifest(package, report)
    serialized = json.dumps(manifest)

    assert "operator_pseudonym_001" not in serialized
    assert "artifacts/proprioception.json" not in serialized
