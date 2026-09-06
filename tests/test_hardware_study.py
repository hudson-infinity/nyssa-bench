from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from nyssa_bench.cli import main
from nyssa_bench.hardware_study import (
    ConditionMismatch,
    HardwareCalibrationStudy,
    HardwareEvidence,
    SafetyPlan,
    evaluate_hardware_study,
)
from nyssa_bench.hardware_study import evaluator as hardware_evaluator
from nyssa_bench.hardware_study.candidate import build_hardware_study_candidate
from nyssa_bench.reference_benchmark import ArtifactReference


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "configs" / "hardware" / "nyssa_hardware_calibration_v0_1.json"


def test_committed_hardware_draft_is_deterministic_and_honest() -> None:
    expected = build_hardware_study_candidate(ROOT).model_dump(mode="json")
    committed = json.loads(CANDIDATE.read_text(encoding="utf-8"))

    assert committed == expected
    study = HardwareCalibrationStudy.model_validate(committed)
    report = evaluate_hardware_study(study, root=ROOT)

    assert report["status"] == "evidence_missing"
    assert report["claim_ready"] is False
    assert report["condition_count"] == 12
    assert report["planned_trial_count"] == 240
    assert report["planned_recovery_trial_count"] == 120
    assert report["planned_evidence_package_count"] == 360
    assert report["status_counts"] == {
        "passed": 0,
        "failed": 0,
        "missing": 5,
        "not_applicable": 0,
    }


def test_factorial_cells_cannot_be_omitted() -> None:
    study = build_hardware_study_candidate(ROOT)
    payload = study.model_dump(mode="json")
    payload["conditions"].pop()

    with pytest.raises(ValidationError, match="complete factorial"):
        HardwareCalibrationStudy.model_validate(payload)


def test_recovery_analysis_requires_matched_trials() -> None:
    study = build_hardware_study_candidate(ROOT)
    payload = study.model_dump(mode="json")
    for condition in payload["conditions"]:
        condition["recovery_design"] = "disabled"
        condition["recovery_trial_count"] = 0

    with pytest.raises(ValidationError, match="recovery analysis"):
        HardwareCalibrationStudy.model_validate(payload)


def test_mismatch_quantification_is_all_or_none() -> None:
    with pytest.raises(ValidationError, match="all-or-none"):
        ConditionMismatch(
            mismatch_id="latency",
            category="latency",
            description="Measured latency mismatch.",
            expected_direction="hardware_harder",
            magnitude=0.02,
            unit="s",
        )


def test_safety_plan_cannot_omit_stop_controls() -> None:
    study = build_hardware_study_candidate(ROOT)
    payload = study.safety.model_dump(mode="json")
    payload["stop_conditions"] = []

    with pytest.raises(ValidationError, match="stop conditions"):
        SafetyPlan.model_validate(payload)


def test_design_hash_ignores_evidence_and_status() -> None:
    study = build_hardware_study_candidate(ROOT)
    receipt = ArtifactReference(path="receipt.json", sha256="a" * 64)
    preregistered = study.model_copy(
        update={
            "status": "preregistered",
            "evidence": HardwareEvidence(preregistration_receipt=receipt),
        }
    )

    assert preregistered.design_sha256 == study.design_sha256


def test_preregistration_receipt_must_precede_trials(tmp_path: Path) -> None:
    study = build_hardware_study_candidate(ROOT)
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "format": "nyssa-preregistration-receipt-v1",
                "study_id": study.study_id,
                "design_sha256": study.design_sha256,
                "registered_at": "2027-01-01T00:00:00Z",
                "registry_uri": "https://example.org/registration/1",
            }
        ),
        encoding="utf-8",
    )
    receipt = ArtifactReference(path="receipt.json", sha256=_sha(receipt_path))
    local = study.model_copy(
        update={
            "status": "preregistered",
            "evidence": HardwareEvidence(preregistration_receipt=receipt),
        }
    )

    check = hardware_evaluator._preregistration_check(local, tmp_path)

    assert check["status"] == "failed"
    assert "does not precede" in check["reason"]


def test_complete_status_requires_every_trial_package() -> None:
    study = build_hardware_study_candidate(ROOT)
    payload = study.model_dump(mode="json")
    payload["status"] = "complete"
    payload["evidence"]["preregistration_receipt"] = {
        "path": "receipt.json",
        "sha256": "a" * 64,
    }

    with pytest.raises(ValidationError, match="one package per planned trial"):
        HardwareCalibrationStudy.model_validate(payload)


def test_cli_writes_hardware_audit_reports(tmp_path: Path) -> None:
    out = tmp_path / "report"

    exit_code = main(
        [
            "audit-hardware-study",
            str(CANDIDATE),
            "--repo-root",
            str(ROOT),
            "--out",
            str(out),
        ]
    )

    assert exit_code == 2
    assert (out / "hardware_calibration.json").is_file()
    assert "evidence_missing" in (out / "hardware_calibration.html").read_text(
        encoding="utf-8"
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
