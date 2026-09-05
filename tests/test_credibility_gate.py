from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from nyssa_bench.cli import main
from nyssa_bench.credibility import (
    CredibilityEvidence,
    CredibilitySpec,
    EvidenceArtifact,
    EvidenceReference,
    evaluate_credibility,
)
from nyssa_bench.metrics.vector import build_metric_vector
from nyssa_bench.nep import PolicyContract
from nyssa_bench.simreal import (
    RealReference,
    SimRealPair,
    SimRealStudySpec,
    SimulationReference,
)
from nyssa_bench.validity import AuditResult, BenchmarkValidityReport


ROOT = Path(__file__).resolve().parents[1]
CLAIMS = ROOT / "claims" / "claim_evidence.json"
DEFAULT_SPEC = ROOT / "claims" / "phase1_credibility.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_repository_gate_passes_measurement_core_only(tmp_path: Path) -> None:
    exit_code = main(
        [
            "credibility-gate",
            str(DEFAULT_SPEC),
            "--repo-root",
            str(ROOT),
            "--out",
            str(tmp_path),
        ]
    )
    report = json.loads(
        (tmp_path / "phase1_credibility.json").read_text(encoding="utf-8")
    )

    assert exit_code == 2
    assert report["gates"]["A"]["status"] == "passed"
    assert report["gates"]["B"]["status"] == "missing"
    assert report["gates"]["C"]["status"] == "missing"
    assert report["highest_completed_gate"] == "measurement_core"
    assert report["phase1_complete"] is False
    assert report["public_wording_claim_id"] == "current_public_positioning"
    assert report["issue_dependencies"] == list(range(13, 24))
    assert report["scope_exclusions"]
    assert "broader Hudson Labs roadmap" in report["scope_statement"]
    assert len(report["report_sha256"]) == 64
    assert "Measurement Core" in (tmp_path / "phase1_credibility.html").read_text(
        encoding="utf-8"
    )


@pytest.mark.parametrize(
    ("interval", "passed_check", "not_applicable_check"),
    [
        ([0.01, 0.08], "positive_incremental_result", "well_powered_negative_result"),
        ([-0.03, 0.01], "well_powered_negative_result", "positive_incremental_result"),
    ],
)
def test_all_gates_require_content_addressed_executable_evidence(
    tmp_path: Path,
    interval: list[float],
    passed_check: str,
    not_applicable_check: str,
) -> None:
    references = _complete_evidence(tmp_path, interval)
    spec = _spec(tmp_path, references)

    report = evaluate_credibility(spec, spec_root=tmp_path, source_root=ROOT)

    assert all(gate["status"] == "passed" for gate in report["gates"].values())
    assert report["phase1_complete"] is True
    assert report["highest_completed_gate"] == "predictive_validity"
    # Passing evidence cannot bypass the separately reviewed claim authorization.
    assert report["public_wording_claim_id"] == "current_public_positioning"
    checks = {item["check_id"]: item for item in report["gates"]["C"]["checks"]}
    assert checks[passed_check]["status"] == "passed"
    assert checks[passed_check]["evidence_references"]
    assert checks[not_applicable_check]["status"] == "not_applicable"
    assert report["check_status_counts"]["not_applicable"] == 1


def test_tampered_evidence_fails_instead_of_becoming_missing(tmp_path: Path) -> None:
    reference = _evidence(
        tmp_path,
        "reference",
        "reference_benchmark",
        [_run_metrics(), _validity_report(), _reference_manifest()],
    )
    artifact = tmp_path / "reference" / "artifact-0.json"
    artifact.write_text("{}", encoding="utf-8")
    spec = _spec(tmp_path, [reference])

    report = evaluate_credibility(spec, spec_root=tmp_path, source_root=ROOT)

    check = next(
        item
        for item in report["gates"]["B"]["checks"]
        if item["check_id"] == "reference_benchmark"
    )
    assert check["status"] == "failed"
    assert report["gates"]["B"]["status"] == "failed"
    assert report["evidence_errors"][0]["evidence_id"] == "reference"


def test_document_and_escaping_paths_are_rejected() -> None:
    with pytest.raises(ValidationError, match="JSON files"):
        EvidenceArtifact(path="claim.md", sha256="a" * 64)
    with pytest.raises(ValidationError, match="cannot escape"):
        EvidenceReference(
            evidence_id="outside",
            category="benchmark_validity",
            path="../outside/evidence.json",
            sha256="a" * 64,
        )


def test_duplicate_policy_families_do_not_satisfy_gate_b(
    tmp_path: Path,
) -> None:
    references = [
        _evidence(
            tmp_path,
            "policy-a",
            "learned_policy_track",
            [
                _run_metrics(),
                _validity_report(),
                _policy_contract("a", "diffusion"),
                _policy_track_report(),
            ],
            {
                "benchmark_id": "reference_v1",
                "benchmark_version": "1.0.0",
                "policy_id": "a",
            },
        ),
        _evidence(
            tmp_path,
            "policy-b",
            "learned_policy_track",
            [
                _run_metrics(),
                _validity_report(),
                _policy_contract("b", "diffusion"),
                _policy_track_report(),
            ],
            {
                "benchmark_id": "reference_v1",
                "benchmark_version": "1.0.0",
                "policy_id": "b",
            },
        ),
    ]
    spec = _spec(tmp_path, references)

    report = evaluate_credibility(spec, spec_root=tmp_path, source_root=ROOT)
    check = next(
        item
        for item in report["gates"]["B"]["checks"]
        if item["check_id"] == "two_learned_policy_families"
    )

    assert check["status"] == "missing"
    assert "found 1 distinct" in check["reason"]


def test_control_policy_is_failed_learned_policy_evidence(tmp_path: Path) -> None:
    reference = _evidence(
        tmp_path,
        "control",
        "learned_policy_track",
        [
            _run_metrics(),
            _validity_report(),
            _policy_contract("control", "random"),
            _policy_track_report(),
        ],
        {
            "benchmark_id": "reference_v1",
            "benchmark_version": "1.0.0",
            "policy_id": "control",
        },
    )
    spec = _spec(tmp_path, [reference])

    report = evaluate_credibility(spec, spec_root=tmp_path, source_root=ROOT)

    assert report["gates"]["B"]["status"] == "failed"
    assert "control policy" in report["evidence_errors"][0]["message"]


def test_missing_claim_matrix_produces_failed_report(tmp_path: Path) -> None:
    spec = CredibilitySpec(
        program_id="missing-matrix",
        program_version="1.0.0",
        claim_matrix_path="missing.json",
        claim_matrix_sha256="a" * 64,
    )

    report = evaluate_credibility(spec, spec_root=tmp_path, source_root=ROOT)

    assert report["claim_matrix_sha256"] is None
    assert report["gates"]["A"]["status"] == "failed"
    assert report["gates"]["B"]["status"] == "failed"
    assert report["highest_completed_gate"] == "none"


def _complete_evidence(root: Path, interval: list[float]) -> list[EvidenceReference]:
    study_spec = _sim_real_spec()
    return [
        _evidence(
            root,
            "reference",
            "reference_benchmark",
            [_run_metrics(), _validity_report(), _reference_manifest()],
        ),
        _evidence(
            root,
            "policy-a",
            "learned_policy_track",
            [
                _run_metrics(),
                _validity_report(),
                _policy_contract("a", "diffusion"),
                _policy_track_report(),
            ],
            {
                "benchmark_id": "reference_v1",
                "benchmark_version": "1.0.0",
                "policy_id": "a",
            },
        ),
        _evidence(
            root,
            "policy-b",
            "learned_policy_track",
            [
                _run_metrics(),
                _validity_report(),
                _policy_contract("b", "transformer_bc"),
                _policy_track_report(),
            ],
            {
                "benchmark_id": "reference_v1",
                "benchmark_version": "1.0.0",
                "policy_id": "b",
            },
        ),
        _evidence(
            root,
            "shifted",
            "paired_clean_shifted",
            [
                _robustness_sweep(),
                _validity_report("statistical_precision", "paired_design"),
            ],
        ),
        _evidence(root, "validity", "benchmark_validity", [_validity_report()]),
        _evidence(
            root,
            "mujoco-ci",
            "simulator_ci",
            [_simulator_smoke("mujoco")],
            {"engine": "mujoco"},
        ),
        _evidence(
            root,
            "maniskill-ci",
            "simulator_ci",
            [_simulator_smoke("maniskill")],
            {"engine": "maniskill"},
        ),
        _evidence(
            root,
            "hardware",
            "hardware_calibration",
            [_hardware_validation()],
            {"prespecified": True},
        ),
        _evidence(
            root,
            "predictive",
            "sim_real_predictive_result",
            [
                study_spec.model_dump(mode="json"),
                _sim_real_report(study_spec, interval),
                _validity_report(
                    "statistical_precision", "sim_real_predictive_validity"
                ),
            ],
        ),
    ]


def _spec(root: Path, references: list[EvidenceReference]) -> CredibilitySpec:
    matrix = root / "claim_evidence.json"
    matrix.write_bytes(CLAIMS.read_bytes())
    return CredibilitySpec(
        program_id="unit-phase1",
        program_version="1.0.0",
        claim_matrix_path=matrix.name,
        claim_matrix_sha256=_sha(matrix),
        evidence=tuple(references),
    )


def _evidence(
    root: Path,
    evidence_id: str,
    category: str,
    payloads: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> EvidenceReference:
    directory = root / evidence_id
    directory.mkdir()
    artifacts = []
    for index, payload in enumerate(payloads):
        path = directory / f"artifact-{index}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        artifacts.append(EvidenceArtifact(path=path.name, sha256=_sha(path)))
    record = CredibilityEvidence(
        evidence_id=evidence_id,
        category=category,  # type: ignore[arg-type]
        status="validated",
        artifacts=tuple(artifacts),
        metadata=metadata or {},
    )
    path = directory / "evidence.json"
    path.write_text(json.dumps(record.model_dump(mode="json")), encoding="utf-8")
    return EvidenceReference(
        evidence_id=evidence_id,
        category=record.category,
        path=path.relative_to(root).as_posix(),
        sha256=_sha(path),
    )


def _run_metrics() -> dict[str, Any]:
    return {
        "format": "nyssa-run-metrics-v2",
        "episodes": 100,
        "successes": 70,
        "metric_vector": build_metric_vector({"episodes": 100, "successes": 70}),
        "public_claim_validation": {
            "status": "validated",
            "public_claim": True,
            "failures": [],
        },
    }


def _validity_report(*audit_ids: str) -> dict[str, Any]:
    ids = audit_ids or (
        "shortcut_solvability",
        "train_evaluation_leakage",
        "language_observation_ablations",
        "statistical_precision",
        "paired_design",
        "rank_stability",
        "hidden_test_integrity",
    )
    audits = tuple(
        AuditResult(
            audit_id=audit_id,
            category="credibility_fixture",
            status="passed",
            severity="blocking",
            inputs={"prespecified": True},
            evidence={"validated": True},
            remediation="No remediation required while the evidence remains unchanged.",
            claim_impact="block",
            summary="The prespecified audit passed.",
        )
        for audit_id in ids
    )
    return BenchmarkValidityReport(
        benchmark_id="reference_v1",
        benchmark_version="1.0.0",
        claim_tier="public_simulation",
        spec_sha256="a" * 64,
        audits=audits,
        metadata={"required_audits": list(ids)},
    ).to_dict()


def _reference_manifest() -> dict[str, Any]:
    return {
        "format": "nyssa-reference-benchmark-report-v1",
        "benchmark_id": "reference_v1",
        "benchmark_version": "1.0.0",
        "status": "release_ready",
        "release_ready": True,
        "task_count": 12,
        "spec_sha256": "b" * 64,
        "oracle_control_policy_ids": ["planner_oracle"],
    }


def _policy_contract(policy_id: str, family: str) -> dict[str, Any]:
    return PolicyContract(
        policy_id=policy_id,
        policy_version="1.0.0",
        policy_family=family,
        checkpoint_id=f"{policy_id}-checkpoint",
        checkpoint_sha256="c" * 64,
        preprocessing_sha256="d" * 64,
        observation_modalities=("state",),
        action_representation="normalized_joint_delta",
        action_dimension=2,
        action_lower_bounds=(-1.0, -1.0),
        action_upper_bounds=(1.0, 1.0),
        prediction_horizon=8,
        execution_horizon=4,
        state_semantics="resettable",
        deterministic_seeding=True,
    ).model_dump(mode="json")


def _policy_track_report() -> dict[str, Any]:
    return {
        "format": "nyssa-policy-track-report-v1",
        "status": "release_ready",
        "release_ready": True,
        "tracks": [
            {
                "policy_id": "a",
                "policy_family": "diffusion",
                "validated": True,
            },
            {
                "policy_id": "b",
                "policy_family": "diffusion",
                "validated": True,
            },
            {
                "policy_id": "b",
                "policy_family": "transformer_bc",
                "validated": True,
            },
            {
                "policy_id": "control",
                "policy_family": "random",
                "validated": False,
            },
        ],
    }


def _robustness_sweep() -> dict[str, Any]:
    return {
        "format": "nyssa-robustness-sweep-v1",
        "paired_episode_coverage": 100,
        "points": [
            {"severity": 0.0, "episodes": 100},
            {"severity": 0.5, "episodes": 100},
        ],
    }


def _simulator_smoke(engine: str) -> dict[str, Any]:
    artifacts = [
        "run.yaml",
        "metrics.json",
        "episodes.json",
        "dataset_manifest.json",
        "stressor_manifest.json",
        "failure_ledger.json",
        "nep_manifest.json",
    ]
    return {
        "format": "nyssa-simulator-ci-smoke-v1",
        "status": "passed",
        "engine": engine,
        "episodes": 2,
        "state_restore_capability": {"supported": True},
        "restore_checks": [
            {"attempt": 0, "action_within_bounds": True},
            {"attempt": 1, "action_within_bounds": True},
        ],
        "stressor_id": "action_gaussian_noise",
        "package_versions": {engine: "1.0.0"},
        "episode_seeds": [7, 8],
        "required_artifacts": artifacts,
        "replay_requested": engine == "maniskill",
        "replay_count": 2 if engine == "maniskill" else 0,
    }


def _hardware_validation() -> dict[str, Any]:
    return {
        "format": "nyssa-real-evidence-validation-v1",
        "package_identity": f"hardware@1.0.0:{'e' * 64}",
        "valid": True,
        "evidence_ready": True,
        "calibration_ready": True,
        "governance_ready": True,
        "comparison_ready": True,
        "claim_ready": True,
        "issues": [],
    }


def _sim_real_spec() -> SimRealStudySpec:
    pairs = []
    for index in range(5):
        shift_id = "held-out" if index >= 3 else "train"
        simulation = SimulationReference(
            run_dir=f"runs/{index}",
            run_id=f"run-{index}",
            artifacts_sha256={
                "run.yaml": "1" * 64,
                "dataset_manifest.json": "2" * 64,
                "metrics.json": "3" * 64,
                "episodes.json": "4" * 64,
            },
            policy_name="policy-a",
            checkpoint_id="checkpoint-a",
            checkpoint_sha256="5" * 64,
            preprocessing_sha256="6" * 64,
            task_id="pick",
            episode_seed=index,
            episode_index=index,
        )
        pairs.append(
            SimRealPair(
                pair_id=f"pair-{index}",
                policy_id="policy-a",
                task_id="pick",
                shift_id=shift_id,
                severity=0.5 if shift_id == "held-out" else 0.0,
                simulation=simulation,
                real=RealReference(
                    package_path=f"real/{index}.json",
                    package_identity=f"real@1.0.0:{'7' * 64}",
                    real_episode_id=f"real-{index}",
                    variant_id=f"variant-{index}",
                    trial_id=f"trial-{index}",
                ),
                sim_step_seconds=0.02,
                real_event_step_seconds=0.02,
            )
        )
    return SimRealStudySpec(
        study_id="predictive-v1",
        study_version="1.0.0",
        prespecified_at="2026-09-05T00:00:00Z",
        pairs=tuple(pairs),
        primary_metrics=("incremental_predictive_value",),
        bootstrap_samples=200,
        bootstrap_seed=7,
        cluster_fields=("trial_id",),
        holdout_shift_ids=("held-out",),
    )


def _sim_real_report(spec: SimRealStudySpec, interval: list[float]) -> dict[str, Any]:
    return {
        "format": "nyssa-sim-real-study-report-v1",
        "status": "complete",
        "study_id": spec.study_id,
        "study_version": spec.study_version,
        "study_sha256": spec.sha256,
        "pair_count": 5,
        "metrics": {
            "incremental_predictive_value": {
                "status": "available",
                "train_pairs": 3,
                "holdout_pairs": 2,
                "holdout_shift_ids": ["held-out"],
                "incremental_brier_improvement_ci95": interval,
            }
        },
    }
