from __future__ import annotations

import hashlib
from datetime import date, datetime, timezone
from pathlib import Path

from nyssa_bench.nep import (
    ClaimContract,
    FailureEvidenceContract,
    InterventionContract,
    StressorContract,
    StressorEntryContract,
)
from nyssa_bench.hardware_study.protocol import (
    AnalysisPlan,
    CalibrationCondition,
    ConditionMismatch,
    ExclusionRule,
    HardwareCalibrationStudy,
    SafetyPlan,
)
from nyssa_bench.policy_tracks import load_policy_track_registry
from nyssa_bench.real_evidence import GovernanceContract
from nyssa_bench.reference_benchmark import load_reference_benchmark
from nyssa_bench.reference_benchmark import ArtifactReference


TASK_IDS = ("maniskill_pick_cube", "maniskill_push_cube")
POLICY_IDS = ("planner_oracle", "robomimic_bc", "diffusion_action_chunk")
STRESSOR_IDS = ("hardware-clean", "hardware-action-delay-s05")


def build_hardware_study_candidate(repo_root: str | Path) -> HardwareCalibrationStudy:
    root = Path(repo_root).resolve()
    reference = load_reference_benchmark(
        root / "configs" / "reference" / "nyssa_reference_v0_1.json"
    )
    tracks = load_policy_track_registry(
        root / "configs" / "policy_tracks" / "nyssa_policy_tracks_v0_1.json"
    )
    reference_path = root / "configs" / "reference" / "nyssa_reference_v0_1.json"
    tracks_path = root / "configs" / "policy_tracks" / "nyssa_policy_tracks_v0_1.json"
    task_by_id = {task.contract.task_id: task.contract for task in reference.tasks}
    policy_by_id = {track.contract.policy_id: track.contract for track in tracks.tracks}
    stressors = (_clean_stressor(), _latency_stressor())
    conditions = tuple(
        _condition(task_id, policy_id, stressor_id)
        for task_id in TASK_IDS
        for policy_id in POLICY_IDS
        for stressor_id in STRESSOR_IDS
    )
    return HardwareCalibrationStudy(
        study_id="nyssa_hardware_calibration_v0_1",
        study_version="0.1.0",
        status="draft",
        protocol_authored_at=datetime(2026, 9, 5, 22, 45, tzinfo=timezone.utc),
        first_trial_not_before=datetime(2027, 1, 1, tzinfo=timezone.utc),
        reference_benchmark=_artifact(root, reference_path),
        policy_track_registry=_artifact(root, tracks_path),
        tasks=tuple(task_by_id[task_id] for task_id in TASK_IDS),
        policies=tuple(policy_by_id[policy_id] for policy_id in POLICY_IDS),
        stressors=stressors,
        failure_evidence=FailureEvidenceContract(
            ledger_artifact_id="real-failure-ledgers",
            detector_contract_artifact_id="hardware-detector-contracts",
            temporal_precision=("exact_step", "step_interval"),
            evidence_visibility=("policy_observable", "privileged", "external"),
            causal_semantics="hypothesis_only",
        ),
        intervention=InterventionContract(
            enabled=True,
            trigger_sources=("operator", "safety_system", "policy_monitor"),
            intervention_types=("emergency_stop", "matched_recovery_trial"),
            cost_metrics=("intervention_count", "intervention_latency_seconds"),
            restoration_requirement="none",
        ),
        claim=ClaimContract(
            requested_tier="sim_real_predictive",
            evidence_artifact_ids=(
                "real-evidence-packages",
                "sim-real-study-report",
                "hardware-validity-report",
            ),
            run_validity_artifact_id="simulation-run-validity",
            benchmark_validity_artifact_id="hardware-validity-report",
            real_evidence_artifact_id="real-evidence-packages",
        ),
        conditions=conditions,
        exclusions=(
            ExclusionRule(
                rule_id="pretrial-hardware-fault",
                criterion="Robot or sensor self-test fails before policy execution.",
                decision_time="before_outcome",
                treatment="exclude",
            ),
            ExclusionRule(
                rule_id="protective-stop",
                criterion="A prespecified safety stop occurs after policy execution begins.",
                decision_time="after_outcome_blinded",
                treatment="retain_with_flag",
            ),
            ExclusionRule(
                rule_id="stream-loss",
                criterion="Required evidence stream is unavailable beyond its tolerance.",
                decision_time="after_outcome_blinded",
                treatment="censor",
            ),
        ),
        analysis=AnalysisPlan(
            primary_metrics=(
                "policy_rank",
                "failure_distribution",
                "shift_response",
                "time_to_failure",
                "recovery_effect",
                "incremental_predictive_value",
            ),
            baseline_features=("intercept", "clean_sim_success"),
            enhanced_features=(
                "intercept",
                "clean_sim_success",
                "shift_severity",
                "failure_category",
                "failure_onset_time",
                "sim_recovery_gain",
            ),
            heldout_shift_ids=("hardware-action-delay-s05",),
            bootstrap_samples=5000,
            bootstrap_seed=20260905,
            cluster_fields=("policy_id", "task_id", "condition_id", "trial_id"),
            sensitivity_analyses=(
                "leave_one_task_out",
                "leave_one_policy_out",
                "exclude_operator_interventions",
                "alternate_failure_taxonomy_coarsening",
                "censoring_worst_case_bounds",
            ),
            negative_result_policy=(
                "Report the prespecified confidence interval and power analysis even "
                "when failure features do not improve held-out prediction."
            ),
        ),
        safety=SafetyPlan(
            risk_assessment_id="nyssa-hardware-risk-v0.1",
            operator_training=(
                "robot-specific emergency stop",
                "workspace reset and lockout",
                "incident and near-miss logging",
            ),
            pretrial_checks=(
                "emergency stop functional",
                "workspace clear",
                "robot and sensor self-tests pass",
                "calibration records current",
            ),
            stop_conditions=(
                "human enters controlled workspace",
                "unexpected high-force contact",
                "robot or fixture damage risk",
                "loss of required control or sensor stream",
            ),
            emergency_stop_test="Test and timestamp the hardware stop before each session.",
            workspace_controls=(
                "physical exclusion zone",
                "speed and force limits",
                "single designated operator",
            ),
            incident_reporting=(
                "Retain all failures, protective stops, near misses, and damage events "
                "with trial identity and evidence access status."
            ),
            damage_measurements=(
                "peak_contact_force",
                "protective_stop_count",
                "fixture_displacement",
                "visible_damage_annotation",
            ),
            operator_intervention_policy=(
                "Intervene only under stop conditions; retain the trial as failed or "
                "censored according to the frozen exclusion rule."
            ),
        ),
        governance=GovernanceContract(
            privacy_classification="restricted",
            consent_basis="institutional hardware study authorization",
            license_id="pending-study-data-license",
            redistribution="metadata_only",
            redactions=("operator identity", "site security details"),
            retention_policy="retain raw evidence through study audit and publication",
            retention_until=date(2032, 1, 1),
            artifact_access_rules=(
                "raw evidence restricted to authorized study personnel",
                "sanitized manifests and aggregate reports may be public",
                "every redaction preserves content hashes and reason codes",
            ),
            operator_ids_pseudonymous=True,
        ),
        metadata={
            "sites": ["pending-site"],
            "robot": "pending-panda-compatible-hardware",
            "dependency_boundary": (
                "Reference benchmark and policy tracks must be released before freezing."
            ),
        },
    )


def _clean_stressor() -> StressorContract:
    return StressorContract(
        condition_id="hardware-clean",
        composition_semantics="ordered",
        stressors=(),
    )


def _latency_stressor() -> StressorContract:
    return StressorContract(
        condition_id="hardware-action-delay-s05",
        composition_semantics="ordered",
        stressors=(
            StressorEntryContract(
                stressor_id="action_delay",
                stressor_version="1.0.0",
                category="action",
                severity=0.5,
                seed=0,
                application_points=("before_step",),
                parameters={"delay_steps": 2},
                observable_by_policy=False,
                privileged=False,
                backend_confirmed=False,
            ),
        ),
    )


def _condition(task_id: str, policy_id: str, stressor_id: str) -> CalibrationCondition:
    shifted = stressor_id != "hardware-clean"
    return CalibrationCondition(
        condition_id=f"{task_id}-{policy_id}-{stressor_id}",
        task_id=task_id,
        policy_id=policy_id,
        stressor_condition_id=stressor_id,
        severity=0.5 if shifted else 0.0,
        hardware_condition_id=f"real-{task_id}-{stressor_id}",
        simulation_condition_id=f"sim-{task_id}-{stressor_id}",
        trial_count=20,
        matched_axes=(
            "task goal",
            "policy checkpoint",
            "controller rate",
            "action delay",
            "initial pose stratum",
        ),
        mismatches=(
            ConditionMismatch(
                mismatch_id="residual-dynamics",
                category="dynamics",
                description="Residual contact and actuator dynamics mismatch after calibration.",
                expected_direction="unknown",
            ),
            ConditionMismatch(
                mismatch_id="visual-rendering",
                category="appearance",
                description="Rendering cannot exactly reproduce hardware illumination and optics.",
                expected_direction="unknown",
            ),
        ),
        recovery_design="matched_trials" if shifted else "disabled",
        recovery_trial_count=20 if shifted else 0,
    )


def _artifact(root: Path, path: Path) -> ArtifactReference:
    return ArtifactReference(
        path=path.relative_to(root).as_posix(),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
