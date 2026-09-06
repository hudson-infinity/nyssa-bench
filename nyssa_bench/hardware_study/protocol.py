from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nyssa_bench.nep import (
    ClaimContract,
    FailureEvidenceContract,
    InterventionContract,
    PolicyContract,
    StressorContract,
    TaskContract,
)
from nyssa_bench.real_evidence import GovernanceContract
from nyssa_bench.reference_benchmark import ArtifactReference


HARDWARE_STUDY_FORMAT = "nyssa-hardware-calibration-study-v1"
PREREGISTRATION_RECEIPT_FORMAT = "nyssa-preregistration-receipt-v1"
StudyStatus = Literal["draft", "preregistered", "complete"]


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ConditionMismatch(ContractModel):
    mismatch_id: str
    category: Literal["geometry", "appearance", "dynamics", "latency", "sensor", "task"]
    description: str
    expected_direction: Literal[
        "simulation_harder", "hardware_harder", "unknown", "approximately_matched"
    ]
    magnitude: float | None = Field(default=None, ge=0.0)
    unit: str | None = None
    uncertainty: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def quantified_together(self) -> "ConditionMismatch":
        _identifier(self.mismatch_id, "mismatch_id")
        if not self.description.strip():
            raise ValueError("condition mismatch description must be non-empty")
        supplied = (
            self.magnitude is not None,
            self.unit is not None,
            self.uncertainty is not None,
        )
        if any(supplied) and not all(supplied):
            raise ValueError(
                "mismatch magnitude, unit, and uncertainty are all-or-none"
            )
        return self


class CalibrationCondition(ContractModel):
    condition_id: str
    task_id: str
    policy_id: str
    stressor_condition_id: str
    severity: float = Field(ge=0.0, le=1.0)
    hardware_condition_id: str
    simulation_condition_id: str
    trial_count: int = Field(ge=20)
    matched_axes: tuple[str, ...]
    mismatches: tuple[ConditionMismatch, ...]
    recovery_design: Literal["disabled", "matched_trials"]
    recovery_trial_count: int = Field(ge=0)

    @model_validator(mode="after")
    def complete_condition(self) -> "CalibrationCondition":
        for value, label in (
            (self.condition_id, "condition_id"),
            (self.task_id, "task_id"),
            (self.policy_id, "policy_id"),
            (self.stressor_condition_id, "stressor_condition_id"),
            (self.hardware_condition_id, "hardware_condition_id"),
            (self.simulation_condition_id, "simulation_condition_id"),
        ):
            _identifier(value, label)
        _unique(self.matched_axes, "matched axes")
        _unique(tuple(item.mismatch_id for item in self.mismatches), "mismatch IDs")
        if self.recovery_design == "disabled" and self.recovery_trial_count:
            raise ValueError("disabled recovery cannot declare recovery trials")
        if self.recovery_design == "matched_trials" and self.recovery_trial_count < 20:
            raise ValueError("matched recovery requires at least 20 trials per branch")
        return self


class ExclusionRule(ContractModel):
    rule_id: str
    criterion: str
    decision_time: Literal["before_outcome", "after_outcome_blinded"]
    treatment: Literal["exclude", "censor", "retain_with_flag"]

    @model_validator(mode="after")
    def complete_rule(self) -> "ExclusionRule":
        _identifier(self.rule_id, "exclusion rule_id")
        if not self.criterion.strip():
            raise ValueError("exclusion criterion must be non-empty")
        return self


class AnalysisPlan(ContractModel):
    primary_metrics: tuple[
        Literal[
            "policy_rank",
            "failure_distribution",
            "shift_response",
            "time_to_failure",
            "recovery_effect",
            "incremental_predictive_value",
        ],
        ...,
    ]
    baseline_features: tuple[str, ...]
    enhanced_features: tuple[str, ...]
    heldout_shift_ids: tuple[str, ...]
    bootstrap_samples: int = Field(ge=1000)
    bootstrap_seed: int = Field(ge=0)
    cluster_fields: tuple[str, ...]
    sensitivity_analyses: tuple[str, ...]
    negative_result_policy: str

    @model_validator(mode="after")
    def complete_analysis(self) -> "AnalysisPlan":
        required = {
            "policy_rank",
            "failure_distribution",
            "shift_response",
            "time_to_failure",
            "incremental_predictive_value",
        }
        if not required <= set(self.primary_metrics):
            raise ValueError("hardware study lacks required primary analyses")
        for values, label in (
            (self.primary_metrics, "primary metrics"),
            (self.baseline_features, "baseline features"),
            (self.enhanced_features, "enhanced features"),
            (self.heldout_shift_ids, "held-out shifts"),
            (self.cluster_fields, "cluster fields"),
            (self.sensitivity_analyses, "sensitivity analyses"),
        ):
            _unique(values, label)
        if "clean_sim_success" not in self.baseline_features:
            raise ValueError("baseline model must include clean simulation success")
        if not set(self.baseline_features) < set(self.enhanced_features):
            raise ValueError("enhanced model must strictly extend baseline features")
        if not self.negative_result_policy.strip():
            raise ValueError("negative result reporting policy must be explicit")
        return self


class SafetyPlan(ContractModel):
    risk_assessment_id: str
    operator_training: tuple[str, ...]
    pretrial_checks: tuple[str, ...]
    stop_conditions: tuple[str, ...]
    emergency_stop_test: str
    workspace_controls: tuple[str, ...]
    incident_reporting: str
    damage_measurements: tuple[str, ...]
    operator_intervention_policy: str

    @model_validator(mode="after")
    def complete_safety(self) -> "SafetyPlan":
        _identifier(self.risk_assessment_id, "risk_assessment_id")
        for values, label in (
            (self.operator_training, "operator training"),
            (self.pretrial_checks, "pretrial checks"),
            (self.stop_conditions, "stop conditions"),
            (self.workspace_controls, "workspace controls"),
            (self.damage_measurements, "damage measurements"),
        ):
            _unique(values, label)
        for value in (
            self.emergency_stop_test,
            self.incident_reporting,
            self.operator_intervention_policy,
        ):
            if not value.strip():
                raise ValueError("safety procedures must be non-empty")
        return self


class HardwareEvidence(ContractModel):
    preregistration_receipt: ArtifactReference | None = None
    real_evidence_packages: tuple[ArtifactReference, ...] = ()
    sim_real_study_spec: ArtifactReference | None = None
    sim_real_study_report: ArtifactReference | None = None
    benchmark_validity_report: ArtifactReference | None = None


class HardwareCalibrationStudy(ContractModel):
    format: Literal["nyssa-hardware-calibration-study-v1"] = (
        "nyssa-hardware-calibration-study-v1"
    )
    schema_version: Literal[1] = 1
    study_id: str
    study_version: str
    status: StudyStatus
    protocol_authored_at: datetime
    first_trial_not_before: datetime
    reference_benchmark: ArtifactReference
    policy_track_registry: ArtifactReference
    tasks: tuple[TaskContract, ...]
    policies: tuple[PolicyContract, ...]
    stressors: tuple[StressorContract, ...]
    failure_evidence: FailureEvidenceContract
    intervention: InterventionContract
    claim: ClaimContract
    conditions: tuple[CalibrationCondition, ...]
    exclusions: tuple[ExclusionRule, ...]
    analysis: AnalysisPlan
    safety: SafetyPlan
    governance: GovernanceContract
    evidence: HardwareEvidence = Field(default_factory=HardwareEvidence)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("study_version")
    @classmethod
    def semver(cls, value: str) -> str:
        if not re.fullmatch(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)", value):
            raise ValueError("study_version must use semantic versioning")
        return value

    @model_validator(mode="after")
    def complete_protocol(self) -> "HardwareCalibrationStudy":
        _identifier(self.study_id, "study_id")
        if (
            self.protocol_authored_at.tzinfo is None
            or self.first_trial_not_before.tzinfo is None
        ):
            raise ValueError("study timestamps must include timezones")
        if self.first_trial_not_before <= self.protocol_authored_at:
            raise ValueError("first trial must occur after protocol authorship")
        task_ids = tuple(task.task_id for task in self.tasks)
        policy_ids = tuple(policy.policy_id for policy in self.policies)
        stressor_ids = tuple(stressor.condition_id for stressor in self.stressors)
        _unique(task_ids, "hardware task IDs")
        _unique(policy_ids, "hardware policy IDs")
        _unique(stressor_ids, "hardware stressor conditions")
        condition_ids = tuple(condition.condition_id for condition in self.conditions)
        _unique(condition_ids, "hardware condition IDs")
        _unique(tuple(rule.rule_id for rule in self.exclusions), "exclusion rules")
        for condition in self.conditions:
            if condition.task_id not in task_ids:
                raise ValueError("condition references an unknown task")
            if condition.policy_id not in policy_ids:
                raise ValueError("condition references an unknown policy")
            if condition.stressor_condition_id not in stressor_ids:
                raise ValueError("condition references an unknown stressor")
            task = next(
                item for item in self.tasks if item.task_id == condition.task_id
            )
            policy = next(
                item for item in self.policies if item.policy_id == condition.policy_id
            )
            if task.action_representation != policy.action_representation:
                raise ValueError(
                    "hardware task and policy action representations differ"
                )
            if not set(policy.observation_modalities) <= set(
                task.observation_modalities
            ):
                raise ValueError("hardware task lacks policy observation modalities")
            if policy.prediction_horizon > task.horizon_steps:
                raise ValueError("hardware policy horizon exceeds task horizon")
        expected_cells = {
            (task_id, policy_id, stressor_id)
            for task_id in task_ids
            for policy_id in policy_ids
            for stressor_id in stressor_ids
        }
        observed_cells = {
            (item.task_id, item.policy_id, item.stressor_condition_id)
            for item in self.conditions
        }
        if observed_cells != expected_cells:
            raise ValueError(
                "hardware conditions do not form a complete factorial design"
            )
        if not set(self.analysis.heldout_shift_ids) <= set(stressor_ids):
            raise ValueError("analysis references an unknown held-out shift")
        recovery_enabled = any(
            condition.recovery_design == "matched_trials"
            for condition in self.conditions
        )
        if recovery_enabled != ("recovery_effect" in self.analysis.primary_metrics):
            raise ValueError("recovery analysis and matched-trial design disagree")
        if self.status in {"preregistered", "complete"} and (
            self.evidence.preregistration_receipt is None
        ):
            raise ValueError("frozen studies require a preregistration receipt")
        if self.status == "complete":
            if len(self.evidence.real_evidence_packages) != sum(
                condition.trial_count + condition.recovery_trial_count
                for condition in self.conditions
            ):
                raise ValueError("complete study lacks one package per planned trial")
            if any(
                item is None
                for item in (
                    self.evidence.sim_real_study_spec,
                    self.evidence.sim_real_study_report,
                    self.evidence.benchmark_validity_report,
                )
            ):
                raise ValueError("complete study lacks analysis or validity evidence")
        _finite_json(self.metadata)
        return self

    @property
    def design_sha256(self) -> str:
        payload = self.model_dump(mode="json")
        payload["status"] = "draft"
        payload["evidence"] = HardwareEvidence().model_dump(mode="json")
        return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()


def _identifier(value: str, label: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise ValueError(f"{label} must be a portable identifier")


def _unique(values: Any, label: str) -> None:
    if not values or len(values) != len(set(values)):
        raise ValueError(f"{label} must be non-empty and unique")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _finite_json(value: Any) -> None:
    try:
        _canonical_json(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("metadata must contain finite JSON data") from exc
