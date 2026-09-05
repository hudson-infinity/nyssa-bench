from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


NEP_VERSION = "0.1.0"
NEP_MANIFEST_FORMAT = "nyssa-evaluation-protocol-v0.1"

ClaimTier = Literal[
    "pipeline",
    "clean_simulation",
    "ood_robustness",
    "recovery_effectiveness",
    "cross_simulator",
    "sim_real_predictive",
]


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def reject_blank_strings(self) -> "ContractModel":
        for field_name in type(self).model_fields:
            value = getattr(self, field_name)
            if isinstance(value, str) and not value.strip():
                raise ValueError(f"{field_name} must be non-empty")
            if isinstance(value, tuple) and any(
                isinstance(item, str) and not item.strip() for item in value
            ):
                raise ValueError(f"{field_name} cannot contain blank strings")
        return self


class ArtifactContract(ContractModel):
    artifact_id: str
    media_type: str
    sha256: str
    uri: str
    required: bool = True

    @field_validator("sha256")
    @classmethod
    def hash_is_sha256(cls, value: str) -> str:
        return _sha256(value, "artifact.sha256")


class AssetContract(ContractModel):
    asset_id: str
    asset_version: str
    sha256: str
    license_id: str
    split: Literal["train", "validation", "public_test", "hidden_test"]

    @field_validator("asset_version")
    @classmethod
    def version_is_semver(cls, value: str) -> str:
        return _semver(value, "asset_version")

    @field_validator("sha256")
    @classmethod
    def hash_is_sha256(cls, value: str) -> str:
        return _sha256(value, "asset.sha256")


class SplitLineageContract(ContractModel):
    split_id: str
    partition: Literal["train", "validation", "public_test", "hidden_test"]
    lineage_sha256: str
    parent_split_ids: tuple[str, ...] = ()
    training_overlap_allowed: bool = False

    @field_validator("lineage_sha256")
    @classmethod
    def hash_is_sha256(cls, value: str) -> str:
        return _sha256(value, "split.lineage_sha256")


class TaskContract(ContractModel):
    format: Literal["nyssa-nep-task-contract-v0.1"] = (
        "nyssa-nep-task-contract-v0.1"
    )
    contract_version: str = NEP_VERSION
    task_id: str
    task_version: str
    engine_ids: tuple[str, ...]
    robot_id: str
    scene_id: str
    horizon_steps: int = Field(gt=0)
    observation_modalities: tuple[str, ...]
    action_representation: str
    success_predicate: dict[str, Any]
    assets: tuple[AssetContract, ...]
    split_lineage: SplitLineageContract

    @field_validator("contract_version", "task_version")
    @classmethod
    def versions_are_semver(cls, value: str) -> str:
        return _semver(value, "task contract version")

    @model_validator(mode="after")
    def validate_task(self) -> "TaskContract":
        _unique(self.engine_ids, "task engine_ids")
        _unique(self.observation_modalities, "task observation modalities")
        if not self.success_predicate:
            raise ValueError("task success_predicate must be non-empty")
        if not self.assets:
            raise ValueError("task contract requires at least one asset")
        asset_ids = [asset.asset_id for asset in self.assets]
        if len(asset_ids) != len(set(asset_ids)):
            raise ValueError("task asset IDs must be unique")
        if self.split_lineage.partition != "train" and (
            self.split_lineage.training_overlap_allowed
        ):
            raise ValueError("evaluation splits cannot allow training overlap")
        _finite_json(self.success_predicate, "task success_predicate")
        return self


class StressorEntryContract(ContractModel):
    stressor_id: str
    stressor_version: str
    category: Literal["visual", "dynamics", "sensor", "system", "action", "language"]
    severity: float = Field(ge=0.0, le=1.0)
    seed: int = Field(ge=0)
    application_points: tuple[str, ...]
    parameters: dict[str, Any]
    observable_by_policy: bool
    privileged: bool
    backend_confirmed: bool
    backend_evidence_artifact_id: str | None = None

    @field_validator("stressor_version")
    @classmethod
    def version_is_semver(cls, value: str) -> str:
        return _semver(value, "stressor_version")

    @model_validator(mode="after")
    def validate_stressor(self) -> "StressorEntryContract":
        _unique(self.application_points, "stressor application_points")
        _finite_json(self.parameters, "stressor parameters")
        if self.backend_confirmed and not self.backend_evidence_artifact_id:
            raise ValueError(
                "backend-confirmed stressors require an evidence artifact"
            )
        if not self.backend_confirmed and self.backend_evidence_artifact_id:
            raise ValueError(
                "unconfirmed stressors cannot reference backend application evidence"
            )
        return self


class StressorContract(ContractModel):
    format: Literal["nyssa-nep-stressor-contract-v0.1"] = (
        "nyssa-nep-stressor-contract-v0.1"
    )
    contract_version: str = NEP_VERSION
    condition_id: str
    composition_semantics: Literal["ordered", "commutative"]
    stressors: tuple[StressorEntryContract, ...] = ()

    @field_validator("contract_version")
    @classmethod
    def version_is_semver(cls, value: str) -> str:
        return _semver(value, "stressor contract version")

    @model_validator(mode="after")
    def unique_stressors(self) -> "StressorContract":
        ids = [stressor.stressor_id for stressor in self.stressors]
        if len(ids) != len(set(ids)):
            raise ValueError("stressor IDs must be unique within a condition")
        return self


class TrainingDataContract(ContractModel):
    dataset_id: str
    dataset_version: str
    sha256: str
    split_ids: tuple[str, ...]
    license_id: str

    @field_validator("dataset_version")
    @classmethod
    def version_is_semver(cls, value: str) -> str:
        return _semver(value, "dataset_version")

    @field_validator("sha256")
    @classmethod
    def hash_is_sha256(cls, value: str) -> str:
        return _sha256(value, "dataset.sha256")


class PolicyContract(ContractModel):
    format: Literal["nyssa-nep-policy-contract-v0.1"] = (
        "nyssa-nep-policy-contract-v0.1"
    )
    contract_version: str = NEP_VERSION
    policy_id: str
    policy_version: str
    policy_family: str
    checkpoint_id: str
    checkpoint_sha256: str
    preprocessing_sha256: str
    observation_modalities: tuple[str, ...]
    action_representation: str
    action_dimension: int = Field(gt=0)
    action_lower_bounds: tuple[float, ...]
    action_upper_bounds: tuple[float, ...]
    prediction_horizon: int = Field(gt=0)
    execution_horizon: int = Field(gt=0)
    state_semantics: Literal["stateless", "resettable", "restorable"]
    deterministic_seeding: bool
    training_data: tuple[TrainingDataContract, ...] = ()

    @field_validator("contract_version", "policy_version")
    @classmethod
    def versions_are_semver(cls, value: str) -> str:
        return _semver(value, "policy contract version")

    @field_validator("checkpoint_sha256", "preprocessing_sha256")
    @classmethod
    def hashes_are_sha256(cls, value: str) -> str:
        return _sha256(value, "policy hash")

    @model_validator(mode="after")
    def validate_policy(self) -> "PolicyContract":
        _unique(self.observation_modalities, "policy observation modalities")
        if self.execution_horizon > self.prediction_horizon:
            raise ValueError("execution horizon cannot exceed prediction horizon")
        if any(
            len(values) != self.action_dimension
            for values in (self.action_lower_bounds, self.action_upper_bounds)
        ):
            raise ValueError("policy action bounds must match action_dimension")
        if any(
            not math.isfinite(low) or not math.isfinite(high) or low >= high
            for low, high in zip(self.action_lower_bounds, self.action_upper_bounds)
        ):
            raise ValueError("policy action bounds must be finite and ordered")
        return self


class FailureEvidenceContract(ContractModel):
    format: Literal["nyssa-nep-failure-evidence-contract-v0.1"] = (
        "nyssa-nep-failure-evidence-contract-v0.1"
    )
    contract_version: str = NEP_VERSION
    event_format: Literal["nyssa-failure-event-v1"] = "nyssa-failure-event-v1"
    ledger_artifact_id: str
    detector_contract_artifact_id: str
    temporal_precision: tuple[
        Literal["exact_step", "step_interval", "terminal_only", "unknown"], ...
    ]
    evidence_visibility: tuple[Literal["policy_observable", "privileged", "external"], ...]
    causal_semantics: Literal["hypothesis_only", "intervention_supported"]

    @field_validator("contract_version")
    @classmethod
    def version_is_semver(cls, value: str) -> str:
        return _semver(value, "failure evidence contract version")

    @model_validator(mode="after")
    def validate_failure_evidence(self) -> "FailureEvidenceContract":
        _unique(self.temporal_precision, "failure temporal precision")
        _unique(self.evidence_visibility, "failure evidence visibility")
        return self


class InterventionContract(ContractModel):
    format: Literal["nyssa-nep-intervention-contract-v0.1"] = (
        "nyssa-nep-intervention-contract-v0.1"
    )
    contract_version: str = NEP_VERSION
    enabled: bool
    trigger_sources: tuple[str, ...] = ()
    intervention_types: tuple[str, ...] = ()
    cost_metrics: tuple[str, ...] = ()
    counterfactual_branch_artifact_id: str | None = None
    restoration_requirement: Literal["none", "qualified", "exact"] = "none"

    @field_validator("contract_version")
    @classmethod
    def version_is_semver(cls, value: str) -> str:
        return _semver(value, "intervention contract version")

    @model_validator(mode="after")
    def validate_intervention(self) -> "InterventionContract":
        for values, label in (
            (self.trigger_sources, "intervention trigger_sources"),
            (self.intervention_types, "intervention types"),
            (self.cost_metrics, "intervention cost_metrics"),
        ):
            if values:
                _unique(values, label)
        if self.enabled and (not self.trigger_sources or not self.intervention_types):
            raise ValueError("enabled interventions require triggers and types")
        if not self.enabled and (
            self.trigger_sources
            or self.intervention_types
            or self.counterfactual_branch_artifact_id
            or self.restoration_requirement != "none"
        ):
            raise ValueError("disabled interventions cannot declare execution evidence")
        return self


class ClaimContract(ContractModel):
    format: Literal["nyssa-nep-claim-contract-v0.1"] = (
        "nyssa-nep-claim-contract-v0.1"
    )
    contract_version: str = NEP_VERSION
    requested_tier: ClaimTier
    evidence_artifact_ids: tuple[str, ...]
    run_validity_artifact_id: str
    benchmark_validity_artifact_id: str | None = None
    real_evidence_artifact_id: str | None = None

    @field_validator("contract_version")
    @classmethod
    def version_is_semver(cls, value: str) -> str:
        return _semver(value, "claim contract version")

    @model_validator(mode="after")
    def unique_evidence(self) -> "ClaimContract":
        _unique(self.evidence_artifact_ids, "claim evidence artifacts")
        return self


class NEPManifest(ContractModel):
    format: Literal["nyssa-evaluation-protocol-v0.1"] = (
        "nyssa-evaluation-protocol-v0.1"
    )
    nep_version: str = NEP_VERSION
    evaluation_id: str
    task: TaskContract
    stressor: StressorContract
    policy: PolicyContract
    failure_evidence: FailureEvidenceContract
    intervention: InterventionContract
    claim: ClaimContract
    artifacts: tuple[ArtifactContract, ...]
    content_sha256: str

    @field_validator("nep_version")
    @classmethod
    def version_is_current(cls, value: str) -> str:
        _semver(value, "nep_version")
        if value != NEP_VERSION:
            raise ValueError(f"unsupported NEP version: {value}")
        return value

    @field_validator("content_sha256")
    @classmethod
    def content_hash_is_sha256(cls, value: str) -> str:
        return _sha256(value, "content_sha256")

    @model_validator(mode="after")
    def validate_cross_contracts(self) -> "NEPManifest":
        if self.policy.action_representation != self.task.action_representation:
            raise ValueError("policy and task action representations differ")
        if not set(self.policy.observation_modalities) <= set(
            self.task.observation_modalities
        ):
            raise ValueError("policy requires task observation modalities that are absent")
        if self.policy.prediction_horizon > self.task.horizon_steps:
            raise ValueError("policy prediction horizon exceeds the task horizon")
        training_splits = {
            split_id
            for dataset in self.policy.training_data
            for split_id in dataset.split_ids
        }
        if self.task.split_lineage.split_id in training_splits:
            raise ValueError("policy training data overlaps the evaluation split")
        artifact_ids = [artifact.artifact_id for artifact in self.artifacts]
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("NEP artifact IDs must be unique")
        known = set(artifact_ids)
        references = {
            self.failure_evidence.ledger_artifact_id,
            self.failure_evidence.detector_contract_artifact_id,
            self.claim.run_validity_artifact_id,
            *self.claim.evidence_artifact_ids,
        }
        for optional in (
            self.intervention.counterfactual_branch_artifact_id,
            self.claim.benchmark_validity_artifact_id,
            self.claim.real_evidence_artifact_id,
        ):
            if optional:
                references.add(optional)
        for stressor in self.stressor.stressors:
            if stressor.backend_evidence_artifact_id:
                references.add(stressor.backend_evidence_artifact_id)
        unknown = references - known
        if unknown:
            raise ValueError(
                "NEP contracts reference unknown artifacts: " + ", ".join(sorted(unknown))
            )
        self._validate_claim_tier()
        if self.compute_content_sha256() != self.content_sha256:
            raise ValueError("NEP content_sha256 does not match canonical content")
        return self

    def _validate_claim_tier(self) -> None:
        tier = self.claim.requested_tier
        if tier != "pipeline" and not self.claim.benchmark_validity_artifact_id:
            raise ValueError(f"{tier} claims require BenchmarkValidity evidence")
        if tier == "clean_simulation" and any(
            stressor.severity > 0 for stressor in self.stressor.stressors
        ):
            raise ValueError("clean simulation claims cannot use positive stressor severity")
        if tier == "ood_robustness":
            if not self.stressor.stressors or not all(
                stressor.severity > 0 and stressor.backend_confirmed
                for stressor in self.stressor.stressors
            ):
                raise ValueError(
                    "OOD robustness claims require positive, backend-confirmed stressors"
                )
        if tier == "recovery_effectiveness" and (
            not self.intervention.enabled
            or not self.intervention.counterfactual_branch_artifact_id
            or self.intervention.restoration_requirement == "none"
        ):
            raise ValueError(
                "recovery claims require enabled counterfactual intervention evidence"
            )
        if tier == "cross_simulator" and len(self.task.engine_ids) < 2:
            raise ValueError("cross-simulator claims require at least two engines")
        if tier == "sim_real_predictive" and not self.claim.real_evidence_artifact_id:
            raise ValueError("sim-real claims require real evidence")

    def compute_content_sha256(self) -> str:
        payload = self.model_dump(mode="json", exclude={"content_sha256"})
        return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()

    @classmethod
    def create(cls, **data: Any) -> "NEPManifest":
        unhashed = cls.model_construct(content_sha256="0" * 64, **data)
        digest = unhashed.compute_content_sha256()
        return cls.model_validate({**data, "content_sha256": digest})


def _semver(value: str, label: str) -> str:
    if not re.fullmatch(
        r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:[-+][0-9A-Za-z.-]+)?",
        value,
    ):
        raise ValueError(f"{label} must use semantic versioning")
    return value


def _sha256(value: str, label: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _unique(values: tuple[Any, ...], label: str) -> None:
    if not values or len(values) != len(set(values)):
        raise ValueError(f"{label} must be non-empty and unique")


def _finite_json(value: Any, label: str) -> None:
    try:
        _canonical_json(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain finite JSON data") from exc


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
