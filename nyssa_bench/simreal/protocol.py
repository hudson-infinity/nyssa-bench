from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SIM_REAL_STUDY_FORMAT = "nyssa-sim-real-study-v1"


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SimulationReference(ContractModel):
    run_dir: str
    run_id: str
    artifacts_sha256: dict[str, str]
    policy_name: str
    checkpoint_id: str
    checkpoint_sha256: str
    preprocessing_sha256: str
    task_id: str
    episode_seed: int = Field(ge=0)
    episode_index: int = Field(ge=0)

    @field_validator("checkpoint_sha256", "preprocessing_sha256")
    @classmethod
    def hashes(cls, value: str) -> str:
        return _sha256(value)

    @field_validator("artifacts_sha256")
    @classmethod
    def artifact_hashes(cls, value: dict[str, str]) -> dict[str, str]:
        required = {"run.yaml", "dataset_manifest.json", "metrics.json", "episodes.json"}
        if not required <= set(value):
            raise ValueError("simulation reference is missing required artifact hashes")
        for digest in value.values():
            _sha256(digest)
        return value


class RealReference(ContractModel):
    package_path: str
    package_identity: str
    real_episode_id: str
    variant_id: str
    trial_id: str

    @field_validator("package_identity")
    @classmethod
    def identity(cls, value: str) -> str:
        if not re.fullmatch(r"[^@:]+@[^:]+:[0-9a-f]{64}", value):
            raise ValueError("real package identity is malformed")
        return value


class SimRealPair(ContractModel):
    pair_id: str
    policy_id: str
    task_id: str
    shift_id: str
    severity: float = Field(ge=0.0, le=1.0)
    simulation: SimulationReference
    real: RealReference
    included: bool = True
    exclusion_reason: str | None = None
    sim_step_seconds: float = Field(gt=0.0)
    real_event_step_seconds: float = Field(gt=0.0)

    @model_validator(mode="after")
    def exclusion_is_declared(self) -> "SimRealPair":
        if self.included and self.exclusion_reason:
            raise ValueError("included pairs cannot have exclusion reasons")
        if not self.included and not self.exclusion_reason:
            raise ValueError("excluded pairs require a prespecified reason")
        if self.task_id != self.simulation.task_id:
            raise ValueError("pair and simulation task IDs differ")
        if self.policy_id != self.simulation.policy_name:
            raise ValueError("pair and simulation policy IDs differ")
        return self


class SimRealStudySpec(ContractModel):
    format: Literal["nyssa-sim-real-study-v1"] = "nyssa-sim-real-study-v1"
    schema_version: Literal[1] = 1
    study_id: str
    study_version: str
    prespecified_at: datetime
    unit_of_analysis: Literal["policy_task_shift_trial"] = "policy_task_shift_trial"
    pairs: tuple[SimRealPair, ...]
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
    bootstrap_samples: int = Field(ge=200, le=100000)
    bootstrap_seed: int = Field(ge=0)
    cluster_fields: tuple[Literal["policy_id", "task_id", "shift_id", "trial_id"], ...]
    holdout_shift_ids: tuple[str, ...]
    recovery_assumption: Literal[
        "disabled", "matched_real_trials", "counterfactual_sim_only"
    ] = "disabled"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("study_version")
    @classmethod
    def semver(cls, value: str) -> str:
        if not re.fullmatch(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)", value):
            raise ValueError("study_version must use semantic versioning")
        return value

    @model_validator(mode="after")
    def validate_study(self) -> "SimRealStudySpec":
        if not self.pairs:
            raise ValueError("sim-real study requires pairs")
        pair_ids = [pair.pair_id for pair in self.pairs]
        if len(pair_ids) != len(set(pair_ids)):
            raise ValueError("sim-real pair IDs must be unique")
        simulation_ids = [
            (
                pair.simulation.run_id,
                pair.simulation.task_id,
                pair.simulation.episode_seed,
                pair.simulation.episode_index,
            )
            for pair in self.pairs
            if pair.included
        ]
        real_ids = [
            (pair.real.package_identity, pair.real.real_episode_id, pair.real.variant_id)
            for pair in self.pairs
            if pair.included
        ]
        if len(simulation_ids) != len(set(simulation_ids)):
            raise ValueError("many-to-one simulation episode mappings are forbidden")
        if len(real_ids) != len(set(real_ids)):
            raise ValueError("many-to-one real episode mappings are forbidden")
        if not self.primary_metrics or len(self.primary_metrics) != len(
            set(self.primary_metrics)
        ):
            raise ValueError("primary metrics must be non-empty and unique")
        if not self.cluster_fields or len(self.cluster_fields) != len(
            set(self.cluster_fields)
        ):
            raise ValueError("bootstrap cluster fields must be non-empty and unique")
        included_shifts = {pair.shift_id for pair in self.pairs if pair.included}
        if not set(self.holdout_shift_ids) <= included_shifts:
            raise ValueError("holdout shifts must exist in included pairs")
        if len(self.holdout_shift_ids) != len(set(self.holdout_shift_ids)):
            raise ValueError("holdout shift IDs must be unique")
        if (
            "incremental_predictive_value" in self.primary_metrics
            and not self.holdout_shift_ids
        ):
            raise ValueError("incremental predictive analysis requires held-out shifts")
        if self.prespecified_at.tzinfo is None:
            raise ValueError("prespecified_at must include a timezone")
        if (
            "recovery_effect" in self.primary_metrics
            and self.recovery_assumption == "disabled"
        ):
            raise ValueError("recovery metric requires a declared matching assumption")
        _canonical_json(self.metadata)
        return self

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.model_dump(mode="json")).encode()).hexdigest()


def _sha256(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("value must be a lowercase SHA-256 digest")
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
