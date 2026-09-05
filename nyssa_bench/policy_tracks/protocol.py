from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nyssa_bench.nep import PolicyContract
from nyssa_bench.reference_benchmark import ArtifactReference


POLICY_TRACK_SPEC_FORMAT = "nyssa-policy-track-registry-v1"
TRAINING_PROVENANCE_FORMAT = "nyssa-policy-training-provenance-v1"
TrackRole = Literal["oracle_control", "learned", "vla", "sanity_control"]
TrackStatus = Literal["integration_only", "validated"]


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ComputeContract(ContractModel):
    hardware: str
    accelerator_count: int = Field(ge=0)
    training_hours: float = Field(ge=0.0)
    peak_memory_gb: float | None = Field(default=None, gt=0.0)
    precision: str

    @model_validator(mode="after")
    def complete_compute(self) -> "ComputeContract":
        if not self.hardware.strip() or not self.precision.strip():
            raise ValueError("compute hardware and precision must be declared")
        return self


class PolicyTrack(ContractModel):
    track_id: str
    role: TrackRole
    status: TrackStatus
    required_for_release: bool
    adapter_id: str
    contract: PolicyContract
    setup_document: ArtifactReference
    checkpoint_artifact: ArtifactReference | None = None
    preprocessing_artifact: ArtifactReference | None = None
    training_provenance: ArtifactReference | None = None
    conformance_reports: tuple[ArtifactReference, ...] = ()
    clean_run_fingerprints: tuple[ArtifactReference, ...] = ()
    shifted_run_fingerprints: tuple[ArtifactReference, ...] = ()
    evaluation_task_ids: tuple[str, ...]
    evaluation_split_id: str
    evaluation_split_sha256: str
    evaluation_seeds: tuple[int, ...]
    evaluation_asset_ids: tuple[str, ...]
    stressor_condition_id: str
    evaluation_compute: ComputeContract

    @field_validator("evaluation_split_sha256")
    @classmethod
    def digest(cls, value: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("evaluation split hash must be a lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def valid_track(self) -> "PolicyTrack":
        _identifier(self.track_id, "track_id")
        _identifier(self.adapter_id, "adapter_id")
        _identifier(self.evaluation_split_id, "evaluation_split_id")
        _identifier(self.stressor_condition_id, "stressor_condition_id")
        _unique(self.evaluation_task_ids, "evaluation task IDs")
        _unique(self.evaluation_seeds, "evaluation seeds")
        _unique(self.evaluation_asset_ids, "evaluation asset IDs")
        if any(seed < 0 for seed in self.evaluation_seeds):
            raise ValueError("evaluation seeds must be non-negative")
        if self.role == "sanity_control" and self.status == "validated":
            raise ValueError("sanity controls cannot be validated headline tracks")
        if self.role in {"learned", "vla"} and not self.contract.training_data:
            raise ValueError("learned tracks must declare training data in NEP")
        if self.status == "validated":
            required = {
                "checkpoint_artifact": self.checkpoint_artifact,
                "preprocessing_artifact": self.preprocessing_artifact,
                "conformance_reports": self.conformance_reports,
                "clean_run_fingerprints": self.clean_run_fingerprints,
                "shifted_run_fingerprints": self.shifted_run_fingerprints,
            }
            if self.role in {"learned", "vla"}:
                required["training_provenance"] = self.training_provenance
            missing = [name for name, value in required.items() if value is None]
            missing.extend(name for name, value in required.items() if value == ())
            if missing:
                raise ValueError(
                    "validated policy track is missing: " + ", ".join(missing)
                )
            if len(self.conformance_reports) != len(self.evaluation_task_ids):
                raise ValueError("validated track requires conformance for every task")
            if not (
                len(self.clean_run_fingerprints)
                == len(self.shifted_run_fingerprints)
                == len(self.evaluation_seeds)
            ):
                raise ValueError("validated track requires clean/shifted runs per seed")
        return self


class PolicyTrackRegistry(ContractModel):
    format: Literal["nyssa-policy-track-registry-v1"] = "nyssa-policy-track-registry-v1"
    schema_version: Literal[1] = 1
    registry_id: str
    registry_version: str
    status: Literal["candidate", "release"]
    reference_benchmark: ArtifactReference
    benchmark_task_subset: tuple[str, ...]
    minimum_episodes_per_task: int = Field(ge=50)
    required_learned_policy_families: int = Field(ge=2)
    tracks: tuple[PolicyTrack, ...]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("registry_version")
    @classmethod
    def semver(cls, value: str) -> str:
        if not re.fullmatch(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)", value):
            raise ValueError("registry_version must use semantic versioning")
        return value

    @model_validator(mode="after")
    def valid_registry(self) -> "PolicyTrackRegistry":
        _identifier(self.registry_id, "registry_id")
        _unique(self.benchmark_task_subset, "benchmark task subset")
        track_ids = [track.track_id for track in self.tracks]
        _unique(track_ids, "policy track IDs")
        if not any(track.role == "oracle_control" for track in self.tracks):
            raise ValueError("policy registry requires an oracle control")
        if not any(track.role == "sanity_control" for track in self.tracks):
            raise ValueError("policy registry requires a sanity control")
        for track in self.tracks:
            if track.evaluation_task_ids != self.benchmark_task_subset:
                raise ValueError(
                    "all policy tracks must use the same task subset and order"
                )
        designs = {
            (
                track.evaluation_split_id,
                track.evaluation_split_sha256,
                track.evaluation_seeds,
                track.evaluation_asset_ids,
                track.stressor_condition_id,
            )
            for track in self.tracks
        }
        if len(designs) != 1:
            raise ValueError("all policy tracks must use the same evaluation design")
        learned = [track for track in self.tracks if track.role in {"learned", "vla"}]
        if self.status == "release":
            families = {
                track.contract.policy_family
                for track in learned
                if track.status == "validated" and track.required_for_release
            }
            if len(families) < self.required_learned_policy_families:
                raise ValueError(
                    "release registry lacks distinct validated learned families"
                )
            if not any(
                track.role == "oracle_control" and track.status == "validated"
                for track in self.tracks
            ):
                raise ValueError("release registry lacks a validated oracle control")
        _finite_json(self.metadata)
        return self

    @property
    def sha256(self) -> str:
        encoded = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


def _identifier(value: str, label: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise ValueError(f"{label} must be a portable identifier")


def _unique(values: Any, label: str) -> None:
    if not values or len(values) != len(set(values)):
        raise ValueError(f"{label} must be non-empty and unique")


def _finite_json(value: Any) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("metadata must contain finite JSON data") from exc
