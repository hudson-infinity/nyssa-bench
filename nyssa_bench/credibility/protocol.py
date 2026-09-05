from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


CREDIBILITY_SPEC_FORMAT = "nyssa-phase1-credibility-spec-v1"
CREDIBILITY_EVIDENCE_FORMAT = "nyssa-credibility-evidence-v1"

EvidenceCategory = Literal[
    "reference_benchmark",
    "learned_policy_track",
    "paired_clean_shifted",
    "benchmark_validity",
    "simulator_ci",
    "hardware_calibration",
    "sim_real_predictive_result",
]
GateId = Literal["A", "B", "C"]
CheckStatus = Literal["passed", "failed", "missing", "not_applicable"]


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _safe_relative_path(value: str, label: str) -> str:
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"{label} must be relative and cannot escape its root")
    return path.as_posix()


def _sha256(value: str, label: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{label} must be a lowercase digest")
    return value


def _identifier(value: str, label: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise ValueError(f"{label} must be a non-empty portable identifier")
    return value


class GateCheckDefinition(ContractModel):
    check_id: str
    description: str
    issue_ids: tuple[int, ...]
    evidence_categories: tuple[EvidenceCategory, ...] = ()
    alternative_group: str | None = None

    @field_validator("check_id")
    @classmethod
    def check_identifier(cls, value: str) -> str:
        return _identifier(value, "check_id")

    @model_validator(mode="after")
    def valid_definition(self) -> "GateCheckDefinition":
        if not self.description.strip():
            raise ValueError("gate check description must be non-empty")
        if not self.issue_ids or len(self.issue_ids) != len(set(self.issue_ids)):
            raise ValueError("gate check issue IDs must be non-empty and unique")
        if any(issue_id < 13 or issue_id > 23 for issue_id in self.issue_ids):
            raise ValueError("Phase 1 issue dependencies must be between #13 and #23")
        if self.alternative_group is not None:
            _identifier(self.alternative_group, "alternative_group")
        return self


class GateDefinition(ContractModel):
    gate_id: GateId
    name: str
    required_checks: tuple[GateCheckDefinition, ...]

    @model_validator(mode="after")
    def valid_gate(self) -> "GateDefinition":
        if not self.name.strip() or not self.required_checks:
            raise ValueError("gate name and checks must be non-empty")
        check_ids = [item.check_id for item in self.required_checks]
        if len(check_ids) != len(set(check_ids)):
            raise ValueError("gate check IDs must be unique")
        return self


class SplitLineage(ContractModel):
    split_manifest_sha256: str
    evaluation_partition: Literal["hidden_test"]
    protected: Literal[True]

    @field_validator("split_manifest_sha256")
    @classmethod
    def hash_value(cls, value: str) -> str:
        return _sha256(value, "split_manifest_sha256")


class ReferenceBenchmarkManifest(ContractModel):
    format: Literal["nyssa-reference-benchmark-manifest-v1"] = (
        "nyssa-reference-benchmark-manifest-v1"
    )
    benchmark_id: str
    benchmark_version: str
    task_ids: tuple[str, ...]
    split_lineage: SplitLineage
    oracle_control_policy_ids: tuple[str, ...]

    @field_validator("benchmark_version")
    @classmethod
    def semver(cls, value: str) -> str:
        if not re.fullmatch(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)", value):
            raise ValueError("benchmark_version must use semantic versioning")
        return value

    @model_validator(mode="after")
    def valid_reference(self) -> "ReferenceBenchmarkManifest":
        _identifier(self.benchmark_id, "benchmark_id")
        if not 1 <= len(self.task_ids) <= 20:
            raise ValueError("reference benchmark must contain 1 to 20 tasks")
        if len(self.task_ids) != len(set(self.task_ids)):
            raise ValueError("reference benchmark task IDs must be unique")
        for task_id in self.task_ids:
            _identifier(task_id, "task_id")
        if not self.oracle_control_policy_ids:
            raise ValueError("reference benchmark requires an oracle control")
        if len(self.oracle_control_policy_ids) != len(
            set(self.oracle_control_policy_ids)
        ):
            raise ValueError("oracle control policy IDs must be unique")
        for policy_id in self.oracle_control_policy_ids:
            _identifier(policy_id, "oracle_control_policy_id")
        return self


class EvidenceReference(ContractModel):
    evidence_id: str
    category: EvidenceCategory
    path: str
    sha256: str

    @field_validator("sha256")
    @classmethod
    def hash_value(cls, value: str) -> str:
        return _sha256(value, "sha256")

    @field_validator("path")
    @classmethod
    def safe_path(cls, value: str) -> str:
        return _safe_relative_path(value, "evidence path")

    @field_validator("evidence_id")
    @classmethod
    def valid_id(cls, value: str) -> str:
        return _identifier(value, "evidence_id")


class CredibilitySpec(ContractModel):
    format: Literal["nyssa-phase1-credibility-spec-v1"] = (
        "nyssa-phase1-credibility-spec-v1"
    )
    schema_version: Literal[1] = 1
    program_id: str
    program_version: str
    claim_matrix_path: str
    claim_matrix_sha256: str
    evidence: tuple[EvidenceReference, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("program_id")
    @classmethod
    def valid_program_id(cls, value: str) -> str:
        return _identifier(value, "program_id")

    @field_validator("claim_matrix_path")
    @classmethod
    def safe_claim_path(cls, value: str) -> str:
        return _safe_relative_path(value, "claim_matrix_path")

    @field_validator("program_version")
    @classmethod
    def semver(cls, value: str) -> str:
        if not re.fullmatch(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)", value):
            raise ValueError("program_version must use semantic versioning")
        return value

    @field_validator("claim_matrix_sha256")
    @classmethod
    def claim_hash(cls, value: str) -> str:
        return _sha256(value, "claim_matrix_sha256")

    @model_validator(mode="after")
    def unique_evidence(self) -> "CredibilitySpec":
        ids = [item.evidence_id for item in self.evidence]
        if len(ids) != len(set(ids)):
            raise ValueError("credibility evidence IDs must be unique")
        return self


class EvidenceArtifact(ContractModel):
    path: str
    sha256: str
    media_type: Literal["application/json"] = "application/json"

    @field_validator("path")
    @classmethod
    def safe_path(cls, value: str) -> str:
        path = _safe_relative_path(value, "evidence artifact path")
        if PurePosixPath(path).suffix.lower() != ".json":
            raise ValueError("credibility evidence artifacts must be JSON files")
        return path

    @field_validator("sha256")
    @classmethod
    def hash_value(cls, value: str) -> str:
        return _sha256(value, "artifact sha256")


class CredibilityEvidence(ContractModel):
    format: Literal["nyssa-credibility-evidence-v1"] = "nyssa-credibility-evidence-v1"
    evidence_id: str
    category: EvidenceCategory
    status: Literal["validated"]
    artifacts: tuple[EvidenceArtifact, ...]
    metadata: dict[str, Any]

    @field_validator("evidence_id")
    @classmethod
    def valid_id(cls, value: str) -> str:
        return _identifier(value, "evidence_id")

    @model_validator(mode="after")
    def nonempty_artifacts(self) -> "CredibilityEvidence":
        if not self.artifacts:
            raise ValueError("credibility evidence requires artifacts")
        paths = [item.path for item in self.artifacts]
        if len(paths) != len(set(paths)):
            raise ValueError("credibility artifact paths must be unique")
        return self
