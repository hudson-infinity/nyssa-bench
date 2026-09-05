from __future__ import annotations

import hashlib
import json
import re
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nyssa_bench.nep import TaskContract


REFERENCE_SPEC_FORMAT = "nyssa-reference-benchmark-spec-v1"
REFERENCE_EVIDENCE_FORMAT = "nyssa-reference-solvability-v1"
SplitPartition = Literal["train", "validation", "public_test", "hidden_test"]
SplitDimension = Literal[
    "assets", "initial_states", "poses", "task_variants", "demonstrations"
]
Mechanism = Literal[
    "grasp_place",
    "nonprehensile",
    "stacking",
    "contact_insertion",
    "articulated",
    "clutter_distractors",
    "multi_stage",
]
REQUIRED_DIMENSIONS = {
    "assets",
    "initial_states",
    "poses",
    "task_variants",
    "demonstrations",
}
REQUIRED_MECHANISMS = {
    "grasp_place",
    "nonprehensile",
    "stacking",
    "contact_insertion",
    "articulated",
    "clutter_distractors",
    "multi_stage",
}


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ArtifactReference(ContractModel):
    path: str
    sha256: str

    @field_validator("path")
    @classmethod
    def safe_path(cls, value: str) -> str:
        path = PurePosixPath(value.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError("artifact path must be relative and confined")
        return path.as_posix()

    @field_validator("sha256")
    @classmethod
    def digest(cls, value: str) -> str:
        return _sha256(value)


class SplitDimensionCommitment(ContractModel):
    dimension: SplitDimension
    content_sha256: str
    item_count: int = Field(gt=0)
    status: Literal["pending", "committed"]
    public_artifact: ArtifactReference | None = None

    @field_validator("content_sha256")
    @classmethod
    def digest(cls, value: str) -> str:
        return _sha256(value)

    @model_validator(mode="after")
    def artifact_matches_commitment(self) -> "SplitDimensionCommitment":
        if (
            self.public_artifact is not None
            and self.public_artifact.sha256 != self.content_sha256
        ):
            raise ValueError("public artifact hash must equal its content commitment")
        return self


class BenchmarkSplit(ContractModel):
    split_id: str
    partition: SplitPartition
    parent_split_ids: tuple[str, ...] = ()
    producer_id: str
    evaluator_id: str
    protected: bool
    contents_published: bool
    contamination_status: Literal["unknown", "clean", "contaminated"]
    dimensions: tuple[SplitDimensionCommitment, ...]

    @model_validator(mode="after")
    def valid_split(self) -> "BenchmarkSplit":
        _identifier(self.split_id, "split_id")
        if not self.producer_id.strip() or not self.evaluator_id.strip():
            raise ValueError("split producer and evaluator IDs must be non-empty")
        dimension_ids = [item.dimension for item in self.dimensions]
        if set(dimension_ids) != REQUIRED_DIMENSIONS or len(dimension_ids) != len(
            REQUIRED_DIMENSIONS
        ):
            raise ValueError("every split must commit all five split dimensions")
        if self.partition == "hidden_test":
            if not self.protected or self.contents_published:
                raise ValueError(
                    "hidden-test contents must be protected and unpublished"
                )
            if self.producer_id == self.evaluator_id:
                raise ValueError(
                    "hidden-test producer and evaluator must be independent"
                )
            if any(item.public_artifact is not None for item in self.dimensions):
                raise ValueError("hidden-test commitments cannot expose artifact paths")
        elif self.protected:
            raise ValueError("only hidden-test splits may be protected")
        elif any(
            item.status == "committed" and item.public_artifact is None
            for item in self.dimensions
        ):
            raise ValueError("committed public split dimensions require artifacts")
        return self


class ReferenceTask(ContractModel):
    contract: TaskContract
    task_spec: ArtifactReference
    mechanisms: tuple[Mechanism, ...]
    supported_stressors: tuple[str, ...]
    failure_capabilities: tuple[str, ...]
    asset_provenance_status: Literal["pending", "verified"]
    success_predicate_status: Literal["pending", "verified"]
    solvability_evidence: ArtifactReference | None = None

    @model_validator(mode="after")
    def valid_task(self) -> "ReferenceTask":
        if not self.mechanisms:
            raise ValueError("reference task requires a failure mechanism category")
        _unique(self.mechanisms, "task mechanisms")
        _unique(self.supported_stressors, "task stressors")
        _unique(self.failure_capabilities, "failure capabilities")
        if len(self.contract.engine_ids) != 1:
            raise ValueError("reference tasks must bind exactly one simulator engine")
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (*self.supported_stressors, *self.failure_capabilities)
        ):
            raise ValueError("task capability identifiers must be non-empty strings")
        if not self.supported_stressors or not self.failure_capabilities:
            raise ValueError(
                "reference task requires stressor and failure capabilities"
            )
        return self


class ExperimentalDesign(ContractModel):
    paired_seeds: Literal[True]
    minimum_episodes_per_condition: int = Field(ge=50)
    target_success_ci95_width: float = Field(gt=0.0, le=0.25)
    bootstrap_samples: int = Field(ge=1000)
    minimum_oracle_success_rate: float = Field(gt=0.0, le=1.0)
    required_learned_policy_families: int = Field(ge=2)
    required_controls: tuple[str, ...]
    primary_metrics: tuple[str, ...]
    rationale: str

    @model_validator(mode="after")
    def complete_design(self) -> "ExperimentalDesign":
        _unique(self.required_controls, "required controls")
        _unique(self.primary_metrics, "primary metrics")
        if "oracle" not in self.required_controls:
            raise ValueError("reference design requires an oracle control")
        if not self.primary_metrics or not self.rationale.strip():
            raise ValueError("reference design requires metrics and a power rationale")
        return self


class ReferenceBenchmarkSpec(ContractModel):
    format: Literal["nyssa-reference-benchmark-spec-v1"] = (
        "nyssa-reference-benchmark-spec-v1"
    )
    schema_version: Literal[1] = 1
    benchmark_id: str
    benchmark_version: str
    status: Literal["candidate", "release"]
    tasks: tuple[ReferenceTask, ...]
    splits: tuple[BenchmarkSplit, ...]
    experimental_design: ExperimentalDesign
    learned_policy_evidence: tuple[ArtifactReference, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("benchmark_version")
    @classmethod
    def semver(cls, value: str) -> str:
        if not re.fullmatch(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)", value):
            raise ValueError("benchmark_version must use semantic versioning")
        return value

    @model_validator(mode="after")
    def complete_benchmark(self) -> "ReferenceBenchmarkSpec":
        _identifier(self.benchmark_id, "benchmark_id")
        if not 12 <= len(self.tasks) <= 20:
            raise ValueError("reference benchmark must contain 12 to 20 tasks")
        task_ids = [item.contract.task_id for item in self.tasks]
        _unique(task_ids, "reference task IDs")
        mechanisms = {mechanism for task in self.tasks for mechanism in task.mechanisms}
        if mechanisms != REQUIRED_MECHANISMS:
            missing = sorted(REQUIRED_MECHANISMS - mechanisms)
            extra = sorted(mechanisms - REQUIRED_MECHANISMS)
            raise ValueError(
                f"mechanism coverage mismatch; missing={missing}, extra={extra}"
            )
        split_ids = [item.split_id for item in self.splits]
        _unique(split_ids, "split IDs")
        partitions = {item.partition for item in self.splits}
        if partitions != {"train", "validation", "public_test", "hidden_test"}:
            raise ValueError(
                "train, validation, public-test, and hidden-test splits are required"
            )
        known = set(split_ids)
        for split in self.splits:
            if not set(split.parent_split_ids) <= known:
                raise ValueError(f"split {split.split_id} has an unknown parent")
            if split.split_id in split.parent_split_ids:
                raise ValueError("split cannot be its own parent")
        for task in self.tasks:
            lineage = task.contract.split_lineage
            if lineage.split_id not in known:
                raise ValueError(
                    f"task {task.contract.task_id} references an unknown split"
                )
            if any(asset.split != lineage.partition for asset in task.contract.assets):
                raise ValueError("task asset and split-lineage partitions differ")
        _reject_split_cycles(self.splits)
        hashes: dict[str, str] = {}
        for split in self.splits:
            for item in split.dimensions:
                key = item.content_sha256
                if key in hashes:
                    raise ValueError(
                        f"split content collision: {hashes[key]} and {split.split_id}"
                    )
                hashes[key] = split.split_id
        if self.status == "release":
            if any(
                item.status != "committed"
                for split in self.splits
                for item in split.dimensions
            ):
                raise ValueError("release benchmark cannot contain pending commitments")
            if any(split.contamination_status != "clean" for split in self.splits):
                raise ValueError("release benchmark requires clean split audits")
            if any(task.solvability_evidence is None for task in self.tasks):
                raise ValueError(
                    "release benchmark requires per-task solvability evidence"
                )
            if any(task.asset_provenance_status != "verified" for task in self.tasks):
                raise ValueError("release benchmark requires verified asset provenance")
            if any(task.success_predicate_status != "verified" for task in self.tasks):
                raise ValueError(
                    "release benchmark requires verified success predicates"
                )
            if len(self.learned_policy_evidence) < (
                self.experimental_design.required_learned_policy_families
            ):
                raise ValueError("release benchmark lacks learned-policy evidence")
        _finite_json(self.metadata)
        return self

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            _canonical_json(self.model_dump(mode="json")).encode()
        ).hexdigest()


def _reject_split_cycles(splits: tuple[BenchmarkSplit, ...]) -> None:
    graph = {split.split_id: set(split.parent_split_ids) for split in splits}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise ValueError("split lineage contains a cycle")
        if node in visited:
            return
        visiting.add(node)
        for parent in graph[node]:
            visit(parent)
        visiting.remove(node)
        visited.add(node)

    for split_id in graph:
        visit(split_id)


def _sha256(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("value must be a lowercase SHA-256 digest")
    return value


def _identifier(value: str, label: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise ValueError(f"{label} must be a portable identifier")


def _unique(values: tuple[Any, ...] | list[Any], label: str) -> None:
    if not values or len(values) != len(set(values)):
        raise ValueError(f"{label} must be non-empty and unique")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _finite_json(value: Any) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("metadata must contain finite JSON data") from exc
