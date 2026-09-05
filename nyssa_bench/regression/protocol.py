from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import PurePosixPath
from typing import Any, Literal, Mapping


REGRESSION_STUDY_FORMAT = "nyssa-policy-regression-study-v1"
POLICY_IDENTITY_FORMAT = "nyssa-regression-policy-identity-v1"
RUN_REFERENCE_FORMAT = "nyssa-regression-run-reference-v1"
REGRESSION_CELL_FORMAT = "nyssa-regression-cell-v1"
REGRESSION_RULE_FORMAT = "nyssa-regression-rule-v1"
BOUNDARY_REFERENCE_FORMAT = "nyssa-regression-boundary-reference-v1"
EVIDENCE_REQUIREMENTS_FORMAT = "nyssa-regression-evidence-requirements-v1"

ConditionKind = Literal["clean", "shifted", "confirmed_boundary"]
RuleSource = Literal[
    "paired_success",
    "episode_metric",
    "metric_vector",
    "failure_category_rate",
    "failure_onset_steps",
    "failure_duration_steps",
]
RuleKind = Literal["non_inferiority", "safety_block"]
RuleDirection = Literal["higher", "lower"]
ArtifactBinding = Literal["pinned", "observe_and_record"]

REQUIRED_RUN_ARTIFACTS = frozenset(
    {"run.yaml", "dataset_manifest.json", "metrics.json", "episodes.json"}
)


@dataclass(frozen=True)
class PolicyCheckpointIdentity:
    policy_name: str
    checkpoint_id: str
    checkpoint_sha256: str
    preprocessing_sha256: str

    def __post_init__(self) -> None:
        _require_text(self.policy_name, "policy_name")
        _require_text(self.checkpoint_id, "checkpoint_id")
        _require_sha256(self.checkpoint_sha256, "checkpoint_sha256")
        _require_sha256(self.preprocessing_sha256, "preprocessing_sha256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": POLICY_IDENTITY_FORMAT,
            "policy_name": self.policy_name,
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "preprocessing_sha256": self.preprocessing_sha256,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PolicyCheckpointIdentity":
        _check_format(data, POLICY_IDENTITY_FORMAT, "policy identity")
        _reject_unknown(
            data,
            {
                "format",
                "policy_name",
                "checkpoint_id",
                "checkpoint_sha256",
                "preprocessing_sha256",
            },
            "policy identity",
        )
        return cls(
            policy_name=str(data.get("policy_name", "")),
            checkpoint_id=str(data.get("checkpoint_id", "")),
            checkpoint_sha256=str(data.get("checkpoint_sha256", "")),
            preprocessing_sha256=str(data.get("preprocessing_sha256", "")),
        )


@dataclass(frozen=True, order=True)
class RegressionEpisodeKey:
    task_id: str
    seed: int
    episode_index: int

    def __post_init__(self) -> None:
        _require_text(self.task_id, "episode task_id")
        if self.seed < 0 or self.episode_index < 0:
            raise ValueError("regression episode seed and index must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "seed": self.seed,
            "episode_index": self.episode_index,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RegressionEpisodeKey":
        _reject_unknown(data, {"task_id", "seed", "episode_index"}, "episode key")
        return cls(
            task_id=str(data.get("task_id", "")),
            seed=_integer(data.get("seed"), "episode seed"),
            episode_index=_integer(data.get("episode_index"), "episode index"),
        )


@dataclass(frozen=True)
class RunArtifactReference:
    run_dir: str
    run_id: str
    artifact_binding: ArtifactBinding
    artifacts_sha256: dict[str, str]

    def __post_init__(self) -> None:
        _require_text(self.run_dir, "run_dir")
        _require_text(self.run_id, "run_id")
        if self.artifact_binding not in {"pinned", "observe_and_record"}:
            raise ValueError(f"unsupported run artifact binding: {self.artifact_binding}")
        missing = sorted(REQUIRED_RUN_ARTIFACTS - set(self.artifacts_sha256))
        if self.artifact_binding == "pinned" and missing:
            raise ValueError(
                "run reference is missing artifact hashes: " + ", ".join(missing)
            )
        if self.artifact_binding == "observe_and_record" and self.artifacts_sha256:
            raise ValueError(
                "observe_and_record run references cannot contain post-run hashes"
            )
        for name, digest in self.artifacts_sha256.items():
            normalized = name.replace("\\", "/")
            path = PurePosixPath(normalized)
            if (
                not normalized
                or normalized != name
                or path.is_absolute()
                or ".." in path.parts
                or ":" in normalized
            ):
                raise ValueError(f"run artifact path must stay within the pack: {name}")
            _require_sha256(digest, f"artifact hash for {name}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": RUN_REFERENCE_FORMAT,
            "run_dir": self.run_dir,
            "run_id": self.run_id,
            "artifact_binding": self.artifact_binding,
            "artifacts_sha256": {
                key: self.artifacts_sha256[key]
                for key in sorted(self.artifacts_sha256)
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RunArtifactReference":
        _check_format(data, RUN_REFERENCE_FORMAT, "run reference")
        _reject_unknown(
            data,
            {
                "format",
                "run_dir",
                "run_id",
                "artifact_binding",
                "artifacts_sha256",
            },
            "run reference",
        )
        hashes = _mapping(data.get("artifacts_sha256"), "run artifact hashes")
        if not all(isinstance(value, str) for value in hashes.values()):
            raise ValueError("run artifact hashes must be strings")
        return cls(
            run_dir=str(data.get("run_dir", "")),
            run_id=str(data.get("run_id", "")),
            artifact_binding=str(data.get("artifact_binding", "")),  # type: ignore[arg-type]
            artifacts_sha256={str(key): str(value) for key, value in hashes.items()},
        )


@dataclass(frozen=True)
class ConfirmedBoundaryReference:
    study_path: str
    artifact_sha256: str
    point: dict[str, Any]

    def __post_init__(self) -> None:
        _require_text(self.study_path, "boundary study_path")
        _require_sha256(self.artifact_sha256, "boundary artifact_sha256")
        if not self.point:
            raise ValueError("boundary reference point must be non-empty")
        _canonical_json(self.point)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": BOUNDARY_REFERENCE_FORMAT,
            "study_path": self.study_path,
            "artifact_sha256": self.artifact_sha256,
            "point": self.point,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ConfirmedBoundaryReference":
        _check_format(data, BOUNDARY_REFERENCE_FORMAT, "boundary reference")
        _reject_unknown(
            data,
            {"format", "study_path", "artifact_sha256", "point"},
            "boundary reference",
        )
        return cls(
            study_path=str(data.get("study_path", "")),
            artifact_sha256=str(data.get("artifact_sha256", "")),
            point=dict(_mapping(data.get("point"), "boundary point")),
        )


@dataclass(frozen=True)
class RegressionCellSpec:
    cell_id: str
    condition_kind: ConditionKind
    condition_id: str
    severity_levels: dict[str, float]
    comparison_contract_sha256: str
    baseline_run: RunArtifactReference
    candidate_run: RunArtifactReference
    episode_keys: tuple[RegressionEpisodeKey, ...]
    boundary_references: tuple[ConfirmedBoundaryReference, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.cell_id, "regression cell_id")
        _require_text(self.condition_id, "regression condition_id")
        if self.condition_kind not in {"clean", "shifted", "confirmed_boundary"}:
            raise ValueError(f"unsupported regression condition kind: {self.condition_kind}")
        _require_sha256(
            self.comparison_contract_sha256, "comparison_contract_sha256"
        )
        if not self.episode_keys:
            raise ValueError("regression cells require pinned episode keys")
        if len(self.episode_keys) != len(set(self.episode_keys)):
            raise ValueError("regression cell episode keys must be unique")
        for stressor_id, severity in self.severity_levels.items():
            _require_text(stressor_id, "severity stressor_id")
            if not math.isfinite(severity) or severity < 0.0:
                raise ValueError("stressor severities must be finite and non-negative")
        if self.condition_kind == "clean" and any(
            severity > 0.0 for severity in self.severity_levels.values()
        ):
            raise ValueError("clean regression cells cannot declare positive severity")
        if self.condition_kind in {"shifted", "confirmed_boundary"} and not any(
            severity > 0.0 for severity in self.severity_levels.values()
        ):
            raise ValueError("shifted regression cells require positive severity")
        if self.condition_kind == "confirmed_boundary" and not self.boundary_references:
            raise ValueError("confirmed-boundary cells require boundary provenance")
        if self.condition_kind != "confirmed_boundary" and self.boundary_references:
            raise ValueError("boundary provenance is only valid for confirmed-boundary cells")
        if self.baseline_run.artifact_binding != "pinned":
            raise ValueError("baseline regression runs must be content-pinned")

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": REGRESSION_CELL_FORMAT,
            "cell_id": self.cell_id,
            "condition_kind": self.condition_kind,
            "condition_id": self.condition_id,
            "severity_levels": {
                key: self.severity_levels[key] for key in sorted(self.severity_levels)
            },
            "comparison_contract_sha256": self.comparison_contract_sha256,
            "baseline_run": self.baseline_run.to_dict(),
            "candidate_run": self.candidate_run.to_dict(),
            "episode_keys": [key.to_dict() for key in self.episode_keys],
            "boundary_references": [
                reference.to_dict() for reference in self.boundary_references
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RegressionCellSpec":
        _check_format(data, REGRESSION_CELL_FORMAT, "regression cell")
        _reject_unknown(
            data,
            {
                "format",
                "cell_id",
                "condition_kind",
                "condition_id",
                "severity_levels",
                "comparison_contract_sha256",
                "baseline_run",
                "candidate_run",
                "episode_keys",
                "boundary_references",
            },
            "regression cell",
        )
        severity = _mapping(data.get("severity_levels"), "severity levels")
        raw_keys = _mapping_list(data.get("episode_keys"), "episode keys")
        boundaries = _mapping_list(
            data.get("boundary_references", []), "boundary references"
        )
        return cls(
            cell_id=str(data.get("cell_id", "")),
            condition_kind=str(data.get("condition_kind", "")),  # type: ignore[arg-type]
            condition_id=str(data.get("condition_id", "")),
            severity_levels={str(key): float(value) for key, value in severity.items()},
            comparison_contract_sha256=str(
                data.get("comparison_contract_sha256", "")
            ),
            baseline_run=RunArtifactReference.from_dict(
                _mapping(data.get("baseline_run"), "baseline run")
            ),
            candidate_run=RunArtifactReference.from_dict(
                _mapping(data.get("candidate_run"), "candidate run")
            ),
            episode_keys=tuple(RegressionEpisodeKey.from_dict(item) for item in raw_keys),
            boundary_references=tuple(
                ConfirmedBoundaryReference.from_dict(item) for item in boundaries
            ),
        )


@dataclass(frozen=True)
class RegressionEvidenceRequirements:
    minimum_pair_coverage: float = 1.0
    require_failure_ledger: bool = True
    require_detector_evidence: bool = True
    require_replays: bool = False
    require_run_validity: bool = True
    require_benchmark_validity: bool = True
    required_metric_vector: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not math.isfinite(self.minimum_pair_coverage) or not (
            0.0 < self.minimum_pair_coverage <= 1.0
        ):
            raise ValueError("minimum_pair_coverage must be within (0, 1]")
        for value, label in (
            (self.require_failure_ledger, "require_failure_ledger"),
            (self.require_detector_evidence, "require_detector_evidence"),
            (self.require_replays, "require_replays"),
            (self.require_run_validity, "require_run_validity"),
            (self.require_benchmark_validity, "require_benchmark_validity"),
        ):
            if not isinstance(value, bool):
                raise ValueError(f"{label} must be a boolean")
        _validate_unique_text(self.required_metric_vector, "required metric vector")

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": EVIDENCE_REQUIREMENTS_FORMAT,
            "minimum_pair_coverage": self.minimum_pair_coverage,
            "require_failure_ledger": self.require_failure_ledger,
            "require_detector_evidence": self.require_detector_evidence,
            "require_replays": self.require_replays,
            "require_run_validity": self.require_run_validity,
            "require_benchmark_validity": self.require_benchmark_validity,
            "required_metric_vector": list(self.required_metric_vector),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RegressionEvidenceRequirements":
        _check_format(data, EVIDENCE_REQUIREMENTS_FORMAT, "evidence requirements")
        _reject_unknown(
            data,
            {
                "format",
                "minimum_pair_coverage",
                "require_failure_ledger",
                "require_detector_evidence",
                "require_replays",
                "require_run_validity",
                "require_benchmark_validity",
                "required_metric_vector",
            },
            "evidence requirements",
        )
        metrics = data.get("required_metric_vector", [])
        if not isinstance(metrics, list) or not all(
            isinstance(item, str) for item in metrics
        ):
            raise ValueError("required_metric_vector must be a list of strings")
        return cls(
            minimum_pair_coverage=float(data.get("minimum_pair_coverage", 1.0)),
            require_failure_ledger=_boolean(
                data.get("require_failure_ledger"), "require_failure_ledger"
            ),
            require_detector_evidence=_boolean(
                data.get("require_detector_evidence"), "require_detector_evidence"
            ),
            require_replays=_boolean(
                data.get("require_replays"), "require_replays"
            ),
            require_run_validity=_boolean(
                data.get("require_run_validity"), "require_run_validity"
            ),
            require_benchmark_validity=_boolean(
                data.get("require_benchmark_validity"),
                "require_benchmark_validity",
            ),
            required_metric_vector=tuple(metrics),
        )


@dataclass(frozen=True)
class RegressionRule:
    rule_id: str
    source: RuleSource
    metric_id: str
    cell_ids: tuple[str, ...]
    kind: RuleKind
    direction: RuleDirection
    non_inferiority_margin: float = 0.0
    minimum_pairs: int = 2
    candidate_limit: float | None = None

    def __post_init__(self) -> None:
        _require_text(self.rule_id, "regression rule_id")
        _require_text(self.metric_id, "regression metric_id")
        if self.source not in {
            "paired_success",
            "episode_metric",
            "metric_vector",
            "failure_category_rate",
            "failure_onset_steps",
            "failure_duration_steps",
        }:
            raise ValueError(f"unsupported regression rule source: {self.source}")
        if self.kind not in {"non_inferiority", "safety_block"}:
            raise ValueError(f"unsupported regression rule kind: {self.kind}")
        if self.direction not in {"higher", "lower"}:
            raise ValueError(f"unsupported regression rule direction: {self.direction}")
        _validate_unique_text(self.cell_ids, "regression rule cell IDs")
        expected_metric_id = {
            "paired_success": "success",
            "failure_onset_steps": "failure_onset_steps",
            "failure_duration_steps": "failure_duration_steps",
        }.get(self.source)
        if expected_metric_id is not None and self.metric_id != expected_metric_id:
            raise ValueError(
                f"{self.source} rules must use metric_id '{expected_metric_id}'"
            )
        if not math.isfinite(self.non_inferiority_margin) or self.non_inferiority_margin < 0:
            raise ValueError("non-inferiority margin must be finite and non-negative")
        if self.minimum_pairs < 2:
            raise ValueError("regression rules require at least two independent pairs")
        if self.kind == "safety_block":
            if self.direction != "lower":
                raise ValueError("safety-block rules require lower-is-better metrics")
            if self.candidate_limit is None or not math.isfinite(self.candidate_limit):
                raise ValueError("safety-block rules require a finite candidate_limit")
        elif self.candidate_limit is not None:
            raise ValueError("candidate_limit is only valid for safety-block rules")

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": REGRESSION_RULE_FORMAT,
            "rule_id": self.rule_id,
            "source": self.source,
            "metric_id": self.metric_id,
            "cell_ids": list(self.cell_ids),
            "kind": self.kind,
            "direction": self.direction,
            "non_inferiority_margin": self.non_inferiority_margin,
            "minimum_pairs": self.minimum_pairs,
            "candidate_limit": self.candidate_limit,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RegressionRule":
        _check_format(data, REGRESSION_RULE_FORMAT, "regression rule")
        _reject_unknown(
            data,
            {
                "format",
                "rule_id",
                "source",
                "metric_id",
                "cell_ids",
                "kind",
                "direction",
                "non_inferiority_margin",
                "minimum_pairs",
                "candidate_limit",
            },
            "regression rule",
        )
        cell_ids = data.get("cell_ids")
        if not isinstance(cell_ids, list) or not all(
            isinstance(item, str) for item in cell_ids
        ):
            raise ValueError("regression rule cell_ids must be a list of strings")
        return cls(
            rule_id=str(data.get("rule_id", "")),
            source=str(data.get("source", "")),  # type: ignore[arg-type]
            metric_id=str(data.get("metric_id", "")),
            cell_ids=tuple(cell_ids),
            kind=str(data.get("kind", "")),  # type: ignore[arg-type]
            direction=str(data.get("direction", "")),  # type: ignore[arg-type]
            non_inferiority_margin=float(data.get("non_inferiority_margin", 0.0)),
            minimum_pairs=_integer(data.get("minimum_pairs", 2), "minimum_pairs"),
            candidate_limit=float(data["candidate_limit"])
            if data.get("candidate_limit") is not None
            else None,
        )


@dataclass(frozen=True)
class RegressionStudySpec:
    study_id: str
    study_version: str
    baseline_policy: PolicyCheckpointIdentity
    candidate_policy: PolicyCheckpointIdentity
    cells: tuple[RegressionCellSpec, ...]
    rules: tuple[RegressionRule, ...]
    evidence_requirements: RegressionEvidenceRequirements
    prespecified_at: str
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        _require_text(self.study_id, "regression study_id")
        _require_text(self.prespecified_at, "prespecified_at")
        try:
            prespecified = datetime.fromisoformat(
                self.prespecified_at.replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise ValueError("prespecified_at must be an ISO-8601 timestamp") from exc
        if prespecified.tzinfo is None:
            raise ValueError("prespecified_at must include a timezone")
        if not re.fullmatch(
            r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:[-+][0-9A-Za-z.-]+)?",
            self.study_version,
        ):
            raise ValueError("study_version must use semantic versioning")
        if self.baseline_policy == self.candidate_policy:
            raise ValueError("baseline and candidate policy identities must differ")
        if not self.cells or not self.rules:
            raise ValueError("regression studies require cells and decision rules")
        cell_ids = [cell.cell_id for cell in self.cells]
        rule_ids = [rule.rule_id for rule in self.rules]
        if len(cell_ids) != len(set(cell_ids)):
            raise ValueError("regression cell IDs must be unique")
        run_pairs = [
            (
                cell.baseline_run.run_id,
                cell.candidate_run.run_id,
                cell.comparison_contract_sha256,
            )
            for cell in self.cells
        ]
        if len(run_pairs) != len(set(run_pairs)):
            raise ValueError("regression cells cannot duplicate a run pair")
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError("regression rule IDs must be unique")
        known_cells = set(cell_ids)
        for rule in self.rules:
            missing = sorted(set(rule.cell_ids) - known_cells)
            if missing:
                raise ValueError(
                    f"regression rule '{rule.rule_id}' references unknown cells: "
                    + ", ".join(missing)
                )
        if self.schema_version != 1:
            raise ValueError(f"unsupported regression study version: {self.schema_version}")
        _canonical_json(self.metadata)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict()).encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": REGRESSION_STUDY_FORMAT,
            "schema_version": self.schema_version,
            "study_id": self.study_id,
            "study_version": self.study_version,
            "prespecified_at": self.prespecified_at,
            "baseline_policy": self.baseline_policy.to_dict(),
            "candidate_policy": self.candidate_policy.to_dict(),
            "cells": [cell.to_dict() for cell in self.cells],
            "rules": [rule.to_dict() for rule in self.rules],
            "evidence_requirements": self.evidence_requirements.to_dict(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RegressionStudySpec":
        _check_format(data, REGRESSION_STUDY_FORMAT, "regression study")
        _reject_unknown(
            data,
            {
                "format",
                "schema_version",
                "study_id",
                "study_version",
                "prespecified_at",
                "baseline_policy",
                "candidate_policy",
                "cells",
                "rules",
                "evidence_requirements",
                "metadata",
            },
            "regression study",
        )
        cells = _mapping_list(data.get("cells"), "regression cells")
        rules = _mapping_list(data.get("rules"), "regression rules")
        metadata = _mapping(data.get("metadata", {}), "regression metadata")
        return cls(
            study_id=str(data.get("study_id", "")),
            study_version=str(data.get("study_version", "")),
            prespecified_at=str(data.get("prespecified_at", "")),
            baseline_policy=PolicyCheckpointIdentity.from_dict(
                _mapping(data.get("baseline_policy"), "baseline policy")
            ),
            candidate_policy=PolicyCheckpointIdentity.from_dict(
                _mapping(data.get("candidate_policy"), "candidate policy")
            ),
            cells=tuple(RegressionCellSpec.from_dict(item) for item in cells),
            rules=tuple(RegressionRule.from_dict(item) for item in rules),
            evidence_requirements=RegressionEvidenceRequirements.from_dict(
                _mapping(data.get("evidence_requirements"), "evidence requirements")
            ),
            metadata=dict(metadata),
            schema_version=_integer(data.get("schema_version", 1), "schema_version"),
        )


def _check_format(data: Mapping[str, Any], expected: str, label: str) -> None:
    if data.get("format") != expected:
        raise ValueError(f"unsupported {label} format: {data.get('format')}")


def _reject_unknown(data: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"unknown {label} fields: {', '.join(unknown)}")


def _require_text(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty")


def _require_sha256(value: Any, label: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        char not in "0123456789abcdef" for char in value
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _mapping_list(value: Any, label: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or not all(
        isinstance(item, Mapping) for item in value
    ):
        raise ValueError(f"{label} must be a list of mappings")
    return value


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{label} must be an integer")
    return result


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def _validate_unique_text(values: tuple[str, ...], label: str) -> None:
    if not values:
        raise ValueError(f"{label} must be non-empty")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{label} must contain non-empty strings")
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
