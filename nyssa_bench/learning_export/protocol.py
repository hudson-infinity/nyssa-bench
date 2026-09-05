from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from pathlib import PurePosixPath
from dataclasses import dataclass
from typing import Any, Literal, Mapping

from nyssa_bench.failures.protocol import FailureLedgerRecord
from nyssa_bench.recovery.protocol import CounterfactualRecoveryRecord


LEARNING_EXPORT_MANIFEST_FORMAT = "nyssa-learning-evidence-manifest-v1"
LEARNING_EPISODE_FORMAT = "nyssa-learning-evidence-episode-v1"
LEARNING_STEP_FORMAT = "nyssa-learning-evidence-step-v1"
LEARNING_EXCLUSION_FORMAT = "nyssa-evaluation-exclusion-v1"
ARTIFACT_REFERENCE_FORMAT = "nyssa-content-addressed-artifact-v1"

PrivacyLevel = Literal["public", "restricted", "private"]


@dataclass(frozen=True)
class ArtifactReference:
    artifact_id: str
    uri: str
    sha256: str
    bytes: int
    media_type: str
    embedded: bool = False

    def __post_init__(self) -> None:
        if not self.artifact_id.strip() or not self.uri.strip() or not self.media_type.strip():
            raise ValueError("artifact identity, URI, and media type are required")
        _require_sha256(self.sha256, "artifact sha256")
        if self.bytes < 0:
            raise ValueError("artifact byte count must be non-negative")
        if self.embedded:
            path = PurePosixPath(self.uri)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("embedded artifact URI must stay inside the package")
        elif "://" not in self.uri:
            raise ValueError("external artifacts require a scheme-qualified URI")

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": ARTIFACT_REFERENCE_FORMAT,
            "artifact_id": self.artifact_id,
            "uri": self.uri,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "media_type": self.media_type,
            "embedded": self.embedded,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArtifactReference":
        _check_format(data, ARTIFACT_REFERENCE_FORMAT, "artifact reference")
        _reject_unknown(
            data,
            {"format", "artifact_id", "uri", "sha256", "bytes", "media_type", "embedded"},
            "artifact reference",
        )
        return cls(
            artifact_id=str(data.get("artifact_id", "")),
            uri=str(data.get("uri", "")),
            sha256=str(data.get("sha256", "")),
            bytes=int(data.get("bytes", -1)),
            media_type=str(data.get("media_type", "")),
            embedded=_boolean(data.get("embedded"), "artifact embedded"),
        )


@dataclass(frozen=True)
class EvaluationExclusion:
    exclusion_id: str
    source_benchmark_id: str
    source_suite_id: str
    source_split_id: str
    source_episode_id: str
    content_sha256: str
    excluded_from_evaluation: bool = True
    reason: str = "exported evaluation evidence must not re-enter held-out evaluation"

    def __post_init__(self) -> None:
        for value, label in (
            (self.exclusion_id, "exclusion_id"),
            (self.source_benchmark_id, "source_benchmark_id"),
            (self.source_suite_id, "source_suite_id"),
            (self.source_split_id, "source_split_id"),
            (self.source_episode_id, "source_episode_id"),
            (self.reason, "exclusion reason"),
        ):
            if not value.strip():
                raise ValueError(f"{label} must be non-empty")
        _require_sha256(self.content_sha256, "exclusion content_sha256")
        if not self.excluded_from_evaluation:
            raise ValueError("learning exports must remain excluded from evaluation")

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": LEARNING_EXCLUSION_FORMAT,
            "exclusion_id": self.exclusion_id,
            "source_benchmark_id": self.source_benchmark_id,
            "source_suite_id": self.source_suite_id,
            "source_split_id": self.source_split_id,
            "source_episode_id": self.source_episode_id,
            "content_sha256": self.content_sha256,
            "excluded_from_evaluation": self.excluded_from_evaluation,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvaluationExclusion":
        _check_format(data, LEARNING_EXCLUSION_FORMAT, "evaluation exclusion")
        _reject_unknown(
            data,
            {
                "format",
                "exclusion_id",
                "source_benchmark_id",
                "source_suite_id",
                "source_split_id",
                "source_episode_id",
                "content_sha256",
                "excluded_from_evaluation",
                "reason",
            },
            "evaluation exclusion",
        )
        return cls(
            exclusion_id=str(data.get("exclusion_id", "")),
            source_benchmark_id=str(data.get("source_benchmark_id", "")),
            source_suite_id=str(data.get("source_suite_id", "")),
            source_split_id=str(data.get("source_split_id", "")),
            source_episode_id=str(data.get("source_episode_id", "")),
            content_sha256=str(data.get("content_sha256", "")),
            excluded_from_evaluation=_boolean(
                data.get("excluded_from_evaluation"),
                "excluded_from_evaluation",
            ),
            reason=str(data.get("reason", "")),
        )


@dataclass(frozen=True)
class LearningStepRecord:
    step_index: int
    observation: Any
    proposed_action: Any
    proposed_action_source: str
    rejected_action: Any
    action_rejected: bool
    executed_action_before_stressors: Any
    executed_action: Any
    executed_action_source: str
    oracle_action: Any
    recovery_action: Any
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]

    def __post_init__(self) -> None:
        if self.step_index < 0:
            raise ValueError("learning step index must be non-negative")
        if not self.proposed_action_source.strip() or not self.executed_action_source.strip():
            raise ValueError("proposed and executed action sources are required")
        allowed_sources = {"policy", "recovery", "expert", "pending"}
        if self.proposed_action_source not in allowed_sources:
            raise ValueError(
                f"unsupported proposed action source: {self.proposed_action_source}"
            )
        if self.executed_action_source not in allowed_sources:
            raise ValueError(
                f"unsupported executed action source: {self.executed_action_source}"
            )
        if self.proposed_action is None or self.executed_action is None:
            raise ValueError("proposed and executed actions are required")
        if not math.isfinite(float(self.reward)):
            raise ValueError("learning step reward must be finite")
        if self.action_rejected and self.rejected_action is None:
            raise ValueError("rejected steps must retain the rejected action")
        if not self.action_rejected and self.rejected_action is not None:
            raise ValueError("accepted steps cannot contain a rejected action")
        if self.executed_action_source == "expert" and self.oracle_action is None:
            raise ValueError("expert steps must retain the oracle action")
        if self.executed_action_source == "recovery" and self.recovery_action is None:
            raise ValueError("recovery steps must retain the recovery action")
        if self.executed_action_source != "expert" and self.oracle_action is not None:
            raise ValueError("non-expert steps cannot contain an oracle action")
        if self.executed_action_source != "recovery" and self.recovery_action is not None:
            raise ValueError("non-recovery steps cannot contain a recovery action")
        _canonical_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": LEARNING_STEP_FORMAT,
            "step_index": self.step_index,
            "observation": _jsonable(self.observation),
            "proposed_action": _jsonable(self.proposed_action),
            "proposed_action_source": self.proposed_action_source,
            "rejected_action": _jsonable(self.rejected_action),
            "action_rejected": self.action_rejected,
            "executed_action_before_stressors": _jsonable(
                self.executed_action_before_stressors
            ),
            "executed_action": _jsonable(self.executed_action),
            "executed_action_source": self.executed_action_source,
            "oracle_action": _jsonable(self.oracle_action),
            "recovery_action": _jsonable(self.recovery_action),
            "reward": self.reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "info": _jsonable(self.info),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LearningStepRecord":
        _check_format(data, LEARNING_STEP_FORMAT, "learning step")
        allowed = {
            "format",
            "step_index",
            "observation",
            "proposed_action",
            "proposed_action_source",
            "rejected_action",
            "action_rejected",
            "executed_action_before_stressors",
            "executed_action",
            "executed_action_source",
            "oracle_action",
            "recovery_action",
            "reward",
            "terminated",
            "truncated",
            "info",
        }
        _reject_unknown(data, allowed, "learning step")
        info = data.get("info")
        if not isinstance(info, Mapping):
            raise ValueError("learning step info must be a mapping")
        return cls(
            step_index=int(data.get("step_index", -1)),
            observation=data.get("observation"),
            proposed_action=data.get("proposed_action"),
            proposed_action_source=str(data.get("proposed_action_source", "")),
            rejected_action=data.get("rejected_action"),
            action_rejected=_boolean(data.get("action_rejected"), "action_rejected"),
            executed_action_before_stressors=data.get("executed_action_before_stressors"),
            executed_action=data.get("executed_action"),
            executed_action_source=str(data.get("executed_action_source", "")),
            oracle_action=data.get("oracle_action"),
            recovery_action=data.get("recovery_action"),
            reward=float(data.get("reward", float("nan"))),
            terminated=_boolean(data.get("terminated"), "terminated"),
            truncated=_boolean(data.get("truncated"), "truncated"),
            info=dict(info),
        )


@dataclass(frozen=True)
class LearningEpisodeRecord:
    episode_id: str
    task_id: str
    policy_id: str
    policy_family: str
    engine_name: str
    episode_index: int
    seed: int
    success: bool
    failure_label: str | None
    failure_ledger: dict[str, Any] | None
    stressor_context: dict[str, Any]
    split_lineage: dict[str, Any]
    provenance: dict[str, Any]
    steps: tuple[LearningStepRecord, ...]
    counterfactual_recovery: tuple[dict[str, Any], ...]
    recovery_metrics: dict[str, Any]
    failure_cluster: dict[str, Any] | None
    boundary_context: dict[str, Any] | None
    artifacts: tuple[ArtifactReference, ...]
    exclusion: EvaluationExclusion

    def __post_init__(self) -> None:
        if not all(
            value.strip()
            for value in (
                self.episode_id,
                self.task_id,
                self.policy_id,
                self.policy_family,
                self.engine_name,
            )
        ):
            raise ValueError("learning episode identity fields must be non-empty")
        if self.episode_index < 0 or self.seed < 0:
            raise ValueError("episode index and seed must be non-negative")
        if not self.steps:
            raise ValueError("learning episode must contain step evidence")
        if self.success and self.failure_label is not None:
            raise ValueError("successful episodes cannot have a terminal failure label")
        if not self.success and (
            not self.failure_label or self.failure_ledger is None
        ):
            raise ValueError(
                "failed episodes require a failure label and temporal failure ledger"
            )
        if self.exclusion.source_episode_id != self.episode_id:
            raise ValueError("episode exclusion identity does not match")
        if [step.step_index for step in self.steps] != list(range(len(self.steps))):
            raise ValueError("learning episode steps must be contiguous")
        artifact_ids = [artifact.artifact_id for artifact in self.artifacts]
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("episode artifact IDs must be unique")
        if self.failure_ledger is not None:
            FailureLedgerRecord.from_dict(self.failure_ledger)
        for record in self.counterfactual_recovery:
            CounterfactualRecoveryRecord.from_dict(record)
        _validate_split_lineage(self.split_lineage)
        if self.exclusion.source_split_id != self.split_lineage.get("split_id"):
            raise ValueError("episode exclusion split does not match split lineage")
        if not self.provenance.get("source_run_id") or not self.provenance.get(
            "source_run_sha256"
        ):
            raise ValueError("episode provenance requires source run identity and hash")
        _require_sha256(
            str(self.provenance["source_run_sha256"]), "source_run_sha256"
        )
        if self.boundary_context is not None:
            _require_sha256(
                str(self.boundary_context.get("study_sha256")),
                "boundary study_sha256",
            )
        _canonical_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": LEARNING_EPISODE_FORMAT,
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "policy_id": self.policy_id,
            "policy_family": self.policy_family,
            "engine_name": self.engine_name,
            "episode_index": self.episode_index,
            "seed": self.seed,
            "success": self.success,
            "failure_label": self.failure_label,
            "failure_ledger": self.failure_ledger,
            "stressor_context": self.stressor_context,
            "split_lineage": self.split_lineage,
            "provenance": self.provenance,
            "steps": [step.to_dict() for step in self.steps],
            "counterfactual_recovery": list(self.counterfactual_recovery),
            "recovery_metrics": self.recovery_metrics,
            "failure_cluster": self.failure_cluster,
            "boundary_context": self.boundary_context,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "evaluation_exclusion": self.exclusion.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LearningEpisodeRecord":
        _check_format(data, LEARNING_EPISODE_FORMAT, "learning episode")
        allowed = {
            "format",
            "episode_id",
            "task_id",
            "policy_id",
            "policy_family",
            "engine_name",
            "episode_index",
            "seed",
            "success",
            "failure_label",
            "failure_ledger",
            "stressor_context",
            "split_lineage",
            "provenance",
            "steps",
            "counterfactual_recovery",
            "recovery_metrics",
            "failure_cluster",
            "boundary_context",
            "artifacts",
            "evaluation_exclusion",
        }
        _reject_unknown(data, allowed, "learning episode")
        mappings = {}
        for key in ("stressor_context", "split_lineage", "provenance", "recovery_metrics"):
            value = data.get(key)
            if not isinstance(value, Mapping):
                raise ValueError(f"learning episode {key} must be a mapping")
            mappings[key] = dict(value)
        raw_steps = _mapping_list(data.get("steps"), "steps")
        raw_branches = _mapping_list(
            data.get("counterfactual_recovery"), "counterfactual_recovery"
        )
        raw_artifacts = _mapping_list(data.get("artifacts"), "artifacts")
        ledger = data.get("failure_ledger")
        if ledger is not None and not isinstance(ledger, Mapping):
            raise ValueError("failure_ledger must be a mapping or null")
        cluster = _optional_mapping(data.get("failure_cluster"), "failure_cluster")
        boundary = _optional_mapping(data.get("boundary_context"), "boundary_context")
        exclusion = data.get("evaluation_exclusion")
        if not isinstance(exclusion, Mapping):
            raise ValueError("evaluation_exclusion must be a mapping")
        return cls(
            episode_id=str(data.get("episode_id", "")),
            task_id=str(data.get("task_id", "")),
            policy_id=str(data.get("policy_id", "")),
            policy_family=str(data.get("policy_family", "")),
            engine_name=str(data.get("engine_name", "")),
            episode_index=int(data.get("episode_index", -1)),
            seed=int(data.get("seed", -1)),
            success=_boolean(data.get("success"), "episode success"),
            failure_label=str(data["failure_label"])
            if data.get("failure_label") is not None
            else None,
            failure_ledger=dict(ledger) if ledger is not None else None,
            stressor_context=mappings["stressor_context"],
            split_lineage=mappings["split_lineage"],
            provenance=mappings["provenance"],
            steps=tuple(LearningStepRecord.from_dict(item) for item in raw_steps),
            counterfactual_recovery=tuple(dict(item) for item in raw_branches),
            recovery_metrics=mappings["recovery_metrics"],
            failure_cluster=cluster,
            boundary_context=boundary,
            artifacts=tuple(ArtifactReference.from_dict(item) for item in raw_artifacts),
            exclusion=EvaluationExclusion.from_dict(exclusion),
        )

    @property
    def content_sha256(self) -> str:
        return _sha256(self.to_dict())


@dataclass(frozen=True)
class LearningExportManifest:
    export_id: str
    created_at: str
    source_runs: tuple[dict[str, Any], ...]
    episode_count: int
    episode_file: ArtifactReference
    exclusion_file: ArtifactReference
    facet_file: ArtifactReference
    facet_index: dict[str, Any]
    licenses: tuple[str, ...]
    privacy_level: PrivacyLevel
    privacy_restrictions: tuple[str, ...]
    intended_use: str = "failure_driven_learning_and_data_selection"
    evaluation_reuse_policy: str = "excluded"
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.export_id.strip() or not self.created_at.strip():
            raise ValueError("export identity and creation time are required")
        if self.episode_count < 0:
            raise ValueError("episode_count must be non-negative")
        if self.schema_version != 1:
            raise ValueError(f"unsupported learning export schema: {self.schema_version}")
        if not self.source_runs:
            raise ValueError("learning export requires source run provenance")
        try:
            datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("created_at must be an ISO-8601 timestamp") from exc
        run_ids = []
        for source in self.source_runs:
            for key in (
                "run_id",
                "suite_id",
                "policy_id",
                "policy_family",
                "engine_name",
                "source_uri",
                "source_run_sha256",
                "dataset_manifest_sha256",
                "provenance",
            ):
                if key not in source:
                    raise ValueError(f"source run is missing {key}")
            run_ids.append(str(source["run_id"]))
            _require_sha256(
                str(source["source_run_sha256"]), "source run sha256"
            )
            _require_sha256(
                str(source["dataset_manifest_sha256"]),
                "dataset manifest sha256",
            )
        if len(run_ids) != len(set(run_ids)):
            raise ValueError("source run IDs must be unique")
        if self.privacy_level not in {"public", "restricted", "private"}:
            raise ValueError(f"unsupported privacy level: {self.privacy_level}")
        if self.privacy_level != "public" and not self.privacy_restrictions:
            raise ValueError("restricted/private exports require privacy restrictions")
        if not self.licenses:
            raise ValueError("learning exports require at least one license declaration")
        if self.evaluation_reuse_policy != "excluded":
            raise ValueError("learning exports must be excluded from evaluation reuse")
        if self.intended_use != "failure_driven_learning_and_data_selection":
            raise ValueError("unsupported learning export intended use")
        if not all(
            reference.embedded
            for reference in (
                self.episode_file,
                self.exclusion_file,
                self.facet_file,
            )
        ):
            raise ValueError("manifest data files must be embedded artifacts")
        if self.facet_index.get("episode_count") != self.episode_count:
            raise ValueError("facet index episode count does not match manifest")
        _canonical_json(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "format": LEARNING_EXPORT_MANIFEST_FORMAT,
            "schema_version": self.schema_version,
            "export_id": self.export_id,
            "created_at": self.created_at,
            "source_runs": list(self.source_runs),
            "episode_count": self.episode_count,
            "episode_file": self.episode_file.to_dict(),
            "exclusion_file": self.exclusion_file.to_dict(),
            "facet_file": self.facet_file.to_dict(),
            "facet_index": self.facet_index,
            "licenses": list(self.licenses),
            "privacy_level": self.privacy_level,
            "privacy_restrictions": list(self.privacy_restrictions),
            "intended_use": self.intended_use,
            "evaluation_reuse_policy": self.evaluation_reuse_policy,
        }
        if include_hash:
            payload["manifest_sha256"] = _sha256(payload)
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LearningExportManifest":
        _check_format(data, LEARNING_EXPORT_MANIFEST_FORMAT, "learning export manifest")
        allowed = {
            "format",
            "schema_version",
            "export_id",
            "created_at",
            "source_runs",
            "episode_count",
            "episode_file",
            "exclusion_file",
            "facet_file",
            "facet_index",
            "licenses",
            "privacy_level",
            "privacy_restrictions",
            "intended_use",
            "evaluation_reuse_policy",
            "manifest_sha256",
        }
        _reject_unknown(data, allowed, "learning export manifest")
        expected_hash = data.get("manifest_sha256")
        unhashed = {key: value for key, value in data.items() if key != "manifest_sha256"}
        if expected_hash != _sha256(unhashed):
            raise ValueError("learning export manifest hash mismatch")
        source_runs = _mapping_list(data.get("source_runs"), "source_runs")
        episode_file = data.get("episode_file")
        exclusion_file = data.get("exclusion_file")
        facet_file = data.get("facet_file")
        facet_index = data.get("facet_index")
        licenses = data.get("licenses")
        restrictions = data.get("privacy_restrictions")
        if (
            not isinstance(episode_file, Mapping)
            or not isinstance(exclusion_file, Mapping)
            or not isinstance(facet_file, Mapping)
        ):
            raise ValueError("manifest episode/exclusion/facet files must be mappings")
        if not isinstance(facet_index, Mapping):
            raise ValueError("manifest facet_index must be a mapping")
        if not isinstance(licenses, list) or not all(isinstance(item, str) for item in licenses):
            raise ValueError("manifest licenses must be a list of strings")
        if not isinstance(restrictions, list) or not all(
            isinstance(item, str) for item in restrictions
        ):
            raise ValueError("privacy_restrictions must be a list of strings")
        return cls(
            export_id=str(data.get("export_id", "")),
            created_at=str(data.get("created_at", "")),
            source_runs=tuple(dict(item) for item in source_runs),
            episode_count=int(data.get("episode_count", -1)),
            episode_file=ArtifactReference.from_dict(episode_file),
            exclusion_file=ArtifactReference.from_dict(exclusion_file),
            facet_file=ArtifactReference.from_dict(facet_file),
            facet_index=dict(facet_index),
            licenses=tuple(licenses),
            privacy_level=str(data.get("privacy_level", "")),  # type: ignore[arg-type]
            privacy_restrictions=tuple(restrictions),
            intended_use=str(data.get("intended_use", "")),
            evaluation_reuse_policy=str(data.get("evaluation_reuse_policy", "")),
            schema_version=int(data.get("schema_version", 1)),
        )


def _check_format(data: Mapping[str, Any], expected: str, label: str) -> None:
    if data.get("format") != expected:
        raise ValueError(f"unsupported {label} format: {data.get('format')}")


def _reject_unknown(data: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"unknown {label} fields: {', '.join(unknown)}")


def _require_sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def _mapping_list(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
        raise ValueError(f"{label} must be a list of mappings")
    return [dict(item) for item in value]


def _optional_mapping(value: Any, label: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping or null")
    return dict(value)


def _validate_split_lineage(value: Mapping[str, Any]) -> None:
    split_id = value.get("split_id")
    partition = value.get("partition")
    content_sha256 = value.get("content_sha256")
    if not isinstance(split_id, str) or not split_id.strip():
        raise ValueError("split lineage requires split_id")
    if partition not in {"train", "validation", "public_test", "hidden_test"}:
        raise ValueError("split lineage has invalid partition")
    _require_sha256(str(content_sha256), "split content_sha256")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"value is not JSON-compatible: {type(value).__name__}")
