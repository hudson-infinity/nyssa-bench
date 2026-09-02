from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from datetime import date, datetime
from typing import Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)


REAL_EVIDENCE_PACKAGE_FORMAT = "nyssa-real-evidence-package-v1"
REAL_EVIDENCE_PROTOCOL_VERSION = 1
REAL_EVIDENCE_MANIFEST = "evidence.yaml"
REAL_EVIDENCE_LEDGER_FORMAT = "nyssa-real-evidence-ledgers-v1"
REAL_EVIDENCE_REPORT_FORMAT = "nyssa-real-evidence-report-v1"

ArtifactAccess = Literal["packaged", "protected", "external"]
StreamModality = Literal[
    "rgb",
    "depth",
    "point_cloud",
    "proprioception",
    "tactile",
    "wrench",
    "audio",
    "other",
]
CalibrationType = Literal["clock", "camera", "geometry", "dynamics", "latency"]


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


class ProtocolReferences(ContractModel):
    nep_version: str
    task_contract_format: str
    failure_event_format: str = "nyssa-failure-event-v1"
    scenario_contract_format: str = "nyssa-external-scenario-package-v1"

    @field_validator("nep_version")
    @classmethod
    def validate_nep_version(cls, value: str) -> str:
        return _semver(value, "protocol.nep_version")


class EvidenceArtifact(ContractModel):
    artifact_id: str
    sha256: str
    media_type: str
    license_id: str
    provenance_uri: str
    access: ArtifactAccess
    path: str | None = None
    external_locator: str | None = None
    redacted: bool = False
    required: bool = True

    @field_validator("artifact_id", "media_type", "license_id", "provenance_uri")
    @classmethod
    def nonempty(cls, value: str) -> str:
        return _text(value, "artifact field")

    @field_validator("sha256")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        return _sha256(value, "artifact.sha256")

    @field_validator("provenance_uri")
    @classmethod
    def provenance_is_uri(cls, value: str) -> str:
        return _uri(value, "artifact.provenance_uri")

    @field_validator("external_locator")
    @classmethod
    def locator_is_uri(cls, value: str | None) -> str | None:
        return _uri(value, "artifact.external_locator") if value else None

    @field_validator("path")
    @classmethod
    def safe_path(cls, value: str | None) -> str | None:
        if value is not None:
            _safe_relative_path(value, "artifact.path")
        return value

    @model_validator(mode="after")
    def validate_location(self) -> "EvidenceArtifact":
        if self.access == "packaged" and not self.path:
            raise ValueError("packaged artifacts require path")
        if self.access != "packaged" and not (self.path or self.external_locator):
            raise ValueError(
                "protected/external artifacts require path or external_locator"
            )
        return self


class ClockContract(ContractModel):
    clock_id: str
    domain: Literal["monotonic", "utc", "device"]
    timestamp_unit: Literal["s", "ms", "us", "ns"]
    epoch: str
    synchronization_method: str
    calibration_id: str
    offset_seconds: float
    offset_uncertainty_seconds: float = Field(ge=0.0)
    drift_ppm: float
    drift_uncertainty_ppm: float = Field(ge=0.0)


class CoordinateFrame(ContractModel):
    frame_id: str
    parent_frame_id: str | None = None
    transform_to_parent: tuple[float, ...]
    translation_unit: Literal["m"] = "m"
    rotation_representation: Literal["matrix4x4"] = "matrix4x4"

    @field_validator("transform_to_parent")
    @classmethod
    def matrix_has_16_finite_values(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if len(value) != 16 or not all(_finite(item) for item in value):
            raise ValueError("frame transform_to_parent must contain 16 finite values")
        return value


class MissingRange(ContractModel):
    start_seconds: float = Field(ge=0.0)
    end_seconds: float = Field(ge=0.0)
    reason: str

    @model_validator(mode="after")
    def ordered(self) -> "MissingRange":
        if self.end_seconds < self.start_seconds:
            raise ValueError("missing range end must not precede start")
        return self


class SensorStream(ContractModel):
    stream_id: str
    modality: StreamModality
    artifact_id: str
    clock_id: str
    frame_id: str | None = None
    sample_count: int = Field(gt=0)
    timestamp_field: str = "timestamps"
    value_field: str = "values"
    units: dict[str, str]
    missing_ranges: tuple[MissingRange, ...] = ()


class ActionContract(ContractModel):
    stream_id: str
    artifact_id: str
    clock_id: str
    frame_id: str | None = None
    representation: str
    control_mode: str
    dimension: int = Field(gt=0)
    units: tuple[str, ...]
    lower_bounds: tuple[float, ...]
    upper_bounds: tuple[float, ...]
    sample_count: int = Field(gt=0)
    timestamp_field: str = "timestamps"
    value_field: str = "values"
    latency_calibration_id: str

    @model_validator(mode="after")
    def dimensions_match(self) -> "ActionContract":
        if any(
            len(values) != self.dimension
            for values in (self.units, self.lower_bounds, self.upper_bounds)
        ):
            raise ValueError("action units and bounds must match dimension")
        if any(
            not _finite(low) or not _finite(high) or low >= high
            for low, high in zip(self.lower_bounds, self.upper_bounds)
        ):
            raise ValueError("action bounds must be finite and ordered")
        return self


class RobotEpisodeIdentity(ContractModel):
    episode_id: str
    trial_id: str
    recorded_at: datetime
    site_id: str
    robot_id: str
    embodiment_id: str
    embodiment_version: str
    controller_id: str
    controller_version: str
    policy_id: str
    checkpoint_sha256: str
    operator_id: str
    operator_role: str

    @field_validator("checkpoint_sha256")
    @classmethod
    def checkpoint_hash(cls, value: str) -> str:
        return _sha256(value, "identity.checkpoint_sha256")

    @field_validator("embodiment_version", "controller_version")
    @classmethod
    def versions(cls, value: str) -> str:
        return _semver(value, "identity version")


class InterventionRecord(ContractModel):
    intervention_id: str
    timestamp_seconds: float = Field(ge=0.0)
    actor: Literal["operator", "safety_system", "policy_monitor"]
    intervention_type: str
    reason: str


class SafetyEventRecord(ContractModel):
    event_id: str
    timestamp_seconds: float = Field(ge=0.0)
    category: str
    severity: Literal["info", "warning", "critical"]
    evidence_artifact_ids: tuple[str, ...] = ()


class MissingDataMarker(ContractModel):
    stream_id: str
    start_seconds: float = Field(ge=0.0)
    end_seconds: float = Field(ge=0.0)
    reason: str


class TaskOutcome(ContractModel):
    task_id: str
    success: bool
    terminated: bool
    truncated: bool
    duration_seconds: float = Field(gt=0.0)
    interventions: tuple[InterventionRecord, ...] = ()
    safety_events: tuple[SafetyEventRecord, ...] = ()
    missing_data: tuple[MissingDataMarker, ...] = ()


class RealRobotEpisode(ContractModel):
    identity: RobotEpisodeIdentity
    clocks: tuple[ClockContract, ...]
    frames: tuple[CoordinateFrame, ...]
    sensors: tuple[SensorStream, ...]
    actions: ActionContract
    outcome: TaskOutcome
    failure_events: tuple[dict[str, Any], ...]


class CalibrationRecord(ContractModel):
    calibration_id: str
    calibration_type: CalibrationType
    method_id: str
    method_version: str
    estimate: dict[str, Any]
    units: dict[str, str]
    uncertainty: dict[str, Any]
    fit_quality: dict[str, float]
    artifact_id: str | None = None
    status: Literal["valid", "estimated", "missing"]

    @field_validator("method_version")
    @classmethod
    def method_semver(cls, value: str) -> str:
        return _semver(value, "calibration.method_version")


class ReconstructionParameter(ContractModel):
    name: str
    value: Any
    unit: str
    uncertainty: float = Field(ge=0.0)
    calibration_id: str


class ReconstructionMismatch(ContractModel):
    mismatch_id: str
    category: Literal["geometry", "appearance", "dynamics", "latency", "sensor", "task"]
    description: str
    magnitude: float = Field(ge=0.0)
    unit: str
    confidence: float = Field(ge=0.0, le=1.0)


class ReconstructionTool(ContractModel):
    tool_id: str
    tool_version: str
    revision: str
    repository_url: str

    @field_validator("tool_version")
    @classmethod
    def tool_semver(cls, value: str) -> str:
        return _semver(value, "reconstruction.tool_version")

    @field_validator("repository_url")
    @classmethod
    def repository_is_uri(cls, value: str) -> str:
        return _uri(value, "reconstruction.repository_url")


class ReconstructedVariant(ContractModel):
    variant_id: str
    scenario_identity: str
    reconstruction: ReconstructionTool
    assumptions: tuple[str, ...]
    estimated_parameters: tuple[ReconstructionParameter, ...]
    fit_quality: dict[str, float]
    mismatches: tuple[ReconstructionMismatch, ...]
    outcome: TaskOutcome
    failure_events: tuple[dict[str, Any], ...]

    @model_validator(mode="after")
    def nonempty_evidence(self) -> "ReconstructedVariant":
        if not self.assumptions:
            raise ValueError("reconstructed variants require assumptions")
        if not self.estimated_parameters:
            raise ValueError("reconstructed variants require estimated_parameters")
        if not self.mismatches:
            raise ValueError("reconstructed variants require explicit mismatch records")
        if not re.fullmatch(r"[^@:]+@[^:]+:[0-9a-f]{64}", self.scenario_identity):
            raise ValueError("variant scenario_identity is malformed")
        return self


class RealSimMapping(ContractModel):
    mapping_id: str
    real_episode_id: str
    variant_ids: tuple[str, ...]
    controlled_axes: tuple[str, ...]
    matching_keys: tuple[str, ...]
    purpose: Literal["calibration", "counterfactual", "failure_replay"]

    @model_validator(mode="after")
    def family_is_nonempty(self) -> "RealSimMapping":
        if not self.variant_ids:
            raise ValueError("real/sim mapping requires at least one variant")
        if len(set(self.variant_ids)) != len(self.variant_ids):
            raise ValueError("real/sim mapping variant_ids must be unique")
        if not self.controlled_axes or not self.matching_keys:
            raise ValueError(
                "real/sim mapping requires controlled_axes and matching_keys"
            )
        return self


class GovernanceContract(ContractModel):
    privacy_classification: Literal["public", "restricted", "confidential"]
    consent_basis: str
    license_id: str
    redistribution: Literal["allowed", "metadata_only", "prohibited"]
    redactions: tuple[str, ...]
    retention_policy: str
    retention_until: date | None = None
    artifact_access_rules: tuple[str, ...]
    operator_ids_pseudonymous: bool

    @model_validator(mode="after")
    def governance_is_actionable(self) -> "GovernanceContract":
        if not self.artifact_access_rules or not self.retention_policy:
            raise ValueError("governance requires retention and artifact access rules")
        return self


class RealEvidencePackage(ContractModel):
    format: Literal["nyssa-real-evidence-package-v1"]
    schema_version: Literal[1]
    package_id: str
    package_version: str
    content_sha256: str
    protocol: ProtocolReferences
    artifacts: tuple[EvidenceArtifact, ...]
    real_episode: RealRobotEpisode
    calibrations: tuple[CalibrationRecord, ...]
    reconstructed_variants: tuple[ReconstructedVariant, ...]
    mapping: RealSimMapping
    governance: GovernanceContract
    metadata: dict[str, Any] = Field(default_factory=dict)

    _source_path: Path | None = PrivateAttr(default=None)

    @field_validator("package_version")
    @classmethod
    def package_semver(cls, value: str) -> str:
        return _semver(value, "package_version")

    @field_validator("content_sha256")
    @classmethod
    def package_hash(cls, value: str) -> str:
        return _sha256(value, "content_sha256")

    @model_validator(mode="after")
    def unique_ids(self) -> "RealEvidencePackage":
        _unique([item.artifact_id for item in self.artifacts], "artifact IDs")
        _unique([item.clock_id for item in self.real_episode.clocks], "clock IDs")
        _unique([item.frame_id for item in self.real_episode.frames], "frame IDs")
        _unique([item.stream_id for item in self.real_episode.sensors], "stream IDs")
        _unique([item.calibration_id for item in self.calibrations], "calibration IDs")
        _unique(
            [item.variant_id for item in self.reconstructed_variants], "variant IDs"
        )
        return self

    @classmethod
    def load(cls, path: str | Path) -> "RealEvidencePackage":
        source = resolve_real_evidence_manifest(path)
        payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        package = cls.model_validate(payload)
        object.__setattr__(package, "_source_path", source)
        return package

    @property
    def source_path(self) -> Path | None:
        return self._source_path

    @property
    def package_root(self) -> Path | None:
        return self._source_path.parent if self._source_path is not None else None

    @property
    def identity(self) -> str:
        return f"{self.package_id}@{self.package_version}:{self.content_sha256}"

    def compute_content_sha256(self) -> str:
        payload = self.model_dump(mode="json")
        payload.pop("content_sha256", None)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def resolve_real_evidence_manifest(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / REAL_EVIDENCE_MANIFEST
    if not candidate.is_file():
        raise FileNotFoundError(f"Real evidence manifest not found: {candidate}")
    return candidate.resolve()


def real_evidence_conformance_fixture_path(
    fixture_name: str = "valid_reconstructed_family",
) -> Path:
    _safe_relative_path(fixture_name, "fixture_name")
    root = Path(__file__).resolve().parents[2] / "conformance" / "real_evidence" / "v1"
    fixture = (root / fixture_name).resolve()
    try:
        fixture.relative_to(root.resolve())
    except ValueError as exc:
        raise FileNotFoundError(
            f"Real evidence fixture not found: {fixture_name}"
        ) from exc
    if not fixture.is_dir():
        raise FileNotFoundError(f"Real evidence fixture not found: {fixture_name}")
    return fixture


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text(value: str, label: str) -> str:
    if not value.strip():
        raise ValueError(f"{label} must be non-empty")
    return value


def _semver(value: str, label: str) -> str:
    if not re.fullmatch(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)", value):
        raise ValueError(f"{label} must be a semantic version")
    return value


def _sha256(value: str, label: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _safe_relative_path(value: str, label: str) -> None:
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        path.is_absolute()
        or re.match(r"^[A-Za-z]:/", normalized)
        or ".." in path.parts
        or not path.parts
    ):
        raise ValueError(f"{label} must be a safe package-relative path")


def _uri(value: str, label: str) -> str:
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9+.-]*://[^\s]+", value):
        raise ValueError(f"{label} must be an absolute URI")
    return value


def _finite(value: float) -> bool:
    return value == value and value not in {float("inf"), float("-inf")}


def _unique(values: list[str], label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique")
