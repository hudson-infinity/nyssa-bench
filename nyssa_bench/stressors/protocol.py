from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


STRESSOR_SPEC_FORMAT = "nyssa-stressor-spec-v1"
STRESSOR_CONFIG_FORMAT = "nyssa-stressor-config-v1"
STRESSOR_CONTEXT_FORMAT = "nyssa-stressor-context-v1"

StressorStatus = Literal["requested", "applied", "skipped", "unsupported"]
UnsupportedPolicy = Literal["error", "record"]


@dataclass(frozen=True)
class StressorSpec:
    stressor_id: str
    severity: float
    parameters: dict[str, Any] = field(default_factory=dict)
    seed: int | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.stressor_id.strip():
            raise ValueError("stressor_id must be non-empty")
        if (
            not math.isfinite(float(self.severity))
            or not 0.0 <= float(self.severity) <= 1.0
        ):
            raise ValueError("stressor severity must be finite and within [0, 1]")
        if self.seed is not None and int(self.seed) < 0:
            raise ValueError("stressor seed must be non-negative")
        if self.schema_version != 1:
            raise ValueError(
                f"Unsupported stressor schema version: {self.schema_version}"
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StressorSpec":
        format_name = data.get("format")
        if format_name not in {None, STRESSOR_SPEC_FORMAT}:
            raise ValueError(f"Unsupported stressor spec format: {format_name}")
        allowed = {
            "format",
            "schema_version",
            "stressor_id",
            "id",
            "severity",
            "parameters",
            "seed",
        }
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(f"Unknown stressor spec fields: {', '.join(unknown)}")
        stressor_id = data.get("stressor_id", data.get("id"))
        if stressor_id is None:
            raise ValueError("Stressor spec requires stressor_id")
        parameters = data.get("parameters", {})
        if not isinstance(parameters, dict):
            raise ValueError("stressor parameters must be a mapping")
        return cls(
            stressor_id=str(stressor_id),
            severity=float(data.get("severity", 0.0)),
            parameters=dict(parameters),
            seed=int(data["seed"]) if data.get("seed") is not None else None,
            schema_version=int(data.get("schema_version", 1)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": STRESSOR_SPEC_FORMAT,
            "schema_version": self.schema_version,
            "stressor_id": self.stressor_id,
            "severity": self.severity,
            "parameters": self.parameters,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class StressorConfig:
    condition_id: str
    stressors: tuple[StressorSpec, ...]
    unsupported_policy: UnsupportedPolicy = "error"

    def __post_init__(self) -> None:
        if not self.condition_id.strip():
            raise ValueError("stressor condition_id must be non-empty")
        if self.unsupported_policy not in {"error", "record"}:
            raise ValueError("unsupported_policy must be 'error' or 'record'")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StressorConfig":
        format_name = data.get("format")
        if format_name not in {None, STRESSOR_CONFIG_FORMAT}:
            raise ValueError(f"Unsupported stressor config format: {format_name}")
        allowed = {"format", "condition_id", "stressors", "unsupported_policy"}
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(f"Unknown stressor config fields: {', '.join(unknown)}")
        raw_stressors = data.get("stressors", [])
        if not isinstance(raw_stressors, list):
            raise ValueError("stressors must be a list")
        return cls(
            condition_id=str(data.get("condition_id", "shifted")),
            stressors=tuple(
                StressorSpec.from_dict(dict(item)) for item in raw_stressors
            ),
            unsupported_policy=str(data.get("unsupported_policy", "error")),  # type: ignore[arg-type]
        )

    @classmethod
    def load(cls, path: str | Path) -> "StressorConfig":
        path = Path(path)
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"Stressor config must contain a mapping: {path}")
        return cls.from_dict(payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": STRESSOR_CONFIG_FORMAT,
            "condition_id": self.condition_id,
            "unsupported_policy": self.unsupported_policy,
            "stressors": [stressor.to_dict() for stressor in self.stressors],
        }


@dataclass(frozen=True)
class StressorContext:
    engine_name: str
    task_id: str
    observation_mode: str | None = None
    action_mode: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine_name": self.engine_name,
            "task_id": self.task_id,
            "observation_mode": self.observation_mode,
            "action_mode": self.action_mode,
        }


@dataclass
class StressorApplication:
    stressor_id: str
    category: str
    composition_index: int
    application_points: tuple[str, ...]
    severity_domain: tuple[float, float]
    lifetime: str
    observable_by_policy: bool
    privileged: bool
    requested: dict[str, Any]
    seed: int
    status: StressorStatus = "requested"
    applied_parameters: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None
    backend_evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stressor_id": self.stressor_id,
            "category": self.category,
            "composition_index": self.composition_index,
            "application_points": list(self.application_points),
            "severity_domain": list(self.severity_domain),
            "lifetime": self.lifetime,
            "observable_by_policy": self.observable_by_policy,
            "privileged": self.privileged,
            "requested": self.requested,
            "seed": self.seed,
            "status": self.status,
            "applied_parameters": self.applied_parameters,
            "reason": self.reason,
            "backend_evidence": self.backend_evidence,
        }
