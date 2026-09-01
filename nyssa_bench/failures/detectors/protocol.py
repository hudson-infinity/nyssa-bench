from __future__ import annotations

import json
import re
from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from nyssa_bench.failures.protocol import (
    EVIDENCE_VISIBILITIES,
    TEMPORAL_PRECISIONS,
    EvidenceVisibility,
    FailureEventDraft,
    TemporalPrecision,
)


FAILURE_DETECTOR_PROTOCOL_FORMAT = "nyssa-failure-detector-v1"
FAILURE_DETECTOR_PROTOCOL_VERSION = 1

DetectorMode = Literal["passive", "instrumented"]
DetectorSupportStatus = Literal["supported", "pending", "unsupported"]


@dataclass(frozen=True)
class DetectorSignalRequirement:
    """A set of interchangeable runtime signals required by a detector."""

    any_of: tuple[str, ...]
    visibility: EvidenceVisibility
    description: str = ""

    def __post_init__(self) -> None:
        if not self.any_of or any(not item.strip() for item in self.any_of):
            raise ValueError("detector signal requirements need non-empty signal IDs")
        if len(set(self.any_of)) != len(self.any_of):
            raise ValueError("detector signal requirement IDs must be unique")
        if self.visibility not in EVIDENCE_VISIBILITIES:
            raise ValueError(
                f"Unsupported detector signal visibility: {self.visibility}"
            )

    def satisfied_by(self, capabilities: set[str]) -> bool:
        return any(signal in capabilities for signal in self.any_of)

    def to_dict(self) -> dict[str, Any]:
        return {
            "any_of": list(self.any_of),
            "visibility": self.visibility,
            "description": self.description,
        }


@dataclass(frozen=True)
class DetectorSupport:
    status: DetectorSupportStatus
    available_signals: tuple[str, ...] = ()
    missing_requirements: tuple[tuple[str, ...], ...] = ()
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "available_signals": list(self.available_signals),
            "missing_requirements": [
                list(requirement) for requirement in self.missing_requirements
            ],
            "reason": self.reason,
        }


@dataclass(frozen=True)
class FailureDetectorContract:
    detector_id: str
    detector_version: str
    signal_requirements: tuple[DetectorSignalRequirement, ...] = ()
    supported_engines: tuple[str, ...] = ("*",)
    supported_tasks: tuple[str, ...] = ("*",)
    evidence_visibility: tuple[EvidenceVisibility, ...] = ("privileged",)
    temporal_precision: tuple[TemporalPrecision, ...] = ("exact_step",)
    mode: DetectorMode = "passive"
    configuration: dict[str, Any] = field(default_factory=dict)
    protocol_version: int = FAILURE_DETECTOR_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if not self.detector_id.strip():
            raise ValueError("detector_id must be non-empty")
        if not re.fullmatch(
            r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:[-+][0-9A-Za-z.-]+)?",
            self.detector_version,
        ):
            raise ValueError("detector_version must be a semantic version")
        if self.protocol_version != FAILURE_DETECTOR_PROTOCOL_VERSION:
            raise ValueError(
                f"Unsupported failure detector protocol version: {self.protocol_version}"
            )
        if self.mode not in {"passive", "instrumented"}:
            raise ValueError("detector mode must be 'passive' or 'instrumented'")
        _validate_patterns(self.supported_engines, "supported_engines")
        _validate_patterns(self.supported_tasks, "supported_tasks")
        if not self.evidence_visibility or any(
            item not in EVIDENCE_VISIBILITIES for item in self.evidence_visibility
        ):
            raise ValueError("detector evidence_visibility contains an invalid value")
        if not self.temporal_precision or any(
            item not in TEMPORAL_PRECISIONS for item in self.temporal_precision
        ):
            raise ValueError("detector temporal_precision contains an invalid value")
        try:
            json.dumps(self.configuration, allow_nan=False, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "detector configuration must be finite JSON-compatible data"
            ) from exc

    def support(
        self,
        *,
        engine_name: str,
        task_id: str,
        capabilities: set[str],
    ) -> DetectorSupport:
        available = tuple(sorted(capabilities))
        if not _matches(self.supported_engines, engine_name):
            return DetectorSupport(
                status="unsupported",
                available_signals=available,
                reason=f"engine '{engine_name}' is not supported",
            )
        if not _matches(self.supported_tasks, task_id):
            return DetectorSupport(
                status="unsupported",
                available_signals=available,
                reason=f"task '{task_id}' is not supported",
            )
        missing = tuple(
            requirement.any_of
            for requirement in self.signal_requirements
            if not requirement.satisfied_by(capabilities)
        )
        if missing:
            return DetectorSupport(
                status="pending",
                available_signals=available,
                missing_requirements=missing,
                reason="required runtime signals have not been observed",
            )
        return DetectorSupport(status="supported", available_signals=available)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": FAILURE_DETECTOR_PROTOCOL_FORMAT,
            "protocol_version": self.protocol_version,
            "detector_id": self.detector_id,
            "detector_version": self.detector_version,
            "mode": self.mode,
            "supported_engines": list(self.supported_engines),
            "supported_tasks": list(self.supported_tasks),
            "signal_requirements": [
                requirement.to_dict() for requirement in self.signal_requirements
            ],
            "evidence_visibility": list(self.evidence_visibility),
            "temporal_precision": list(self.temporal_precision),
            "configuration": dict(self.configuration),
        }


class FailureDetector(ABC):
    """Versioned base class for streaming temporal failure detectors."""

    detector_id = "failure_detector"
    detector_version = "1.0.0"
    mode: DetectorMode = "passive"
    supported_engines: tuple[str, ...] = ("*",)
    supported_tasks: tuple[str, ...] = ("*",)
    signal_requirements: tuple[DetectorSignalRequirement, ...] = ()
    evidence_visibility: tuple[EvidenceVisibility, ...] = ("privileged",)
    temporal_precision: tuple[TemporalPrecision, ...] = ("exact_step",)

    def configuration(self) -> dict[str, Any]:
        return {}

    def contract(self) -> FailureDetectorContract:
        return FailureDetectorContract(
            detector_id=self.detector_id,
            detector_version=self.detector_version,
            mode=self.mode,
            supported_engines=self.supported_engines,
            supported_tasks=self.supported_tasks,
            signal_requirements=self.signal_requirements,
            evidence_visibility=self.evidence_visibility,
            temporal_precision=self.temporal_precision,
            configuration=self.configuration(),
        )

    def request_instrumentation(self, *, task: Any, engine: Any) -> set[str] | None:
        """Enable optional signals for an instrumented detector."""

        return None

    def reset(
        self,
        *,
        task: Any,
        engine: Any,
        observation: Mapping[str, Any] | None,
        stressor_context: Mapping[str, Any] | None,
    ) -> None:
        return None

    def observe_before_action(
        self,
        *,
        step_index: int,
        observation: Mapping[str, Any] | None,
        action: Any,
        task: Any,
        engine: Any,
        stressor_context: Mapping[str, Any] | None,
    ) -> list[FailureEventDraft | dict[str, Any]] | None:
        return None

    def observe_after_action(
        self,
        *,
        step_index: int,
        pre_observation: Mapping[str, Any] | None,
        post_observation: Mapping[str, Any] | None,
        action: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Mapping[str, Any],
        task: Any,
        engine: Any,
        stressor_context: Mapping[str, Any] | None,
    ) -> list[FailureEventDraft | dict[str, Any]] | None:
        return None

    def detect(
        self,
        *,
        step_index: int,
        observation: Mapping[str, Any] | None,
        action: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Mapping[str, Any],
        task: Any,
        engine: Any,
        stressor_context: Mapping[str, Any] | None,
    ) -> list[FailureEventDraft | dict[str, Any]]:
        return []

    def finalize(
        self,
        *,
        step_index: int,
        final_observation: Mapping[str, Any] | None,
        reward: float,
        terminated: bool,
        truncated: bool,
        success: bool,
        info: Mapping[str, Any],
        task: Any,
        engine: Any,
        stressor_context: Mapping[str, Any] | None,
    ) -> list[FailureEventDraft | dict[str, Any]]:
        return []


def _matches(patterns: tuple[str, ...], value: str) -> bool:
    return "*" in patterns or value in patterns


def _validate_patterns(patterns: tuple[str, ...], field_name: str) -> None:
    if not patterns or any(not item.strip() for item in patterns):
        raise ValueError(f"{field_name} must contain non-empty values")
    if len(set(patterns)) != len(patterns):
        raise ValueError(f"{field_name} values must be unique")
