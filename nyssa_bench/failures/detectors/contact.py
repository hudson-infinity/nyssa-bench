from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from nyssa_bench.failures.protocol import FailureEvidence, FailureEventDraft

from .utils import as_bool, as_float
from .protocol import DetectorSignalRequirement, FailureDetector


class ContactDetector(FailureDetector):
    detector_id = "contact_detector"
    detector_version = "1.0.0"
    signal_requirements = (
        DetectorSignalRequirement(
            any_of=(
                "info.collision_count",
                "info.num_collisions",
                "info.collision",
                "info.contact_violation",
                "info.safety_violation",
            ),
            visibility="privileged",
            description="explicit collision or safety-contact signal",
        ),
    )
    evidence_visibility = ("privileged",)
    temporal_precision = ("exact_step",)

    def __init__(self, *, collision_threshold: float = 0.0) -> None:
        self.collision_threshold = float(collision_threshold)
        if not math.isfinite(self.collision_threshold) or self.collision_threshold < 0:
            raise ValueError("collision_threshold must be finite and non-negative")
        self._in_collision = False

    def configuration(self) -> dict[str, Any]:
        return {"collision_threshold": self.collision_threshold}

    def reset(
        self,
        *,
        task: Any,
        engine: Any,
        observation: Mapping[str, Any] | None,
        stressor_context: Mapping[str, Any] | None,
    ) -> None:
        self._in_collision = False

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
        stressor_context: Mapping[str, Any] | None = None,
    ) -> list[FailureEventDraft | dict[str, Any]]:
        collision_count = _as_float_optional(
            info.get("collision_count"),
            info.get("num_collisions"),
            info.get("collision"),
            info.get("contact_violation"),
            info.get("safety_violation"),
        )
        if collision_count is None:
            if as_bool(info.get("collision")):
                collision_count = 1.0
            else:
                collision_count = 0.0
        collision = collision_count > self.collision_threshold
        if collision and not self._in_collision:
            self._in_collision = True
            return [
                FailureEventDraft(
                    role="mechanism",
                    category="interaction",
                    subtype="collision",
                    onset_step=step_index,
                    temporal_precision="exact_step",
                    confidence=1.0,
                    summary_label="collision",
                    evidence=(
                        FailureEvidence(
                            evidence_id=f"contact:{step_index}:collision_count",
                            evidence_type="collision_count",
                            payload={"collision_count": collision_count},
                            source="simulator_state",
                            annotation_source="contact_detector",
                            confidence=1.0,
                            visibility="privileged",
                            captured_step=step_index,
                        ),
                    ),
                    recovery_eligibility="eligible",
                    consequences=("contact_event",),
                    deduplication_key="contact:collision",
                )
            ]

        if not collision:
            self._in_collision = False
        return []


def _as_float_optional(*values: Any) -> float | None:
    for value in values:
        result = as_float(value)
        if result is not None:
            return result
    return None
