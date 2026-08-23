from __future__ import annotations

from typing import Any

from nyssa_bench.failures.protocol import FailureEvidence, FailureEventDraft

from .utils import as_bool, as_float
from .protocol import FailureDetector


class ContactDetector(FailureDetector):
    detector_id = "contact_detector"

    def __init__(self, *, collision_threshold: float = 0.0) -> None:
        self.collision_threshold = float(collision_threshold)
        self._in_collision = False

    def reset(
        self,
        *,
        task: Any,
        engine: Any,
        observation: dict[str, Any] | None,
        stressor_context: dict[str, Any] | None,
    ) -> None:
        self._in_collision = False

    def detect(
        self,
        *,
        step_index: int,
        observation: dict[str, Any] | None,
        action: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
        task: Any,
        engine: Any,
        stressor_context: dict[str, Any] | None = None,
    ) -> list[FailureEventDraft | dict[str, Any]]:
        collision_count = _as_float_optional(
            info.get("collision_count"),
            info.get("num_collisions"),
            info.get("collision"),
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
                            source="external_monitor",
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
