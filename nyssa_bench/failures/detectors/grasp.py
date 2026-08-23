from __future__ import annotations

from typing import Any

from nyssa_bench.failures.protocol import FailureEvidence, FailureEventDraft

from .utils import as_bool, as_float
from .protocol import FailureDetector


class GraspDetector(FailureDetector):
    detector_id = "grasp_detector"

    def __init__(self) -> None:
        self._emitted_wrong_object = False
        self._emitted_slip = False
        self._emitted_bad_grasp = False

    def reset(
        self,
        *,
        task: Any,
        engine: Any,
        observation: dict[str, Any] | None,
        stressor_context: dict[str, Any] | None,
    ) -> None:
        self._emitted_wrong_object = False
        self._emitted_slip = False
        self._emitted_bad_grasp = False

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
        drafts: list[FailureEventDraft | dict[str, Any]] = []

        if not self._emitted_wrong_object and (
            as_bool(info.get("wrong_object"))
            or as_bool(info.get("wrong_object_selected"))
        ):
            self._emitted_wrong_object = True
            drafts.append(
                _failure_draft(
                    step_index=step_index,
                    subtype="wrong_object",
                    summary_label="wrong_object",
                    message="wrong_object_selected",
                )
            )

        if not self._emitted_slip and as_bool(info.get("object_slip")):
            self._emitted_slip = True
            drafts.append(
                _failure_draft(
                    step_index=step_index,
                    subtype="object_slip",
                    summary_label="object_slip",
                    message="object_slip",
                    payload={
                        "grasp_grip": _as_float_optional(info.get("grip_strength"))
                    },
                )
            )

        if not self._emitted_bad_grasp and (
            as_bool(info.get("bad_grasp"))
            or as_bool(info.get("grasp_failed"))
            or (info.get("grasp_success") is False and info.get("grasp") is not None)
        ):
            self._emitted_bad_grasp = True
            drafts.append(
                _failure_draft(
                    step_index=step_index,
                    subtype="bad_grasp",
                    summary_label="bad_grasp",
                    message="grasp_failed",
                    payload={
                        "grasp_success": info.get("grasp_success"),
                        "grasp_failed": info.get("grasp_failed"),
                    },
                )
            )

        return drafts


def _failure_draft(
    *,
    step_index: int,
    subtype: str,
    summary_label: str,
    message: str,
    payload: dict[str, Any] | None = None,
) -> FailureEventDraft:
    return FailureEventDraft(
        role="mechanism",
        category="manipulation",
        subtype=subtype,
        onset_step=step_index,
        temporal_precision="exact_step",
        confidence=0.85,
        summary_label=summary_label,
        evidence=(
            FailureEvidence(
                evidence_id=f"grasp:{step_index}:{subtype}",
                evidence_type="manipulation_signal",
                payload={
                    "signal": message,
                    **(payload or {}),
                },
                source="external_monitor",
                annotation_source="grasp_detector",
                confidence=0.85,
                visibility="privileged",
                captured_step=step_index,
            ),
        ),
        recovery_eligibility="eligible",
        consequences=(f"{subtype}_event",),
        deduplication_key=f"grasp:{subtype}",
    )


def _as_float_optional(*values: Any) -> float | None:
    for value in values:
        candidate = as_float(value)
        if candidate is not None:
            return candidate
    return None
