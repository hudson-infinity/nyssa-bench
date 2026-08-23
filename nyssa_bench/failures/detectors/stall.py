from __future__ import annotations

from typing import Any

from nyssa_bench.failures.protocol import FailureEvidence, FailureEventDraft

from .utils import as_float
from .protocol import FailureDetector


class StallDetector(FailureDetector):
    detector_id = "stall_detector"

    def __init__(
        self,
        *,
        stall_window: int = 6,
        reward_tolerance: float = 1e-3,
        min_steps: int = 2,
    ) -> None:
        self.stall_window = int(stall_window)
        self.reward_tolerance = float(reward_tolerance)
        self.min_steps = int(min_steps)
        self._last_reward: float | None = None
        self._stagnant_steps = 0
        self._stalled = False
        self._stall_onset: int | None = None

    def reset(
        self,
        *,
        task: Any,
        engine: Any,
        observation: dict[str, Any] | None,
        stressor_context: dict[str, Any] | None,
    ) -> None:
        self._last_reward = None
        self._stagnant_steps = 0
        self._stalled = False
        self._stall_onset = None

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
        if self._stalled:
            return []

        current_reward = as_float(reward)
        if current_reward is None:
            return []
        if self._last_reward is None:
            self._last_reward = current_reward
            return []

        if abs(current_reward - self._last_reward) <= self.reward_tolerance:
            self._stagnant_steps += 1
            if (
                self._stagnant_steps >= self.stall_window
                and step_index + 1 >= self.min_steps
            ):
                self._stalled = True
                self._stall_onset = max(0, step_index - self._stagnant_steps + 1)
                return [
                    _failure_draft(
                        step_index=self._stall_onset,
                        duration_steps=self._stagnant_steps,
                    )
                ]
            return []

        self._stagnant_steps = 0
        self._last_reward = current_reward
        return []

    def finalize(
        self,
        *,
        step_index: int,
        final_observation: dict[str, Any] | None,
        reward: float,
        terminated: bool,
        truncated: bool,
        success: bool,
        info: dict[str, Any],
        task: Any,
        engine: Any,
        stressor_context: dict[str, Any] | None = None,
    ) -> list[FailureEventDraft | dict[str, Any]]:
        if success:
            return []
        if self._stalled:
            return []
        if self._last_reward is None:
            return []
        if (
            self._stagnant_steps < self.stall_window
            or self._stagnant_steps < self.min_steps
        ):
            return []
        return [
            _failure_draft(
                step_index=max(0, step_index - self._stagnant_steps + 1),
                duration_steps=self._stagnant_steps,
            )
        ]


def _failure_draft(*, step_index: int, duration_steps: int) -> FailureEventDraft:
    return FailureEventDraft(
        role="mechanism",
        category="control",
        subtype="planner_stuck",
        onset_step=step_index,
        end_step=step_index,
        temporal_precision="step_interval" if duration_steps > 1 else "exact_step",
        confidence=0.72,
        summary_label="planner_stuck",
        evidence=(
            FailureEvidence(
                evidence_id=f"stall:{step_index}:reward",
                evidence_type="reward_stagnation",
                payload={"duration_steps": duration_steps},
                source="external_monitor",
                annotation_source="stall_detector",
                confidence=0.72,
                visibility="policy_observable",
                captured_step=step_index,
            ),
        ),
        recovery_eligibility="unknown",
        consequences=("no_progress_detected",),
        deduplication_key="planner:stalled",
    )
