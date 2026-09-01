from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from nyssa_bench.failures.protocol import FailureEvidence, FailureEventDraft

from .utils import as_float
from .protocol import DetectorSignalRequirement, FailureDetector


class StallDetector(FailureDetector):
    detector_id = "stall_detector"
    detector_version = "1.0.0"
    signal_requirements = (
        DetectorSignalRequirement(
            any_of=("reward",),
            visibility="policy_observable",
            description="per-transition reward or progress scalar",
        ),
    )
    evidence_visibility = ("policy_observable",)
    temporal_precision = ("exact_step", "step_interval")

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
        if self.stall_window <= 0:
            raise ValueError("stall_window must be a positive integer")
        if self.min_steps <= 0:
            raise ValueError("min_steps must be a positive integer")
        if not math.isfinite(self.reward_tolerance) or self.reward_tolerance < 0:
            raise ValueError("reward_tolerance must be finite and non-negative")
        self._last_reward: float | None = None
        self._stagnant_steps = 0
        self._stalled = False
        self._stall_onset: int | None = None

    def configuration(self) -> dict[str, Any]:
        return {
            "stall_window": self.stall_window,
            "reward_tolerance": self.reward_tolerance,
            "min_steps": self.min_steps,
        }

    def reset(
        self,
        *,
        task: Any,
        engine: Any,
        observation: Mapping[str, Any] | None,
        stressor_context: Mapping[str, Any] | None,
    ) -> None:
        self._last_reward = None
        self._stagnant_steps = 0
        self._stalled = False
        self._stall_onset = None

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
                        onset_step=self._stall_onset,
                        detected_step=step_index,
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
        final_observation: Mapping[str, Any] | None,
        reward: float,
        terminated: bool,
        truncated: bool,
        success: bool,
        info: Mapping[str, Any],
        task: Any,
        engine: Any,
        stressor_context: Mapping[str, Any] | None = None,
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
                onset_step=max(0, step_index - self._stagnant_steps + 1),
                detected_step=step_index,
                duration_steps=self._stagnant_steps,
            )
        ]


def _failure_draft(
    *, onset_step: int, detected_step: int, duration_steps: int
) -> FailureEventDraft:
    return FailureEventDraft(
        role="mechanism",
        category="control",
        subtype="planner_stuck",
        onset_step=onset_step,
        end_step=detected_step,
        temporal_precision="step_interval" if duration_steps > 1 else "exact_step",
        confidence=0.72,
        summary_label="planner_stuck",
        evidence=(
            FailureEvidence(
                evidence_id=f"stall:{detected_step}:reward",
                evidence_type="reward_stagnation",
                payload={"duration_steps": duration_steps},
                source="task_logic",
                annotation_source="stall_detector",
                confidence=0.72,
                visibility="policy_observable",
                captured_step=detected_step,
            ),
        ),
        recovery_eligibility="unknown",
        consequences=("no_progress_detected",),
        deduplication_key="planner:stalled",
    )
