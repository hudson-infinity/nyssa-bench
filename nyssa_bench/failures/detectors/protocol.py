from __future__ import annotations

from abc import ABC
from typing import Any

from nyssa_bench.failures.protocol import FailureEventDraft


class FailureDetector(ABC):
    """Base class for per-step temporal failure detectors.

    Detectors observe transitions and can emit `FailureEventDraft` values at
    reset points, before/after action selection, per-step and at episode end.
    """

    detector_id = "failure_detector"

    def reset(
        self,
        *,
        task: Any,
        engine: Any,
        observation: dict[str, Any] | None,
        stressor_context: dict[str, Any] | None,
    ) -> None:
        """Prepare for a fresh episode."""
        return None

    def observe_before_action(
        self,
        *,
        step_index: int,
        observation: dict[str, Any] | None,
        action: Any,
        task: Any,
        engine: Any,
        stressor_context: dict[str, Any] | None,
    ) -> None:
        """Observe policy output before the environment transition."""
        return None

    def observe_after_action(
        self,
        *,
        step_index: int,
        pre_observation: dict[str, Any] | None,
        post_observation: dict[str, Any] | None,
        action: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
        task: Any,
        engine: Any,
        stressor_context: dict[str, Any] | None,
    ) -> None:
        """Observe the post-step state and transition payload."""
        return None

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
        stressor_context: dict[str, Any] | None,
    ) -> list[FailureEventDraft | dict[str, Any]]:
        """Return zero or more events detected at this step."""
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
        stressor_context: dict[str, Any] | None,
    ) -> list[FailureEventDraft | dict[str, Any]]:
        """Return terminal events (timeouts, unresolved stalls, etc.)."""
        return []
