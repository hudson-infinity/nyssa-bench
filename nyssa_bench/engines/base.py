from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from nyssa_bench.core.task import TaskSpec
from nyssa_bench.failures.protocol import FailureEventDraft


class NyssaEngine(ABC):
    """Base interface for simulator adapters.

    Step and reset intentionally follow Gymnasium's shape:
    reset(seed) -> observation, info
    step(action) -> observation, reward, terminated, truncated, info
    """

    @abstractmethod
    def load_task(self, task_spec: TaskSpec) -> None:
        raise NotImplementedError

    @abstractmethod
    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def step(
        self, action: Any
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def render(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def get_state(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    def apply_stressor(
        self, stressor_id: str, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Apply a backend stressor and return explicit support evidence."""

        return {
            "status": "unsupported",
            "stressor_id": stressor_id,
            "reason": f"engine adapter {self.__class__.__name__} does not implement this stressor",
        }

    def drain_failure_events(self) -> list[FailureEventDraft | dict[str, Any]]:
        """Return and clear queued simulator-originated failure event drafts."""

        return []

    def failure_signal_capabilities(
        self, *, info: dict[str, Any] | None = None
    ) -> set[str]:
        """Return canonical signals available to passive failure detectors.

        Adapters should include signals they guarantee on every transition. The
        base implementation also exposes keys observed in the current info
        payload so third-party engines can participate without adapter changes.
        """

        capabilities = {"reward"}
        if info:
            capabilities.update(f"info.{key}" for key in info)
        return capabilities
