from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from nyssa_bench.core.task import TaskSpec


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
