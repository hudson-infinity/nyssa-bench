from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from nyssa_bench.monitors.protocol import (
    FailureMonitorContract,
    MonitorInput,
    MonitorPrediction,
)


class FailureMonitor(ABC):
    @abstractmethod
    def contract(self) -> FailureMonitorContract: ...

    def reset(self, *, task: Any, episode_index: int, seed: int) -> None:
        return None

    @abstractmethod
    def predict(self, monitor_input: MonitorInput) -> MonitorPrediction: ...

    def get_state(self) -> Any:
        return None

    def set_state(self, state: Any) -> None:
        if state is not None:
            raise RuntimeError(
                f"{self.__class__.__name__} does not support monitor state restore"
            )

    def close(self) -> None:
        return None
