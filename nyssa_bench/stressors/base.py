from __future__ import annotations

from abc import ABC
from typing import Any, ClassVar

import numpy as np

from nyssa_bench.failures.protocol import FailureEventDraft
from nyssa_bench.stressors.protocol import StressorContext, StressorSpec


class StressorUnsupportedError(RuntimeError):
    pass


class Stressor(ABC):
    stressor_id: ClassVar[str]
    category: ClassVar[str]
    application_points: ClassVar[tuple[str, ...]]
    severity_domain: ClassVar[tuple[float, float]] = (0.0, 1.0)
    lifetime: ClassVar[str] = "episode"
    supported_engines: ClassVar[frozenset[str]] = frozenset({"maniskill", "mujoco"})
    supported_tasks: ClassVar[frozenset[str] | None] = None
    supported_observation_modes: ClassVar[frozenset[str] | None] = None
    supported_action_modes: ClassVar[frozenset[str] | None] = None
    observable_by_policy: ClassVar[bool] = False
    privileged: ClassVar[bool] = True
    conflicts_with: ClassVar[frozenset[str]] = frozenset()

    def __init__(self) -> None:
        self.spec: StressorSpec | None = None
        self.seed = 0
        self.rng = np.random.default_rng(0)
        self.applied_parameters: dict[str, Any] = {}

    def reset(self, spec: StressorSpec, *, seed: int) -> None:
        self.spec = spec
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.applied_parameters = self.resolve_parameters(spec)

    def resolve_parameters(self, spec: StressorSpec) -> dict[str, Any]:
        return dict(spec.parameters)

    def support_reason(self, context: StressorContext) -> str | None:
        assert self.spec is not None
        lower, upper = self.severity_domain
        if not lower <= self.spec.severity <= upper:
            return (
                f"severity {self.spec.severity} is outside the supported domain "
                f"[{lower}, {upper}]"
            )
        if (
            "*" not in self.supported_engines
            and context.engine_name not in self.supported_engines
        ):
            return f"engine '{context.engine_name}' is not supported"
        if (
            self.supported_tasks is not None
            and context.task_id not in self.supported_tasks
        ):
            return f"task '{context.task_id}' is not supported"
        if (
            self.supported_observation_modes is not None
            and context.observation_mode not in self.supported_observation_modes
        ):
            return f"observation mode '{context.observation_mode}' is not supported"
        if (
            self.supported_action_modes is not None
            and context.action_mode not in self.supported_action_modes
        ):
            return f"action mode '{context.action_mode}' is not supported"
        return None

    def before_reset(self, engine: Any) -> None:
        return None

    def after_reset(self, engine: Any, observation: Any) -> dict[str, Any]:
        return {}

    def transform_observation(self, observation: Any, *, step_index: int) -> Any:
        return observation

    def transform_action(
        self, action: Any, *, observation: Any, step_index: int
    ) -> Any:
        return action

    def before_step(self, engine: Any, *, step_index: int) -> None:
        return None

    def after_step(self, engine: Any, info: dict[str, Any], *, step_index: int) -> None:
        return None

    def get_state(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "rng_state": _jsonable(self.rng.bit_generator.state),
            "runtime": self.runtime_state(),
        }

    def runtime_state(self) -> dict[str, Any]:
        return {}

    def set_state(self, state: dict[str, Any], *, engine: Any | None = None) -> None:
        self.seed = int(state.get("seed", self.seed))
        rng_state = state.get("rng_state")
        if isinstance(rng_state, dict):
            self.rng.bit_generator.state = rng_state
        runtime = state.get("runtime", {})
        if isinstance(runtime, dict):
            self.restore_runtime_state(runtime, engine=engine)

    def restore_runtime_state(
        self, state: dict[str, Any], *, engine: Any | None = None
    ) -> None:
        return None

    def drain_failure_events(self) -> list[FailureEventDraft | dict[str, Any]]:
        """Return and clear queued stressor-originated event drafts."""

        return []


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
