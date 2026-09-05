from __future__ import annotations

from importlib import import_module
from typing import Any

import numpy as np

from nyssa_bench.core.task import TaskSpec
from nyssa_bench.engines.base import NyssaEngine


class GenesisEngine(NyssaEngine):
    """Best-effort Genesis adapter.

    Genesis tasks should provide ``success.engine_factory.genesis`` as
    ``module:function``. The factory receives the TaskSpec and returns an
    environment with reset, step, render, and close methods.
    """

    def __init__(self) -> None:
        self.task_spec: TaskSpec | None = None
        self.env: Any | None = None
        self.max_steps = 1000

    def load_task(self, task_spec: TaskSpec) -> None:
        self.task_spec = task_spec
        self.max_steps = int(task_spec.success.get("max_steps", self.max_steps))
        factory = _resolve_factory(task_spec, "genesis")
        if factory:
            self.env = factory(task_spec)
            return

        try:
            import genesis as gs  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Genesis validation requires: "
                'python -m pip install "nyssa-bench[experimental]"'
            ) from exc

        raise RuntimeError(
            "Genesis is installed, but this task does not define success.engine_factory.genesis. "
            "Add a module:function factory that builds the Genesis scene for this task."
        )

    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        self._require_env()
        if hasattr(self.env, "reset"):
            try:
                observation = self.env.reset(seed=seed)
            except TypeError:
                observation = self.env.reset()
            if isinstance(observation, tuple) and len(observation) == 2:
                obs, info = observation
                return {"raw": obs}, dict(info or {})
            return {"raw": observation}, {"seed": seed}
        return {"raw": {}}, {"seed": seed}

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self._require_env()
        result = self.env.step(_coerce_action(action))
        if len(result) == 5:
            observation, reward, terminated, truncated, info = result
        elif len(result) == 4:
            observation, reward, done, info = result
            terminated, truncated = bool(done), False
        else:
            raise RuntimeError("Genesis factory env.step must return 4 or 5 values")
        info = dict(info or {})
        info.setdefault("success", bool(info.get("success", False)))
        info.setdefault("completion_time", float(info.get("step", 0.0)))
        info.setdefault("collision_count", float(info.get("collision_count", 0.0)))
        info.setdefault("path_efficiency", float(info.get("path_efficiency", max(0.0, min(1.0, float(reward))))))
        return {"raw": observation}, float(reward), bool(terminated), bool(truncated), info

    def render(self) -> Any:
        if self.env is not None and hasattr(self.env, "render"):
            return self.env.render()
        return None

    def get_state(self) -> dict[str, Any]:
        if self.env is not None and hasattr(self.env, "get_state"):
            return {"raw": self.env.get_state()}
        return {}

    def close(self) -> None:
        if self.env is not None and hasattr(self.env, "close"):
            self.env.close()

    def _require_env(self) -> None:
        if self.env is None:
            raise RuntimeError("No Genesis environment loaded. Call load_task first.")


def _resolve_factory(task_spec: TaskSpec, engine: str):
    factories = task_spec.success.get("engine_factory", {})
    factory_path = factories.get(engine) if isinstance(factories, dict) else None
    if not factory_path:
        return None
    module_name, _, attr = str(factory_path).partition(":")
    if not module_name or not attr:
        raise ValueError(f"Invalid {engine} engine factory '{factory_path}'. Use module:function.")
    return getattr(import_module(module_name), attr)


def _coerce_action(action: Any) -> Any:
    if isinstance(action, np.ndarray):
        return action
    if isinstance(action, (list, tuple)):
        return np.asarray(action, dtype=float)
    return action
