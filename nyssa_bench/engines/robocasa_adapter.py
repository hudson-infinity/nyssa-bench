from __future__ import annotations

from importlib import import_module
from typing import Any

import numpy as np

from nyssa_bench.core.task import TaskSpec
from nyssa_bench.engines.base import NyssaEngine


class RoboCasaEngine(NyssaEngine):
    """Best-effort RoboCasa/robosuite adapter.

    Tasks can provide either ``success.engine_factory.robocasa`` as
    ``module:function`` or ``success.engine_env_ids.robocasa`` for robosuite.
    """

    def __init__(self) -> None:
        self.task_spec: TaskSpec | None = None
        self.env: Any | None = None
        self.max_steps = 1000

    def load_task(self, task_spec: TaskSpec) -> None:
        self.task_spec = task_spec
        self.max_steps = int(task_spec.success.get("max_steps", self.max_steps))
        factory = _resolve_factory(task_spec, "robocasa")
        if factory:
            self.env = factory(task_spec)
            return

        env_id = _resolve_env_id(task_spec, "robocasa")
        if not env_id:
            raise RuntimeError(
                "RoboCasa task mapping is missing. Add success.engine_env_ids.robocasa "
                "or success.engine_factory.robocasa before running this adapter."
            )

        try:
            import robosuite as suite
            import robocasa  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "RoboCasa validation requires upstream RoboCasa/robosuite setup. "
                "Install RoboCasa from its upstream source, configure its macros and "
                "assets, then retry. See docs/installation.md#extras."
            ) from exc

        self.env = suite.make(
            env_name=env_id,
            robots=task_spec.robot,
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
        )

    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        self._require_env()
        if seed is not None and hasattr(self.env, "seed"):
            self.env.seed(seed)
        observation = self.env.reset()
        return {"raw": observation}, {"seed": seed}

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self._require_env()
        observation, reward, done, info = self.env.step(_coerce_action(self.env, action))
        info = dict(info or {})
        info.setdefault("success", bool(info.get("success", info.get("task_success", False))))
        info.setdefault("completion_time", float(info.get("timestep", 0.0)))
        info.setdefault("collision_count", float(info.get("collision_count", 0.0)))
        info.setdefault("path_efficiency", float(info.get("path_efficiency", max(0.0, min(1.0, float(reward))))))
        return {"raw": observation}, float(reward), bool(done), False, info

    def render(self) -> Any:
        if self.env is None:
            return None
        if hasattr(self.env, "render"):
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
            raise RuntimeError("No RoboCasa environment loaded. Call load_task first.")


def _resolve_env_id(task_spec: TaskSpec, engine: str) -> str | None:
    engine_env_ids = task_spec.success.get("engine_env_ids", {})
    if isinstance(engine_env_ids, dict) and engine_env_ids.get(engine):
        return str(engine_env_ids[engine])
    value = task_spec.success.get(f"{engine}_env_id")
    return str(value) if value else None


def _resolve_factory(task_spec: TaskSpec, engine: str):
    factories = task_spec.success.get("engine_factory", {})
    factory_path = factories.get(engine) if isinstance(factories, dict) else None
    if not factory_path:
        return None
    module_name, _, attr = str(factory_path).partition(":")
    if not module_name or not attr:
        raise ValueError(f"Invalid {engine} engine factory '{factory_path}'. Use module:function.")
    return getattr(import_module(module_name), attr)


def _coerce_action(env: Any, action: Any) -> Any:
    if isinstance(action, np.ndarray):
        return action
    if isinstance(action, (list, tuple)):
        return np.asarray(action, dtype=float)
    action_dim = getattr(env, "action_dim", None)
    if action_dim:
        return np.full(int(action_dim), float(action), dtype=float)
    return action
