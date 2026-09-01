from __future__ import annotations

import os
from typing import Any

import numpy as np

from nyssa_bench.core.task import TaskSpec
from nyssa_bench.engines.base import NyssaEngine
from nyssa_bench.engines.spaces import wrap_observation


class MuJoCoEngine(NyssaEngine):
    """Adapter boundary for Gymnasium/MuJoCo environments."""

    def __init__(self) -> None:
        self.env: Any | None = None
        self.task_spec: TaskSpec | None = None
        self.max_steps = 1000
        self.episode_return = 0.0
        self.elapsed_steps = 0
        self._baseline_geom_friction: np.ndarray | None = None

    def load_task(self, task_spec: TaskSpec) -> None:
        self.task_spec = task_spec
        self.max_steps = int(task_spec.success.get("max_steps", self.max_steps))
        render_mode = task_spec.success.get("render_mode", "rgb_array")
        _configure_headless_mujoco_rendering(render_mode)
        try:
            import gymnasium as gym
            import mujoco  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Install NyssaBench with the MuJoCo extra: pip install -e '.[mujoco]'"
            ) from exc

        env_id = _resolve_env_id(task_spec, "mujoco")
        env_kwargs = {"render_mode": render_mode}
        self.env = _make_mujoco_env(gym, env_id, env_kwargs)
        self._capture_friction_baseline()

    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        self._require_env()
        self._restore_friction_baseline()
        self.episode_return = 0.0
        self.elapsed_steps = 0
        observation, info = self.env.reset(seed=seed)
        return wrap_observation(self.env, observation), dict(info)

    def step(
        self, action: Any
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self._require_env()
        action = self._coerce_action(action)
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.episode_return += float(reward)
        self.elapsed_steps = int(
            getattr(self.env, "_elapsed_steps", self.elapsed_steps + 1)
        )
        info = dict(info)
        info.setdefault("completion_time", float(self.elapsed_steps))
        info.setdefault("collision_count", 0.0)
        info.setdefault(
            "path_efficiency", max(0.0, min(1.0, (float(reward) + 10.0) / 10.0))
        )
        info["episode_return"] = self.episode_return
        info["success"] = _extract_success(
            info=info,
            reward=float(reward),
            episode_return=self.episode_return,
            elapsed_steps=self.elapsed_steps,
            terminated=bool(terminated),
            task_spec=self.task_spec,
        )
        return (
            wrap_observation(self.env, observation),
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    def render(self) -> Any:
        self._require_env()
        return self.env.render()

    def get_state(self) -> dict[str, Any]:
        target = getattr(self.env, "unwrapped", self.env)
        if target is not None and hasattr(target, "data"):
            return {"time": float(getattr(target.data, "time", 0.0))}
        return {}

    def failure_signal_capabilities(
        self, *, info: dict[str, Any] | None = None
    ) -> set[str]:
        capabilities = super().failure_signal_capabilities(info=info)
        capabilities.update(
            {
                "info.collision_count",
                "info.completion_time",
                "info.episode_return",
                "info.path_efficiency",
                "info.success",
            }
        )
        return capabilities

    def apply_stressor(
        self, stressor_id: str, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        if stressor_id != "friction_scale":
            return super().apply_stressor(stressor_id, parameters)
        self._require_env()
        target = getattr(self.env, "unwrapped", self.env)
        model = getattr(target, "model", None)
        friction = getattr(model, "geom_friction", None)
        if friction is None:
            return {
                "status": "unsupported",
                "stressor_id": stressor_id,
                "reason": "MuJoCo model does not expose geom_friction",
            }
        current = np.asarray(friction, dtype=float)
        if (
            self._baseline_geom_friction is None
            or self._baseline_geom_friction.shape != current.shape
        ):
            self._baseline_geom_friction = current.copy()
        scale = float(parameters["scale"])
        applied = self._baseline_geom_friction * scale
        friction[...] = applied
        return {
            "status": "applied",
            "stressor_id": stressor_id,
            "backend": "mujoco",
            "geom_count": int(applied.shape[0]) if applied.ndim else 1,
            "scale": scale,
            "baseline_min": float(self._baseline_geom_friction.min()),
            "baseline_max": float(self._baseline_geom_friction.max()),
            "applied_min": float(applied.min()),
            "applied_max": float(applied.max()),
        }

    def close(self) -> None:
        if self.env is not None:
            self.env.close()

    def _require_env(self) -> None:
        if self.env is None:
            raise RuntimeError("No MuJoCo environment loaded. Call load_task first.")

    def _coerce_action(self, action: Any) -> Any:
        if self.env is None or not hasattr(self.env, "action_space"):
            return action
        action_space = self.env.action_space
        if (
            hasattr(action_space, "shape")
            and action_space.shape
            and isinstance(action, (int, float))
        ):
            try:
                import numpy as np
            except ImportError:
                return action
            low = getattr(action_space, "low", None)
            high = getattr(action_space, "high", None)
            value = np.full(
                action_space.shape,
                float(action),
                dtype=getattr(action_space, "dtype", float),
            )
            if low is not None and high is not None:
                value = np.clip(value, low, high)
            return value
        return action

    def _capture_friction_baseline(self) -> None:
        target = getattr(self.env, "unwrapped", self.env)
        friction = getattr(getattr(target, "model", None), "geom_friction", None)
        if friction is not None:
            self._baseline_geom_friction = np.asarray(friction, dtype=float).copy()

    def _restore_friction_baseline(self) -> None:
        if self._baseline_geom_friction is None:
            return
        target = getattr(self.env, "unwrapped", self.env)
        friction = getattr(getattr(target, "model", None), "geom_friction", None)
        if (
            friction is not None
            and np.asarray(friction).shape == self._baseline_geom_friction.shape
        ):
            friction[...] = self._baseline_geom_friction


def _resolve_env_id(task_spec: TaskSpec, engine: str) -> str:
    engine_env_ids = task_spec.success.get("engine_env_ids", {})
    if isinstance(engine_env_ids, dict) and engine_env_ids.get(engine):
        return str(engine_env_ids[engine])
    legacy = task_spec.success.get(f"{engine}_env_id")
    if legacy:
        return str(legacy)
    raise RuntimeError(
        f"Task '{task_spec.task_id}' is missing success.engine_env_ids.{engine}. "
        "Real simulator tasks must define explicit environment mappings."
    )


def _configure_headless_mujoco_rendering(render_mode: Any) -> None:
    if str(render_mode) != "rgb_array":
        return
    if os.name == "nt" or os.environ.get("DISPLAY") or os.environ.get("MUJOCO_GL"):
        return
    os.environ["MUJOCO_GL"] = "egl"
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def _make_mujoco_env(gym: Any, env_id: str, env_kwargs: dict[str, Any]) -> Any:
    errors: list[BaseException] = []
    for candidate in _mujoco_env_id_candidates(env_id):
        try:
            return gym.make(candidate, **env_kwargs)
        except Exception as exc:
            if not _is_missing_gym_env_error(gym, exc):
                raise
            errors.append(exc)

    tried = ", ".join(_mujoco_env_id_candidates(env_id))
    cause = errors[-1] if errors else None
    raise RuntimeError(
        f"Could not create MuJoCo environment for '{env_id}'. Tried: {tried}. "
        "Install a Gymnasium/MuJoCo version that provides one of these environment IDs."
    ) from cause


def _mujoco_env_id_candidates(env_id: str) -> list[str]:
    fallback_versions = {
        "Reacher": ["v5", "v4", "v2"],
        "Pusher": ["v5", "v4", "v2"],
        "InvertedPendulum": ["v5", "v4", "v2"],
    }
    if "-" not in env_id:
        return [env_id]

    name, _, version = env_id.rpartition("-")
    versions = fallback_versions.get(name)
    if not versions:
        return [env_id]

    requested_version = _gym_env_version_number(version)
    candidates = [env_id]
    for fallback_version in versions:
        candidate = f"{name}-{fallback_version}"
        fallback_number = _gym_env_version_number(fallback_version)
        if candidate not in candidates and (
            requested_version is None
            or fallback_number is None
            or fallback_number <= requested_version
        ):
            candidates.append(candidate)
    return candidates


def _gym_env_version_number(version: str) -> int | None:
    if not version.startswith("v"):
        return None
    try:
        return int(version[1:])
    except ValueError:
        return None


def _is_missing_gym_env_error(gym: Any, exc: BaseException) -> bool:
    error_module = getattr(gym, "error", None)
    missing_error_types = tuple(
        error_type
        for error_type in (
            getattr(error_module, "VersionNotFound", None),
            getattr(error_module, "NameNotFound", None),
            getattr(error_module, "NamespaceNotFound", None),
        )
        if isinstance(error_type, type)
    )
    return bool(missing_error_types) and isinstance(exc, missing_error_types)


def _extract_success(
    *,
    info: dict[str, Any],
    reward: float,
    episode_return: float,
    elapsed_steps: int,
    terminated: bool,
    task_spec: TaskSpec | None,
) -> bool:
    configured_keys = []
    if task_spec is not None:
        configured = task_spec.success.get("success_info_keys", [])
        if isinstance(configured, str):
            configured_keys.append(configured)
        elif isinstance(configured, list):
            configured_keys.extend(str(key) for key in configured)
    for key in [*configured_keys, "success", "is_success"]:
        if key in info:
            return _as_bool(info[key])

    success_config = task_spec.success if task_spec is not None else {}
    metric = str(success_config.get("success_metric", "")).lower()
    if (
        metric in {"reward_threshold", "final_reward_threshold"}
        or "reward_threshold" in success_config
    ):
        return reward >= float(success_config.get("reward_threshold", 0.0))
    if (
        metric in {"return_threshold", "episode_return_threshold"}
        or "return_threshold" in success_config
    ):
        return episode_return >= float(success_config.get("return_threshold", 0.0))
    if (
        metric in {"survival_steps", "min_episode_steps"}
        or "min_success_steps" in success_config
    ):
        min_steps = int(
            success_config.get("min_success_steps", success_config.get("max_steps", 0))
        )
        return elapsed_steps >= min_steps and not terminated
    return False


def _as_bool(value: Any) -> bool:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "item"):
        try:
            return bool(value.item())
        except ValueError:
            pass
    if hasattr(value, "all"):
        return bool(value.all())
    return bool(value)
