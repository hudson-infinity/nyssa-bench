from __future__ import annotations

import os
from typing import Any

from nyssa_bench.core.task import TaskSpec
from nyssa_bench.engines.base import NyssaEngine
from nyssa_bench.engines.spaces import wrap_observation


class ManiSkillEngine(NyssaEngine):
    """Adapter boundary for ManiSkill environments."""

    def __init__(self) -> None:
        self.env: Any | None = None
        self.task_spec: TaskSpec | None = None
        self.max_steps = 1000
        self._baseline_friction_materials: list[tuple[Any, float, float]] = []

    def load_task(self, task_spec: TaskSpec) -> None:
        self.task_spec = task_spec
        self.max_steps = int(task_spec.success.get("max_steps", self.max_steps))
        try:
            import gymnasium as gym
            import mani_skill  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Install NyssaBench with the ManiSkill extra: pip install -e '.[maniskill]'"
            ) from exc

        env_id = _resolve_env_id(task_spec, "maniskill")
        env_kwargs = _maniskill_env_kwargs(task_spec)
        env_kwargs.setdefault("max_episode_steps", self.max_steps)
        self.env = gym.make(env_id, **env_kwargs)
        self._capture_friction_baseline()

    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        self._require_env()
        self._restore_friction_baseline()
        observation, info = self.env.reset(seed=seed)
        self._capture_friction_baseline()
        return wrap_observation(self.env, observation), dict(info)

    def step(
        self, action: Any
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self._require_env()
        action = self._coerce_action(action)
        observation, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["success"] = _extract_success(info, self.task_spec)
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
        if self.env is not None and hasattr(self.env, "get_state"):
            return {"raw": self.env.get_state()}
        return {}

    def set_state(self, state: Any) -> dict[str, Any] | None:
        self._require_env()
        target = getattr(self.env, "unwrapped", self.env)
        state_payload = _to_numpy_state(state)
        if isinstance(state_payload, dict) and hasattr(target, "set_state_dict"):
            target.set_state_dict(state_payload)
        elif hasattr(target, "set_state"):
            target.set_state(state_payload)
        else:
            raise RuntimeError(
                "Loaded ManiSkill environment does not support state restore."
            )
        observation = _get_observation_after_state_restore(self.env)
        return (
            wrap_observation(self.env, observation) if observation is not None else None
        )

    def apply_stressor(
        self, stressor_id: str, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        if stressor_id != "friction_scale":
            return super().apply_stressor(stressor_id, parameters)
        self._require_env()
        target = getattr(self.env, "unwrapped", self.env)
        scene = getattr(target, "scene", None)
        if bool(getattr(scene, "gpu_sim_enabled", False)):
            return {
                "status": "unsupported",
                "stressor_id": stressor_id,
                "reason": "ManiSkill GPU simulation cannot safely mutate PhysX materials after scene creation",
            }
        self._capture_friction_baseline()
        if not self._baseline_friction_materials:
            return {
                "status": "unsupported",
                "stressor_id": stressor_id,
                "reason": "ManiSkill scene exposes no mutable collision materials",
            }
        scale = float(parameters["scale"])
        before = []
        after = []
        for (
            material,
            static_friction,
            dynamic_friction,
        ) in self._baseline_friction_materials:
            before.extend([static_friction, dynamic_friction])
            scaled_static = static_friction * scale
            scaled_dynamic = dynamic_friction * scale
            _set_material_friction(
                material, static_friction=scaled_static, dynamic_friction=scaled_dynamic
            )
            after.extend([scaled_static, scaled_dynamic])
        return {
            "status": "applied",
            "stressor_id": stressor_id,
            "backend": "maniskill_cpu",
            "material_count": len(self._baseline_friction_materials),
            "scale": scale,
            "baseline_min": min(before),
            "baseline_max": max(before),
            "applied_min": min(after),
            "applied_max": max(after),
        }

    def close(self) -> None:
        if self.env is not None:
            self.env.close()

    def _require_env(self) -> None:
        if self.env is None:
            raise RuntimeError("No ManiSkill environment loaded. Call load_task first.")

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
        materials = _maniskill_collision_materials(self.env)
        existing = {
            id(material): (material, static, dynamic)
            for material, static, dynamic in self._baseline_friction_materials
        }
        self._baseline_friction_materials = [
            existing.get(id(material), (material, static, dynamic))
            for material, static, dynamic in materials
        ]

    def _restore_friction_baseline(self) -> None:
        for (
            material,
            static_friction,
            dynamic_friction,
        ) in self._baseline_friction_materials:
            _set_material_friction(
                material,
                static_friction=static_friction,
                dynamic_friction=dynamic_friction,
            )


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


def _maniskill_env_kwargs(task_spec: TaskSpec) -> dict[str, Any]:
    render_mode = _env_or_task_value(
        "NYSSA_MANISKILL_RENDER_MODE",
        task_spec,
        "render_mode",
        "rgb_array",
    )
    env_kwargs: dict[str, Any] = {}
    if str(render_mode).lower() not in {"", "none", "null"}:
        env_kwargs["render_mode"] = render_mode
    for env_name, key in (
        ("NYSSA_MANISKILL_OBS_MODE", "obs_mode"),
        ("NYSSA_MANISKILL_CONTROL_MODE", "control_mode"),
        ("NYSSA_MANISKILL_ROBOT_UIDS", "robot_uids"),
        ("NYSSA_MANISKILL_SIM_BACKEND", "sim_backend"),
        ("NYSSA_MANISKILL_RENDER_DEVICE", "render_device"),
        ("NYSSA_MANISKILL_SHADER_DIR", "shader_dir"),
    ):
        value = _env_or_task_value(env_name, task_spec, key, None)
        if value is not None and str(value).lower() not in {"", "none", "null"}:
            env_kwargs[key] = value
    max_episode_steps = _env_or_task_value(
        "NYSSA_MANISKILL_MAX_EPISODE_STEPS", task_spec, "max_steps", None
    )
    if max_episode_steps is not None and str(max_episode_steps).lower() not in {
        "",
        "none",
        "null",
    }:
        env_kwargs["max_episode_steps"] = int(max_episode_steps)
    return env_kwargs


def _env_or_task_value(
    env_name: str, task_spec: TaskSpec, key: str, default: Any
) -> Any:
    value = os.getenv(env_name)
    if value is not None:
        return value
    return task_spec.success.get(key, default)


def _extract_success(info: dict[str, Any], task_spec: TaskSpec | None) -> bool:
    configured_keys = []
    if task_spec is not None:
        configured = task_spec.success.get("success_info_keys", [])
        if isinstance(configured, str):
            configured_keys.append(configured)
        elif isinstance(configured, list):
            configured_keys.extend(str(key) for key in configured)
    for key in [*configured_keys, "success", "is_success", "success_once"]:
        if key in info:
            return _as_bool(info[key])
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


def _to_numpy_state(value: Any) -> Any:
    try:
        import numpy as np
    except ImportError:
        return value
    if isinstance(value, dict):
        if set(value) == {"raw"}:
            return _to_numpy_state(value["raw"])
        for key in ("env_states", "states", "state"):
            if key in value:
                return _to_numpy_state(value[key])
        return {key: _to_numpy_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return np.asarray(value)
    return value


def _get_observation_after_state_restore(env: Any) -> Any | None:
    for target in (env, getattr(env, "unwrapped", None)):
        if target is None:
            continue
        for name in ("get_obs", "_get_obs"):
            method = getattr(target, name, None)
            if method is None:
                continue
            try:
                return method()
            except TypeError:
                continue
    return None


def _maniskill_collision_materials(env: Any) -> list[tuple[Any, float, float]]:
    target = getattr(env, "unwrapped", env)
    scene = getattr(target, "scene", None)
    if scene is None:
        return []
    bodies: list[Any] = []
    for views_name in ("actor_views", "articulation_views"):
        views = getattr(scene, views_name, {})
        values = views.values() if isinstance(views, dict) else views or []
        for view in values:
            bodies.extend(getattr(view, "_bodies", []) or [])
            for link in getattr(view, "links", []) or []:
                bodies.extend(getattr(link, "_bodies", []) or [])

    materials: dict[int, tuple[Any, float, float]] = {}
    for body in bodies:
        get_shapes = getattr(body, "get_collision_shapes", None)
        shapes = (
            get_shapes()
            if callable(get_shapes)
            else getattr(body, "collision_shapes", [])
        )
        for shape in shapes or []:
            material = _collision_material(shape)
            if material is None or id(material) in materials:
                continue
            static_friction = _get_material_value(material, "static_friction")
            dynamic_friction = _get_material_value(material, "dynamic_friction")
            if static_friction is None or dynamic_friction is None:
                continue
            materials[id(material)] = (material, static_friction, dynamic_friction)
    return list(materials.values())


def _collision_material(shape: Any) -> Any | None:
    for attribute in ("physical_material", "material"):
        material = getattr(shape, attribute, None)
        if material is not None:
            return material
    getter = getattr(shape, "get_physical_material", None)
    return getter() if callable(getter) else None


def _get_material_value(material: Any, name: str) -> float | None:
    value = getattr(material, name, None)
    if value is None:
        getter = getattr(material, f"get_{name}", None)
        value = getter() if callable(getter) else None
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _set_material_friction(
    material: Any, *, static_friction: float, dynamic_friction: float
) -> None:
    for name, value in (
        ("static_friction", static_friction),
        ("dynamic_friction", dynamic_friction),
    ):
        setter = getattr(material, f"set_{name}", None)
        if callable(setter):
            setter(float(value))
        else:
            setattr(material, name, float(value))
