from __future__ import annotations

import copy
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
            data = target.data
            state = {
                "qpos": np.asarray(data.qpos).copy(),
                "qvel": np.asarray(data.qvel).copy(),
                "time": float(getattr(data, "time", 0.0)),
                "episode_return": self.episode_return,
                "elapsed_steps": self.elapsed_steps,
                "wrapper_elapsed_steps": _capture_wrapper_elapsed_steps(self.env),
            }
            for name in (
                "act",
                "ctrl",
                "mocap_pos",
                "mocap_quat",
                "userdata",
                "qacc_warmstart",
                "qfrc_applied",
                "xfrc_applied",
            ):
                value = getattr(data, name, None)
                if value is not None:
                    state[name] = np.asarray(value).copy()
            rng = getattr(target, "np_random", None)
            if rng is not None and hasattr(rng, "bit_generator"):
                state["env_rng_state"] = copy.deepcopy(rng.bit_generator.state)
            integration_state = _capture_mujoco_integration_state(target)
            if integration_state is not None:
                state["integration_state"] = integration_state
            return state
        return {}

    def set_state(self, state: Any) -> dict[str, Any] | None:
        self._require_env()
        if not isinstance(state, dict) or not {"qpos", "qvel"} <= set(state):
            raise ValueError("MuJoCo state requires qpos and qvel")
        target = getattr(self.env, "unwrapped", self.env)
        data = target.data
        integration_state = state.get("integration_state")
        if integration_state is not None:
            _restore_mujoco_integration_state(target, integration_state)
        else:
            _restore_array(data.qpos, state["qpos"], name="qpos")
            _restore_array(data.qvel, state["qvel"], name="qvel")
            for name in (
                "act",
                "ctrl",
                "mocap_pos",
                "mocap_quat",
                "userdata",
                "qacc_warmstart",
                "qfrc_applied",
                "xfrc_applied",
            ):
                target_array = getattr(data, name, None)
                if name in state and target_array is not None:
                    _restore_array(target_array, state[name], name=name)
            data.time = float(state.get("time", data.time))
        self.episode_return = float(state.get("episode_return", 0.0))
        self.elapsed_steps = int(state.get("elapsed_steps", 0))
        _restore_wrapper_elapsed_steps(
            self.env,
            state.get("wrapper_elapsed_steps", [self.elapsed_steps]),
        )
        rng_state = state.get("env_rng_state")
        rng = getattr(target, "np_random", None)
        if rng_state is not None and rng is not None and hasattr(rng, "bit_generator"):
            rng.bit_generator.state = copy.deepcopy(rng_state)
        try:
            import mujoco

            if target.model is not None:
                mujoco.mj_forward(target.model, data)
        except ImportError:
            pass
        observation = _mujoco_observation(target)
        return (
            wrap_observation(self.env, observation) if observation is not None else None
        )

    def state_restore_capability(self) -> dict[str, Any]:
        target = getattr(self.env, "unwrapped", self.env)
        supported = target is not None and all(
            hasattr(target, name) for name in ("data", "model")
        )
        exact_integration = bool(
            supported and _mujoco_integration_state_supported(target)
        )
        captures_rng = bool(
            supported and hasattr(getattr(target, "np_random", None), "bit_generator")
        )
        exact = exact_integration and captures_rng
        return {
            "supported": supported,
            "fidelity": (
                "exact_mujoco_integration_state_and_rng"
                if exact
                else "qualified_mujoco_physics_state"
                if supported
                else "unsupported"
            ),
            "captures_rng": captures_rng,
            "exact": exact,
            "reason": None
            if exact
            else "MuJoCo integration-state API or environment RNG is unavailable"
            if supported
            else "MuJoCo model/data are unavailable",
        }

    def seed_branch_rng(self, seed: int) -> bool:
        target = getattr(self.env, "unwrapped", self.env)
        rng = getattr(target, "np_random", None)
        bit_generator = getattr(rng, "bit_generator", None)
        if bit_generator is None:
            return False
        bit_generator.state = copy.deepcopy(
            np.random.default_rng(int(seed)).bit_generator.state
        )
        return True

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


def _mujoco_observation(target: Any) -> Any | None:
    for name in ("_get_obs", "get_obs"):
        method = getattr(target, name, None)
        if callable(method):
            try:
                return method()
            except TypeError:
                continue
    return None


def _capture_mujoco_integration_state(target: Any) -> dict[str, Any] | None:
    model = getattr(target, "model", None)
    data = getattr(target, "data", None)
    if model is None or data is None:
        return None
    try:
        import mujoco

        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        values = np.empty(mujoco.mj_stateSize(model, spec), dtype=np.float64)
        mujoco.mj_getState(model, data, values, spec)
    except (ImportError, AttributeError, TypeError, ValueError):
        return None
    return {"spec": int(spec), "values": values}


def _restore_mujoco_integration_state(target: Any, state: Any) -> None:
    if not isinstance(state, dict) or "spec" not in state or "values" not in state:
        raise ValueError("MuJoCo integration state must contain spec and values")
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError(
            "MuJoCo is required to restore a captured integration state"
        ) from exc
    spec = mujoco.mjtState(int(state["spec"]))
    values = np.asarray(state["values"], dtype=np.float64)
    expected_size = mujoco.mj_stateSize(target.model, spec)
    if values.shape != (expected_size,):
        raise ValueError(
            "MuJoCo integration state has shape "
            f"{values.shape}; expected {(expected_size,)}"
        )
    mujoco.mj_setState(target.model, target.data, values, spec)


def _mujoco_integration_state_supported(target: Any) -> bool:
    return _capture_mujoco_integration_state(target) is not None


def _restore_array(target: Any, value: Any, *, name: str) -> None:
    source = np.asarray(value, dtype=target.dtype)
    if source.shape != target.shape:
        raise ValueError(
            f"MuJoCo {name} state has shape {source.shape}; expected {target.shape}"
        )
    target[...] = source


def _wrapper_chain(env: Any) -> list[Any]:
    chain: list[Any] = []
    seen: set[int] = set()
    current = env
    while current is not None and id(current) not in seen:
        chain.append(current)
        seen.add(id(current))
        current = getattr(current, "env", None)
    return chain


def _capture_wrapper_elapsed_steps(env: Any) -> list[Any]:
    return [
        copy.deepcopy(getattr(wrapper, "_elapsed_steps", None))
        for wrapper in _wrapper_chain(env)
    ]


def _restore_wrapper_elapsed_steps(env: Any, values: Any) -> None:
    if not isinstance(values, list):
        raise ValueError("MuJoCo wrapper elapsed-step state must be a list")
    chain = _wrapper_chain(env)
    if len(values) == 1 and len(chain) > 1:
        values = [values[0], *([None] * (len(chain) - 1))]
    if len(chain) != len(values):
        raise RuntimeError("MuJoCo wrapper chain changed after state capture")
    for wrapper, value in zip(chain, values, strict=True):
        if value is not None:
            setattr(wrapper, "_elapsed_steps", copy.deepcopy(value))
