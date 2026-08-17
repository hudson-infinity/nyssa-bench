from __future__ import annotations

from collections import deque
from copy import deepcopy
from typing import Any

import numpy as np

from nyssa_bench.stressors.base import Stressor, StressorUnsupportedError, _jsonable
from nyssa_bench.stressors.protocol import StressorSpec
from nyssa_bench.stressors.registry import register_stressor


@register_stressor
class ImageBrightnessStressor(Stressor):
    stressor_id = "image_brightness"
    category = "visual"
    application_points = ("observation",)
    observable_by_policy = True
    privileged = False
    supported_engines = frozenset({"*"})
    supported_observation_modes = frozenset({"rgb", "rgbd", "sensor_data"})

    def resolve_parameters(self, spec: StressorSpec) -> dict[str, Any]:
        target_scale = float(spec.parameters.get("target_scale", 0.25))
        if not 0.0 <= target_scale <= 4.0:
            raise ValueError("target_scale must be within [0, 4]")
        scale = 1.0 + float(spec.severity) * (target_scale - 1.0)
        return {"scale": scale, "target_scale": target_scale}

    def transform_observation(self, observation: Any, *, step_index: int) -> Any:
        if not isinstance(observation, dict) or "raw" not in observation:
            raise StressorUnsupportedError(
                "wrapped observation does not expose a 'raw' payload"
            )
        raw, transformed_images = _scale_image_brightness(
            observation["raw"],
            scale=float(self.applied_parameters["scale"]),
            field_name=None,
        )
        if transformed_images == 0:
            raise StressorUnsupportedError(
                "observation contains no RGB image arrays for brightness scaling"
            )
        transformed = dict(observation)
        transformed["raw"] = raw
        return transformed


@register_stressor
class ObservationGaussianNoiseStressor(Stressor):
    stressor_id = "observation_gaussian_noise"
    category = "sensor"
    application_points = ("observation",)
    observable_by_policy = True
    privileged = False
    supported_engines = frozenset({"*"})

    def resolve_parameters(self, spec: StressorSpec) -> dict[str, Any]:
        max_std = _positive_float(spec.parameters.get("max_std", 0.1), "max_std")
        return {
            "std": float(spec.severity) * max_std,
            "max_std": max_std,
            "clip_min": _optional_float(spec.parameters.get("clip_min"), "clip_min"),
            "clip_max": _optional_float(spec.parameters.get("clip_max"), "clip_max"),
        }

    def transform_observation(self, observation: Any, *, step_index: int) -> Any:
        if not isinstance(observation, dict) or "raw" not in observation:
            raise StressorUnsupportedError(
                "wrapped observation does not expose a 'raw' payload"
            )
        transformed = dict(observation)
        transformed["raw"] = _add_gaussian_noise(
            observation["raw"],
            rng=self.rng,
            std=float(self.applied_parameters["std"]),
            clip_min=self.applied_parameters["clip_min"],
            clip_max=self.applied_parameters["clip_max"],
        )
        return transformed


@register_stressor
class ActionGaussianNoiseStressor(Stressor):
    stressor_id = "action_gaussian_noise"
    category = "action"
    application_points = ("action",)
    observable_by_policy = False
    privileged = True
    supported_engines = frozenset({"*"})

    def resolve_parameters(self, spec: StressorSpec) -> dict[str, Any]:
        max_std = _positive_float(spec.parameters.get("max_std", 0.15), "max_std")
        return {
            "std": float(spec.severity) * max_std,
            "max_std": max_std,
            "clip_to_action_space": True,
        }

    def transform_action(
        self, action: Any, *, observation: Any, step_index: int
    ) -> Any:
        transformed = _add_gaussian_noise(
            action,
            rng=self.rng,
            std=float(self.applied_parameters["std"]),
        )
        return _clip_action(transformed, observation)


@register_stressor
class ActionDelayStressor(Stressor):
    stressor_id = "action_delay"
    category = "system"
    application_points = ("action",)
    observable_by_policy = False
    privileged = True
    supported_engines = frozenset({"*"})

    def __init__(self) -> None:
        super().__init__()
        self._buffer: deque[Any] = deque()

    def reset(self, spec: StressorSpec, *, seed: int) -> None:
        super().reset(spec, seed=seed)
        self._buffer.clear()

    def resolve_parameters(self, spec: StressorSpec) -> dict[str, Any]:
        max_delay_steps = int(spec.parameters.get("max_delay_steps", 5))
        if max_delay_steps < 1:
            raise ValueError("max_delay_steps must be at least 1")
        delay_steps = int(round(float(spec.severity) * max_delay_steps))
        return {
            "delay_steps": delay_steps,
            "max_delay_steps": max_delay_steps,
            "initial_action": "zeros",
        }

    def transform_action(
        self, action: Any, *, observation: Any, step_index: int
    ) -> Any:
        delay_steps = int(self.applied_parameters["delay_steps"])
        self._buffer.append(deepcopy(action))
        if len(self._buffer) <= delay_steps:
            return _zeros_like(action)
        return self._buffer.popleft()

    def runtime_state(self) -> dict[str, Any]:
        return {"buffer": [_jsonable(action) for action in self._buffer]}

    def restore_runtime_state(
        self, state: dict[str, Any], *, engine: Any | None = None
    ) -> None:
        self._buffer = deque(deepcopy(state.get("buffer", [])))


@register_stressor
class FrictionScaleStressor(Stressor):
    stressor_id = "friction_scale"
    category = "dynamics"
    application_points = ("after_reset",)
    observable_by_policy = False
    privileged = True

    def __init__(self) -> None:
        super().__init__()
        self._backend_evidence: dict[str, Any] = {}

    def resolve_parameters(self, spec: StressorSpec) -> dict[str, Any]:
        target_scale = float(spec.parameters.get("target_scale", 0.25))
        if not 0.0 < target_scale <= 4.0:
            raise ValueError("target_scale must be within (0, 4]")
        scale = 1.0 + float(spec.severity) * (target_scale - 1.0)
        return {"scale": scale, "target_scale": target_scale}

    def after_reset(self, engine: Any, observation: Any) -> dict[str, Any]:
        apply_stressor = getattr(engine, "apply_stressor", None)
        if not callable(apply_stressor):
            raise StressorUnsupportedError(
                "engine does not implement runtime stressor application"
            )
        result = apply_stressor(self.stressor_id, dict(self.applied_parameters))
        if not isinstance(result, dict) or result.get("status") != "applied":
            reason = result.get("reason") if isinstance(result, dict) else None
            raise StressorUnsupportedError(
                str(reason or "backend did not confirm friction application")
            )
        self._backend_evidence = dict(result)
        return self._backend_evidence

    def runtime_state(self) -> dict[str, Any]:
        return {"backend_evidence": self._backend_evidence}

    def restore_runtime_state(
        self, state: dict[str, Any], *, engine: Any | None = None
    ) -> None:
        self._backend_evidence = dict(state.get("backend_evidence", {}))
        if engine is not None:
            self.after_reset(engine, observation=None)


def _add_gaussian_noise(
    value: Any,
    *,
    rng: np.random.Generator,
    std: float,
    clip_min: float | None = None,
    clip_max: float | None = None,
) -> Any:
    if isinstance(value, dict):
        return {
            key: _add_gaussian_noise(
                item, rng=rng, std=std, clip_min=clip_min, clip_max=clip_max
            )
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(
            _add_gaussian_noise(
                item, rng=rng, std=std, clip_min=clip_min, clip_max=clip_max
            )
            for item in value
        )
    if isinstance(value, list):
        return [
            _add_gaussian_noise(
                item, rng=rng, std=std, clip_min=clip_min, clip_max=clip_max
            )
            for item in value
        ]
    if isinstance(value, float):
        return _clip_scalar(value + float(rng.normal(0.0, std)), clip_min, clip_max)
    if isinstance(value, np.ndarray):
        if not np.issubdtype(value.dtype, np.floating):
            return value.copy()
        noisy = value + rng.normal(0.0, std, size=value.shape)
        return np.clip(noisy, clip_min, clip_max).astype(value.dtype, copy=False)
    if _is_floating_tensor(value):
        noise = rng.normal(0.0, std, size=tuple(value.shape))
        noisy = value + value.new_tensor(noise)
        if clip_min is not None or clip_max is not None:
            minimum = clip_min if clip_min is not None else float("-inf")
            maximum = clip_max if clip_max is not None else float("inf")
            noisy = noisy.clamp(min=minimum, max=maximum)
        return noisy
    return value


def _scale_image_brightness(
    value: Any, *, scale: float, field_name: str | None
) -> tuple[Any, int]:
    if isinstance(value, dict):
        transformed: dict[Any, Any] = {}
        count = 0
        for key, item in value.items():
            transformed_item, item_count = _scale_image_brightness(
                item,
                scale=scale,
                field_name=str(key),
            )
            transformed[key] = transformed_item
            count += item_count
        return transformed, count
    if isinstance(value, tuple):
        transformed_items = [
            _scale_image_brightness(item, scale=scale, field_name=field_name)
            for item in value
        ]
        return tuple(item for item, _ in transformed_items), sum(
            count for _, count in transformed_items
        )
    if isinstance(value, list):
        transformed_items = [
            _scale_image_brightness(item, scale=scale, field_name=field_name)
            for item in value
        ]
        return [item for item, _ in transformed_items], sum(
            count for _, count in transformed_items
        )
    if not _is_image_field(field_name) or not _looks_like_image(value):
        return value, 0
    if isinstance(value, np.ndarray):
        maximum = 255.0 if np.issubdtype(value.dtype, np.integer) else 1.0
        scaled_array = np.clip(value.astype(float) * scale, 0.0, maximum)
        return scaled_array.astype(value.dtype, copy=False), 1
    if _is_tensor(value):
        maximum = 255.0 if "uint8" in str(value.dtype) else 1.0
        scaled_tensor = (value * scale).clamp(min=0.0, max=maximum)
        return scaled_tensor.to(dtype=value.dtype), 1
    return value, 0


def _is_image_field(field_name: str | None) -> bool:
    if field_name is None:
        return True
    normalized = field_name.lower()
    return any(token in normalized for token in ("rgb", "image", "pixel", "color"))


def _looks_like_image(value: Any) -> bool:
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) < 3:
        return False
    dimensions = tuple(int(item) for item in shape)
    return dimensions[-1] in {1, 3, 4} or dimensions[-3] in {1, 3, 4}


def _is_tensor(value: Any) -> bool:
    return bool(
        hasattr(value, "dtype")
        and hasattr(value, "shape")
        and hasattr(value, "clamp")
        and hasattr(value, "to")
    )


def _clip_action(action: Any, observation: Any) -> Any:
    if not isinstance(observation, dict):
        return action
    action_space = observation.get("action_space")
    if not isinstance(action_space, dict):
        return action
    low = action_space.get("low")
    high = action_space.get("high")
    if low is None or high is None:
        return action
    try:
        clipped = np.clip(
            np.asarray(action, dtype=float),
            np.asarray(low, dtype=float),
            np.asarray(high, dtype=float),
        )
    except (TypeError, ValueError):
        return action
    return _restore_container_type(action, clipped)


def _restore_container_type(original: Any, value: np.ndarray) -> Any:
    if isinstance(original, np.ndarray):
        return value.astype(original.dtype, copy=False)
    if _is_floating_tensor(original):
        return original.new_tensor(value)
    if isinstance(original, tuple):
        return tuple(value.tolist())
    if isinstance(original, list):
        return value.tolist()
    if isinstance(original, (float, int)) and value.size == 1:
        return float(value.reshape(-1)[0])
    return value


def _zeros_like(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return np.zeros_like(value)
    if hasattr(value, "new_zeros") and hasattr(value, "shape"):
        return value.new_zeros(value.shape)
    if isinstance(value, tuple):
        return tuple(_zeros_like(item) for item in value)
    if isinstance(value, list):
        return [_zeros_like(item) for item in value]
    if isinstance(value, (int, float)):
        return type(value)(0)
    raise StressorUnsupportedError(
        f"cannot construct delayed initial action for {type(value).__name__}"
    )


def _is_floating_tensor(value: Any) -> bool:
    return bool(
        hasattr(value, "is_floating_point")
        and callable(value.is_floating_point)
        and value.is_floating_point()
        and hasattr(value, "new_tensor")
    )


def _clip_scalar(value: float, minimum: float | None, maximum: float | None) -> float:
    if minimum is not None:
        value = max(value, minimum)
    if maximum is not None:
        value = min(value, maximum)
    return value


def _positive_float(value: Any, name: str) -> float:
    result = float(value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _optional_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
