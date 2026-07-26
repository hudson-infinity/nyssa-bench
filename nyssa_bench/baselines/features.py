from __future__ import annotations

from typing import Any

import numpy as np


def action_bounds(observation: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    action_space = observation.get("action_space", {})
    shape = tuple(int(value) for value in action_space.get("shape", [1]))
    low = np.asarray(action_space.get("low", [-1.0] * int(np.prod(shape))), dtype=float).reshape(shape)
    high = np.asarray(action_space.get("high", [1.0] * int(np.prod(shape))), dtype=float).reshape(shape)
    low = np.where(np.isfinite(low), low, -1.0)
    high = np.where(np.isfinite(high), high, 1.0)
    return low, high, shape


def flatten_observation(observation: dict[str, Any], max_dim: int = 256) -> np.ndarray:
    values = observation_numeric_values(observation)
    if len(values) >= max_dim:
        return np.asarray(values[:max_dim], dtype=float)
    return np.asarray([*values, *([0.0] * (max_dim - len(values)))], dtype=float)


def observation_numeric_values(observation: dict[str, Any]) -> list[float]:
    values: list[float] = []
    _collect_numbers(observation.get("raw", observation), values)
    return values


def normalize_action(action: Any, size: int) -> np.ndarray:
    values: list[float] = []
    _collect_numbers(action, values)
    if len(values) >= size:
        return np.asarray(values[:size], dtype=float)
    return np.asarray([*values, *([0.0] * (size - len(values)))], dtype=float)


def fit_action_to_observation(action: Any, observation: dict[str, Any]) -> np.ndarray:
    low, high, shape = action_bounds(observation)
    flat = normalize_action(action, int(np.prod(shape)))
    return np.clip(flat.reshape(shape), low, high)


def find_vector(observation: dict[str, Any], names: tuple[str, ...], min_size: int = 3) -> np.ndarray | None:
    found = _find_named_value(observation.get("raw", observation), names)
    if found is None:
        return None
    values: list[float] = []
    _collect_numbers(found, values)
    if len(values) < min_size:
        return None
    return np.asarray(values[:min_size], dtype=float)


def _find_named_value(value: Any, names: tuple[str, ...]) -> Any | None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).lower()
            if any(name in normalized for name in names):
                return item
        for item in value.values():
            found = _find_named_value(item, names)
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for item in value:
            found = _find_named_value(item, names)
            if found is not None:
                return found
    return None


def _collect_numbers(value: Any, output: list[float]) -> None:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        for key in sorted(value):
            _collect_numbers(value[key], output)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _collect_numbers(item, output)
    elif isinstance(value, (int, float, bool)):
        output.append(float(value))
