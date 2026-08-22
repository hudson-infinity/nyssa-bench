from __future__ import annotations

from typing import Any

import numpy as np

ACTION_TRANSFORM_FORMAT = "nyssa-action-minmax-v1"


def action_bounds(
    observation: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    action_space = observation.get("action_space", {})
    shape = tuple(int(value) for value in action_space.get("shape", [1]))
    low = np.asarray(
        action_space.get("low", [-1.0] * int(np.prod(shape))), dtype=float
    ).reshape(shape)
    high = np.asarray(
        action_space.get("high", [1.0] * int(np.prod(shape))), dtype=float
    ).reshape(shape)
    low = np.where(np.isfinite(low), low, -1.0)
    high = np.where(np.isfinite(high), high, 1.0)
    return low, high, shape


def flatten_observation(observation: dict[str, Any], max_dim: int = 256) -> np.ndarray:
    if max_dim < 0:
        raise ValueError("max_dim must be non-negative")
    values: list[float] = []
    _collect_numbers(observation.get("raw", observation), values, limit=max_dim)
    result = np.zeros(max_dim, dtype=float)
    if values:
        result[: len(values)] = values
    return result


def observation_numeric_values(
    observation: dict[str, Any],
    *,
    max_values: int | None = None,
) -> list[float]:
    if max_values is not None and max_values < 0:
        raise ValueError("max_values must be non-negative")
    values: list[float] = []
    _collect_numbers(
        observation.get("raw", observation),
        values,
        limit=max_values,
    )
    return values


def normalize_action(action: Any, size: int) -> np.ndarray:
    values: list[float] = []
    _collect_numbers(action, values, limit=size)
    if len(values) >= size:
        return np.asarray(values[:size], dtype=float)
    return np.asarray([*values, *([0.0] * (size - len(values)))], dtype=float)


def action_space_contract(observation: dict[str, Any]) -> dict[str, Any]:
    action_space = observation.get("action_space")
    if not isinstance(action_space, dict) or action_space.get("type") != "box":
        raise ValueError(
            "RoboMimic requires an explicit Box action_space observation contract"
        )
    missing = [key for key in ("shape", "low", "high") if key not in action_space]
    if missing:
        raise ValueError(
            f"RoboMimic action_space contract is missing: {', '.join(missing)}"
        )

    shape = tuple(int(value) for value in action_space["shape"])
    if any(value <= 0 for value in shape):
        raise ValueError(f"RoboMimic action_space has invalid shape {shape}")
    size = int(np.prod(shape)) if shape else 1
    low = np.asarray(action_space["low"], dtype=float).reshape(-1)
    high = np.asarray(action_space["high"], dtype=float).reshape(-1)
    if low.size != size or high.size != size:
        raise ValueError(
            f"RoboMimic action bounds do not match shape {shape}: "
            f"low={low.size}, high={high.size}, expected={size}"
        )
    if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
        raise ValueError("RoboMimic action normalization requires finite action bounds")
    if np.any(high <= low):
        raise ValueError(
            "RoboMimic action normalization requires high > low in every dimension"
        )
    return {
        "format": ACTION_TRANSFORM_FORMAT,
        "shape": list(shape),
        "low": low.tolist(),
        "high": high.tolist(),
    }


def normalize_action_to_unit(action: Any, observation: dict[str, Any]) -> np.ndarray:
    """Map a bounded environment action to RoboMimic's [-1, 1] convention."""

    contract = action_space_contract(observation)
    shape, low, high = _contract_arrays(contract)
    flat = _strict_action_vector(action, low.size)
    tolerance = 1e-6 * np.maximum(1.0, high - low)
    if np.any(flat < low - tolerance) or np.any(flat > high + tolerance):
        raise ValueError(
            "Demonstration action falls outside its recorded action_space bounds"
        )
    clipped = np.clip(flat, low, high)
    unit = 2.0 * (clipped - low) / (high - low) - 1.0
    return unit.reshape(shape)


def denormalize_action_from_unit(
    action: Any,
    observation: dict[str, Any],
    *,
    expected_contract: dict[str, Any] | None = None,
) -> np.ndarray:
    """Map a RoboMimic [-1, 1] action into the live environment action space."""

    live_contract = action_space_contract(observation)
    if expected_contract is not None:
        validate_action_space_contract(live_contract, expected_contract)
    shape, low, high = _contract_arrays(live_contract)
    unit = np.clip(_strict_action_vector(action, low.size), -1.0, 1.0)
    environment_action = low + 0.5 * (unit + 1.0) * (high - low)
    return environment_action.reshape(shape)


def validate_action_space_contract(
    actual: dict[str, Any], expected: dict[str, Any]
) -> None:
    if expected.get("format") != ACTION_TRANSFORM_FORMAT:
        raise ValueError(
            f"Unsupported RoboMimic action transform: {expected.get('format')!r}"
        )
    actual_shape, actual_low, actual_high = _contract_arrays(actual)
    expected_shape, expected_low, expected_high = _contract_arrays(expected)
    if actual_shape != expected_shape:
        raise ValueError(
            f"Live action shape {actual_shape} does not match training action shape {expected_shape}"
        )
    if not np.allclose(
        actual_low, expected_low, rtol=1e-6, atol=1e-6
    ) or not np.allclose(actual_high, expected_high, rtol=1e-6, atol=1e-6):
        raise ValueError(
            "Live action bounds do not match the RoboMimic training action bounds"
        )


def fit_action_to_observation(action: Any, observation: dict[str, Any]) -> np.ndarray:
    low, high, shape = action_bounds(observation)
    flat = normalize_action(action, int(np.prod(shape)))
    return np.clip(flat.reshape(shape), low, high)


def _contract_arrays(
    contract: dict[str, Any],
) -> tuple[tuple[int, ...], np.ndarray, np.ndarray]:
    shape = tuple(int(value) for value in contract.get("shape", []))
    size = int(np.prod(shape)) if shape else 1
    low = np.asarray(contract.get("low", []), dtype=float).reshape(-1)
    high = np.asarray(contract.get("high", []), dtype=float).reshape(-1)
    if low.size != size or high.size != size:
        raise ValueError(f"Invalid action contract bounds for shape {shape}")
    return shape, low, high


def _strict_action_vector(action: Any, size: int) -> np.ndarray:
    values: list[float] = []
    _collect_numbers(action, values)
    if len(values) != size:
        raise ValueError(f"Action has {len(values)} values; expected exactly {size}")
    vector = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(vector)):
        raise ValueError("Action contains non-finite values")
    return vector


def find_vector(
    observation: dict[str, Any], names: tuple[str, ...], min_size: int = 3
) -> np.ndarray | None:
    found = _find_named_value(observation.get("raw", observation), names)
    if found is None:
        return None
    values: list[float] = []
    _collect_numbers(found, values, limit=min_size)
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


def _collect_numbers(
    value: Any,
    output: list[float],
    *,
    limit: int | None = None,
) -> None:
    if limit is not None and len(output) >= limit:
        return
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if isinstance(value, np.ndarray) and (
        np.issubdtype(value.dtype, np.integer)
        or np.issubdtype(value.dtype, np.floating)
        or np.issubdtype(value.dtype, np.bool_)
    ):
        flat = value.reshape(-1)
        if limit is not None:
            flat = flat[: max(0, limit - len(output))]
        output.extend(flat.astype(float, copy=False).tolist())
        return
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        for key in sorted(value):
            _collect_numbers(value[key], output, limit=limit)
            if limit is not None and len(output) >= limit:
                break
    elif isinstance(value, (list, tuple)):
        for item in value:
            _collect_numbers(item, output, limit=limit)
            if limit is not None and len(output) >= limit:
                break
    elif isinstance(value, (int, float, bool)):
        output.append(float(value))
