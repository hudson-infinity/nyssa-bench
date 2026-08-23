from __future__ import annotations

from typing import Any


def as_bool(value: Any) -> bool | None:
    if value is None:
        return None
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
            return None
    if hasattr(value, "all"):
        try:
            return bool(value.all())
        except ValueError:
            return None
    return bool(value)


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "__iter__") and not isinstance(value, (str, bytes, dict)):
        try:
            values = [float(item) for item in value]
            if len(values) == 1:
                return values[0]
        except (TypeError, ValueError, OverflowError):
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
