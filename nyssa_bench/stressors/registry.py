from __future__ import annotations

from typing import TypeVar

from nyssa_bench.stressors.base import Stressor


STRESSOR_REGISTRY: dict[str, type[Stressor]] = {}
_StressorType = TypeVar("_StressorType", bound=type[Stressor])


def register_stressor(stressor_cls: _StressorType) -> _StressorType:
    stressor_id = stressor_cls.stressor_id
    if not stressor_id:
        raise ValueError("Registered stressors require a stable stressor_id")
    existing = STRESSOR_REGISTRY.get(stressor_id)
    if existing is not None and existing is not stressor_cls:
        raise ValueError(f"Stressor '{stressor_id}' is already registered")
    STRESSOR_REGISTRY[stressor_id] = stressor_cls
    return stressor_cls


def make_stressor(stressor_id: str) -> Stressor:
    try:
        return STRESSOR_REGISTRY[stressor_id]()
    except KeyError as exc:
        available = ", ".join(sorted(STRESSOR_REGISTRY))
        raise ValueError(
            f"Unknown stressor '{stressor_id}'. Available stressors: {available}"
        ) from exc


def list_stressors() -> list[str]:
    return sorted(STRESSOR_REGISTRY)
