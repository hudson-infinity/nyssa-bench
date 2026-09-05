from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Sequence

from nyssa_bench.monitors.base import FailureMonitor
from nyssa_bench.monitors.reference import ActionMagnitudeFailureMonitor


def load_failure_monitor(value: str | Path | FailureMonitor) -> FailureMonitor:
    if isinstance(value, FailureMonitor):
        return value
    name = str(value)
    if name in {"action-magnitude", "action_magnitude_reference"}:
        return ActionMagnitudeFailureMonitor()
    path = Path(name)
    if not path.exists() and path.suffix != ".py":
        raise ValueError(f"unknown failure monitor: {name}")
    if not path.is_file():
        raise FileNotFoundError(f"failure monitor module not found: {path}")
    spec = importlib.util.spec_from_file_location(
        f"nyssa_failure_monitor_{path.stem}", path
    )
    if spec is None or spec.loader is None:
        raise ValueError(f"could not load failure monitor module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    factory = getattr(module, "create_failure_monitor", None)
    monitor: Any = factory() if callable(factory) else None
    if not isinstance(monitor, FailureMonitor):
        raise TypeError(
            "failure monitor module must define create_failure_monitor() returning FailureMonitor"
        )
    return monitor


def load_failure_monitors(
    values: Sequence[str | Path | FailureMonitor] | None,
) -> tuple[FailureMonitor, ...]:
    return tuple(load_failure_monitor(value) for value in values or ())
