from __future__ import annotations

from typing import Any


SUPPORTED_RANDOMIZATION_KEYS: dict[str, set[str]] = {
    "maniskill": {"seed"},
    "mujoco": {"seed"},
}


def summarize_randomization(randomization: dict[str, Any]) -> dict[str, Any]:
    return {
        "enabled_keys": sorted(
            key for key, value in randomization.items() if bool(value)
        ),
        "raw": randomization,
    }


def summarize_stressor_support(
    randomization: dict[str, Any], engine: str
) -> dict[str, Any]:
    enabled = sorted(
        key
        for key, value in randomization.items()
        if key != "stressors" and _is_enabled(value)
    )
    supported = SUPPORTED_RANDOMIZATION_KEYS.get(engine, set())
    supported_ids = {key for key in enabled if key in supported}
    unsupported_ids = {key for key in enabled if key not in supported}
    for stressor_id in _declared_stressor_ids(randomization.get("stressors")):
        enabled.append(stressor_id)
        try:
            from nyssa_bench.stressors import make_stressor

            stressor = make_stressor(stressor_id)
        except ValueError:
            unsupported_ids.add(stressor_id)
            continue
        if "*" in stressor.supported_engines or engine in stressor.supported_engines:
            supported_ids.add(stressor_id)
        else:
            unsupported_ids.add(stressor_id)
    return {
        "enabled_stressors": sorted(set(enabled)),
        "supported_stressors": sorted(supported_ids),
        "unsupported_stressors": sorted(unsupported_ids),
    }


def aggregate_stressor_support(
    task_summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    unsupported: dict[str, list[str]] = {}
    supported: dict[str, list[str]] = {}
    for task_id, summary in task_summaries.items():
        if summary.get("unsupported_stressors"):
            unsupported[task_id] = list(summary["unsupported_stressors"])
        if summary.get("supported_stressors"):
            supported[task_id] = list(summary["supported_stressors"])
    return {
        "supported_by_task": supported,
        "unsupported_by_task": unsupported,
        "unsupported_stressors": sorted(
            {item for values in unsupported.values() for item in values}
        ),
    }


def _is_enabled(value: Any) -> bool:
    if value is False or value is None:
        return False
    if isinstance(value, (list, tuple, dict)):
        return bool(value)
    return bool(value)


def _declared_stressor_ids(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    ids = []
    for item in value:
        if not isinstance(item, dict):
            continue
        stressor_id = item.get("stressor_id", item.get("id"))
        if stressor_id:
            ids.append(str(stressor_id))
    return ids
