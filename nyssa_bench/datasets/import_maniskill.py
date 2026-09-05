from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from nyssa_bench.core.episode import EpisodeResult, StepRecord
from nyssa_bench.datasets.export_json import export_json
from nyssa_bench.datasets.export_jsonl import export_jsonl


ENV_TO_TASK_ID = {
    "PickCube-v1": "maniskill_pick_cube",
    "StackCube-v1": "maniskill_stack_cube",
    "PushCube-v1": "maniskill_push_cube",
}


def import_maniskill_demos(input_dir: str | Path, out_dir: str | Path) -> dict[str, Path]:
    """Convert ManiSkill trajectory HDF5 files into Nyssa episode artifacts.

    ManiSkill's motion-planning examples write HDF5 trajectory files with one or
    more trajectory groups. The exact observation payload can vary by ManiSkill
    version and recording options, so this importer keeps observations generic
    and preserves whatever timed datasets are available under `obs`,
    `env_states`, or `states`.
    """

    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - exercised in minimal envs
        raise RuntimeError(
            "ManiSkill demo import requires: "
            'python -m pip install "nyssa-bench[dataset]"'
        ) from exc

    input_dir = Path(input_dir)
    out_dir = Path(out_dir)
    h5_files = sorted(input_dir.rglob("*.h5")) + sorted(input_dir.rglob("*.hdf5"))
    if not h5_files:
        raise FileNotFoundError(f"No ManiSkill HDF5 trajectories found under {input_dir}")

    episodes: list[EpisodeResult] = []
    source_files: list[str] = []
    for h5_path in h5_files:
        source_files.append(h5_path.as_posix())
        with h5py.File(h5_path, "r") as handle:
            file_env_id = _infer_env_id(handle, h5_path)
            for group_name, group in _trajectory_groups(handle):
                env_id = _env_id_from_attrs(group.attrs) or file_env_id
                episode = _episode_from_group(
                    group,
                    env_id=env_id,
                    task_id=ENV_TO_TASK_ID.get(env_id, _task_id_from_env_id(env_id)),
                    episode_index=len(episodes),
                    source=f"{h5_path.as_posix()}::{group_name}",
                )
                episodes.append(episode)

    if not episodes:
        raise ValueError(f"No trajectory groups with actions were found under {input_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    all_json = export_json(episodes, out_dir / "episodes.json")
    all_jsonl = export_jsonl(episodes, out_dir / "episodes.jsonl")

    per_task_paths: dict[str, Path] = {}
    for task_id in sorted({episode.task_id for episode in episodes}):
        task_episodes = [episode for episode in episodes if episode.task_id == task_id]
        task_dir = out_dir / task_id
        per_task_paths[f"{task_id}_episodes"] = export_json(task_episodes, task_dir / "episodes.json")
        export_jsonl(task_episodes, task_dir / "episodes.jsonl")

    manifest = {
        "format": "nyssa-maniskill-demo-import-v1",
        "input_dir": input_dir.as_posix(),
        "episode_count": len(episodes),
        "task_counts": _task_counts(episodes),
        "source_files": source_files,
        "episodes": all_json.as_posix(),
        "episodes_jsonl": all_jsonl.as_posix(),
        "per_task": {key: value.as_posix() for key, value in per_task_paths.items()},
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return {"manifest": manifest_path, "episodes": all_json, "episodes_jsonl": all_jsonl, **per_task_paths}


def _trajectory_groups(handle: Any) -> list[tuple[str, Any]]:
    groups: list[tuple[str, Any]] = []

    def visit(name: str, obj: Any) -> None:
        if hasattr(obj, "keys") and "actions" in obj:
            groups.append((name, obj))

    handle.visititems(visit)
    if "actions" in handle:
        groups.insert(0, ("", handle))
    return groups


def _episode_from_group(
    group: Any,
    *,
    env_id: str,
    task_id: str,
    episode_index: int,
    source: str,
) -> EpisodeResult:
    from nyssa_bench.failures import failure_ledger_from_episode_dict

    actions = np.asarray(group["actions"])
    if actions.ndim == 1:
        actions = actions.reshape(-1, 1)
    step_count = int(actions.shape[0])
    action_space = _action_space_spec(actions)
    rewards = _optional_array(group, ("rewards", "reward"), step_count)
    terminated = _optional_bool_array(group, ("terminated", "terminations", "dones"), step_count)
    truncated = _optional_bool_array(group, ("truncated", "truncations"), step_count)
    successes = _optional_bool_array(group, ("success", "successes", "is_success"), step_count)
    success = bool(successes.any()) if successes is not None else True

    steps = []
    for index in range(step_count):
        obs = _observation_at(group, index, step_count)
        info = {
            "success": bool(successes[index]) if successes is not None and index < len(successes) else success,
            "source": "maniskill_motionplanning",
            "source_trajectory": source,
            "env_id": env_id,
        }
        steps.append(
            StepRecord(
                observation={"raw": obs, "action_space": action_space},
                action=actions[index],
                reward=float(rewards[index]) if rewards is not None and index < len(rewards) else 0.0,
                terminated=bool(terminated[index]) if terminated is not None and index < len(terminated) else index == step_count - 1,
                truncated=bool(truncated[index]) if truncated is not None and index < len(truncated) else False,
                info=info,
            )
        )

    seed = _int_attr(group, ("episode_seed", "seed"), default=episode_index)
    failure_label = None if success else "unknown_failure"
    failure_label_source = None if success else "maniskill_import"
    failure_ledger = failure_ledger_from_episode_dict(
        {
            "task_id": task_id,
            "episode_index": episode_index,
            "seed": seed,
            "success": success,
            "failure_label": failure_label,
            "failure_label_source": failure_label_source,
            "steps": [None] * len(steps),
        },
        engine_name="maniskill",
    )
    return EpisodeResult(
        task_id=task_id,
        episode_index=episode_index,
        seed=seed,
        success=success,
        failure_label=failure_label,
        failure_label_source=failure_label_source,
        metrics={
            "completion_time": float(step_count),
            "path_efficiency": 0.0,
            "grasp_success_rate": 1.0 if success else 0.0,
            "recovery_success_rate": 0.0,
            "drop_rate": 0.0,
        },
        steps=steps,
        failure_ledger=failure_ledger,
    )


def _observation_at(group: Any, index: int, step_count: int) -> dict[str, Any]:
    observation_payload: Any | None = None
    for key in ("obs", "observation", "observations"):
        if key in group:
            observation = _slice_timed_value(group[key], index, step_count)
            if _has_payload(observation):
                observation_payload = observation
                break

    state_payloads: dict[str, Any] = {}
    for key in ("env_states", "states", "state"):
        if key in group:
            state = _slice_timed_value(group[key], index, step_count)
            if _has_payload(state):
                state_payloads[key] = state

    if observation_payload is not None and state_payloads:
        if isinstance(observation_payload, dict):
            return {**observation_payload, **state_payloads}
        return {"observation": observation_payload, **state_payloads}
    if observation_payload is not None:
        return observation_payload
    if state_payloads:
        return state_payloads
    return {"timestep": index}


def _slice_timed_value(value: Any, index: int, step_count: int) -> Any:
    if hasattr(value, "keys"):
        return {str(key): _slice_timed_value(value[key], index, step_count) for key in sorted(value.keys())}
    array = np.asarray(value)
    if array.ndim > 0 and array.shape[0] in (step_count, step_count + 1):
        return array[min(index, array.shape[0] - 1)]
    return array


def _has_payload(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_has_payload(item) for item in value.values())
    if hasattr(value, "shape"):
        return bool(value.shape) or value.size > 0
    if isinstance(value, (list, tuple)):
        return bool(value)
    return value is not None


def _optional_array(group: Any, names: tuple[str, ...], step_count: int) -> np.ndarray | None:
    for name in names:
        if name in group:
            array = np.asarray(group[name])
            if array.ndim == 0:
                return np.full(step_count, array.item())
            return array.reshape(-1)
    return None


def _optional_bool_array(group: Any, names: tuple[str, ...], step_count: int) -> np.ndarray | None:
    array = _optional_array(group, names, step_count)
    if array is None:
        return None
    return array.astype(bool)


def _action_space_spec(actions: np.ndarray) -> dict[str, Any]:
    size = int(np.prod(actions.shape[1:])) if actions.ndim > 1 else 1
    return {
        "type": "box",
        "shape": [size],
        "low": [-1.0] * size,
        "high": [1.0] * size,
        "dtype": str(actions.dtype),
    }


def _infer_env_id(handle: Any, path: Path) -> str:
    for attrs in (handle.attrs,):
        env_id = _env_id_from_attrs(attrs)
        if env_id:
            return env_id
    text = path.as_posix()
    for env_id in ENV_TO_TASK_ID:
        if env_id in text:
            return env_id
    stem = path.stem
    for env_id in ENV_TO_TASK_ID:
        if env_id.replace("-v1", "").lower() in stem.lower():
            return env_id
    return stem


def _env_id_from_attrs(attrs: Any) -> str | None:
    for key in ("env_id", "env_name", "env"):
        if key in attrs:
            return _decode_attr(attrs[key])
    if "env_info" in attrs:
        try:
            payload = json.loads(_decode_attr(attrs["env_info"]))
            for key in ("env_id", "env_name", "id"):
                if key in payload:
                    return str(payload[key])
        except (TypeError, ValueError):
            return None
    return None


def _int_attr(group: Any, names: tuple[str, ...], *, default: int) -> int:
    for name in names:
        if name in group.attrs:
            try:
                return int(group.attrs[name])
            except (TypeError, ValueError):
                return default
    return default


def _decode_attr(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "item"):
        value = value.item()
        if isinstance(value, bytes):
            return value.decode("utf-8")
    return str(value)


def _task_id_from_env_id(env_id: str) -> str:
    return env_id.replace("-v1", "").replace("-", "_").lower()


def _task_counts(episodes: list[EpisodeResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for episode in episodes:
        counts[episode.task_id] = counts.get(episode.task_id, 0) + 1
    return counts
