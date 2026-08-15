from __future__ import annotations

import json
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nyssa_bench.baselines.features import action_bounds
from nyssa_bench.baselines.simple_bc import task_checkpoint_key, train_linear_bc
from nyssa_bench.datasets.recovery import RECOVERY_TARGET_SOURCES


@dataclass(frozen=True)
class RecoveryBCTrainingResult:
    source_paths: list[Path]
    merged_path: Path | None
    checkpoints: dict[str, Path]
    episodes: int
    steps: int
    routing: str
    action_sizes: dict[str, int]


def collect_recovery_episode_paths(sources: list[str | Path]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for source in sources:
        source_path = Path(source)
        candidates: list[Path] = []
        if source_path.is_file():
            candidates.append(source_path)
        elif source_path.is_dir():
            direct = source_path / "recovery_dataset" / "episodes.json"
            if direct.exists():
                candidates.append(direct)
            recovery_dir = source_path / "episodes.json"
            if source_path.name == "recovery_dataset" and recovery_dir.exists():
                candidates.append(recovery_dir)
            candidates.extend(sorted(source_path.glob("**/recovery_dataset/episodes.json")))
        else:
            raise FileNotFoundError(f"Recovery source not found: {source_path}")

        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in seen:
                paths.append(candidate)
                seen.add(resolved)

    if not paths:
        raise FileNotFoundError("No recovery_dataset/episodes.json files found in the provided sources")
    return paths


def load_recovery_episodes(paths: list[str | Path], *, min_steps: int = 1) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    for path in paths:
        source_path = Path(path)
        data = json.loads(source_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Recovery episodes file must contain a list: {path}")
        for episode_index, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            steps = item.get("steps", [])
            if not isinstance(steps, list):
                continue
            eligible_steps = []
            for step_index, step in enumerate(steps):
                if not isinstance(step, dict):
                    continue
                training_step = _eligible_training_step(
                    step,
                    source_path=source_path,
                    episode_index=episode_index,
                    step_index=step_index,
                )
                if training_step is not None:
                    eligible_steps.append(training_step)
            if len(eligible_steps) < min_steps:
                continue
            episode = dict(item)
            episode["steps"] = eligible_steps
            episodes.append(episode)
    if not episodes:
        raise ValueError("No eligible expert or recovery targets with enough steps were found")
    return episodes


def _eligible_training_step(
    step: dict[str, Any],
    *,
    source_path: Path,
    episode_index: int,
    step_index: int,
) -> dict[str, Any] | None:
    location = f"{source_path}: episode {episode_index}, step {step_index}"
    has_target_schema = any(
        key in step for key in ("target_action", "target_source", "target_valid", "record_type")
    )
    if has_target_schema:
        return _eligible_v2_training_step(step, location=location)
    return _eligible_legacy_training_step(step, location=location)


def _eligible_v2_training_step(step: dict[str, Any], *, location: str) -> dict[str, Any] | None:
    required = {"target_action", "target_source", "target_valid"}
    missing = sorted(required.difference(step))
    if missing:
        raise ValueError(f"Incomplete recovery target metadata at {location}: missing {', '.join(missing)}")

    target_valid = step["target_valid"]
    if type(target_valid) is not bool:
        raise ValueError(f"Recovery target_valid must be a boolean at {location}")

    target_source = step["target_source"]
    target_action = step["target_action"]
    record_type = step.get("record_type")
    if not target_valid:
        if target_action is not None or target_source is not None:
            raise ValueError(f"Invalid recovery context must not contain a supervised target at {location}")
        if record_type not in {None, "negative_context"}:
            raise ValueError(f"Invalid recovery record_type at {location}: {record_type!r}")
        return None

    if target_source not in RECOVERY_TARGET_SOURCES:
        raise ValueError(f"Ineligible recovery target source at {location}: {target_source!r}")
    if target_action is None:
        raise ValueError(f"Eligible recovery target is missing target_action at {location}")
    if record_type not in {None, "supervised_target"}:
        raise ValueError(f"Invalid recovery record_type at {location}: {record_type!r}")
    executed_action_source = step.get("executed_action_source")
    if executed_action_source is not None and executed_action_source != target_source:
        raise ValueError(
            f"Recovery target source does not match the executed action source at {location}: "
            f"{target_source!r} != {executed_action_source!r}"
        )

    training_step = dict(step)
    training_step["action"] = target_action
    return training_step


def _eligible_legacy_training_step(step: dict[str, Any], *, location: str) -> dict[str, Any] | None:
    info = step.get("info")
    info = info if isinstance(info, dict) else {}
    action_source = str(info.get("action_source") or "").strip().lower()
    if not action_source:
        if info.get("recovery_applied") or info.get("recovery_cached_action"):
            action_source = "recovery"
        elif info.get("expert_intervention"):
            action_source = "expert"
    if action_source not in RECOVERY_TARGET_SOURCES:
        return None

    target_action = step.get("action")
    if target_action is None:
        raise ValueError(f"Eligible legacy recovery target is missing action at {location}")
    training_step = dict(step)
    training_step.update(
        {
            "executed_action": target_action,
            "executed_action_source": action_source,
            "target_action": target_action,
            "target_source": action_source,
            "target_valid": True,
            "target_invalid_reason": None,
            "record_type": "supervised_target",
        }
    )
    return training_step


def train_recovery_bc(
    sources: list[str | Path],
    *,
    out: str | Path = "checkpoints/recovery_bc_policy.json",
    by_task: bool = False,
    routing: str = "auto",
    out_dir: str | Path = "checkpoints/bc_by_task",
    merged_out: str | Path | None = None,
    feature_dim: int = 256,
    ridge: float = 1e-3,
    min_steps: int = 1,
) -> RecoveryBCTrainingResult:
    source_paths = collect_recovery_episode_paths(sources)
    episodes = load_recovery_episodes(source_paths, min_steps=min_steps)
    merged_path = _write_json(episodes, merged_out) if merged_out else None
    routing = _resolve_routing(episodes, by_task=by_task, routing=routing)
    action_sizes = _action_sizes_by_task(episodes)
    checkpoints = (
        _train_by_task(episodes, out_dir=out_dir, feature_dim=feature_dim, ridge=ridge)
        if routing == "task"
        else {"global": _train_global(episodes, out=out, feature_dim=feature_dim, ridge=ridge)}
    )
    return RecoveryBCTrainingResult(
        source_paths=source_paths,
        merged_path=merged_path,
        checkpoints=checkpoints,
        episodes=len(episodes),
        steps=sum(len(item.get("steps", [])) for item in episodes),
        routing=routing,
        action_sizes=action_sizes,
    )


def _resolve_routing(episodes: list[dict[str, Any]], *, by_task: bool, routing: str) -> str:
    if by_task:
        routing = "task"
    if routing not in {"auto", "global", "task"}:
        raise ValueError(f"Unsupported recovery BC routing mode: {routing}")
    if routing == "auto":
        return "global" if len(set(_action_sizes_by_task(episodes).values())) <= 1 else "task"
    return routing


def _action_sizes_by_task(episodes: list[dict[str, Any]]) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for episode in episodes:
        task_id = str(episode.get("task_id") or "unknown_task")
        for step in episode.get("steps", []):
            size = _action_size_from_step(step)
            if size is None:
                continue
            previous = sizes.get(task_id)
            if previous is not None and previous != size:
                raise ValueError(
                    f"Recovery data for task '{task_id}' mixes action sizes {previous} and {size}. "
                    "Train separate checkpoints for each task/action-space contract."
                )
            sizes[task_id] = size
    return sizes


def _action_size_from_step(step: dict[str, Any]) -> int | None:
    observation = step.get("observation")
    if not isinstance(observation, dict):
        return None
    try:
        _, _, shape = action_bounds(observation)
    except Exception:
        return None
    size = 1
    for value in shape:
        size *= int(value)
    return size


def _train_global(
    episodes: list[dict[str, Any]],
    *,
    out: str | Path,
    feature_dim: int,
    ridge: float,
) -> Path:
    action_sizes = set(_action_sizes_by_task(episodes).values())
    if len(action_sizes) > 1:
        formatted = ", ".join(str(size) for size in sorted(action_sizes))
        raise ValueError(
            f"Global recovery BC cannot train on mixed action sizes: {formatted}. "
            "Use --routing task or --by-task and evaluate with task_bc_policy."
        )
    return _train_episodes(episodes, out=out, feature_dim=feature_dim, ridge=ridge)


def _train_by_task(
    episodes: list[dict[str, Any]],
    *,
    out_dir: str | Path,
    feature_dim: int,
    ridge: float,
) -> dict[str, Path]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        task_id = str(episode.get("task_id") or "unknown_task")
        grouped[task_id].append(episode)

    checkpoint_dir = Path(out_dir)
    checkpoints: dict[str, Path] = {}
    for task_id, task_episodes in sorted(grouped.items()):
        key = task_checkpoint_key(task_id)
        checkpoints[task_id] = _train_episodes(
            task_episodes,
            out=checkpoint_dir / f"{key}.json",
            feature_dim=feature_dim,
            ridge=ridge,
        )
    return checkpoints


def _train_episodes(
    episodes: list[dict[str, Any]],
    *,
    out: str | Path,
    feature_dim: int,
    ridge: float,
) -> Path:
    with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8", delete=False) as handle:
        json.dump(episodes, handle)
        episodes_path = Path(handle.name)
    try:
        return train_linear_bc(episodes_path, out, feature_dim=feature_dim, ridge=ridge)
    finally:
        try:
            episodes_path.unlink()
        except OSError:
            pass


def _write_json(episodes: list[dict[str, Any]], out: str | Path | None) -> Path:
    if out is None:
        raise ValueError("merged output path is required")
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(episodes, indent=2) + "\n", encoding="utf-8")
    return path
