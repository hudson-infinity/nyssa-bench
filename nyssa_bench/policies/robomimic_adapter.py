from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from nyssa_bench.baselines.features import fit_action_to_observation, flatten_observation
from nyssa_bench.baselines.robomimic_bc import create_robomimic_policy, load_robomimic_checkpoint
from nyssa_bench.policies.base import Policy
from nyssa_bench.policies.loaders import call_model, load_callable_from_env, normalize_action, require_model

DEFAULT_ROBOMIMIC_FEATURE_DIM = 256


class RoboMimicPolicy(Policy):
    def __init__(self, model: Any | None = None) -> None:
        loaded = model if model is not None else load_callable_from_env("NYSSA_ROBOMIMIC_POLICY")
        self.model = require_model(
            loaded if loaded is not None else create_robomimic_policy(),
            policy_name="RoboMimicPolicy",
            env_var="NYSSA_ROBOMIMIC_POLICY",
        )
        self.feature_dim = _feature_dim_override()

    def reset(self, task: Any | None = None, seed: int | None = None) -> None:
        _reset_robomimic_model(self.model)

    def act(self, observation: dict[str, Any]) -> Any:
        return _robomimic_action(self.model, observation, feature_dim=self.feature_dim)


class TaskRoboMimicPolicy(Policy):
    """Task-routed RoboMimic policy for one checkpoint per task."""

    def __init__(self) -> None:
        self.checkpoint_dir = Path(os.getenv("NYSSA_TASK_ROBOMIMIC_DIR", "checkpoints/robomimic_by_task"))
        self.feature_dim = _feature_dim_override()
        self.current_task_id: str | None = None
        self._models: dict[str, Any] = {}

    def reset(self, task: Any | None = None, seed: int | None = None) -> None:
        self.current_task_id = str(getattr(task, "task_id", "")) or None
        if self.current_task_id:
            _reset_robomimic_model(self._model_for_task(self.current_task_id))

    def act(self, observation: dict[str, Any]) -> Any:
        if not self.current_task_id:
            raise RuntimeError("Task-routed RoboMimic policy was used before reset(task=...)")
        return _robomimic_action(self._model_for_task(self.current_task_id), observation, feature_dim=self.feature_dim)

    def _model_for_task(self, task_id: str) -> Any:
        key = _checkpoint_key(task_id)
        if key not in self._models:
            path = _find_task_checkpoint(self.checkpoint_dir, key)
            self._models[key] = load_robomimic_checkpoint(path)
        return self._models[key]


def _robomimic_action(model: Any, observation: dict[str, Any], *, feature_dim: int | None) -> Any:
    resolved_feature_dim = _model_feature_dim(model, override=feature_dim)
    flat_observation = {"flat": flatten_observation(observation, resolved_feature_dim)}
    action = _call_robomimic_model(model, flat_observation)
    return fit_action_to_observation(action, observation)


def _feature_dim_override() -> int | None:
    value = os.getenv("NYSSA_ROBOMIMIC_FEATURE_DIM")
    if value is None:
        return None
    feature_dim = int(value)
    if feature_dim <= 0:
        raise ValueError("NYSSA_ROBOMIMIC_FEATURE_DIM must be a positive integer")
    return feature_dim


def _model_feature_dim(model: Any, *, override: int | None) -> int:
    if override is not None:
        return override
    value = getattr(model, "_nyssa_flat_feature_dim", None)
    try:
        feature_dim = int(value)
    except (TypeError, ValueError):
        return DEFAULT_ROBOMIMIC_FEATURE_DIM
    return feature_dim if feature_dim > 0 else DEFAULT_ROBOMIMIC_FEATURE_DIM


def _reset_robomimic_model(model: Any) -> None:
    start_episode = getattr(model, "start_episode", None)
    if callable(start_episode):
        start_episode()
    reset = getattr(model, "reset", None)
    if callable(reset):
        reset()


def _call_robomimic_model(model: Any, observation: dict[str, Any]) -> Any:
    for method_name in ("get_action", "predict_action", "select_action", "act"):
        method = getattr(model, method_name, None)
        if callable(method):
            return normalize_action(method(observation))
    if callable(model):
        return normalize_action(model(observation))
    return call_model(model, observation, ("get_action", "predict_action", "select_action", "act"))


def _checkpoint_key(task_id: str) -> str:
    aliases = {
        "maniskill_pick_cube_joint": "maniskill_pick_cube",
        "maniskill_stack_cube_joint": "maniskill_stack_cube",
        "maniskill_push_cube_joint": "maniskill_push_cube",
    }
    return aliases.get(task_id, task_id.removesuffix("_joint"))


def _find_task_checkpoint(checkpoint_dir: Path, task_key: str) -> Path:
    direct = checkpoint_dir / f"{task_key}.pth"
    if direct.exists():
        return direct

    candidates = _task_checkpoint_candidates(checkpoint_dir, task_key)
    if candidates:
        return _latest_model_epoch(candidates)

    expected = [
        direct,
        checkpoint_dir / "checkpoints" / task_key,
        checkpoint_dir / task_key,
    ]
    expected_text = ", ".join(path.as_posix() for path in expected)
    raise RuntimeError(
        f"RoboMimic checkpoint not found for task '{task_key}' under {checkpoint_dir}. "
        "Set NYSSA_TASK_ROBOMIMIC_DIR to either a directory containing task .pth files "
        "or the output directory from `nyssa export-task-robomimic`. "
        f"Expected one of: {expected_text}."
    )


def _task_checkpoint_candidates(checkpoint_dir: Path, task_key: str) -> list[Path]:
    candidates: list[Path] = []
    candidates.extend(_manifest_checkpoint_candidates(checkpoint_dir, task_key))
    if checkpoint_dir.exists():
        task_token = task_key.lower()
        for path in checkpoint_dir.rglob("*.pth"):
            text = path.as_posix().lower()
            if task_token in text and (path.name.startswith("model_epoch_") or path.stem == task_key):
                candidates.append(path)
    return sorted(set(candidates))


def _manifest_checkpoint_candidates(checkpoint_dir: Path, task_key: str) -> list[Path]:
    manifest_path = checkpoint_dir / "task_robomimic_manifest.json"
    if not manifest_path.exists():
        return []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    tasks = manifest.get("tasks", {})
    if not isinstance(tasks, dict):
        return []
    task_artifacts = tasks.get(task_key)
    if not isinstance(task_artifacts, dict):
        return []
    config_path_value = task_artifacts.get("config")
    if not config_path_value:
        return []
    config_path = _resolve_manifest_path(checkpoint_dir, str(config_path_value))
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    output_dir = Path(str(config.get("train", {}).get("output_dir", "")))
    if not output_dir.exists():
        return []
    return [path for path in output_dir.rglob("model_epoch_*.pth") if path.is_file()]


def _resolve_manifest_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    for candidate in (base_dir / value, base_dir.parent / value):
        if candidate.exists():
            return candidate
    return path


def _latest_model_epoch(candidates: list[Path]) -> Path:
    def key(path: Path) -> tuple[int, float, str]:
        match = re.search(r"model_epoch_(\d+)\.pth$", path.name)
        epoch = int(match.group(1)) if match else -1
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        return epoch, mtime, path.as_posix()

    return max(candidates, key=key)
