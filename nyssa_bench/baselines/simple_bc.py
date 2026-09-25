from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, TypeAlias

import numpy as np

from nyssa_bench.baselines.features import action_bounds, fit_action_to_observation, flatten_observation, normalize_action


BCModel: TypeAlias = "LinearBCPolicy | KNNBCPolicy | SequenceKNNBCPolicy"


class LinearBCPolicy:
    def __init__(self, weights: np.ndarray, bias: np.ndarray, feature_dim: int, action_size: int) -> None:
        self.weights = weights
        self.bias = bias
        self.feature_dim = feature_dim
        self.action_size = action_size

    def predict_action(self, observation: dict[str, Any]) -> Any:
        features = _flatten_bc_observation(observation, self.feature_dim)
        action = features @ self.weights + self.bias
        return fit_action_to_observation(action, observation)

    @classmethod
    def load(cls, path: str | Path) -> "LinearBCPolicy":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls._from_payload(payload)

    @classmethod
    def _from_payload(cls, payload: dict[str, Any]) -> "LinearBCPolicy":
        return cls(
            weights=np.asarray(payload["weights"], dtype=float),
            bias=np.asarray(payload["bias"], dtype=float),
            feature_dim=int(payload["feature_dim"]),
            action_size=int(payload["action_size"]),
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "nyssa-linear-bc-v1",
            "feature_dim": self.feature_dim,
            "action_size": self.action_size,
            "weights": self.weights.tolist(),
            "bias": self.bias.tolist(),
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return path


class KNNBCPolicy:
    def __init__(
        self,
        features: np.ndarray,
        actions: np.ndarray,
        feature_mean: np.ndarray,
        feature_scale: np.ndarray,
        feature_dim: int,
        action_size: int,
        k: int = 1,
    ) -> None:
        self.features = features
        self.actions = actions
        self.feature_mean = feature_mean
        self.feature_scale = np.where(np.abs(feature_scale) > 1e-8, feature_scale, 1.0)
        self.feature_dim = feature_dim
        self.action_size = action_size
        self.k = max(1, int(k))

    def predict_action(self, observation: dict[str, Any]) -> Any:
        features = _flatten_bc_observation(observation, self.feature_dim)
        normalized = (features - self.feature_mean) / self.feature_scale
        distances = np.sum((self.features - normalized) ** 2, axis=1)
        k = min(self.k, len(distances))
        indices = np.argpartition(distances, k - 1)[:k]
        weights = 1.0 / np.maximum(distances[indices], 1e-8)
        action = np.average(self.actions[indices], axis=0, weights=weights)
        return fit_action_to_observation(action, observation)

    @classmethod
    def load(cls, path: str | Path) -> "KNNBCPolicy":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls._from_payload(payload)

    @classmethod
    def _from_payload(cls, payload: dict[str, Any]) -> "KNNBCPolicy":
        return cls(
            features=np.asarray(payload["features"], dtype=float),
            actions=np.asarray(payload["actions"], dtype=float),
            feature_mean=np.asarray(payload["feature_mean"], dtype=float),
            feature_scale=np.asarray(payload["feature_scale"], dtype=float),
            feature_dim=int(payload["feature_dim"]),
            action_size=int(payload["action_size"]),
            k=int(payload.get("k", 1)),
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "nyssa-knn-bc-v1",
            "feature_dim": self.feature_dim,
            "action_size": self.action_size,
            "k": self.k,
            "features": self.features.tolist(),
            "actions": self.actions.tolist(),
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return path


class SequenceKNNBCPolicy:
    def __init__(
        self,
        features: np.ndarray,
        action_sequences: np.ndarray,
        feature_mean: np.ndarray,
        feature_scale: np.ndarray,
        feature_dim: int,
        action_size: int,
        action_horizon: int,
        k: int = 1,
    ) -> None:
        self.features = features
        self.action_sequences = action_sequences
        self.feature_mean = feature_mean
        self.feature_scale = np.where(np.abs(feature_scale) > 1e-8, feature_scale, 1.0)
        self.feature_dim = feature_dim
        self.action_size = action_size
        self.action_horizon = max(1, int(action_horizon))
        self.k = max(1, int(k))

    def predict_action(self, observation: dict[str, Any]) -> Any:
        features = _flatten_bc_observation(observation, self.feature_dim)
        normalized = (features - self.feature_mean) / self.feature_scale
        distances = np.sum((self.features - normalized) ** 2, axis=1)
        k = min(self.k, len(distances))
        indices = np.argpartition(distances, k - 1)[:k]
        weights = 1.0 / np.maximum(distances[indices], 1e-8)
        sequence = np.average(self.action_sequences[indices], axis=0, weights=weights)
        return np.stack([fit_action_to_observation(action, observation) for action in sequence])

    @classmethod
    def load(cls, path: str | Path) -> "SequenceKNNBCPolicy":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls._from_payload(payload)

    @classmethod
    def _from_payload(cls, payload: dict[str, Any]) -> "SequenceKNNBCPolicy":
        return cls(
            features=np.asarray(payload["features"], dtype=float),
            action_sequences=np.asarray(payload["action_sequences"], dtype=float),
            feature_mean=np.asarray(payload["feature_mean"], dtype=float),
            feature_scale=np.asarray(payload["feature_scale"], dtype=float),
            feature_dim=int(payload["feature_dim"]),
            action_size=int(payload["action_size"]),
            action_horizon=int(payload.get("action_horizon", 1)),
            k=int(payload.get("k", 1)),
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "nyssa-sequence-knn-bc-v1",
            "feature_dim": self.feature_dim,
            "action_size": self.action_size,
            "action_horizon": self.action_horizon,
            "k": self.k,
            "features": self.features.tolist(),
            "action_sequences": self.action_sequences.tolist(),
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return path


class TaskRoutedLinearBCPolicy:
    def __init__(self, checkpoint_dir: str | Path, *, missing_task: str = "error") -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.missing_task = missing_task
        self.current_task_id: str | None = None
        self._models: dict[str, BCModel] = {}

    def reset(self, task: Any | None = None, seed: int | None = None) -> None:
        self.current_task_id = str(getattr(task, "task_id", "")) or None

    def predict_action(self, observation: dict[str, Any]) -> Any:
        if not self.current_task_id:
            raise RuntimeError("Task-routed BC policy was used before reset(task=...)")
        model = self._model_for_task(self.current_task_id)
        if model is None:
            return _zero_action(observation)
        return model.predict_action(observation)

    def _model_for_task(self, task_id: str) -> BCModel | None:
        key = _checkpoint_key(task_id)
        if key not in self._models:
            path = self.checkpoint_dir / f"{key}.json"
            if not path.exists():
                if self.missing_task == "zero":
                    return None
                raise RuntimeError(
                    f"Task BC checkpoint not found for task '{task_id}': {path}. "
                    "Train one checkpoint per task under NYSSA_TASK_BC_DIR or set NYSSA_TASK_BC_MISSING=zero."
                )
            self._models[key] = load_bc_policy(path)
        return self._models[key]


def train_linear_bc(
    episodes_path: str | Path,
    out: str | Path,
    *,
    feature_dim: int = 256,
    ridge: float = 1e-3,
) -> Path:
    rows_x, rows_y, action_size = _episode_training_rows(episodes_path, feature_dim=feature_dim)
    x = np.vstack(rows_x)
    y = np.vstack(rows_y)
    x_aug = np.hstack([x, np.ones((x.shape[0], 1))])
    regularizer = ridge * np.eye(x_aug.shape[1])
    solution = np.linalg.solve(x_aug.T @ x_aug + regularizer, x_aug.T @ y)
    policy = LinearBCPolicy(
        weights=solution[:-1, :],
        bias=solution[-1, :],
        feature_dim=feature_dim,
        action_size=action_size,
    )
    return policy.save(out)


def train_knn_bc(
    episodes_path: str | Path,
    out: str | Path,
    *,
    feature_dim: int = 256,
    k: int = 1,
) -> Path:
    rows_x, rows_y, action_size = _episode_training_rows(episodes_path, feature_dim=feature_dim)
    x = np.vstack(rows_x)
    y = np.vstack(rows_y)
    feature_mean = x.mean(axis=0)
    feature_scale = x.std(axis=0)
    normalized = (x - feature_mean) / np.where(np.abs(feature_scale) > 1e-8, feature_scale, 1.0)
    policy = KNNBCPolicy(
        features=normalized,
        actions=y,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        feature_dim=feature_dim,
        action_size=action_size,
        k=k,
    )
    return policy.save(out)


def train_sequence_knn_bc(
    episodes_path: str | Path,
    out: str | Path,
    *,
    feature_dim: int = 256,
    k: int = 1,
    action_horizon: int = 8,
) -> Path:
    policy = train_sequence_knn_bc_from_episodes(
        json.loads(Path(episodes_path).read_text(encoding="utf-8")),
        feature_dim=feature_dim,
        k=k,
        action_horizon=action_horizon,
    )
    return policy.save(out)


def train_task_bc(
    episodes_paths: list[str | Path],
    out_dir: str | Path,
    *,
    feature_dim: int = 256,
    ridge: float = 1e-3,
    model: str = "linear",
    k: int = 1,
    action_horizon: int = 8,
    success_only: bool = True,
) -> dict[str, Path]:
    episodes = _load_episode_sources(episodes_paths)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for episode in episodes:
        if success_only and not bool(episode.get("success", True)):
            continue
        task_id = str(episode.get("task_id", "")).strip()
        if not task_id:
            continue
        grouped.setdefault(task_checkpoint_key(task_id), []).append(episode)
    if not grouped:
        raise ValueError("No task-labeled episodes found for task BC training")

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints: dict[str, Path] = {}
    for task_key, task_episodes in sorted(grouped.items()):
        checkpoint = output_dir / f"{task_key}.json"
        if model == "linear":
            policy = train_linear_bc_from_episodes(task_episodes, feature_dim=feature_dim, ridge=ridge)
        elif model == "knn":
            policy = train_knn_bc_from_episodes(task_episodes, feature_dim=feature_dim, k=k)
        elif model == "sequence-knn":
            policy = train_sequence_knn_bc_from_episodes(
                task_episodes,
                feature_dim=feature_dim,
                k=k,
                action_horizon=action_horizon,
            )
        else:
            raise ValueError(f"Unsupported BC model: {model}")
        checkpoints[task_key] = policy.save(checkpoint)
    return checkpoints


def train_linear_bc_from_episodes(
    episodes: list[dict[str, Any]],
    *,
    feature_dim: int = 256,
    ridge: float = 1e-3,
) -> LinearBCPolicy:
    rows_x, rows_y, action_size = _episode_training_rows_from_payload(episodes, feature_dim=feature_dim)
    x = np.vstack(rows_x)
    y = np.vstack(rows_y)
    x_aug = np.hstack([x, np.ones((x.shape[0], 1))])
    regularizer = ridge * np.eye(x_aug.shape[1])
    solution = np.linalg.solve(x_aug.T @ x_aug + regularizer, x_aug.T @ y)
    return LinearBCPolicy(
        weights=solution[:-1, :],
        bias=solution[-1, :],
        feature_dim=feature_dim,
        action_size=action_size,
    )


def train_knn_bc_from_episodes(
    episodes: list[dict[str, Any]],
    *,
    feature_dim: int = 256,
    k: int = 1,
) -> KNNBCPolicy:
    rows_x, rows_y, action_size = _episode_training_rows_from_payload(episodes, feature_dim=feature_dim)
    x = np.vstack(rows_x)
    y = np.vstack(rows_y)
    feature_mean = x.mean(axis=0)
    feature_scale = x.std(axis=0)
    normalized = (x - feature_mean) / np.where(np.abs(feature_scale) > 1e-8, feature_scale, 1.0)
    return KNNBCPolicy(
        features=normalized,
        actions=y,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        feature_dim=feature_dim,
        action_size=action_size,
        k=k,
    )


def train_sequence_knn_bc_from_episodes(
    episodes: list[dict[str, Any]],
    *,
    feature_dim: int = 256,
    k: int = 1,
    action_horizon: int = 8,
) -> SequenceKNNBCPolicy:
    rows_x, rows_y, action_size = _episode_sequence_training_rows_from_payload(
        episodes,
        feature_dim=feature_dim,
        action_horizon=action_horizon,
    )
    x = np.vstack(rows_x)
    y = np.stack(rows_y)
    feature_mean = x.mean(axis=0)
    feature_scale = x.std(axis=0)
    normalized = (x - feature_mean) / np.where(np.abs(feature_scale) > 1e-8, feature_scale, 1.0)
    return SequenceKNNBCPolicy(
        features=normalized,
        action_sequences=y,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        feature_dim=feature_dim,
        action_size=action_size,
        action_horizon=action_horizon,
        k=k,
    )


def load_bc_policy(path: str | Path) -> BCModel:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    checkpoint_format = str(payload.get("format", "nyssa-linear-bc-v1"))
    if checkpoint_format == "nyssa-linear-bc-v1":
        return LinearBCPolicy._from_payload(payload)
    if checkpoint_format == "nyssa-knn-bc-v1":
        return KNNBCPolicy._from_payload(payload)
    if checkpoint_format == "nyssa-sequence-knn-bc-v1":
        return SequenceKNNBCPolicy._from_payload(payload)
    raise RuntimeError(f"Unsupported BC checkpoint format: {checkpoint_format}")


def create_bc_policy() -> BCModel:
    checkpoint = os.getenv("NYSSA_BC_CHECKPOINT", "checkpoints/bc_policy.json")
    if not Path(checkpoint).exists():
        raise RuntimeError(
            f"BC checkpoint not found: {checkpoint}. Run `nyssa train-bc ... --out {checkpoint}` "
            "or set NYSSA_BC_POLICY=module:factory."
        )
    return load_bc_policy(checkpoint)


def _episode_training_rows(
    episodes_path: str | Path,
    *,
    feature_dim: int,
) -> tuple[list[np.ndarray], list[np.ndarray], int]:
    episodes = json.loads(Path(episodes_path).read_text(encoding="utf-8"))
    return _episode_training_rows_from_payload(episodes, feature_dim=feature_dim)


def _episode_training_rows_from_payload(
    episodes: list[dict[str, Any]],
    *,
    feature_dim: int,
) -> tuple[list[np.ndarray], list[np.ndarray], int]:
    rows_x: list[np.ndarray] = []
    rows_y: list[np.ndarray] = []
    action_size: int | None = None
    for episode in episodes:
        for step in episode.get("steps", []):
            observation = step.get("observation", {})
            _, _, shape = action_bounds(observation)
            size = int(np.prod(shape))
            action_size = action_size or size
            rows_x.append(_flatten_bc_observation(observation, feature_dim))
            rows_y.append(normalize_action(step.get("action"), action_size))
    if not rows_x or action_size is None:
        raise ValueError("No training steps found")
    return rows_x, rows_y, action_size


def _episode_sequence_training_rows_from_payload(
    episodes: list[dict[str, Any]],
    *,
    feature_dim: int,
    action_horizon: int,
) -> tuple[list[np.ndarray], list[np.ndarray], int]:
    rows_x: list[np.ndarray] = []
    rows_y: list[np.ndarray] = []
    action_size: int | None = None
    horizon = max(1, int(action_horizon))
    for episode in episodes:
        steps = list(episode.get("steps", []))
        if not steps:
            continue
        observations = [step.get("observation", {}) for step in steps]
        first_observation = observations[0]
        _, _, shape = action_bounds(first_observation)
        size = int(np.prod(shape))
        action_size = action_size or size
        actions = [normalize_action(step.get("action"), action_size) for step in steps]
        for index, observation in enumerate(observations):
            window = actions[index : index + horizon]
            if len(window) < horizon:
                window = [*window, *([actions[-1]] * (horizon - len(window)))]
            rows_x.append(_flatten_bc_observation(observation, feature_dim))
            rows_y.append(np.stack(window))
    if not rows_x or action_size is None:
        raise ValueError("No sequence training steps found")
    return rows_x, rows_y, action_size


def _load_episode_sources(paths: list[str | Path]) -> list[dict[str, Any]]:
    return load_episode_sources(paths)


def load_episode_sources(paths: list[str | Path]) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    for source in paths:
        path = Path(source)
        if path.is_file() and path.suffix.lower() == ".zip":
            episodes.extend(_load_episode_zip(path))
            continue
        files = _episode_files(path)
        if not files:
            raise FileNotFoundError(f"No episodes.json files found for task BC source: {path}")
        for file_path in files:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                episodes.extend(item for item in payload if isinstance(item, dict))
            else:
                raise ValueError(f"Episode source must contain a JSON list: {file_path}")
    return episodes


def _load_episode_zip(path: Path) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    with zipfile.ZipFile(path) as archive:
        names = _episode_zip_members(archive)
        if not names:
            raise FileNotFoundError(f"No episodes.json files found in task BC archive: {path}")
        for name in names:
            payload = json.loads(archive.read(name).decode("utf-8"))
            if isinstance(payload, list):
                episodes.extend(item for item in payload if isinstance(item, dict))
            else:
                raise ValueError(f"Episode archive member must contain a JSON list: {path}!{name}")
    return episodes


def _episode_zip_members(archive: zipfile.ZipFile) -> list[str]:
    candidates = sorted(
        entry.filename
        for entry in archive.infolist()
        if not entry.is_dir()
        and PurePosixPath(entry.filename).name == "episodes.json"
        and "recovery_dataset" not in {part.lower() for part in PurePosixPath(entry.filename).parts}
    )
    # An aggregate supersedes only its own descendants, regardless of task name.
    directories = {PurePosixPath(name).parent for name in candidates}
    return [
        name for name in candidates
        if not any(parent in directories for parent in PurePosixPath(name).parent.parents)
    ]


def _episode_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        return []
    files = []
    for directory, subdirectories, filenames in os.walk(path):
        subdirectories[:] = [name for name in subdirectories if name.lower() != "recovery_dataset"]
        if "episodes.json" in filenames:
            files.append(Path(directory) / "episodes.json")
            subdirectories.clear()  # Skip per-task copies beneath this aggregate.
    return sorted(files)


def _flatten_bc_observation(observation: dict[str, Any], feature_dim: int) -> np.ndarray:
    return flatten_observation(_without_simulator_state(observation), feature_dim)


def _without_simulator_state(observation: dict[str, Any]) -> dict[str, Any]:
    raw = observation.get("raw", observation)
    if not isinstance(raw, dict):
        return observation
    filtered_raw = {key: value for key, value in raw.items() if key not in {"env_states", "states", "state"}}
    if "raw" not in observation:
        return filtered_raw
    return {**observation, "raw": filtered_raw}


def create_task_bc_policy() -> TaskRoutedLinearBCPolicy:
    checkpoint_dir = os.getenv("NYSSA_TASK_BC_DIR", "checkpoints/bc_by_task")
    missing_task = os.getenv("NYSSA_TASK_BC_MISSING", "error").strip().lower()
    if missing_task not in {"error", "zero"}:
        raise RuntimeError("NYSSA_TASK_BC_MISSING must be either 'error' or 'zero'")
    return TaskRoutedLinearBCPolicy(checkpoint_dir, missing_task=missing_task)


def task_checkpoint_key(task_id: str) -> str:
    aliases = {
        "maniskill_pick_cube_joint": "maniskill_pick_cube",
        "maniskill_stack_cube_joint": "maniskill_stack_cube",
        "maniskill_push_cube_joint": "maniskill_push_cube",
    }
    return aliases.get(task_id, task_id.removesuffix("_joint"))


def _zero_action(observation: dict[str, Any]) -> np.ndarray:
    low, high, shape = action_bounds(observation)
    return np.clip(np.zeros(shape, dtype=float), low, high)


def _checkpoint_key(task_id: str) -> str:
    return task_checkpoint_key(task_id)
