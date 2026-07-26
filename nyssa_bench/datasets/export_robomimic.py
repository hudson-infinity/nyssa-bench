from __future__ import annotations

import json
from pathlib import Path

from typing import Any

import numpy as np

from nyssa_bench.baselines.features import flatten_observation, normalize_action, observation_numeric_values
from nyssa_bench.core.episode import EpisodeResult

MIN_OBSERVATION_PAYLOAD_COVERAGE = 0.95
MIN_FEATURE_VARIANCE = 1e-12
MIN_FEATURE_VARIANCE_STEPS = 32


def export_robomimic_hdf5(
    episodes: list[EpisodeResult],
    path: str | Path,
    *,
    feature_dim: int = 256,
) -> Path:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("RoboMimic export requires: uv sync --extra dataset") from exc

    validate_robomimic_observations(episodes, feature_dim=feature_dim)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        data.attrs["env_args"] = json.dumps(
            {
                "env_name": "NyssaFlat-v0",
                "type": 0,
                "env_kwargs": {
                    "description": "Flat low-dimensional NyssaBench export for offline RoboMimic BC training.",
                },
            }
        )
        total = 0
        for index, episode in enumerate(episodes):
            group = data.create_group(f"demo_{index}")
            observations = [flatten_observation(_without_simulator_state(step.observation), feature_dim) for step in episode.steps]
            action_size = _action_size(episode)
            actions = [normalize_action(step.action, action_size) for step in episode.steps]
            rewards = [step.reward for step in episode.steps]
            dones = [bool(step.terminated or step.truncated) for step in episode.steps]
            if dones:
                dones[-1] = True
            group.create_dataset("actions", data=np.asarray(actions, dtype=float))
            group.create_dataset("rewards", data=np.asarray(rewards, dtype=float))
            group.create_dataset("dones", data=np.asarray(dones, dtype=bool))
            obs_group = group.create_group("obs")
            next_obs_group = group.create_group("next_obs")
            obs_array = np.asarray(observations, dtype=float)
            obs_group.create_dataset("flat", data=obs_array)
            if len(obs_array) > 1:
                next_flat = np.vstack([obs_array[1:], obs_array[-1:]])
            else:
                next_flat = obs_array
            next_obs_group.create_dataset("flat", data=next_flat)
            group.attrs["num_samples"] = len(episode.steps)
            group.attrs["task_id"] = episode.task_id
            total += len(episode.steps)
        data.attrs["total"] = total
    return path


def validate_robomimic_observations(
    episodes: list[EpisodeResult],
    *,
    feature_dim: int,
    context: str = "RoboMimic dataset",
) -> dict[str, Any]:
    quality = robomimic_observation_quality(episodes, feature_dim=feature_dim)
    total_steps = int(quality["steps"])
    coverage = float(quality["observation_payload_coverage"])
    variance = float(quality["feature_variance_max"])
    if total_steps == 0:
        raise ValueError(f"{context} has no training steps")
    if coverage < MIN_OBSERVATION_PAYLOAD_COVERAGE:
        raise ValueError(
            f"{context} observation payload coverage is {coverage:.4f}; "
            f"at least {MIN_OBSERVATION_PAYLOAD_COVERAGE:.2f} is required. "
            "Use state-aligned rollout episodes with live policy observations, not action-only demonstrations."
        )
    if total_steps >= MIN_FEATURE_VARIANCE_STEPS and variance <= MIN_FEATURE_VARIANCE:
        raise ValueError(
            f"{context} has degenerate observation features; maximum feature variance is {variance:.3e}. "
            "Refusing to export a policy dataset with effectively constant observations."
        )
    return quality


def robomimic_observation_quality(episodes: list[EpisodeResult], *, feature_dim: int) -> dict[str, Any]:
    total_steps = 0
    payload_steps = 0
    feature_sum = np.zeros(feature_dim, dtype=float)
    feature_square_sum = np.zeros(feature_dim, dtype=float)
    for episode in episodes:
        for step in episode.steps:
            observation = _without_simulator_state(step.observation)
            total_steps += 1
            if observation_numeric_values(observation):
                payload_steps += 1
            features = flatten_observation(observation, feature_dim)
            feature_sum += features
            feature_square_sum += np.square(features)

    if total_steps:
        mean = feature_sum / total_steps
        variance = np.maximum(feature_square_sum / total_steps - np.square(mean), 0.0)
        variance_max = float(np.max(variance))
        active_features = int(np.count_nonzero(variance > MIN_FEATURE_VARIANCE))
    else:
        variance_max = 0.0
        active_features = 0
    return {
        "steps": total_steps,
        "steps_with_observation_payload": payload_steps,
        "observation_payload_coverage": payload_steps / total_steps if total_steps else 0.0,
        "feature_dim": feature_dim,
        "active_feature_dimensions": active_features,
        "feature_variance_max": variance_max,
    }


def _action_size(episode: EpisodeResult) -> int:
    for step in episode.steps:
        spec = step.observation.get("action_space", {})
        shape = spec.get("shape")
        if shape:
            size = 1
            for value in shape:
                size *= int(value)
            return size
    return 1


def _without_simulator_state(observation: dict[str, Any]) -> dict[str, Any]:
    raw = observation.get("raw", observation)
    if not isinstance(raw, dict):
        return observation
    filtered_raw = {key: value for key, value in raw.items() if key not in {"env_states", "states", "state"}}
    if "raw" not in observation:
        return filtered_raw
    return {**observation, "raw": filtered_raw}
