from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from nyssa_bench.baselines.features import fit_action_to_observation, flatten_observation
from nyssa_bench.baselines.simple_bc import task_checkpoint_key
from nyssa_bench.policies.base import Policy


class DemoReplayPolicy(Policy):
    """Task-routed replay of imported demonstration actions.

    This is a strong teacher/reference baseline for checking that imported
    planner demonstrations can execute through Nyssa's live evaluator. It is not
    a learned policy: it replays successful demo action sequences by task.
    """

    def __init__(self, demo_dir: str | Path | None = None) -> None:
        self.demo_dir = Path(demo_dir or os.getenv("NYSSA_DEMO_REPLAY_DIR", "benchmark_results/maniskill_official_demos_import_v2"))
        self.feature_dim = int(os.getenv("NYSSA_DEMO_REPLAY_FEATURE_DIM", "512"))
        self.current_task_id: str | None = None
        self.current_actions: list[Any] = []
        self.current_episode: _DemoEpisode | None = None
        self.pending_seed: int | None = None
        self.cursor = 0
        self._episodes_by_task: dict[str, list[_DemoEpisode]] = {}

    def reset(self, task: Any | None = None, seed: int | None = None) -> None:
        self.current_task_id = task_checkpoint_key(str(getattr(task, "task_id", "")))
        self.pending_seed = seed
        self.current_actions = []
        self.current_episode = None
        self.cursor = 0

    def initial_state(self, observation: dict[str, Any] | None = None) -> Any | None:
        if self.current_episode is None and self.current_task_id:
            self.current_episode = self._select_episode(self.current_task_id, observation or {})
            self.current_actions = self.current_episode.actions if self.current_episode else []
        return self.current_episode.initial_state if self.current_episode else None

    def act(self, observation: dict[str, Any]) -> Any:
        if not self.current_actions and self.current_task_id:
            self.current_episode = self._select_episode(self.current_task_id, observation)
            self.current_actions = self.current_episode.actions if self.current_episode else []
        if not self.current_actions:
            return fit_action_to_observation(0.0, observation)
        index = min(self.cursor, len(self.current_actions) - 1)
        self.cursor += 1
        return fit_action_to_observation(self.current_actions[index], observation)

    def get_state(self) -> dict[str, Any]:
        return {
            "current_task_id": self.current_task_id,
            "current_actions": copy.deepcopy(self.current_actions),
            "current_episode": copy.deepcopy(self.current_episode),
            "pending_seed": self.pending_seed,
            "cursor": self.cursor,
        }

    def set_state(self, state: Any) -> None:
        if not isinstance(state, dict):
            raise TypeError("demo replay policy state must be a mapping")
        self.current_task_id = state.get("current_task_id")
        self.current_actions = copy.deepcopy(state.get("current_actions", []))
        self.current_episode = copy.deepcopy(state.get("current_episode"))
        self.pending_seed = state.get("pending_seed")
        self.cursor = int(state.get("cursor", 0))

    def state_restore_capability(self) -> dict[str, Any]:
        return {
            "supported": True,
            "fidelity": "exact_demo_replay_cursor",
            "captures_rng": False,
            "exact": True,
            "reason": None,
        }

    def _select_episode(self, task_id: str, observation: dict[str, Any]) -> "_DemoEpisode | None":
        episodes = self._load_task_episodes(task_id)
        if not episodes:
            return None
        query = flatten_observation(_without_initial_state(observation), self.feature_dim)
        distances = [float(np.sum((episode.initial_features - query) ** 2)) for episode in episodes]
        best_distance = min(distances)
        candidates = [index for index, distance in enumerate(distances) if np.isclose(distance, best_distance)]
        selected = candidates[int(self.pending_seed or 0) % len(candidates)]
        return episodes[selected]

    def _load_task_episodes(self, task_id: str) -> list["_DemoEpisode"]:
        if task_id not in self._episodes_by_task:
            path = self.demo_dir / task_id / "episodes.json"
            if not path.exists():
                raise RuntimeError(
                    f"Demo replay episodes not found for task '{task_id}': {path}. "
                    "Set NYSSA_DEMO_REPLAY_DIR to an imported ManiSkill demo directory."
                )
            payload = json.loads(path.read_text(encoding="utf-8"))
            self._episodes_by_task[task_id] = [
                _demo_episode(episode, feature_dim=self.feature_dim)
                for episode in payload
                if bool(episode.get("success", False)) and episode.get("steps")
            ]
        return self._episodes_by_task[task_id]


class _DemoEpisode:
    def __init__(self, initial_features: np.ndarray, initial_state: Any | None, actions: list[Any]) -> None:
        self.initial_features = initial_features
        self.initial_state = initial_state
        self.actions = actions


def _demo_episode(episode: dict[str, Any], *, feature_dim: int) -> _DemoEpisode:
    steps = list(episode.get("steps", []))
    initial_observation = steps[0].get("observation", {}) if steps else {}
    return _DemoEpisode(
        initial_features=flatten_observation(_without_initial_state(initial_observation), feature_dim),
        initial_state=_initial_state(initial_observation),
        actions=[step.get("action") for step in steps],
    )


def _initial_state(observation: dict[str, Any]) -> Any | None:
    raw = observation.get("raw", observation)
    if isinstance(raw, dict):
        for key in ("env_states", "states", "state"):
            if key in raw:
                return raw[key]
    for key in ("env_states", "states", "state"):
        if key in observation:
            return observation[key]
    return None


def _without_initial_state(observation: dict[str, Any]) -> dict[str, Any]:
    raw = observation.get("raw", observation)
    if not isinstance(raw, dict):
        return observation
    filtered_raw = {key: value for key, value in raw.items() if key not in {"env_states", "states", "state"}}
    if "raw" not in observation:
        return filtered_raw
    return {**observation, "raw": filtered_raw}
