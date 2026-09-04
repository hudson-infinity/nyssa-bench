from __future__ import annotations

import random
from typing import Any

from nyssa_bench.policies.base import Policy


class RandomPolicy(Policy):
    def __init__(self) -> None:
        self.rng = random.Random()

    def reset(self, task: Any | None = None, seed: int | None = None) -> None:
        self.rng.seed(seed)

    def get_state(self) -> object:
        return self.rng.getstate()

    def set_state(self, state: Any) -> None:
        self.rng.setstate(state)

    def state_restore_capability(self) -> dict[str, Any]:
        return {
            "supported": True,
            "fidelity": "exact_policy_rng",
            "reason": None,
        }

    def seed_branch_rng(self, seed: int) -> bool:
        self.rng.seed(int(seed))
        return True

    def act(self, observation: dict[str, Any]) -> Any:
        action_space = observation.get("action_space")
        if isinstance(action_space, dict):
            return self._sample_action_space(action_space)
        return self.rng.uniform(-0.35, 0.35)

    def _sample_action_space(self, action_space: dict[str, Any]) -> Any:
        if action_space.get("type") == "discrete":
            return self.rng.randrange(int(action_space["n"]))

        if action_space.get("type") == "box":
            try:
                import numpy as np
            except ImportError:
                return self.rng.uniform(-0.35, 0.35)
            low = np.array(action_space["low"], dtype=float)
            high = np.array(action_space["high"], dtype=float)
            low = np.where(np.isfinite(low), low, -1.0)
            high = np.where(np.isfinite(high), high, 1.0)
            values = [self.rng.random() for _ in range(int(low.size))]
            sample = low + np.array(values).reshape(low.shape) * (high - low)
            dtype = action_space.get("dtype")
            if dtype:
                try:
                    sample = sample.astype(dtype)
                except TypeError:
                    pass
            return sample

        return self.rng.uniform(-0.35, 0.35)
