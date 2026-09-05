from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from nyssa_bench.policies.base import Policy


CHECKPOINT = Path(__file__).parent / "checkpoints" / "state_policy.json"
PREPROCESSING = b"state-policy-preprocess-v1"


class PolicyAdapter(Policy):
    def __init__(self) -> None:
        payload = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
        if payload.get("format") != "nyssa-example-state-policy-v1":
            raise ValueError("unsupported example state checkpoint")
        self.action_dimension = int(payload["action_dimension"])
        self.calls = 0

    def reset(self, task: Any | None = None, seed: int | None = None) -> None:
        self.calls = 0

    def act(self, observation: dict[str, Any]) -> np.ndarray:
        self.calls += 1
        contract = observation.get("action_space", {})
        if tuple(contract.get("shape", ())) != (self.action_dimension,):
            raise ValueError("live action dimension does not match checkpoint")
        return np.zeros((self.action_dimension,), dtype=np.float32)

    def metadata(self) -> dict[str, Any]:
        return {
            "policy_id": "nyssa_example_state_policy",
            "policy_version": "1.0.0",
            "policy_family": "deterministic_control",
            "checkpoint_id": "state-policy-v1",
            "checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
            "preprocessing_sha256": hashlib.sha256(PREPROCESSING).hexdigest(),
            "observation_modalities": ["state"],
            "action_representation": "environment_action",
            "action_dimension": self.action_dimension,
            "prediction_horizon": 1,
            "execution_horizon": 1,
            "device": "cpu",
        }


def create_policy() -> PolicyAdapter:
    return PolicyAdapter()
