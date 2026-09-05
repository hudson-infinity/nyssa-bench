from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from nyssa_bench.policies.base import Policy


CHECKPOINT = Path(__file__).parent / "checkpoints" / "image_chunk_policy.json"
PREPROCESSING = b"rgb-uint8-hwc-to-float32-v1"


class PolicyAdapter(Policy):
    def __init__(self) -> None:
        payload = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
        if payload.get("format") != "nyssa-example-image-chunk-policy-v1":
            raise ValueError("unsupported example image checkpoint")
        self.action_dimension = int(payload["action_dimension"])
        self.prediction_horizon = int(payload["prediction_horizon"])
        self.calls = 0

    def reset(self, task: Any | None = None, seed: int | None = None) -> None:
        self.calls = 0

    def act(self, observation: dict[str, Any]) -> np.ndarray:
        self.calls += 1
        raw = observation.get("raw", {})
        image = raw.get("rgb") if isinstance(raw, dict) else None
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError("image policy requires one uint8 HWC RGB observation")
        contract = observation.get("action_space", {})
        if tuple(contract.get("shape", ())) != (self.action_dimension,):
            raise ValueError("live action dimension does not match checkpoint")
        return np.zeros(
            (self.prediction_horizon, self.action_dimension), dtype=np.float32
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "policy_id": "nyssa_example_image_chunk_policy",
            "policy_version": "1.0.0",
            "policy_family": "action_chunking",
            "checkpoint_id": "image-chunk-policy-v1",
            "checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
            "preprocessing_sha256": hashlib.sha256(PREPROCESSING).hexdigest(),
            "observation_modalities": ["rgb"],
            "action_representation": "environment_action",
            "action_dimension": self.action_dimension,
            "prediction_horizon": self.prediction_horizon,
            "execution_horizon": 2,
            "device": "cpu",
        }


def create_policy() -> PolicyAdapter:
    return PolicyAdapter()
