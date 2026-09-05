from __future__ import annotations

from typing import Any

from nyssa_bench.baselines.lerobot_bc import create_lerobot_policy
from nyssa_bench.policies.base import Policy
from nyssa_bench.policies.loaders import (
    call_model,
    close_model,
    load_callable_from_env,
    model_metadata,
    require_model,
    reset_model,
)


class LeRobotPolicy(Policy):
    """LeRobot-compatible adapter for real loaded policies."""

    def __init__(self, model: Any | None = None) -> None:
        loaded = (
            model
            if model is not None
            else load_callable_from_env("NYSSA_LEROBOT_POLICY")
        )
        self.model = require_model(
            loaded if loaded is not None else create_lerobot_policy(),
            policy_name="LeRobotPolicy",
            env_var="NYSSA_LEROBOT_POLICY",
        )

    def reset(self, task: Any | None = None, seed: int | None = None) -> None:
        reset_model(self.model, task=task, seed=seed)

    def act(self, observation: dict[str, Any]) -> Any:
        return call_model(
            self.model,
            observation,
            ("select_action", "predict_action", "get_action", "act"),
        )

    def close(self) -> None:
        close_model(self.model)

    def metadata(self) -> dict[str, Any]:
        return model_metadata(self.model, adapter="lerobot")
