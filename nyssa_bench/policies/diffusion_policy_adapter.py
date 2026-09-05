from __future__ import annotations

from typing import Any

from nyssa_bench.policies.base import Policy
from nyssa_bench.policies.loaders import (
    call_model,
    close_model,
    load_callable_from_env,
    model_metadata,
    require_model,
    reset_model,
)


class DiffusionPolicyAdapter(Policy):
    def __init__(self, model: Any | None = None) -> None:
        self.model = require_model(
            model
            if model is not None
            else load_callable_from_env("NYSSA_DIFFUSION_POLICY"),
            policy_name="DiffusionPolicyAdapter",
            env_var="NYSSA_DIFFUSION_POLICY",
        )

    def reset(self, task: Any | None = None, seed: int | None = None) -> None:
        reset_model(self.model, task=task, seed=seed)

    def act(self, observation: dict[str, Any]) -> Any:
        return call_model(
            self.model,
            observation,
            ("predict_action", "select_action", "get_action", "act"),
        )

    def close(self) -> None:
        close_model(self.model)

    def metadata(self) -> dict[str, Any]:
        return model_metadata(self.model, adapter="diffusion")
