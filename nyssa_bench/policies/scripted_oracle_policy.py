from __future__ import annotations

from typing import Any

from nyssa_bench.baselines.scripted_maniskill import create_scripted_oracle
from nyssa_bench.policies.base import Policy
from nyssa_bench.policies.loaders import (
    call_model,
    load_callable_from_env,
    require_model,
)


class ScriptedOraclePolicy(Policy):
    """Adapter for task-specific scripted/oracle controllers."""

    def __init__(self, controller: Any | None = None) -> None:
        loaded = (
            controller
            if controller is not None
            else load_callable_from_env("NYSSA_SCRIPTED_ORACLE_POLICY")
        )
        self.controller = require_model(
            loaded if loaded is not None else create_scripted_oracle(),
            policy_name="ScriptedOraclePolicy",
            env_var="NYSSA_SCRIPTED_ORACLE_POLICY",
        )

    def reset(self, task: Any | None = None, seed: int | None = None) -> None:
        reset = getattr(self.controller, "reset", None)
        if callable(reset):
            reset(task=task, seed=seed)

    def act(self, observation: dict[str, Any]) -> Any:
        return call_model(
            self.controller,
            observation,
            ("act", "get_action", "select_action", "predict_action"),
        )

    def close(self) -> None:
        close = getattr(self.controller, "close", None)
        if callable(close):
            close()

    def get_state(self) -> Any:
        getter = getattr(self.controller, "get_state", None)
        return getter() if callable(getter) else None

    def set_state(self, state: Any) -> None:
        setter = getattr(self.controller, "set_state", None)
        if not callable(setter):
            if state is not None:
                raise RuntimeError("scripted controller does not support state restore")
            return
        setter(state)

    def state_restore_capability(self) -> dict[str, Any]:
        method = getattr(self.controller, "state_restore_capability", None)
        if callable(method):
            return dict(method())
        supported = all(
            callable(getattr(self.controller, name, None))
            for name in ("get_state", "set_state")
        )
        return {
            "supported": supported,
            "fidelity": "declared_controller_state" if supported else "unsupported",
            "captures_rng": False,
            "exact": False,
            "reason": None if supported else "controller exposes no state contract",
        }

    def seed_branch_rng(self, seed: int) -> bool:
        method = getattr(self.controller, "seed_branch_rng", None)
        return bool(method(seed)) if callable(method) else False
