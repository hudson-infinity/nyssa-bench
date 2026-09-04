from __future__ import annotations

from typing import Any

from nyssa_bench.baselines.simple_bc import (
    TaskRoutedLinearBCPolicy,
    create_task_bc_policy,
)
from nyssa_bench.policies.bc_policy import _delegated_capability
from nyssa_bench.policies.base import Policy
from nyssa_bench.policies.loaders import (
    call_model,
    load_callable_from_env,
    require_model,
)


class TaskBCPolicy(Policy):
    """Task-routed behavior cloning policy for suites with task-specific checkpoints."""

    def __init__(self, model: Any | None = None) -> None:
        loaded = (
            model
            if model is not None
            else load_callable_from_env("NYSSA_TASK_BC_POLICY")
        )
        self.model = require_model(
            loaded if loaded is not None else create_task_bc_policy(),
            policy_name="TaskBCPolicy",
            env_var="NYSSA_TASK_BC_POLICY",
        )

    def reset(self, task: Any | None = None, seed: int | None = None) -> None:
        reset = getattr(self.model, "reset", None)
        if callable(reset):
            reset(task=task, seed=seed)

    def act(self, observation: dict[str, Any]) -> Any:
        return call_model(
            self.model,
            observation,
            ("predict_action", "select_action", "get_action", "act"),
        )

    def close(self) -> None:
        close = getattr(self.model, "close", None)
        if callable(close):
            close()

    def get_state(self) -> Any:
        if isinstance(self.model, TaskRoutedLinearBCPolicy):
            return {
                "kind": "task_routed_bc",
                "current_task_id": self.model.current_task_id,
            }
        getter = getattr(self.model, "get_state", None)
        if callable(getter):
            return {"kind": "delegated", "state": getter()}
        raise RuntimeError("loaded task BC model does not expose restorable state")

    def set_state(self, state: Any) -> None:
        if not isinstance(state, dict):
            raise TypeError("task BC policy state must be a mapping")
        if state.get("kind") == "task_routed_bc" and isinstance(
            self.model, TaskRoutedLinearBCPolicy
        ):
            self.model.current_task_id = state.get("current_task_id")
            return
        if state.get("kind") != "delegated":
            raise ValueError(
                f"unsupported task BC policy state kind: {state.get('kind')}"
            )
        setter = getattr(self.model, "set_state", None)
        if not callable(setter):
            raise RuntimeError("loaded task BC model cannot restore delegated state")
        setter(state.get("state"))

    def state_restore_capability(self) -> dict[str, Any]:
        if isinstance(self.model, TaskRoutedLinearBCPolicy):
            return {
                "supported": True,
                "fidelity": "exact_task_routed_bc_state",
                "captures_rng": False,
                "exact": True,
                "reason": None,
            }
        return _delegated_capability(self.model)

    def seed_branch_rng(self, seed: int) -> bool:
        method = getattr(self.model, "seed_branch_rng", None)
        return bool(method(seed)) if callable(method) else False
