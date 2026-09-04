from __future__ import annotations

from typing import Any

from nyssa_bench.baselines.simple_bc import (
    KNNBCPolicy,
    LinearBCPolicy,
    SequenceKNNBCPolicy,
    create_bc_policy,
)
from nyssa_bench.policies.base import Policy
from nyssa_bench.policies.loaders import (
    call_model,
    load_callable_from_env,
    require_model,
)


class BCPolicy(Policy):
    """Adapter for a real behavior-cloning policy checkpoint or factory."""

    def __init__(self, model: Any | None = None) -> None:
        loaded = (
            model if model is not None else load_callable_from_env("NYSSA_BC_POLICY")
        )
        self.model = require_model(
            loaded if loaded is not None else create_bc_policy(),
            policy_name="BCPolicy",
            env_var="NYSSA_BC_POLICY",
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
        getter = getattr(self.model, "get_state", None)
        if callable(getter):
            return {"kind": "delegated", "state": getter()}
        if isinstance(self.model, (LinearBCPolicy, KNNBCPolicy, SequenceKNNBCPolicy)):
            return {"kind": "declared_stateless"}
        raise RuntimeError("loaded BC model does not expose restorable state")

    def set_state(self, state: Any) -> None:
        if not isinstance(state, dict):
            raise TypeError("BC policy state must be a mapping")
        if state.get("kind") == "declared_stateless":
            return
        if state.get("kind") != "delegated":
            raise ValueError(f"unsupported BC policy state kind: {state.get('kind')}")
        setter = getattr(self.model, "set_state", None)
        if not callable(setter):
            raise RuntimeError("loaded BC model cannot restore delegated state")
        setter(state.get("state"))

    def state_restore_capability(self) -> dict[str, Any]:
        if isinstance(self.model, (LinearBCPolicy, KNNBCPolicy, SequenceKNNBCPolicy)):
            return _exact_stateless_capability()
        return _delegated_capability(self.model)

    def seed_branch_rng(self, seed: int) -> bool:
        method = getattr(self.model, "seed_branch_rng", None)
        return bool(method(seed)) if callable(method) else False


def _delegated_capability(model: Any) -> dict[str, Any]:
    if not all(
        callable(getattr(model, name, None)) for name in ("get_state", "set_state")
    ):
        return {
            "supported": False,
            "fidelity": "unsupported",
            "captures_rng": False,
            "exact": False,
            "reason": "loaded BC model exposes no state contract",
        }
    method = getattr(model, "state_restore_capability", None)
    if callable(method):
        return dict(method())
    return {
        "supported": True,
        "fidelity": "qualified_delegated_model_state",
        "captures_rng": False,
        "exact": False,
        "reason": "model state is restorable but does not declare fidelity or RNG coverage",
    }


def _exact_stateless_capability() -> dict[str, Any]:
    return {
        "supported": True,
        "fidelity": "exact_declared_stateless_bc",
        "captures_rng": False,
        "exact": True,
        "reason": None,
    }
