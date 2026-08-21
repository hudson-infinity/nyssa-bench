from __future__ import annotations

from abc import ABC, abstractmethod
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from nyssa_bench.failures.protocol import FailureEventDraft


@runtime_checkable
class PolicyLike(Protocol):
    def act(self, observation: dict[str, Any]) -> Any:
        raise NotImplementedError


class Policy(ABC):
    """Policy adapter base class."""

    def reset(self, task: Any | None = None, seed: int | None = None) -> None:
        return None

    @abstractmethod
    def act(self, observation: dict[str, Any]) -> Any:
        raise NotImplementedError

    def close(self) -> None:
        return None

    def drain_failure_events(self) -> list[FailureEventDraft | dict[str, Any]]:
        """Return and clear queued policy-originated failure event drafts."""

        return []


def load_policy_from_path(path: str | Path) -> PolicyLike:
    path = Path(path)
    spec = spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load policy module from {path}")

    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    if hasattr(module, "create_policy"):
        policy = module.create_policy()
    elif hasattr(module, "PolicyAdapter"):
        policy = module.PolicyAdapter()
    else:
        raise ValueError("Policy module must define create_policy() or PolicyAdapter")

    required = getattr(policy, "act", None)
    if not callable(required):
        raise TypeError("Loaded policy must implement act(observation)")
    return policy
