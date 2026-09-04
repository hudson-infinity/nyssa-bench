from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nyssa_bench.failures.protocol import FailureLedgerRecord
    from nyssa_bench.recovery.protocol import CounterfactualRecoveryRecord


@dataclass
class StepRecord:
    observation: dict[str, Any]
    action: Any
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation": _to_jsonable(self.observation),
            "action": _to_jsonable(self.action),
            "reward": _to_jsonable(self.reward),
            "terminated": self.terminated,
            "truncated": self.truncated,
            "info": _to_jsonable(self.info),
        }


@dataclass
class EpisodeResult:
    task_id: str
    episode_index: int
    seed: int
    success: bool
    failure_label: str | None
    metrics: dict[str, float]
    failure_label_source: str | None = None
    steps: list[StepRecord] = field(default_factory=list)
    replay_path: str | None = None
    failure_clip_path: str | None = None
    stressor_context: dict[str, Any] = field(default_factory=dict)
    failure_detector_context: dict[str, Any] = field(default_factory=dict)
    failure_ledger: FailureLedgerRecord | None = None
    counterfactual_recovery: list[CounterfactualRecoveryRecord] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "episode_index": self.episode_index,
            "seed": self.seed,
            "success": self.success,
            "failure_label": self.failure_label,
            "failure_label_source": self.failure_label_source,
            "metrics": self.metrics,
            "replay_path": self.replay_path,
            "failure_clip_path": self.failure_clip_path,
            "stressor_context": _to_jsonable(self.stressor_context),
            "failure_detector_context": _to_jsonable(self.failure_detector_context),
            "failure_ledger": self.failure_ledger.to_dict()
            if self.failure_ledger is not None
            else None,
            "counterfactual_recovery": [
                record.to_dict() for record in self.counterfactual_recovery
            ],
            "steps": [step.to_dict() for step in self.steps],
        }


def _to_jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)
