from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


PreferenceChoice = Literal["policy_a", "policy_b", "tie", "invalid"]


@dataclass(frozen=True)
class PreferenceRecord:
    """Human or automated preference for a paired policy rollout."""

    task_id: str
    seed: int
    episode_index: int
    choice: PreferenceChoice
    reason: str = ""
    evaluator_id: str | None = None
    blinded: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "seed": self.seed,
            "episode_index": self.episode_index,
            "choice": self.choice,
            "reason": self.reason,
            "evaluator_id": self.evaluator_id,
            "blinded": self.blinded,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PreferenceRecord":
        choice = str(data.get("choice", "invalid"))
        if choice not in {"policy_a", "policy_b", "tie", "invalid"}:
            choice = "invalid"
        return cls(
            task_id=str(data["task_id"]),
            seed=int(data["seed"]),
            episode_index=int(data["episode_index"]),
            choice=choice,  # type: ignore[arg-type]
            reason=str(data.get("reason", "")),
            evaluator_id=data.get("evaluator_id"),
            blinded=bool(data.get("blinded", True)),
        )
