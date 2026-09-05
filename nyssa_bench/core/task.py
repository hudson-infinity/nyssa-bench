from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from nyssa_bench.package_resources import resource_root


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent


@dataclass(frozen=True)
class TaskSpec:
    """Parsed NyssaBench task specification."""

    task_id: str
    engine: str
    robot: str
    scene: str
    description: str
    objects: list[dict[str, Any]] = field(default_factory=list)
    success: dict[str, Any] = field(default_factory=dict)
    randomization: dict[str, Any] = field(default_factory=dict)
    observation: dict[str, Any] = field(default_factory=dict)
    action: dict[str, Any] = field(default_factory=dict)
    goal: dict[str, Any] = field(default_factory=dict)
    experts: dict[str, Any] = field(default_factory=dict)
    ood_splits: dict[str, Any] = field(default_factory=dict)
    metrics: list[str] = field(default_factory=list)
    failure_labels: list[str] = field(default_factory=list)
    source_path: Path | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], source_path: Path | None = None) -> "TaskSpec":
        required = ["task_id", "engine", "robot", "scene", "description"]
        missing = [key for key in required if not data.get(key)]
        if missing:
            raise ValueError(f"Task spec missing required fields: {', '.join(missing)}")

        return cls(
            task_id=str(data["task_id"]),
            engine=str(data["engine"]),
            robot=str(data["robot"]),
            scene=str(data["scene"]),
            description=str(data["description"]),
            objects=list(data.get("objects", [])),
            success=dict(data.get("success", {})),
            randomization=dict(data.get("randomization", {})),
            observation=dict(data.get("observation", {})),
            action=dict(data.get("action", {})),
            goal=dict(data.get("goal", {})),
            experts=dict(data.get("experts", {})),
            ood_splits=dict(data.get("ood_splits", {})),
            metrics=list(data.get("metrics", [])),
            failure_labels=list(data.get("failure_labels", [])),
            source_path=source_path,
        )

    @classmethod
    def load(cls, task_id_or_path: str | Path) -> "TaskSpec":
        path = resolve_task_path(task_id_or_path)
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return cls.from_dict(data, source_path=path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "engine": self.engine,
            "robot": self.robot,
            "scene": self.scene,
            "description": self.description,
            "objects": self.objects,
            "success": self.success,
            "randomization": self.randomization,
            "observation": self.observation,
            "action": self.action,
            "goal": self.goal,
            "experts": self.experts,
            "ood_splits": self.ood_splits,
            "metrics": self.metrics,
            "failure_labels": self.failure_labels,
        }


def resolve_task_path(task_id_or_path: str | Path) -> Path:
    candidate = Path(task_id_or_path)
    if candidate.exists():
        return candidate

    task_id = str(task_id_or_path)
    search_roots = _task_roots()
    for root in search_roots:
        for path in root.rglob("*.yaml"):
            if path.stem == task_id:
                return path
    raise FileNotFoundError(f"Could not find task spec '{task_id_or_path}'")


def list_tasks() -> list[str]:
    roots = _task_roots()
    task_ids: set[str] = set()
    for root in roots:
        if root.exists():
            task_ids.update(path.stem for path in root.rglob("*.yaml"))
    return sorted(task_ids)


def _task_roots() -> list[Path]:
    roots = [PACKAGE_ROOT / "tasks"]
    legacy = resource_root("configs") / "tasks"
    if legacy.is_dir():
        roots.append(legacy)
    return roots
