from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from nyssa_bench.core.task import TaskSpec
from nyssa_bench.package_resources import config_root


@dataclass(frozen=True)
class Suite:
    """A collection of NyssaBench task specs evaluated as one benchmark."""

    suite_id: str
    description: str
    tasks: tuple[TaskSpec, ...]
    source_path: Path | None = None

    @classmethod
    def load(cls, suite_id_or_path: str | Path) -> "Suite":
        path = resolve_suite_path(suite_id_or_path)
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}

        task_ids = data.get("tasks", [])
        if not task_ids:
            raise ValueError(f"Suite '{path}' does not define any tasks")

        tasks = tuple(TaskSpec.load(task_id) for task_id in task_ids)
        return cls(
            suite_id=str(data.get("suite_id") or path.stem),
            description=str(data.get("description", "")),
            tasks=tasks,
            source_path=path,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_id": self.suite_id,
            "description": self.description,
            "tasks": [task.task_id for task in self.tasks],
        }

    def filter_tasks(self, task_ids: list[str] | tuple[str, ...] | None) -> "Suite":
        if not task_ids:
            return self
        requested = [str(task_id) for task_id in task_ids]
        requested_set = set(requested)
        tasks = tuple(
            task
            for task in self.tasks
            if task.task_id in requested_set or (task.source_path is not None and task.source_path.stem in requested_set)
        )
        found = {task.task_id for task in tasks}
        found.update(task.source_path.stem for task in tasks if task.source_path is not None)
        missing = [task_id for task_id in requested if task_id not in found]
        if missing:
            available = ", ".join(task.task_id for task in self.tasks)
            raise ValueError(
                f"Suite '{self.suite_id}' does not contain requested task(s): {', '.join(missing)}. "
                f"Available tasks: {available}"
            )
        return Suite(
            suite_id=self.suite_id,
            description=self.description,
            tasks=tasks,
            source_path=self.source_path,
        )


def resolve_suite_path(suite_id_or_path: str | Path) -> Path:
    candidate = Path(suite_id_or_path)
    if candidate.exists():
        return candidate

    suite_id = str(suite_id_or_path)
    search_roots = [config_root("suites")]
    for root in search_roots:
        path = root / f"{suite_id}.yaml"
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find suite '{suite_id_or_path}'")


def list_suites() -> list[str]:
    root = config_root("suites")
    if not root.exists():
        return []
    return sorted(path.stem for path in root.glob("*.yaml"))
