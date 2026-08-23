from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def export_metrics_csv(summary: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, Any]] = [
        ("episodes", summary.get("episodes", 0)),
        ("success_count", summary.get("success_count", 0)),
        ("success_rate", summary.get("success_rate", 0.0)),
        ("success_rate_ci95.low", _ci_value(summary.get("success_rate_ci95"), 0)),
        ("success_rate_ci95.high", _ci_value(summary.get("success_rate_ci95"), 1)),
        (
            "prototype_reliability_score",
            summary.get("prototype_reliability_score", 0.0),
        ),
        ("score_kind", summary.get("score_kind", "prototype_reliability_heuristic")),
        ("benchmark_tier", summary.get("benchmark_tier")),
        ("public_claim", summary.get("public_claim")),
        ("primary_failure_mode", summary.get("primary_failure_mode")),
    ]
    rows.extend(
        (f"metrics.{key}", value)
        for key, value in sorted(dict(summary.get("metrics", {})).items())
    )
    rows.extend(
        (f"failure_counts.{key}", value)
        for key, value in sorted(dict(summary.get("failure_counts", {})).items())
    )
    for task_id, task_summary in sorted(dict(summary.get("per_task", {})).items()):
        task_summary = dict(task_summary)
        rows.extend(
            [
                (f"per_task.{task_id}.episodes", task_summary.get("episodes", 0)),
                (
                    f"per_task.{task_id}.success_count",
                    task_summary.get("success_count", 0),
                ),
                (
                    f"per_task.{task_id}.success_rate",
                    task_summary.get("success_rate", 0.0),
                ),
                (
                    f"per_task.{task_id}.success_rate_ci95.low",
                    _ci_value(task_summary.get("success_rate_ci95"), 0),
                ),
                (
                    f"per_task.{task_id}.success_rate_ci95.high",
                    _ci_value(task_summary.get("success_rate_ci95"), 1),
                ),
                (
                    f"per_task.{task_id}.primary_failure_mode",
                    task_summary.get("primary_failure_mode"),
                ),
            ]
        )

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows(rows)
    return path


def _ci_value(value: Any, index: int) -> Any:
    if isinstance(value, (list, tuple)) and len(value) > index:
        return value[index]
    return None
