from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def export_metrics_csv(summary: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, Any]] = [
        ("format", summary.get("format")),
        ("episodes", summary.get("episodes", 0)),
        ("success_count", summary.get("success_count", 0)),
        ("success_rate", summary.get("success_rate", 0.0)),
        ("success_rate_ci95.low", _ci_value(summary.get("success_rate_ci95"), 0)),
        ("success_rate_ci95.high", _ci_value(summary.get("success_rate_ci95"), 1)),
        ("benchmark_tier", summary.get("benchmark_tier")),
        ("public_claim", summary.get("public_claim")),
        ("primary_failure_mode", summary.get("primary_failure_mode")),
    ]
    vector = summary.get("metric_vector", {})
    values = vector.get("values", {}) if isinstance(vector, dict) else {}
    definitions = vector.get("definitions", {}) if isinstance(vector, dict) else {}
    if isinstance(values, dict):
        rows.append(("metric_vector.format", vector.get("format")))
        for metric_id, measurement in sorted(values.items()):
            if not isinstance(measurement, dict):
                continue
            definition = (
                definitions.get(metric_id, {}) if isinstance(definitions, dict) else {}
            )
            rows.extend(
                [
                    (f"metric_vector.{metric_id}.status", measurement.get("status")),
                    (
                        f"metric_vector.{metric_id}.value",
                        _csv_value(measurement.get("value")),
                    ),
                    (
                        f"metric_vector.{metric_id}.ci95.low",
                        _ci_value(measurement.get("ci95"), 0),
                    ),
                    (
                        f"metric_vector.{metric_id}.ci95.high",
                        _ci_value(measurement.get("ci95"), 1),
                    ),
                    (
                        f"metric_vector.{metric_id}.sample_size",
                        measurement.get("sample_size"),
                    ),
                    (
                        f"metric_vector.{metric_id}.denominator",
                        _csv_value(measurement.get("denominator")),
                    ),
                    (
                        f"metric_vector.{metric_id}.direction",
                        definition.get("direction")
                        if isinstance(definition, dict)
                        else None,
                    ),
                    (
                        f"metric_vector.{metric_id}.reason",
                        measurement.get("reason"),
                    ),
                ]
            )
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


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value
