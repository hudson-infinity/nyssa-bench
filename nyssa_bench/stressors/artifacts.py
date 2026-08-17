from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from nyssa_bench.core.episode import EpisodeResult


STRESSOR_MANIFEST_FORMAT = "nyssa-stressor-manifest-v1"


def summarize_stressor_execution(episodes: list[EpisodeResult]) -> dict[str, Any]:
    status_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    requested_ids: set[str] = set()
    applied_ids: set[str] = set()
    unsupported_ids: set[str] = set()
    skipped_ids: set[str] = set()
    by_task: dict[str, dict[str, Any]] = {}

    for episode in episodes:
        context = episode.stressor_context or {}
        task_summary = by_task.setdefault(
            episode.task_id,
            {
                "conditions": set(),
                "requested_stressors": set(),
                "applied_stressors": set(),
                "unsupported_stressors": set(),
                "skipped_stressors": set(),
            },
        )
        task_summary["conditions"].add(str(context.get("condition_id", "clean")))
        for application in context.get("applications", []):
            if not isinstance(application, dict):
                continue
            stressor_id = str(application.get("stressor_id", "unknown"))
            status = str(application.get("status", "requested"))
            category = str(application.get("category", "unknown"))
            status_counts[status] += 1
            category_counts[category] += 1
            requested_ids.add(stressor_id)
            task_summary["requested_stressors"].add(stressor_id)
            if status == "applied":
                applied_ids.add(stressor_id)
                task_summary["applied_stressors"].add(stressor_id)
            elif status == "unsupported":
                unsupported_ids.add(stressor_id)
                task_summary["unsupported_stressors"].add(stressor_id)
            elif status == "skipped":
                skipped_ids.add(stressor_id)
                task_summary["skipped_stressors"].add(stressor_id)

    serialized_by_task = {
        task_id: {key: sorted(value) for key, value in values.items()}
        for task_id, values in sorted(by_task.items())
    }
    return {
        "format": STRESSOR_MANIFEST_FORMAT,
        "episodes": len(episodes),
        "requested_stressors": sorted(requested_ids),
        "applied_stressors": sorted(applied_ids),
        "unsupported_stressors": sorted(unsupported_ids),
        "skipped_stressors": sorted(skipped_ids),
        "status_counts": dict(sorted(status_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "all_requests_resolved": not unsupported_ids,
        "by_task": serialized_by_task,
    }


def write_stressor_manifest(
    episodes: list[EpisodeResult],
    out_dir: str | Path,
    *,
    configured: dict[str, Any] | None,
) -> Path:
    out_dir = Path(out_dir)
    payload = {
        "format": STRESSOR_MANIFEST_FORMAT,
        "configured": configured,
        "summary": summarize_stressor_execution(episodes),
        "episodes": [
            {
                "task_id": episode.task_id,
                "episode_index": episode.episode_index,
                "seed": episode.seed,
                "stressor_context": episode.stressor_context,
            }
            for episode in episodes
        ],
    }
    path = out_dir / "stressor_manifest.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path
