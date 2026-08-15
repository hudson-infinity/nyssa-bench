from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nyssa_bench.core.episode import EpisodeResult


RECOVERY_DATASET_FORMAT = "nyssa-recovery-dataset-v2"
RECOVERY_TARGET_SOURCES = frozenset({"expert", "recovery"})


def write_recovery_dataset(episodes: list[EpisodeResult], out_dir: str | Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    recovery_dir = out_dir / "recovery_dataset"
    recovery_dir.mkdir(parents=True, exist_ok=True)

    items = []
    target_steps = 0
    context_steps = 0
    target_sources: dict[str, int] = {}
    for episode in episodes:
        recovery_steps = []
        for step_index, step in enumerate(episode.steps):
            payload = step.to_dict()
            info = payload["info"] or {}
            executed_action_source = str(info.get("action_source") or "unknown").strip().lower()
            is_target = executed_action_source in RECOVERY_TARGET_SOURCES
            is_recovery_context = bool(
                info.get("expert_intervention")
                or info.get("recovery_attempted")
                or info.get("recovery_applied")
                or is_target
            )
            if not is_recovery_context:
                continue

            target_source = executed_action_source if is_target else None
            target_action = payload["action"] if is_target else None
            invalid_reason = None if is_target else _target_invalid_reason(info, executed_action_source)
            recovery_steps.append(
                {
                    "step_index": step_index,
                    "observation": payload["observation"],
                    "executed_action": payload["action"],
                    "executed_action_source": executed_action_source,
                    "target_action": target_action,
                    "target_source": target_source,
                    "target_valid": is_target,
                    "target_invalid_reason": invalid_reason,
                    "record_type": "supervised_target" if is_target else "negative_context",
                    "reward": step.reward,
                    "terminated": step.terminated,
                    "truncated": step.truncated,
                    "info": payload["info"],
                }
            )
            if is_target:
                target_steps += 1
                target_sources[executed_action_source] = target_sources.get(executed_action_source, 0) + 1
            else:
                context_steps += 1
        if recovery_steps:
            items.append(
                {
                    "task_id": episode.task_id,
                    "episode_index": episode.episode_index,
                    "seed": episode.seed,
                    "success": episode.success,
                    "failure_label": episode.failure_label,
                    "steps": recovery_steps,
                }
            )

    manifest: dict[str, Any] = {
        "format": RECOVERY_DATASET_FORMAT,
        "episodes": len(items),
        "steps": sum(len(item["steps"]) for item in items),
        "target_episodes": sum(any(step["target_valid"] for step in item["steps"]) for item in items),
        "target_steps": target_steps,
        "context_steps": context_steps,
        "target_sources": dict(sorted(target_sources.items())),
        "source": "recovery_context_and_supervised_targets",
    }
    manifest_path = recovery_dir / "manifest.json"
    json_path = recovery_dir / "episodes.json"
    jsonl_path = recovery_dir / "episodes.jsonl"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(items, indent=2) + "\n", encoding="utf-8")
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item) + "\n")
    return {"manifest": manifest_path, "episodes": json_path, "episodes_jsonl": jsonl_path}


def _target_invalid_reason(info: dict[str, Any], executed_action_source: str) -> str:
    if info.get("action_rejected") and executed_action_source == "policy":
        return "rejected_policy_action"
    if info.get("recovery_attempted") and not info.get("recovery_applied"):
        return "recovery_not_applied"
    return "ineligible_action_source"
