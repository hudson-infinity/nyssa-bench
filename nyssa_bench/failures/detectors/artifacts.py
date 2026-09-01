from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


FAILURE_DETECTOR_RUN_MANIFEST_FORMAT = "nyssa-failure-detector-run-manifest-v1"


def summarize_failure_detectors(episodes: list[Any]) -> dict[str, Any]:
    status_counts: Counter[str] = Counter()
    event_counts: Counter[str] = Counter()
    contracts: dict[tuple[str, str], dict[str, Any]] = {}
    episodes_with_supported_detectors = 0

    for episode in episodes:
        context = getattr(episode, "failure_detector_context", {}) or {}
        supported_in_episode = False
        for entry in context.get("detectors", []):
            contract = dict(entry.get("contract", {}))
            detector_id = str(contract.get("detector_id", "unknown"))
            detector_version = str(contract.get("detector_version", "unknown"))
            key = (detector_id, detector_version)
            existing = contracts.get(key)
            if existing is not None and existing != contract:
                raise ValueError(
                    f"Conflicting detector contract for {detector_id}@{detector_version}"
                )
            contracts[key] = contract
            status = str(entry.get("support", {}).get("status", "unknown"))
            status_counts[status] += 1
            supported_in_episode |= status == "supported"
            event_counts[detector_id] += int(entry.get("emitted_event_count", 0))
        episodes_with_supported_detectors += supported_in_episode

    return {
        "format": FAILURE_DETECTOR_RUN_MANIFEST_FORMAT,
        "episodes": len(episodes),
        "episodes_with_supported_detectors": episodes_with_supported_detectors,
        "status_counts": dict(sorted(status_counts.items())),
        "emitted_event_counts": dict(sorted(event_counts.items())),
        "contracts": [contracts[key] for key in sorted(contracts)],
    }


def write_failure_detector_manifest(episodes: list[Any], out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    payload = {
        "format": FAILURE_DETECTOR_RUN_MANIFEST_FORMAT,
        "summary": summarize_failure_detectors(episodes),
        "episodes": [
            {
                "task_id": episode.task_id,
                "episode_index": episode.episode_index,
                "seed": episode.seed,
                "failure_detector_context": getattr(
                    episode, "failure_detector_context", {}
                ),
            }
            for episode in episodes
        ],
    }
    path = out_dir / "failure_detector_manifest.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path
