from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from nyssa_bench.datasets.export_json import EpisodePayload, episode_payload


def export_jsonl(episodes: Sequence[EpisodePayload], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(json.dumps(episode_payload(episode)) + "\n")
    return path
