from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from nyssa_bench.core.episode import EpisodeResult


EpisodePayload = EpisodeResult | dict[str, Any]


def episode_payload(episode: EpisodePayload) -> dict[str, Any]:
    return episode if isinstance(episode, dict) else episode.to_dict()


def episode_payloads(episodes: Sequence[EpisodePayload]) -> list[dict[str, Any]]:
    return [episode_payload(episode) for episode in episodes]


def export_json(episodes: Sequence[EpisodePayload], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(episode_payloads(episodes), handle, indent=2)
    return path
