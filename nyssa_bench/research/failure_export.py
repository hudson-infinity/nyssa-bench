from __future__ import annotations

from nyssa_bench.core.episode import EpisodeResult


def failure_episode_indices(episodes: list[EpisodeResult], *, failure_label: str | None = None) -> list[int]:
    """Return episode indices for failed episodes, optionally filtered by label."""

    indices: list[int] = []
    for episode in episodes:
        if episode.success:
            continue
        if failure_label is not None and episode.failure_label != failure_label:
            continue
        indices.append(episode.episode_index)
    return indices
