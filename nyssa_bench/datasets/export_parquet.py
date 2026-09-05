from __future__ import annotations

from pathlib import Path

from nyssa_bench.core.episode import EpisodeResult


def export_parquet(episodes: list[EpisodeResult], path: str | Path) -> Path:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            'Parquet export requires: python -m pip install "nyssa-bench[dataset]"'
        ) from exc

    rows = []
    for episode in episodes:
        for index, step in enumerate(episode.steps):
            rows.append(
                {
                    "task_id": episode.task_id,
                    "episode_index": episode.episode_index,
                    "step_index": index,
                    "action": step.action,
                    "reward": step.reward,
                    "terminated": step.terminated,
                    "truncated": step.truncated,
                }
            )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path)
    return path
