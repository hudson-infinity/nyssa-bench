from __future__ import annotations

from pathlib import Path

from nyssa_bench.core.episode import EpisodeResult


def export_hdf5(episodes: list[EpisodeResult], path: str | Path) -> Path:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            'HDF5 export requires: python -m pip install "nyssa-bench[dataset]"'
        ) from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        for episode in episodes:
            group = handle.create_group(f"episode_{episode.episode_index:04d}")
            group.attrs["task_id"] = episode.task_id
            group.attrs["success"] = episode.success
            group.attrs["failure_label"] = episode.failure_label or ""
            group.create_dataset("rewards", data=[step.reward for step in episode.steps])
    return path
