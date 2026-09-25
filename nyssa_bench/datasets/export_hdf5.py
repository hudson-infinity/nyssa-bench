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
        occurrences: dict[int, int] = {}
        for episode in episodes:
            occurrence = occurrences.get(episode.episode_index, 0)
            occurrences[episode.episode_index] = occurrence + 1
            name = f"episode_{episode.episode_index:04d}"
            if occurrence:
                name += f"_duplicate_{occurrence:04d}"
            group = handle.create_group(name)
            group.attrs["task_id"] = episode.task_id
            group.attrs["episode_index"] = episode.episode_index
            group.attrs["seed"] = episode.seed
            group.attrs["success"] = episode.success
            group.attrs["failure_label"] = episode.failure_label or ""
            group.create_dataset("rewards", data=[step.reward for step in episode.steps])
    return path
