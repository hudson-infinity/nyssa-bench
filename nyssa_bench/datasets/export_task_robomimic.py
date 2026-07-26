from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nyssa_bench.baselines.robomimic_bc import write_robomimic_bc_config
from nyssa_bench.baselines.simple_bc import load_episode_sources, task_checkpoint_key
from nyssa_bench.core.episode import EpisodeResult, StepRecord
from nyssa_bench.datasets.export_robomimic import export_robomimic_hdf5, validate_robomimic_observations


def export_task_robomimic(
    sources: list[str | Path],
    *,
    out_dir: str | Path,
    config_dir: str | Path | None = None,
    feature_dim: int = 256,
    epochs: int = 50,
    batch_size: int = 64,
    seed: int = 1,
    learning_rate: float = 1e-4,
    success_only: bool = True,
) -> dict[str, dict[str, Path]]:
    episodes = load_episode_sources(sources)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for episode in episodes:
        if success_only and not bool(episode.get("success", True)):
            continue
        task_id = str(episode.get("task_id", "")).strip()
        if not task_id:
            continue
        grouped.setdefault(task_checkpoint_key(task_id), []).append(episode)
    if not grouped:
        raise ValueError("No task-labeled episodes found for RoboMimic export")

    episode_results = {task: _episode_results(task_episodes) for task, task_episodes in grouped.items()}
    observation_quality = {
        task: validate_robomimic_observations(
            results,
            feature_dim=feature_dim,
            context=f"Task '{task}' RoboMimic dataset",
        )
        for task, results in episode_results.items()
    }

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir = Path(config_dir) if config_dir is not None else output_dir / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, dict[str, Path]] = {}
    for task_key, task_episodes in sorted(grouped.items()):
        hdf5_path = output_dir / f"{task_key}.hdf5"
        config_path = cfg_dir / f"{task_key}_bc.json"
        export_robomimic_hdf5(episode_results[task_key], hdf5_path, feature_dim=feature_dim)
        write_robomimic_bc_config(
            data=hdf5_path,
            out=config_path,
            output_dir=output_dir / "checkpoints" / task_key,
            name=f"nyssa_{task_key}_bc",
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
            learning_rate=learning_rate,
        )
        artifacts[task_key] = {"hdf5": hdf5_path, "config": config_path}

    manifest_path = output_dir / "task_robomimic_manifest.json"
    manifest = {
        "format": "nyssa-task-robomimic-export-v2",
        "sources": [str(Path(source)) for source in sources],
        "feature_dim": feature_dim,
        "success_only": success_only,
        "observation_quality": observation_quality,
        "tasks": {
            task: {label: path.as_posix() for label, path in paths.items()} for task, paths in artifacts.items()
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    artifacts["_manifest"] = {"manifest": manifest_path}
    return artifacts


def _episode_results(episodes: list[dict[str, Any]]) -> list[EpisodeResult]:
    return [_episode_result(episode, index) for index, episode in enumerate(episodes)]


def _episode_result(episode: dict[str, Any], index: int) -> EpisodeResult:
    steps = [_step_record(step) for step in episode.get("steps", [])]
    return EpisodeResult(
        task_id=str(episode.get("task_id", "")),
        episode_index=int(episode.get("episode_index", index)),
        seed=int(episode.get("seed", index)),
        success=bool(episode.get("success", True)),
        failure_label=episode.get("failure_label"),
        failure_label_source=episode.get("failure_label_source"),
        metrics={str(key): float(value) for key, value in dict(episode.get("metrics", {})).items()},
        replay_path=episode.get("replay_path"),
        failure_clip_path=episode.get("failure_clip_path"),
        steps=steps,
    )


def _step_record(step: dict[str, Any]) -> StepRecord:
    return StepRecord(
        observation=dict(step.get("observation", {})),
        action=step.get("action"),
        reward=float(step.get("reward", 0.0)),
        terminated=bool(step.get("terminated", False)),
        truncated=bool(step.get("truncated", False)),
        info=dict(step.get("info", {})),
    )
