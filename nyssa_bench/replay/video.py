from __future__ import annotations

import json
import html
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from nyssa_bench.core.episode import EpisodeResult


def write_episode_video(frames: list[Any], out_dir: str | Path, task_id: str, episode_index: int, fps: int = 20) -> str | None:
    normalized = [_normalize_frame(frame) for frame in frames]
    normalized = [frame for frame in normalized if frame is not None]
    if not normalized:
        return None

    try:
        import imageio.v3 as iio
    except ImportError:
        return None

    path = Path(out_dir) / "videos" / f"{task_id}_episode_{episode_index:06d}.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        iio.imwrite(path, normalized, fps=fps)
    except Exception:
        return None
    return path.relative_to(Path(out_dir)).as_posix()


def _normalize_frame(frame: Any) -> np.ndarray | None:
    if frame is None:
        return None
    if isinstance(frame, dict):
        frame = _frame_from_dict(frame)
        if frame is None:
            return None
    if hasattr(frame, "detach") and callable(frame.detach):
        frame = frame.detach()
    if hasattr(frame, "cpu") and callable(frame.cpu):
        frame = frame.cpu()
    if hasattr(frame, "numpy") and callable(frame.numpy):
        frame = frame.numpy()

    array = np.asarray(frame)
    if array.size == 0:
        return None
    while array.ndim > 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 4:
        array = array.reshape((-1, *array.shape[-3:]))[0]
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.ndim != 3:
        return None
    if array.shape[0] in {1, 3, 4} and array.shape[-1] not in {1, 3, 4}:
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=2)
    if array.shape[-1] > 3:
        array = array[..., :3]
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating):
            max_value = float(np.nanmax(array)) if array.size else 1.0
            if max_value <= 1.0:
                array = array * 255.0
        array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _frame_from_dict(frame: dict[Any, Any]) -> Any | None:
    preferred_keys = (
        "rgb",
        "rgba",
        "color",
        "Color",
        "image",
        "render",
        "human",
        "world",
    )
    for key in preferred_keys:
        if key in frame:
            return frame[key]
    for value in frame.values():
        if isinstance(value, dict):
            nested = _frame_from_dict(value)
            if nested is not None:
                return nested
        elif value is not None:
            return value
    return None


def write_failure_clip(out_dir: str | Path, episode: EpisodeResult) -> str | None:
    if episode.success or not episode.replay_path:
        return None

    out_dir = Path(out_dir)
    source = out_dir / episode.replay_path
    if not source.exists():
        return None

    label = episode.failure_label or "failure"
    target = out_dir / "failures" / f"{label}_{episode.task_id}_{episode.episode_index:06d}.mp4"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    episode.failure_clip_path = target.relative_to(out_dir).as_posix()
    return episode.failure_clip_path


def write_replay_manifest(episodes: list[EpisodeResult], out_dir: str | Path) -> Path:
    """Write replay metadata and any available MP4 paths."""

    out_dir = Path(out_dir)
    videos_dir = out_dir / "videos"
    failures_dir = out_dir / "failures"
    videos_dir.mkdir(parents=True, exist_ok=True)
    failures_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "format": "nyssa-replay-lite",
        "video_export": "available" if any(episode.replay_path for episode in episodes) else "unavailable_for_current_engine",
        "episodes": [
            {
                "task_id": episode.task_id,
                "episode_index": episode.episode_index,
                "success": episode.success,
                "failure_label": episode.failure_label,
                "steps": len(episode.steps),
                "replay_path": episode.replay_path,
                "failure_clip_path": episode.failure_clip_path,
                "stressor_context": episode.stressor_context,
            }
            for episode in episodes
        ],
    }
    path = out_dir / "replay_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def write_failure_gallery(episodes: list[EpisodeResult], out_dir: str | Path, *, limit: int = 24) -> Path:
    out_dir = Path(out_dir)
    failures = [episode for episode in episodes if not episode.success]
    failures = sorted(failures, key=lambda episode: len(episode.steps), reverse=True)[:limit]
    rows = []
    for episode in failures:
        clip = episode.failure_clip_path or episode.replay_path
        replay = f'<a href="{html.escape(str(clip))}">video</a>' if clip else "none"
        rows.append(
            "<tr>"
            f"<td>{html.escape(episode.task_id)}</td>"
            f"<td>{episode.episode_index}</td>"
            f"<td>{episode.seed}</td>"
            f"<td>{html.escape(str(episode.failure_label or 'unknown'))}</td>"
            f"<td>{len(episode.steps)}</td>"
            f"<td>{replay}</td>"
            "</tr>"
        )
    body = "".join(rows) if rows else "<tr><td colspan=\"6\">No failures recorded.</td></tr>"
    text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>NyssaBench Failure Gallery</title>
  <style>
    body {{ font-family: Inter, Arial, sans-serif; margin: 40px; color: #17202a; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #d8dee4; padding: 8px; text-align: left; }}
  </style>
</head>
<body>
  <h1>Failure Gallery</h1>
  <table>
    <thead><tr><th>Task</th><th>Episode</th><th>Seed</th><th>Failure</th><th>Steps</th><th>Replay</th></tr></thead>
    <tbody>{body}</tbody>
  </table>
</body>
</html>
"""
    path = out_dir / "failure_gallery.html"
    path.write_text(text, encoding="utf-8")
    return path
