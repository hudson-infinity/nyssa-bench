from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


REPLAY_VALIDATION_FORMAT = "nyssa-result-pack-replay-validation-v1"
PUBLIC_REPLAY_SUFFIXES = {".mp4"}
MEDIA_SUFFIXES = {".mp4", ".webm", ".gif"}


def validate_result_pack_replays(run_dirs: list[str | Path]) -> dict[str, Any]:
    """Revalidate replay evidence from artifacts currently present on disk."""

    runs = [_validate_run(Path(run_dir)) for run_dir in run_dirs]
    counts = _sum_counts(runs)
    checks = {
        "run_directories_present": bool(runs) and all(run["checks"]["run_directory_present"] for run in runs),
        "metrics_episode_counts_present": bool(runs)
        and all(run["checks"]["metrics_episode_count_present"] for run in runs),
        "run_episode_counts_present": bool(runs)
        and all(run["checks"]["run_episode_count_present"] for run in runs),
        "episode_count_sources_consistent": bool(runs)
        and all(run["checks"]["episode_count_sources_consistent"] for run in runs),
        "episode_artifacts_present": bool(runs)
        and all(run["checks"]["episodes_json_present"] for run in runs),
        "complete_episode_records": bool(runs)
        and all(run["checks"]["complete_episode_records"] for run in runs),
        "unique_episode_records": bool(runs)
        and all(run["checks"]["unique_episode_records"] for run in runs),
        "episode_replay_paths_declared": bool(runs)
        and all(run["checks"]["episode_replay_paths_declared"] for run in runs),
        "episode_replay_paths_safe": bool(runs)
        and all(run["checks"]["episode_replay_paths_safe"] for run in runs),
        "episode_replay_media_allowed": bool(runs)
        and all(run["checks"]["episode_replay_media_allowed"] for run in runs),
        "episode_replay_files_present": bool(runs)
        and all(run["checks"]["episode_replay_files_present"] for run in runs),
        "episode_replay_paths_unique": bool(runs)
        and all(run["checks"]["episode_replay_paths_unique"] for run in runs),
        "replay_manifests_present": bool(runs)
        and all(run["checks"]["replay_manifest_present"] for run in runs),
        "replay_manifests_consistent": bool(runs)
        and all(run["checks"]["replay_manifest_consistent"] for run in runs),
        "declared_failure_clips_valid": bool(runs)
        and all(run["checks"]["declared_failure_clips_valid"] for run in runs),
        "failure_galleries_present": bool(runs)
        and all(run["checks"]["failure_gallery_present"] for run in runs),
    }
    failures = [name for name, passed in checks.items() if not passed]
    warnings: list[str] = []
    if counts.get("extra_media_files", 0):
        warnings.append("unreferenced_media_files_present")
    if counts.get("duplicate_media_files", 0):
        warnings.append("duplicate_media_content_present")
    public_claim = not failures
    return {
        "format": REPLAY_VALIDATION_FORMAT,
        "status": "validated" if public_claim else "not_validated",
        "public_claim": public_claim,
        "checks": checks,
        "failures": failures,
        "warnings": warnings,
        "counts": counts,
        "runs": runs,
    }


def _validate_run(root: Path) -> dict[str, Any]:
    root_exists = root.is_dir()
    metrics = _load_json_mapping(root / "metrics.json")
    run_metadata = _load_yaml_mapping(root / "run.yaml")
    metrics_expected_episodes = _positive_int(metrics.get("episodes")) if metrics else None
    run_expected_episodes = _run_expected_episodes(run_metadata)
    expected_episodes = run_expected_episodes or metrics_expected_episodes
    episode_records = _load_json_list(root / "episodes.json")
    replay_manifest = _load_json_mapping(root / "replay_manifest.json")
    manifest_records = replay_manifest.get("episodes") if replay_manifest else None
    if not isinstance(manifest_records, list):
        manifest_records = []

    episode_keys = [_episode_key(record) for record in episode_records]
    duplicate_episode_records = sum(count - 1 for count in Counter(episode_keys).values() if count > 1)
    episode_paths: list[str] = []
    missing_episode_replays: list[dict[str, Any]] = []
    unsafe_episode_paths: list[str] = []
    invalid_episode_media: list[str] = []
    declared_episode_replays = 0
    for record in episode_records:
        raw_path = record.get("replay_path") if isinstance(record, dict) else None
        if raw_path:
            declared_episode_replays += 1
        inspection = _inspect_declared_media(root, raw_path)
        if inspection["valid"]:
            episode_paths.append(str(inspection["path"]))
            continue
        reason = str(inspection["reason"])
        missing_episode_replays.append(
            {
                "episode": _episode_identity(record),
                "replay_path": raw_path,
                "reason": reason,
            }
        )
        if reason == "unsafe_path":
            unsafe_episode_paths.append(str(raw_path))
        elif reason == "unsupported_media_type":
            invalid_episode_media.append(str(raw_path))

    episode_path_counts = Counter(episode_paths)
    duplicate_episode_replay_references = sum(count - 1 for count in episode_path_counts.values() if count > 1)
    unique_episode_paths = set(episode_paths)

    failure_clip_paths: list[str] = []
    missing_failure_clips: list[dict[str, Any]] = []
    unsafe_failure_clip_paths: list[str] = []
    invalid_failure_clip_media: list[str] = []
    declared_failure_clips = 0
    for record in episode_records:
        raw_path = record.get("failure_clip_path") if isinstance(record, dict) else None
        if not raw_path:
            continue
        declared_failure_clips += 1
        inspection = _inspect_declared_media(root, raw_path)
        if inspection["valid"]:
            failure_clip_paths.append(str(inspection["path"]))
            continue
        reason = str(inspection["reason"])
        missing_failure_clips.append(
            {
                "episode": _episode_identity(record),
                "failure_clip_path": raw_path,
                "reason": reason,
            }
        )
        if reason == "unsafe_path":
            unsafe_failure_clip_paths.append(str(raw_path))
        elif reason == "unsupported_media_type":
            invalid_failure_clip_media.append(str(raw_path))

    unique_failure_clip_paths = set(failure_clip_paths)
    media_files = _media_files(root) if root_exists else []
    referenced_media = unique_episode_paths | unique_failure_clip_paths
    extra_media = sorted(set(media_files) - referenced_media)
    duplicate_media_groups = _duplicate_media_groups(root, media_files)
    duplicate_media_files = sum(len(group) - 1 for group in duplicate_media_groups)

    expected = expected_episodes or 0
    present_episode_replays = min(len(unique_episode_paths), expected)
    replay_manifest_consistent = _replay_manifest_matches(episode_records, manifest_records)
    counts = {
        "expected_episode_replays": expected,
        "episode_records_present": len(episode_records),
        "episode_replays_declared": declared_episode_replays,
        "episode_replays_present": present_episode_replays,
        "episode_replays_missing": max(expected - present_episode_replays, 0),
        "episode_replays_invalid_type": len(invalid_episode_media),
        "episode_replays_unsafe": len(unsafe_episode_paths),
        "episode_replay_unique_paths": len(unique_episode_paths),
        "duplicate_episode_records": duplicate_episode_records,
        "duplicate_episode_replay_references": duplicate_episode_replay_references,
        "failure_clips_declared": declared_failure_clips,
        "failure_clips_present": len(set(failure_clip_paths)),
        "failure_clips_missing": len(missing_failure_clips),
        "failure_clips_invalid_type": len(invalid_failure_clip_media),
        "failure_clips_unsafe": len(unsafe_failure_clip_paths),
        "failure_galleries_expected": 1,
        "failure_galleries_present": int((root / "failure_gallery.html").is_file()),
        "replay_manifests_expected": 1,
        "replay_manifests_present": int((root / "replay_manifest.json").is_file()),
        "media_files_total": len(media_files),
        "extra_media_files": len(extra_media),
        "duplicate_media_groups": len(duplicate_media_groups),
        "duplicate_media_files": duplicate_media_files,
    }
    checks = {
        "run_directory_present": root_exists,
        "metrics_episode_count_present": metrics_expected_episodes is not None,
        "run_episode_count_present": run_expected_episodes is not None,
        "episode_count_sources_consistent": metrics_expected_episodes is not None
        and run_expected_episodes is not None
        and metrics_expected_episodes == run_expected_episodes,
        "episodes_json_present": (root / "episodes.json").is_file(),
        "complete_episode_records": expected_episodes is not None and len(episode_records) == expected_episodes,
        "unique_episode_records": duplicate_episode_records == 0,
        "episode_replay_paths_declared": expected_episodes is not None
        and declared_episode_replays == expected_episodes,
        "episode_replay_paths_safe": not unsafe_episode_paths,
        "episode_replay_media_allowed": not invalid_episode_media,
        "episode_replay_files_present": expected_episodes is not None
        and present_episode_replays == expected_episodes,
        "episode_replay_paths_unique": duplicate_episode_replay_references == 0,
        "replay_manifest_present": (root / "replay_manifest.json").is_file(),
        "replay_manifest_consistent": replay_manifest_consistent,
        "declared_failure_clips_valid": not (
            missing_failure_clips or invalid_failure_clip_media or unsafe_failure_clip_paths
        ),
        "failure_gallery_present": (root / "failure_gallery.html").is_file(),
    }
    failures = [name for name, passed in checks.items() if not passed]
    warnings: list[str] = []
    if extra_media:
        warnings.append("unreferenced_media_files_present")
    if duplicate_media_groups:
        warnings.append("duplicate_media_content_present")
    return {
        "run_dir": root.as_posix(),
        "status": "validated" if not failures else "not_validated",
        "public_claim": not failures,
        "checks": checks,
        "failures": failures,
        "warnings": warnings,
        "counts": counts,
        "missing_episode_replays": missing_episode_replays,
        "missing_failure_clips": missing_failure_clips,
        "extra_media": extra_media,
        "duplicate_media": duplicate_media_groups,
    }


def _inspect_declared_media(root: Path, raw_path: Any) -> dict[str, Any]:
    if not isinstance(raw_path, str) or not raw_path.strip():
        return {"valid": False, "path": None, "reason": "path_not_declared"}
    path = Path(raw_path)
    if path.suffix.lower() not in PUBLIC_REPLAY_SUFFIXES:
        return {"valid": False, "path": None, "reason": "unsupported_media_type"}
    resolved_root = root.resolve()
    resolved_path = (root / path).resolve()
    try:
        relative = resolved_path.relative_to(resolved_root).as_posix()
    except ValueError:
        return {"valid": False, "path": None, "reason": "unsafe_path"}
    if not resolved_path.is_file():
        return {"valid": False, "path": relative, "reason": "file_missing"}
    return {"valid": True, "path": relative, "reason": None}


def _replay_manifest_matches(episodes: list[dict[str, Any]], manifest: list[Any]) -> bool:
    if len(episodes) != len(manifest):
        return False
    expected = [_manifest_projection(record) for record in episodes]
    observed = [_manifest_projection(record) for record in manifest if isinstance(record, dict)]
    return len(observed) == len(manifest) and Counter(expected) == Counter(observed)


def _manifest_projection(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _hashable_value(record.get("task_id")),
        _hashable_value(record.get("episode_index")),
        _portable_path(record.get("replay_path")),
        _portable_path(record.get("failure_clip_path")),
    )


def _episode_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _hashable_value(record.get("task_id")),
        _hashable_value(record.get("seed")),
        _hashable_value(record.get("episode_index")),
    )


def _episode_identity(record: Any) -> dict[str, Any]:
    if not isinstance(record, dict):
        return {"task_id": None, "seed": None, "episode_index": None}
    return {
        "task_id": record.get("task_id"),
        "seed": record.get("seed"),
        "episode_index": record.get("episode_index"),
    }


def _portable_path(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return Path(value).as_posix()


def _media_files(root: Path) -> list[str]:
    files: list[str] = []
    resolved_root = root.resolve()
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in MEDIA_SUFFIXES:
            continue
        try:
            files.append(path.resolve().relative_to(resolved_root).as_posix())
        except ValueError:
            continue
    return sorted(set(files))


def _duplicate_media_groups(root: Path, media_files: list[str]) -> list[list[str]]:
    by_size: dict[int, list[str]] = defaultdict(list)
    for relative in media_files:
        try:
            by_size[(root / relative).stat().st_size].append(relative)
        except OSError:
            continue
    duplicate_groups: list[list[str]] = []
    for same_size in by_size.values():
        if len(same_size) < 2:
            continue
        by_hash: dict[str, list[str]] = defaultdict(list)
        for relative in same_size:
            try:
                by_hash[_sha256(root / relative)].append(relative)
            except OSError:
                continue
        duplicate_groups.extend(sorted(paths) for paths in by_hash.values() if len(paths) > 1)
    return sorted(duplicate_groups, key=lambda paths: tuple(paths))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sum_counts(runs: list[dict[str, Any]]) -> dict[str, int]:
    keys = {
        key
        for run in runs
        for key, value in run.get("counts", {}).items()
        if isinstance(value, int)
    }
    return {key: sum(int(run.get("counts", {}).get(key, 0)) for run in runs) for key in sorted(keys)}


def _load_json_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return value if isinstance(value, dict) else {}


def _run_expected_episodes(run_metadata: dict[str, Any]) -> int | None:
    episodes_per_task = _positive_int(run_metadata.get("episodes_per_task"))
    task_ids = run_metadata.get("task_ids")
    if episodes_per_task is None or not isinstance(task_ids, list) or not task_ids:
        return None
    normalized_task_ids = {str(task_id) for task_id in task_ids}
    if len(normalized_task_ids) != len(task_ids):
        return None
    return episodes_per_task * len(normalized_task_ids)


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, float) and not value.is_integer():
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def _hashable_value(value: Any) -> Any:
    try:
        hash(value)
    except TypeError:
        return json.dumps(value, sort_keys=True, ensure_ascii=True)
    return value
