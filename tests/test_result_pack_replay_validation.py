import json
from pathlib import Path

import pytest
import yaml

from nyssa_bench.reports.replay_validation import validate_result_pack_replays
from nyssa_bench.reports.result_pack import write_experiment_manifest, write_results_markdown


def test_empty_pack_is_rejected_without_crashing():
    validation = validate_result_pack_replays([])

    assert validation["public_claim"] is False
    assert validation["status"] == "not_validated"
    assert validation["counts"] == {}
    assert "run_directories_present" in validation["failures"]


def test_complete_pack_validates_episode_replays_separately_from_failure_clips(tmp_path: Path):
    run_dir = _make_run(tmp_path / "run")

    validation = validate_result_pack_replays([run_dir])

    assert validation["status"] == "validated"
    assert validation["public_claim"] is True
    assert validation["counts"]["expected_episode_replays"] == 2
    assert validation["counts"]["episode_replays_present"] == 2
    assert validation["counts"]["episode_replays_missing"] == 0
    assert validation["counts"]["failure_clips_declared"] == 1
    assert validation["counts"]["failure_clips_present"] == 1
    assert validation["counts"]["duplicate_media_groups"] == 1
    assert validation["counts"]["duplicate_media_files"] == 1
    assert validation["warnings"] == ["duplicate_media_content_present"]


def test_missing_original_is_not_replaced_by_failure_clip_or_cached_validation(tmp_path: Path):
    run_dir = _make_run(tmp_path / "run")
    (run_dir / "videos" / "task_episode_000001.mp4").unlink()

    validation = validate_result_pack_replays([run_dir])
    results_path = _write_results(tmp_path, run_dir)
    manifest_path = _write_manifest(tmp_path, run_dir)

    assert validation["status"] == "not_validated"
    assert validation["public_claim"] is False
    assert validation["counts"]["expected_episode_replays"] == 2
    assert validation["counts"]["episode_replays_present"] == 1
    assert validation["counts"]["episode_replays_missing"] == 1
    assert validation["counts"]["failure_clips_present"] == 1
    assert "episode_replay_files_present" in validation["failures"]
    results = results_path.read_text(encoding="utf-8")
    assert "Public-claim validation: `not validated`" in results
    assert "Failure clips: `1` present" in results
    assert "`1` required episode replay artifacts are missing" in results
    assert "do not describe this pack as video-backed" in results
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["validation"]["public_claim"] is False
    assert manifest["validation"]["replay_artifacts"]["counts"]["episode_replays_missing"] == 1


def test_failure_clip_only_pack_has_zero_episode_replay_coverage(tmp_path: Path):
    run_dir = _make_run(tmp_path / "run", failure_clip_for_every_episode=True)
    for path in (run_dir / "videos").glob("*.mp4"):
        path.unlink()

    validation = validate_result_pack_replays([run_dir])

    assert validation["public_claim"] is False
    assert validation["counts"]["expected_episode_replays"] == 2
    assert validation["counts"]["episode_replays_present"] == 0
    assert validation["counts"]["episode_replays_missing"] == 2
    assert validation["counts"]["failure_clips_present"] == 2


def test_duplicate_episode_replay_reference_does_not_inflate_coverage(tmp_path: Path):
    run_dir = _make_run(tmp_path / "run")
    records = _episodes(run_dir)
    records[1]["replay_path"] = records[0]["replay_path"]
    _write_episode_artifacts(run_dir, records)

    validation = validate_result_pack_replays([run_dir])

    assert validation["public_claim"] is False
    assert validation["counts"]["episode_replays_present"] == 1
    assert validation["counts"]["episode_replays_missing"] == 1
    assert validation["counts"]["duplicate_episode_replay_references"] == 1
    assert "episode_replay_paths_unique" in validation["failures"]
    assert validation["counts"]["extra_media_files"] == 1


@pytest.mark.parametrize(
    ("replay_path", "expected_failure", "expected_reason"),
    [
        ("videos/task_episode_000000.webm", "episode_replay_media_allowed", "unsupported_media_type"),
        ("../outside.mp4", "episode_replay_paths_safe", "unsafe_path"),
    ],
)
def test_invalid_or_unsafe_episode_replay_paths_are_rejected(
    tmp_path: Path,
    replay_path: str,
    expected_failure: str,
    expected_reason: str,
):
    run_dir = _make_run(tmp_path / "run")
    records = _episodes(run_dir)
    records[0]["replay_path"] = replay_path
    _write_episode_artifacts(run_dir, records)
    if replay_path.endswith(".webm"):
        (run_dir / replay_path).write_bytes(b"webm")
    else:
        (tmp_path / "outside.mp4").write_bytes(b"outside")

    validation = validate_result_pack_replays([run_dir])
    run = validation["runs"][0]

    assert validation["public_claim"] is False
    assert expected_failure in validation["failures"]
    assert run["missing_episode_replays"][0]["reason"] == expected_reason


def test_pruned_episode_records_and_stale_replay_manifest_are_rejected(tmp_path: Path):
    run_dir = _make_run(tmp_path / "run")
    records = _episodes(run_dir)[:1]
    (run_dir / "episodes.json").write_text(json.dumps(records), encoding="utf-8")

    validation = validate_result_pack_replays([run_dir])

    assert validation["public_claim"] is False
    assert validation["counts"]["expected_episode_replays"] == 2
    assert validation["counts"]["episode_records_present"] == 1
    assert validation["counts"]["episode_replays_missing"] == 1
    assert "complete_episode_records" in validation["failures"]
    assert "replay_manifests_consistent" in validation["failures"]


def test_extra_and_duplicate_media_are_reported_without_counting_as_replays(tmp_path: Path):
    run_dir = _make_run(tmp_path / "run")
    (run_dir / "videos" / "extra.gif").write_bytes(b"extra")

    validation = validate_result_pack_replays([run_dir])
    results = _write_results(tmp_path, run_dir).read_text(encoding="utf-8")

    assert validation["public_claim"] is True
    assert validation["counts"]["episode_replays_present"] == 2
    assert validation["counts"]["extra_media_files"] == 1
    assert validation["counts"]["duplicate_media_files"] == 1
    assert "videos/extra.gif" in validation["runs"][0]["extra_media"]
    assert "Extra media: `1` unreferenced files" in results
    assert "Duplicate media: `1` duplicate-content files" in results


@pytest.mark.parametrize(
    ("relative_path", "expected_failure"),
    [
        ("replay_manifest.json", "replay_manifests_present"),
        ("failure_gallery.html", "failure_galleries_present"),
        ("failures/task_episode_000001.mp4", "declared_failure_clips_valid"),
    ],
)
def test_missing_replay_support_artifacts_downgrade_pack_validation(
    tmp_path: Path,
    relative_path: str,
    expected_failure: str,
):
    run_dir = _make_run(tmp_path / "run")
    (run_dir / relative_path).unlink()

    validation = validate_result_pack_replays([run_dir])

    assert validation["public_claim"] is False
    assert expected_failure in validation["failures"]


def test_run_and_metrics_episode_denominators_must_agree(tmp_path: Path):
    run_dir = _make_run(tmp_path / "run")
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics["episodes"] = 1
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    validation = validate_result_pack_replays([run_dir])

    assert validation["public_claim"] is False
    assert validation["counts"]["expected_episode_replays"] == 2
    assert "episode_count_sources_consistent" in validation["failures"]


def _make_run(run_dir: Path, *, failure_clip_for_every_episode: bool = False) -> Path:
    run_dir.mkdir(parents=True)
    records = [
        {
            "task_id": "task",
            "episode_index": 0,
            "seed": 100,
            "success": True,
            "replay_path": "videos/task_episode_000000.mp4",
            "failure_clip_path": "failures/task_episode_000000.mp4" if failure_clip_for_every_episode else None,
            "steps": [{"reward": 1.0}],
        },
        {
            "task_id": "task",
            "episode_index": 1,
            "seed": 101,
            "success": False,
            "failure_label": "missed_target",
            "replay_path": "videos/task_episode_000001.mp4",
            "failure_clip_path": "failures/task_episode_000001.mp4",
            "steps": [{"reward": 0.0}],
        },
    ]
    metrics = {
        "policy": "bc_policy",
        "episodes": 2,
        "success_rate": 0.5,
        "failure_counts": {"missed_target": 1},
        "public_claim": True,
        "public_claim_validation": {
            "status": "validated",
            "public_claim": True,
            "failures": [],
        },
    }
    metadata = {
        "task_ids": ["task"],
        "episodes_per_task": 2,
        "seed": 0,
        "seed_protocol": {
            "format": "nyssa-episode-seed-v2",
            "run_seed": 0,
            "episode_seed_stride": 10_000_000,
        },
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (run_dir / "run.yaml").write_text(yaml.safe_dump(metadata), encoding="utf-8")
    (run_dir / "failure_gallery.html").write_text("<html></html>", encoding="utf-8")
    (run_dir / "videos").mkdir()
    (run_dir / "failures").mkdir()
    (run_dir / "videos" / "task_episode_000000.mp4").write_bytes(b"episode-zero")
    (run_dir / "videos" / "task_episode_000001.mp4").write_bytes(b"episode-one")
    if failure_clip_for_every_episode:
        (run_dir / "failures" / "task_episode_000000.mp4").write_bytes(b"episode-zero")
    (run_dir / "failures" / "task_episode_000001.mp4").write_bytes(b"episode-one")
    _write_episode_artifacts(run_dir, records)
    return run_dir


def _episodes(run_dir: Path) -> list[dict]:
    return json.loads((run_dir / "episodes.json").read_text(encoding="utf-8"))


def _write_episode_artifacts(run_dir: Path, records: list[dict]) -> None:
    (run_dir / "episodes.json").write_text(json.dumps(records), encoding="utf-8")
    (run_dir / "episodes.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    manifest = {
        "format": "nyssa-replay-lite",
        "video_export": "available",
        "episodes": [
            {
                "task_id": record["task_id"],
                "episode_index": record["episode_index"],
                "success": record["success"],
                "failure_label": record.get("failure_label"),
                "steps": len(record["steps"]),
                "replay_path": record.get("replay_path"),
                "failure_clip_path": record.get("failure_clip_path"),
            }
            for record in records
        ],
    }
    (run_dir / "replay_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _write_results(tmp_path: Path, run_dir: Path) -> Path:
    return write_results_markdown(
        out_dir=tmp_path,
        suite_id="test_suite",
        engine="mujoco",
        policies=["bc_policy"],
        seeds=[0],
        episodes_per_task=2,
        run_dirs=[run_dir],
        comparison_report=tmp_path / "comparison.html",
        leaderboard=tmp_path / "leaderboard.json",
        scorecard=tmp_path / "scorecard.json",
    )


def _write_manifest(tmp_path: Path, run_dir: Path) -> Path:
    return write_experiment_manifest(
        out_dir=tmp_path,
        suite_id="test_suite",
        engine="mujoco",
        policies=["bc_policy"],
        seeds=[0],
        episodes_per_task=2,
        run_dirs=[run_dir],
        artifacts={"results": tmp_path / "RESULTS.md"},
    )
