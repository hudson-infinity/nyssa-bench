import json
from pathlib import Path

import yaml

from nyssa_bench.cli import main
from nyssa_bench.reports.scorecard import build_scorecard, write_scorecard


def test_build_scorecard_from_real_run_artifacts(tmp_path: Path):
    run_dir = _make_run(tmp_path / "run_a", policy="random")

    scorecard = build_scorecard(
        [run_dir],
        benchmark="Test scorecard",
        scorecard_date="2026-06-29",
        comparison_report=tmp_path / "compare.html",
        leaderboard=tmp_path / "leaderboard.json",
    )

    assert scorecard["benchmark"] == "Test scorecard"
    assert scorecard["date"] == "2026-06-29"
    assert scorecard["public_claim"] is False
    assert scorecard["results"][0]["suite"] == "mujoco_control_v0"
    assert scorecard["results"][0]["policy"] == "random"
    assert scorecard["results"][0]["per_task"]["mujoco_reacher"]["success_count"] == 1
    assert scorecard["next_required_result"]["policy"] == "robomimic_or_diffusion"
    assert scorecard["status"] == "artifact_validation_failed"
    assert scorecard["artifacts"]["replay_validation"]["status"] == "not_validated"
    assert scorecard["results"][0]["run_public_claim"] is True
    assert scorecard["results"][0]["public_claim"] is False
    assert scorecard["results"][0]["metric_vector"]["scalar_composite"] is None
    assert (
        scorecard["results"][0]["legacy_metrics"]["values"][
            "prototype_reliability_score"
        ]
        == 0.55
    )
    assert "prototype_reliability_score" not in scorecard["results"][0]
    assert (
        "replay:episode_replay_files_present"
        in scorecard["results"][0]["public_claim_validation"]["failures"]
    )


def test_write_scorecard_outputs_related_artifacts(tmp_path: Path):
    run_a = _make_run(tmp_path / "run_a", policy="random", success_rate=0.1)
    run_b = _make_run(tmp_path / "run_b", policy="random", success_rate=0.2)
    out = tmp_path / "scorecard.json"
    comparison = tmp_path / "comparison.html"
    leaderboard = tmp_path / "leaderboard.json"

    paths = write_scorecard(
        [run_a, run_b],
        out=out,
        scorecard_date="2026-06-29",
        comparison_report=comparison,
        leaderboard=leaderboard,
    )

    assert paths["scorecard"] == out
    assert out.exists()
    assert comparison.exists()
    assert leaderboard.exists()
    ranking = json.loads(leaderboard.read_text(encoding="utf-8"))
    assert ranking["format"] == "nyssa-leaderboard-v3"
    assert ranking["comparable"] is True
    assert ranking["ranking"][0]["run_dir"] == run_b.as_posix()


def test_cli_scorecard(tmp_path: Path):
    run_dir = _make_run(tmp_path / "run_a", policy="random")
    out = tmp_path / "scorecard.json"
    comparison = tmp_path / "comparison.html"
    leaderboard = tmp_path / "leaderboard.json"

    assert (
        main(
            [
                "scorecard",
                str(run_dir),
                "--out",
                str(out),
                "--date",
                "2026-06-29",
                "--comparison-out",
                str(comparison),
                "--leaderboard-out",
                str(leaderboard),
            ]
        )
        == 0
    )
    assert (
        json.loads(out.read_text(encoding="utf-8"))["results"][0]["run_dir"]
        == run_dir.as_posix()
    )


def test_exploratory_scorecard_is_not_a_public_claim(tmp_path: Path):
    run_a = _make_run(tmp_path / "run_a", policy="bc_policy", engine="mujoco")
    run_b = _make_run(tmp_path / "run_b", policy="bc_policy", engine="maniskill")
    out = tmp_path / "scorecard.json"

    write_scorecard(
        [run_a, run_b],
        out=out,
        comparison_report=tmp_path / "comparison.html",
        leaderboard=tmp_path / "leaderboard.json",
        allow_incompatible=True,
    )

    scorecard = json.loads(out.read_text(encoding="utf-8"))
    assert scorecard["status"] == "exploratory_non_comparable"
    assert scorecard["public_claim"] is False
    assert scorecard["comparison"]["comparable"] is False
    assert scorecard["comparison"]["mismatches"][0]["field"] == "engine_name"


def _make_run(
    run_dir: Path, *, policy: str, success_rate: float = 0.1, engine: str = "mujoco"
) -> Path:
    run_dir.mkdir(parents=True)
    metadata = {
        "run_id": f"mujoco_control_v0_{policy}_test",
        "suite_id": "mujoco_control_v0",
        "task_ids": ["mujoco_reacher"],
        "policy_name": policy,
        "engine_name": engine,
        "episodes_per_task": 10,
        "seed": 42,
        "seed_protocol": {
            "format": "nyssa-episode-seed-v1",
            "run_seed": 42,
            "episode_seed_stride": 10_000_000,
            "formula": "run_seed * episode_seed_stride + episode_index",
            "shared_across_tasks": True,
        },
        "started_at": "2026-06-29T00:00:00+00:00",
        "finished_at": "2026-06-29T00:01:00+00:00",
    }
    metrics = {
        "episodes": 10,
        "success_count": int(success_rate * 10),
        "success_rate": success_rate,
        "success_rate_ci95": [0.0, 0.3],
        "failure_counts": {"missed_target": 9},
        "primary_failure_mode": "missed_target",
        "prototype_reliability_score": 0.55,
        "score_kind": "prototype_reliability_heuristic",
        "benchmark_tier": "real",
        "public_claim": True,
        "public_claim_validation": {"status": "validated", "failures": []},
        "per_task": {
            "mujoco_reacher": {
                "episodes": 10,
                "success_count": int(success_rate * 10),
                "success_rate": success_rate,
                "success_rate_ci95": [0.0, 0.3],
                "failure_counts": {"missed_target": 9},
                "primary_failure_mode": "missed_target",
                "metrics": {"completion_time": 1.0},
                "metric_ci95": {"completion_time": [1.0, 1.0]},
            }
        },
        "per_seed": {
            "42": {
                "episodes": 10,
                "success_count": int(success_rate * 10),
                "success_rate": success_rate,
                "success_rate_ci95": [0.0, 0.3],
                "failure_counts": {"missed_target": 9},
                "primary_failure_mode": "missed_target",
            }
        },
    }
    (run_dir / "run.yaml").write_text(
        yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8"
    )
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "suite": {"suite_id": "mujoco_control_v0", "tasks": ["mujoco_reacher"]},
                "engine": engine,
                "episodes_per_task": 10,
                "seed_protocol": metadata["seed_protocol"],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (run_dir / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "format": "nyssa-dataset-manifest-v1",
                "run": metadata,
                "suite": {"suite_id": "mujoco_control_v0", "tasks": ["mujoco_reacher"]},
                "tasks": [
                    {
                        "task_id": "mujoco_reacher",
                        "success": {
                            "type": "threshold",
                            "metric": "distance",
                            "value": 0.05,
                        },
                        "randomization": {"seed": True},
                        "ood_splits": {},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    for name in [
        "environment.json",
        "package_versions.json",
        "git_info.json",
        "episodes.jsonl",
        "failure_gallery.html",
        "report.html",
    ]:
        (run_dir / name).write_text("{}\n", encoding="utf-8")
    return run_dir
