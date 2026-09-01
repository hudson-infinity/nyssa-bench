import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from nyssa_bench.cli import main
from nyssa_bench.reports.comparison import (
    ComparisonMetadataError,
    IncompatibleRunsError,
    compare_runs,
    load_comparison_contract,
    save_comparison_report,
    save_leaderboard,
)


def test_compatible_runs_share_a_deterministic_contract(tmp_path: Path):
    run_a = _make_run(tmp_path / "run_a", policy="policy_a", run_seed=0)
    run_b = _make_run(tmp_path / "run_b", policy="policy_b", run_seed=918)

    comparison = compare_runs([run_a, run_b])

    assert comparison["comparable"] is True
    assert comparison["comparison_mode"] == "strict"
    assert comparison["mismatches"] == []
    assert len(comparison["comparison_contract_sha256"]) == 64
    assert (
        comparison["run_contracts"][0]["sha256"]
        == comparison["run_contracts"][1]["sha256"]
    )
    assert (
        "run_seed"
        not in comparison["comparison_contract"]["shared_contract"]["seed_protocol"]
    )
    assert comparison["ordering"]["primary_metric"] == "success_rate"
    assert "prototype_reliability_score" not in comparison["ranking"][0]
    assert comparison["ranking"][0]["metric_vector"]["scalar_composite"] is None
    assert (
        comparison["runs"][0]["legacy_metrics"]["values"][
            "prototype_reliability_score"
        ]
        == 0.5
    )


@pytest.mark.parametrize(
    ("overrides", "expected_field"),
    [
        ({"suite_id": "other_suite"}, "suite_id"),
        ({"engine_name": "maniskill"}, "engine_name"),
        ({"task_ids": ["task_b"]}, "task_ids"),
        (
            {"success": {"type": "threshold", "metric": "distance", "value": 0.05}},
            "tasks.task_a.success",
        ),
        (
            {"randomization": {"seed": True, "friction": [0.2, 1.5]}},
            "tasks.task_a.stressors.randomization",
        ),
        ({"episodes_per_task": 50}, "episodes_per_task"),
        ({"seed_stride": 1_000_000}, "seed_protocol.episode_seed_stride"),
    ],
)
def test_strict_comparison_rejects_each_major_incompatibility(
    tmp_path: Path,
    overrides: dict[str, Any],
    expected_field: str,
):
    run_a = _make_run(tmp_path / "run_a")
    run_b = _make_run(tmp_path / "run_b", **overrides)

    with pytest.raises(IncompatibleRunsError) as exc_info:
        compare_runs([run_a, run_b])

    mismatched_fields = [item["field"] for item in exc_info.value.mismatches]
    assert any(field.startswith(expected_field) for field in mismatched_fields)
    assert run_a.as_posix() in str(exc_info.value)
    assert run_b.as_posix() in str(exc_info.value)


def test_exploratory_override_labels_all_outputs_non_comparable(tmp_path: Path):
    run_a = _make_run(tmp_path / "run_a", engine_name="mujoco")
    run_b = _make_run(tmp_path / "run_b", engine_name="maniskill")
    report_path = tmp_path / "comparison.html"
    leaderboard_path = tmp_path / "leaderboard.json"

    comparison = compare_runs([run_a, run_b], allow_incompatible=True)
    save_comparison_report(comparison, report_path)
    save_leaderboard(comparison, leaderboard_path)

    assert comparison["comparable"] is False
    assert comparison["comparison_mode"] == "exploratory"
    assert comparison["comparison_contract"]["shared_contract"] is None
    assert comparison["comparison_contract"]["mismatched_fields"] == ["engine_name"]
    report = report_path.read_text(encoding="utf-8")
    assert "NON-COMPARABLE EXPLORATORY OUTPUT" in report
    assert comparison["comparison_contract_sha256"] in report
    leaderboard = json.loads(leaderboard_path.read_text(encoding="utf-8"))
    assert leaderboard["format"] == "nyssa-leaderboard-v3"
    assert leaderboard["comparable"] is False
    assert leaderboard["comparison_mode"] == "exploratory"
    assert (
        leaderboard["comparison_contract_sha256"]
        == comparison["comparison_contract_sha256"]
    )
    assert leaderboard["mismatches"][0]["field"] == "engine_name"


def test_cli_requires_explicit_exploratory_override(tmp_path: Path):
    run_a = _make_run(tmp_path / "run_a", engine_name="mujoco")
    run_b = _make_run(tmp_path / "run_b", engine_name="maniskill")
    out = tmp_path / "leaderboard.json"

    with pytest.raises(IncompatibleRunsError):
        main(["leaderboard", str(run_a), str(run_b), "--out", str(out)])

    assert (
        main(
            [
                "leaderboard",
                str(run_a),
                str(run_b),
                "--allow-incompatible",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    assert json.loads(out.read_text(encoding="utf-8"))["comparable"] is False


def test_missing_comparison_metadata_is_rejected(tmp_path: Path):
    run_dir = tmp_path / "incomplete"
    run_dir.mkdir()
    (run_dir / "metrics.json").write_text('{"success_rate": 0.0}\n', encoding="utf-8")
    (run_dir / "run.yaml").write_text("suite_id: test_suite\n", encoding="utf-8")

    with pytest.raises(ComparisonMetadataError) as exc_info:
        compare_runs([run_dir])

    assert "engine_name" in exc_info.value.missing_fields
    assert "seed_protocol" in exc_info.value.missing_fields
    assert "task_ids" in exc_info.value.missing_fields


def test_conflicting_run_artifacts_are_rejected(tmp_path: Path):
    run_dir = _make_run(tmp_path / "conflicting")
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    config["engine"] = "maniskill"
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )

    with pytest.raises(ComparisonMetadataError) as exc_info:
        compare_runs([run_dir])

    assert any(
        field.startswith("engine_name (conflict")
        for field in exc_info.value.missing_fields
    )


def test_contract_normalizes_task_and_mapping_order(tmp_path: Path):
    success_a = {"type": "threshold", "metric": "distance", "value": 0.1}
    success_b = {"value": 0.1, "metric": "distance", "type": "threshold"}
    run_a = _make_run(
        tmp_path / "run_a", task_ids=["task_b", "task_a"], success=success_a
    )
    run_b = _make_run(
        tmp_path / "run_b", task_ids=["task_a", "task_b"], success=success_b
    )

    comparison = compare_runs([run_a, run_b])

    assert comparison["comparable"] is True
    assert load_comparison_contract(run_a)["task_ids"] == ["task_a", "task_b"]


def test_comparison_contract_rejects_different_executable_stressor_conditions(
    tmp_path: Path,
):
    run_a = _make_run(tmp_path / "run_a")
    run_b = _make_run(tmp_path / "run_b")
    _set_stressor_config(run_a, severity=0.0, condition_id="clean")
    _set_stressor_config(run_b, severity=1.0, condition_id="shifted")

    with pytest.raises(IncompatibleRunsError) as exc_info:
        compare_runs([run_a, run_b])

    fields = [item["field"] for item in exc_info.value.mismatches]
    assert "stressor_config.stressors" in fields


def _make_run(
    run_dir: Path,
    *,
    suite_id: str = "test_suite",
    engine_name: str = "mujoco",
    policy: str = "policy",
    task_ids: list[str] | None = None,
    success: dict[str, Any] | None = None,
    randomization: dict[str, Any] | None = None,
    episodes_per_task: int = 20,
    run_seed: int = 0,
    seed_stride: int = 10_000_000,
) -> Path:
    task_ids = task_ids or ["task_a"]
    success = success or {"type": "threshold", "metric": "distance", "value": 0.1}
    randomization = randomization or {"seed": True}
    seed_protocol = {
        "format": "nyssa-episode-seed-v1",
        "run_seed": run_seed,
        "episode_seed_stride": seed_stride,
        "formula": "run_seed * episode_seed_stride + episode_index",
        "shared_across_tasks": True,
    }
    metadata = {
        "run_id": f"{suite_id}_{policy}_{run_seed}",
        "suite_id": suite_id,
        "task_ids": task_ids,
        "policy_name": policy,
        "engine_name": engine_name,
        "episodes_per_task": episodes_per_task,
        "seed": run_seed,
        "seed_protocol": seed_protocol,
    }
    config = {
        "suite": {"suite_id": suite_id, "tasks": task_ids},
        "engine": engine_name,
        "episodes_per_task": episodes_per_task,
        "seed_protocol": seed_protocol,
    }
    manifest = {
        "format": "nyssa-dataset-manifest-v1",
        "run": metadata,
        "suite": config["suite"],
        "tasks": [
            {
                "task_id": task_id,
                "success": success,
                "randomization": randomization,
                "ood_splits": {"pose": "held_out"},
            }
            for task_id in reversed(task_ids)
        ],
    }
    metrics = {
        "episodes": episodes_per_task * len(task_ids),
        "success_rate": 0.5,
        "success_rate_ci95": [0.3, 0.7],
        "prototype_reliability_score": 0.5,
    }

    run_dir.mkdir(parents=True)
    (run_dir / "run.yaml").write_text(
        yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8"
    )
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (run_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return run_dir


def _set_stressor_config(run_dir: Path, *, severity: float, condition_id: str) -> None:
    stressor_config = {
        "format": "nyssa-stressor-config-v1",
        "condition_id": condition_id,
        "unsupported_policy": "error",
        "stressors": [
            {
                "format": "nyssa-stressor-spec-v1",
                "schema_version": 1,
                "stressor_id": "action_delay",
                "severity": severity,
                "parameters": {"max_delay_steps": 4},
                "seed": None,
            }
        ],
    }
    for name in ("run.yaml", "config.yaml"):
        path = run_dir / name
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        payload["stressor_config"] = stressor_config
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    manifest_path = run_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["run"]["stressor_config"] = stressor_config
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
