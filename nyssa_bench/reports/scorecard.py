from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from nyssa_bench.reports.comparison import (
    compare_runs,
    save_comparison_report,
    save_leaderboard,
)
from nyssa_bench.reports.replay_validation import validate_result_pack_replays


REQUIRED_RUN_FILES = [
    "config.yaml",
    "run.yaml",
    "environment.json",
    "package_versions.json",
    "git_info.json",
    "metrics.json",
    "episodes.jsonl",
    "dataset_manifest.json",
    "failure_gallery.html",
    "report.html",
]

DEFAULT_NOTES = [
    "These results are generated from NyssaBench run artifacts on disk.",
    "Public claims require both run-level validation and assembled artifact revalidation to pass.",
    "Random policies are baseline sanity checks, not strong learned-policy results.",
]


def build_scorecard(
    run_dirs: list[str | Path],
    *,
    benchmark: str = "NyssaBench v0 baselines",
    scorecard_date: str | None = None,
    comparison_report: str | Path | None = None,
    leaderboard: str | Path | None = None,
    notes: list[str] | None = None,
    replay_validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not run_dirs:
        raise ValueError("At least one run directory is required")

    normalized_run_dirs = [Path(path) for path in run_dirs]
    results = [_load_scorecard_result(path) for path in normalized_run_dirs]
    replay_validation = replay_validation or validate_result_pack_replays(
        normalized_run_dirs
    )
    replay_by_run = {run["run_dir"]: run for run in replay_validation.get("runs", [])}
    for result in results:
        run_validation = replay_by_run.get(str(result.get("run_dir")), {})
        original_claim = bool(result.get("public_claim", False))
        original_validation = result.get("public_claim_validation", {})
        if not isinstance(original_validation, dict):
            original_validation = {}
        artifact_claim = bool(run_validation.get("public_claim", False))
        original_failures = original_validation.get("failures", [])
        combined_failures = (
            list(original_failures) if isinstance(original_failures, list) else []
        )
        combined_failures.extend(
            f"replay:{failure}" for failure in run_validation.get("failures", [])
        )
        result["run_public_claim"] = original_claim
        result["run_public_claim_validation"] = original_validation
        result["artifact_validation"] = run_validation
        result["public_claim"] = original_claim and artifact_claim
        result["public_claim_validation"] = {
            "status": "validated" if result["public_claim"] else "not_validated",
            "public_claim": result["public_claim"],
            "failures": combined_failures,
        }
    scorecard: dict[str, Any] = {
        "benchmark": benchmark,
        "date": scorecard_date or date.today().isoformat(),
        "status": "generated",
        "public_claim": all(
            bool(result.get("public_claim", False)) for result in results
        )
        and not _needs_learned_baseline(results)
        and bool(replay_validation["public_claim"]),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "notes": notes or DEFAULT_NOTES,
        "results": results,
        "artifacts": {
            "comparison_report": _display_path(comparison_report)
            if comparison_report
            else None,
            "leaderboard": _display_path(leaderboard) if leaderboard else None,
            "required_per_run": REQUIRED_RUN_FILES,
            "replay_validation": replay_validation,
        },
    }
    if not replay_validation["public_claim"]:
        scorecard["status"] = "artifact_validation_failed"
    if _needs_learned_baseline(results):
        scorecard["next_required_result"] = {
            "suite": "maniskill_smoke_v0",
            "engine": "maniskill",
            "policy": "robomimic_or_diffusion",
            "episodes_per_task": 100,
            "reason": "Random baselines are real but weak; benchmark credibility requires a trained policy baseline.",
        }
    return scorecard


def write_scorecard(
    run_dirs: list[str | Path],
    *,
    out: str | Path,
    benchmark: str = "NyssaBench v0 baselines",
    scorecard_date: str | None = None,
    comparison_report: str | Path | None = None,
    leaderboard: str | Path | None = None,
    allow_incompatible: bool = False,
    replay_validation: dict[str, Any] | None = None,
) -> dict[str, Path]:
    comparison_path = Path(comparison_report) if comparison_report else None
    leaderboard_path = Path(leaderboard) if leaderboard else None

    comparison = (
        compare_runs(run_dirs, allow_incompatible=allow_incompatible)
        if comparison_path or leaderboard_path
        else None
    )
    if comparison_path and comparison:
        save_comparison_report(comparison, comparison_path)
    if leaderboard_path and comparison:
        save_leaderboard(comparison, leaderboard_path)

    scorecard = build_scorecard(
        run_dirs,
        benchmark=benchmark,
        scorecard_date=scorecard_date,
        comparison_report=comparison_path,
        leaderboard=leaderboard_path,
        replay_validation=replay_validation,
    )
    if comparison:
        scorecard["comparison"] = {
            "comparable": comparison["comparable"],
            "mode": comparison["comparison_mode"],
            "contract": comparison["comparison_contract"],
            "contract_sha256": comparison["comparison_contract_sha256"],
            "mismatches": comparison["mismatches"],
        }
        if not comparison["comparable"]:
            scorecard["status"] = "exploratory_non_comparable"
            scorecard["public_claim"] = False
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(scorecard, indent=2) + "\n", encoding="utf-8")

    paths = {"scorecard": out_path}
    if comparison_path:
        paths["comparison_report"] = comparison_path
    if leaderboard_path:
        paths["leaderboard"] = leaderboard_path
    return paths


def _load_scorecard_result(run_dir: Path) -> dict[str, Any]:
    _validate_run_artifacts(run_dir)
    metadata = yaml.safe_load((run_dir / "run.yaml").read_text(encoding="utf-8")) or {}
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    aggregate_metrics = (
        metrics.get("metrics", {})
        if isinstance(metrics.get("metrics", {}), dict)
        else {}
    )
    return {
        "run_dir": _display_path(run_dir),
        "run_id": metadata.get("run_id"),
        "suite": metadata.get("suite_id"),
        "engine": metadata.get("engine_name"),
        "policy": metadata.get("policy_name"),
        "episodes_per_task": metadata.get("episodes_per_task"),
        "episodes": metrics.get("episodes"),
        "seed": metadata.get("seed"),
        "started_at": metadata.get("started_at"),
        "finished_at": metadata.get("finished_at"),
        "wall_time_seconds": metadata.get("wall_time_seconds"),
        "compute": metrics.get("compute", {}),
        "expert_provider": metadata.get("expert_provider", {"provider_id": "none"}),
        "recovery_enabled": bool(metadata.get("recovery_enabled", False)),
        "verifier_enabled": bool(metadata.get("verifier_enabled", False)),
        "success_rate": metrics.get("success_rate", 0.0),
        "success_rate_ci95": metrics.get("success_rate_ci95", [0.0, 0.0]),
        "expert_intervention_rate": aggregate_metrics.get(
            "expert_intervention_rate", 0.0
        ),
        "recovery_success_rate": aggregate_metrics.get("recovery_success_rate", 0.0),
        "recovery_success_count": aggregate_metrics.get("recovery_success_count", 0.0),
        "recovery_applied_count": aggregate_metrics.get("recovery_applied_count", 0.0),
        "recovery_episode_success_rate": aggregate_metrics.get(
            "recovery_episode_success_rate", 0.0
        ),
        "recovery_outcomes": metrics.get(
            "recovery_outcomes", metadata.get("recovery_outcomes")
        ),
        "verifier_rejection_rate": aggregate_metrics.get(
            "verifier_rejection_rate", 0.0
        ),
        "prototype_reliability_score": float(
            metrics.get("prototype_reliability_score", 0.0)
        ),
        "score_kind": metrics.get("score_kind", "prototype_reliability_heuristic"),
        "benchmark_tier": metrics.get("benchmark_tier", "unknown"),
        "public_claim": bool(metrics.get("public_claim", False)),
        "public_claim_validation": metrics.get("public_claim_validation", {}),
        "stressor_support": metrics.get("stressor_support", {}),
        "primary_failure_mode": metrics.get("primary_failure_mode"),
        "failure_counts": metrics.get("failure_counts", {}),
        "per_task": _compact_per_task(metrics.get("per_task", {})),
        "per_seed": _compact_per_seed(metrics.get("per_seed", {})),
    }


def _validate_run_artifacts(run_dir: Path) -> None:
    missing = [name for name in REQUIRED_RUN_FILES if not (run_dir / name).exists()]
    if missing:
        missing_text = ", ".join(str(run_dir / name) for name in missing)
        raise FileNotFoundError(f"Run is missing required artifact(s): {missing_text}")


def _compact_per_task(per_task: Any) -> dict[str, Any]:
    if not isinstance(per_task, dict):
        return {}
    fields = [
        "episodes",
        "success_count",
        "success_rate",
        "success_rate_ci95",
        "failure_counts",
        "primary_failure_mode",
        "metrics",
        "metric_ci95",
    ]
    return {
        str(task_id): {field: values[field] for field in fields if field in values}
        for task_id, values in per_task.items()
        if isinstance(values, dict)
    }


def _compact_per_seed(per_seed: Any) -> dict[str, Any]:
    if not isinstance(per_seed, dict):
        return {}
    fields = [
        "episodes",
        "success_count",
        "success_rate",
        "success_rate_ci95",
        "failure_counts",
        "primary_failure_mode",
    ]
    return {
        str(seed): {field: values[field] for field in fields if field in values}
        for seed, values in per_seed.items()
        if isinstance(values, dict)
    }


def _needs_learned_baseline(results: list[dict[str, Any]]) -> bool:
    return all(str(result.get("policy")) == "random" for result in results)


def _display_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    return Path(path).as_posix()
