from __future__ import annotations

from pathlib import Path
from typing import Any


def write_results_markdown(
    *,
    out_dir: str | Path,
    suite_id: str,
    engine: str,
    policies: list[str],
    seeds: list[int],
    episodes_per_task: int,
    run_dirs: list[Path],
    comparison_report: Path,
    leaderboard: Path,
    scorecard: Path,
) -> Path:
    summaries = _summarize_runs(run_dirs)
    policy_rows = _policy_rows(summaries)
    validation_rows = _validation_rows(summaries)
    unsupported = sorted(
        {
            item
            for summary in summaries
            for item in (summary.get("stressor_support", {}).get("unsupported_stressors", []) or [])
        }
    )
    primary_failures = _primary_failures(summaries)
    video_count = _count_video_files(run_dirs)
    seed_protocol_valid = len(set(seeds)) <= 1 or all(
        (summary.get("_seed_protocol") or {}).get("format") == "nyssa-episode-seed-v2"
        for summary in summaries
    )
    public_claim_status = "validated" if summaries and seed_protocol_valid and all(
        (summary.get("public_claim_validation") or {}).get("status") == "validated" for summary in summaries
    ) else "not validated"
    out_dir = Path(out_dir)
    path = out_dir / "RESULTS.md"
    run_lines = "\n".join(f"- `{run_dir.as_posix()}`" for run_dir in run_dirs)
    policy_lines = "\n".join(f"- `{policy}`" for policy in policies)
    seed_text = ", ".join(str(seed) for seed in seeds)
    text = f"""# NyssaBench Results

## Run Definition

- Suite: `{suite_id}`
- Engine: `{engine}`
- Episodes per task per seed: `{episodes_per_task}`
- Seeds: `{seed_text}`
- Public-claim validation: `{public_claim_status}`

## Policies

{policy_lines}

## Results Summary

| Policy | Episodes | Successes | Success rate | Expert intervention | Recovery success (successful/applied) | Verifier rejection | Primary failure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
{policy_rows}

## Validation Summary

| Run | Status | Public claim | Failures |
| --- | --- | --- | --- |
{validation_rows}

## Failure Modes

{primary_failures}

## Stressor Support

- Unsupported stressors: `{", ".join(unsupported) if unsupported else "none"}`
- Active stressors in this result pack are limited to those listed as supported in each run's `metrics.json`.

## Evidence

- Episode artifacts: `episodes.json` and `episodes.jsonl` are written for every run.
- Replay videos: `{video_count}` video files found.
- Dataset manifests: `dataset_manifest.json` is written for every run.
- Failure galleries: `failure_gallery.html` is written for every run.
- Reproducibility metadata: `run.yaml`, `config.yaml`, `environment.json`, `package_versions.json`, and `git_info.json` are written for every run.

## Publication Caveats

{_publication_caveats(summaries, video_count, policies)}

## Artifacts

- Comparison report: `{comparison_report.as_posix()}`
- Leaderboard: `{leaderboard.as_posix()}`
- Scorecard: `{scorecard.as_posix()}`

## Run Directories

{run_lines}

## Interpretation Notes

- `random` is a weak sanity baseline.
- `scripted_oracle` should be treated as an upper-bound/control baseline only when backed by a real task-specific oracle factory.
- `bc_policy` should be treated as a learned baseline only when backed by a real checkpoint or factory.
- Public benchmark claims require each run's `public_claim_validation` to pass.
- Unsupported stressors are reported in each run summary and should not be claimed as active perturbations.
- `prototype_reliability_score` is a simulator reliability heuristic, not real-world robot validation.
- If replay videos are absent, the result has episode evidence but is not replay-first.
"""
    path.write_text(text, encoding="utf-8")
    return path


def write_experiment_manifest(
    *,
    out_dir: str | Path,
    suite_id: str,
    engine: str,
    policies: list[str],
    seeds: list[int],
    episodes_per_task: int,
    run_dirs: list[Path],
    artifacts: dict[str, Path],
) -> Path:
    import json

    out_dir = Path(out_dir)
    path = out_dir / "manifest.json"
    payload: dict[str, Any] = {
        "suite": suite_id,
        "engine": engine,
        "policies": policies,
        "seeds": seeds,
        "episode_seed_protocol": "nyssa-episode-seed-v2",
        "episodes_per_task": episodes_per_task,
        "run_dirs": [run_dir.as_posix() for run_dir in run_dirs],
        "artifacts": {key: value.as_posix() for key, value in artifacts.items()},
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _summarize_runs(run_dirs: list[Path]) -> list[dict[str, Any]]:
    import json
    import yaml

    summaries: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        path = Path(run_dir) / "metrics.json"
        if not path.exists():
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        summary["_run_dir"] = Path(run_dir).as_posix()
        run_path = Path(run_dir) / "run.yaml"
        if run_path.exists():
            try:
                run_metadata = yaml.safe_load(run_path.read_text(encoding="utf-8")) or {}
            except (OSError, yaml.YAMLError):
                run_metadata = {}
            if isinstance(run_metadata, dict):
                summary["_run_seed"] = run_metadata.get("seed")
                summary["_seed_protocol"] = run_metadata.get("seed_protocol")
        summaries.append(summary)
    return summaries


def _policy_rows(summaries: list[dict[str, Any]]) -> str:
    by_policy: dict[str, dict[str, Any]] = {}
    for summary in summaries:
        policy = str(summary.get("policy") or _policy_from_run_dir(summary.get("_run_dir", "")))
        row = by_policy.setdefault(
            policy,
            {
                "episodes": 0,
                "successes": 0,
                "failures": {},
                "intervention_weighted": 0.0,
                "recovery_successes": 0.0,
                "recovery_applied": 0.0,
                "verifier_weighted": 0.0,
            },
        )
        episodes = int(summary.get("episodes", 0))
        metrics = summary.get("metrics") or {}
        row["episodes"] += episodes
        row["successes"] += round(float(summary.get("success_rate", 0.0)) * episodes)
        row["intervention_weighted"] += float(metrics.get("expert_intervention_rate", 0.0)) * episodes
        recovery_successes, recovery_applied = _recovery_success_totals(summary)
        row["recovery_successes"] += recovery_successes
        row["recovery_applied"] += recovery_applied
        row["verifier_weighted"] += float(metrics.get("verifier_rejection_rate", 0.0)) * episodes
        for label, count in (summary.get("failure_counts") or {}).items():
            row["failures"][label] = row["failures"].get(label, 0) + int(count)
    rows = []
    for policy, row in sorted(by_policy.items()):
        episodes = int(row["episodes"])
        successes = int(row["successes"])
        rate = successes / episodes if episodes else 0.0
        intervention_rate = float(row["intervention_weighted"]) / episodes if episodes else 0.0
        recovery_applied = float(row["recovery_applied"])
        recovery_rate = float(row["recovery_successes"]) / recovery_applied if recovery_applied else 0.0
        verifier_rate = float(row["verifier_weighted"]) / episodes if episodes else 0.0
        primary = _primary_failure(row["failures"])
        rows.append(
            f"| `{policy}` | {episodes} | {successes} | {rate:.4f} | "
            f"{intervention_rate:.4f} | {recovery_rate:.4f} "
            f"({int(float(row['recovery_successes']))}/{int(recovery_applied)}) | "
            f"{verifier_rate:.4f} | `{primary}` |"
        )
    return "\n".join(rows) if rows else "| n/a | 0 | 0 | 0.0000 | 0.0000 | 0.0000 (0/0) | 0.0000 | n/a |"


def _validation_rows(summaries: list[dict[str, Any]]) -> str:
    rows = []
    for summary in summaries:
        validation = summary.get("public_claim_validation") or {}
        failures = validation.get("failures") or []
        rows.append(
            "| `{run}` | `{status}` | `{claim}` | `{failures}` |".format(
                run=summary.get("_run_dir", "unknown"),
                status=validation.get("status", "unknown"),
                claim=summary.get("public_claim", False),
                failures=", ".join(failures) if failures else "none",
            )
        )
    return "\n".join(rows) if rows else "| n/a | unknown | False | missing metrics |"


def _primary_failures(summaries: list[dict[str, Any]]) -> str:
    counts: dict[str, int] = {}
    for summary in summaries:
        for label, count in (summary.get("failure_counts") or {}).items():
            counts[label] = counts.get(label, 0) + int(count)
    if not counts:
        return "No failures recorded."
    return "\n".join(f"- `{label}`: {count}" for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _count_video_files(run_dirs: list[Path]) -> int:
    suffixes = {".mp4", ".webm", ".gif"}
    count = 0
    for run_dir in run_dirs:
        if Path(run_dir).exists():
            count += sum(1 for path in Path(run_dir).rglob("*") if path.suffix.lower() in suffixes)
    return count


def _primary_failure(counts: dict[str, int]) -> str:
    if not counts:
        return "none"
    return max(counts.items(), key=lambda item: item[1])[0]


def _policy_from_run_dir(run_dir: str) -> str:
    parts = Path(run_dir).parts
    if len(parts) >= 2:
        return parts[-2]
    return "unknown"


def _publication_caveats(summaries: list[dict[str, Any]], video_count: int, policies: list[str]) -> str:
    caveats: list[str] = []
    if video_count == 0:
        caveats.append("- Replay videos are absent; do not describe this pack as video-backed.")
    if any(policy.startswith("demo_replay_policy") for policy in policies):
        caveats.append(
            "- `demo_replay_policy` is a teacher replay or oracle-reference baseline, not a learned deployable policy."
        )
    run_seeds = {summary.get("_run_seed") for summary in summaries if summary.get("_run_seed") is not None}
    if len(run_seeds) > 1 and any(
        (summary.get("_seed_protocol") or {}).get("format") != "nyssa-episode-seed-v2"
        for summary in summaries
    ):
        caveats.append(
            "- Multiple run seeds are present without `nyssa-episode-seed-v2`; adjacent legacy run seeds may "
            "share most simulator episodes, so this pack is not independent multi-seed evidence."
        )
    for policy, rate in _policy_success_rates(summaries).items():
        if policy == "random":
            continue
        if rate < 0.05:
            caveats.append(f"- `{policy}` is a failed learned/control baseline in this pack; success rate is {rate:.4f}.")
        elif rate < 0.20:
            caveats.append(f"- `{policy}` is a weak baseline in this pack; success rate is {rate:.4f}.")
    for policy, rate in _policy_metric_rates(summaries, "expert_intervention_rate").items():
        if rate > 0.25:
            caveats.append(
                f"- `{policy}` relies on frequent expert intervention in this pack; "
                f"intervention rate is {rate:.4f}."
            )
    for policy, rate in _policy_metric_rates(summaries, "recovery_success_rate").items():
        intervention = _policy_metric_rates(summaries, "expert_intervention_rate").get(policy, 0.0)
        if intervention > 0.0 and rate < 0.05:
            caveats.append(
                f"- `{policy}` has low recovery yield relative to intervention use; "
                f"recovery success rate is {rate:.4f}."
            )
    success_rates = _policy_success_rates(summaries)
    requested_variants = {policy.rsplit(":", 1)[-1] for policy in policies if ":" in policy}
    verifier_rates = _policy_metric_rates(summaries, "verifier_rejection_rate")
    for variant in sorted(requested_variants & {"verifier", "verifier_recovery"}):
        if verifier_rates.get(variant, 0.0) == 0.0 and success_rates.get(variant, 0.0) < 0.95:
            caveats.append(
                f"- `{variant}` enabled the verifier but rejected no actions despite unresolved failures; "
                "treat this as an inactive verifier, not a verifier ablation."
            )
    recovery_attempts = _policy_metric_rates(summaries, "recovery_attempt_count")
    for variant in sorted(requested_variants & {"recovery", "verifier_recovery"}):
        if recovery_attempts.get(variant, 0.0) == 0.0 and success_rates.get(variant, 0.0) < 0.95:
            caveats.append(
                f"- `{variant}` enabled recovery but attempted none despite unresolved failures; "
                "treat this as an inactive recovery path, not a recovery ablation."
            )
    if not caveats:
        return "- No publication caveats detected by the result-pack writer."
    return "\n".join(caveats)


def _policy_success_rates(summaries: list[dict[str, Any]]) -> dict[str, float]:
    totals: dict[str, list[int]] = {}
    for summary in summaries:
        policy = str(summary.get("policy") or _policy_from_run_dir(summary.get("_run_dir", "")))
        episodes = int(summary.get("episodes", 0))
        successes = round(float(summary.get("success_rate", 0.0)) * episodes)
        item = totals.setdefault(policy, [0, 0])
        item[0] += successes
        item[1] += episodes
    return {policy: successes / episodes if episodes else 0.0 for policy, (successes, episodes) in totals.items()}


def _policy_metric_rates(summaries: list[dict[str, Any]], metric_name: str) -> dict[str, float]:
    if metric_name == "recovery_success_rate":
        totals: dict[str, list[float]] = {}
        for summary in summaries:
            policy = str(summary.get("policy") or _policy_from_run_dir(summary.get("_run_dir", "")))
            successes, applied = _recovery_success_totals(summary)
            item = totals.setdefault(policy, [0.0, 0.0])
            item[0] += successes
            item[1] += applied
        return {
            policy: successes / applied if applied else 0.0
            for policy, (successes, applied) in totals.items()
        }

    totals: dict[str, list[float]] = {}
    for summary in summaries:
        policy = str(summary.get("policy") or _policy_from_run_dir(summary.get("_run_dir", "")))
        episodes = int(summary.get("episodes", 0))
        metrics = summary.get("metrics") or {}
        item = totals.setdefault(policy, [0.0, 0.0])
        item[0] += float(metrics.get(metric_name, 0.0)) * episodes
        item[1] += float(episodes)
    return {policy: weighted / episodes if episodes else 0.0 for policy, (weighted, episodes) in totals.items()}


def _recovery_success_totals(summary: dict[str, Any]) -> tuple[float, float]:
    metrics = summary.get("metrics") or {}
    if "recovery_applied_count" in metrics:
        return (
            float(metrics.get("recovery_success_count", 0.0)),
            float(metrics.get("recovery_applied_count", 0.0)),
        )

    episodes = int(summary.get("episodes", 0))
    attempts = float(metrics.get("recovery_attempt_count", 0.0)) * episodes
    return float(metrics.get("recovery_success_rate", 0.0)) * attempts, attempts
