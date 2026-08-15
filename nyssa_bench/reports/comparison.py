from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from nyssa_bench.metrics.sim_to_real import score_summary


def load_run_summary(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    metrics_path = run_dir / "metrics.json"
    config_path = run_dir / "config.yaml"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Run metrics not found: {metrics_path}")

    summary = json.loads(metrics_path.read_text(encoding="utf-8"))
    summary.setdefault("run_dir", run_dir.as_posix())
    if config_path.exists():
        summary["config_path"] = str(config_path)
    score = float(summary.get("prototype_reliability_score", summary.get("sim_to_real_score", score_summary(summary))))
    summary["prototype_reliability_score"] = score
    return summary


def compare_runs(run_dirs: list[str | Path]) -> dict[str, Any]:
    runs = [load_run_summary(path) for path in run_dirs]
    ranked = sorted(
        runs,
        key=lambda item: (
            float(item.get("success_rate", 0.0)),
            float(item.get("prototype_reliability_score", 0.0)),
        ),
        reverse=True,
    )
    return {
        "runs": runs,
        "ranking": [
            {
                "rank": index + 1,
                "run_dir": item.get("run_dir"),
                "success_rate": item.get("success_rate", 0.0),
                "success_rate_ci95": item.get("success_rate_ci95", [0.0, 0.0]),
                "prototype_reliability_score": item.get("prototype_reliability_score", 0.0),
                "benchmark_tier": item.get("benchmark_tier", "unknown"),
                "public_claim": item.get("public_claim", False),
                "public_claim_status": (item.get("public_claim_validation") or {}).get("status", "unknown"),
                "primary_failure_mode": item.get("primary_failure_mode"),
                "expert_intervention_rate": (item.get("metrics") or {}).get("expert_intervention_rate", 0.0),
                "recovery_success_rate": (item.get("metrics") or {}).get("recovery_success_rate", 0.0),
                "recovery_success_count": (item.get("metrics") or {}).get("recovery_success_count", 0.0),
                "recovery_applied_count": (item.get("metrics") or {}).get("recovery_applied_count", 0.0),
                "recovery_episode_success_rate": (item.get("metrics") or {}).get(
                    "recovery_episode_success_rate", 0.0
                ),
                "verifier_rejection_rate": (item.get("metrics") or {}).get("verifier_rejection_rate", 0.0),
                "wall_time_seconds": (item.get("compute") or {}).get("wall_time_seconds", 0.0),
            }
            for index, item in enumerate(ranked)
        ],
    }


def save_comparison_report(comparison: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_comparison_html(comparison), encoding="utf-8")
    return path


def save_leaderboard(comparison: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(comparison["ranking"], indent=2), encoding="utf-8")
    return path


def _comparison_html(comparison: dict[str, Any]) -> str:
    rows = "\n".join(
        "<tr>"
        f"<td>{item['rank']}</td>"
        f"<td>{html.escape(str(item['run_dir']))}</td>"
        f"<td>{float(item['success_rate']) * 100:.1f}%</td>"
        f"<td>{html.escape(_format_ci_percent(item.get('success_rate_ci95')))}</td>"
        f"<td>{float(item['prototype_reliability_score']):.3f}</td>"
        f"<td>{html.escape(str(item.get('benchmark_tier') or 'unknown'))}</td>"
        f"<td>{html.escape(str(item.get('public_claim_status') or 'unknown'))}</td>"
        f"<td>{float(item.get('expert_intervention_rate', 0.0)) * 100:.1f}%</td>"
        f"<td>{float(item.get('recovery_success_rate', 0.0)) * 100:.1f}% "
        f"({int(float(item.get('recovery_success_count', 0.0)))}/"
        f"{int(float(item.get('recovery_applied_count', 0.0)))})</td>"
        f"<td>{float(item.get('verifier_rejection_rate', 0.0)) * 100:.1f}%</td>"
        f"<td>{float(item.get('wall_time_seconds', 0.0)):.1f}s</td>"
        f"<td>{html.escape(str(item.get('primary_failure_mode') or 'none'))}</td>"
        "</tr>"
        for item in comparison["ranking"]
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>NyssaBench Policy Comparison</title>
  <style>
    body {{ font-family: Inter, Arial, sans-serif; margin: 40px; color: #17202a; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #d8dee4; padding: 8px; text-align: left; }}
  </style>
</head>
<body>
  <h1>Policy Comparison</h1>
  <table>
    <thead>
        <tr><th>Rank</th><th>Run</th><th>Success</th><th>95% CI</th><th>Prototype reliability</th><th>Tier</th><th>Claim status</th><th>Expert intervention</th><th>Recovery success (successful/applied)</th><th>Verifier rejection</th><th>Wall time</th><th>Primary failure</th></tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
</body>
</html>
"""


def _format_ci_percent(value: Any) -> str:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return "n/a"
    return f"{float(value[0]) * 100:.1f}-{float(value[1]) * 100:.1f}%"
