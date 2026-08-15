from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Report:
    suite_id: str
    policy: str
    engine: str
    summary: dict[str, Any]
    run_dir: Path | None = None

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_html(), encoding="utf-8")
        return path

    def to_html(self) -> str:
        success_rate = float(self.summary.get("success_rate", 0.0)) * 100
        success_ci = self.summary.get("success_rate_ci95", [0.0, 0.0])
        prototype_score = float(self.summary.get("prototype_reliability_score", self.summary.get("sim_to_real_score", 0.0)))
        primary_failure = self.summary.get("primary_failure_mode") or "none"
        benchmark_tier = self.summary.get("benchmark_tier", "unknown")
        validation = self.summary.get("public_claim_validation", {})
        stressor_support = self.summary.get("stressor_support", {})
        metrics = self.summary.get("metrics", {})
        compute = self.summary.get("compute", {})
        intervention_rate = float(metrics.get("expert_intervention_rate", 0.0)) * 100
        recovery_rate = float(metrics.get("recovery_success_rate", 0.0)) * 100
        recovery_success_count = int(float(metrics.get("recovery_success_count", 0.0)))
        recovery_applied_count = int(float(metrics.get("recovery_applied_count", 0.0)))
        recovery_episode_rate = float(metrics.get("recovery_episode_success_rate", 0.0)) * 100
        recovery_episode_success_count = int(float(metrics.get("recovery_episode_success_count", 0.0)))
        recovery_episode_applied_count = int(float(metrics.get("recovery_episode_applied_count", 0.0)))
        verifier_rejection_rate = float(metrics.get("verifier_rejection_rate", 0.0)) * 100
        failure_counts = self.summary.get("failure_counts", {})
        per_task = self.summary.get("per_task", {})
        per_seed = self.summary.get("per_seed", {})
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>NyssaBench Report - {html.escape(self.suite_id)}</title>
  <style>
    body {{ font-family: Inter, Arial, sans-serif; margin: 40px; color: #17202a; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 16px; }}
    th, td {{ border-bottom: 1px solid #d8dee4; padding: 8px; text-align: left; }}
    .metric {{ display: inline-block; margin: 12px 24px 12px 0; }}
    .value {{ font-size: 28px; font-weight: 700; }}
    pre {{ background: #f6f8fa; padding: 16px; overflow: auto; }}
  </style>
</head>
<body>
  <h1>NyssaBench Report</h1>
  <p><strong>Policy:</strong> {html.escape(self.policy)}<br>
  <strong>Suite:</strong> {html.escape(self.suite_id)}<br>
  <strong>Engine:</strong> {html.escape(self.engine)}<br>
  <strong>Episodes:</strong> {self.summary.get("episodes", 0)}</p>

  <section>
    <div class="metric"><div>Success rate</div><div class="value">{success_rate:.1f}%</div></div>
    <div class="metric"><div>95% CI</div><div class="value">{_format_ci_percent(success_ci)}</div></div>
    <div class="metric"><div>Prototype reliability</div><div class="value">{prototype_score:.3f}</div></div>
    <div class="metric"><div>Primary failure mode</div><div class="value">{html.escape(str(primary_failure))}</div></div>
    <div class="metric"><div>Benchmark tier</div><div class="value">{html.escape(str(benchmark_tier))}</div></div>
    <div class="metric"><div>Public claim</div><div class="value">{html.escape(str(self.summary.get("public_claim", False)))}</div></div>
    <div class="metric"><div>Expert intervention</div><div class="value">{intervention_rate:.1f}%</div></div>
    <div class="metric"><div>Recovery attempt success ({recovery_success_count}/{recovery_applied_count} applied)</div><div class="value">{recovery_rate:.1f}%</div></div>
    <div class="metric"><div>Recovery episode success ({recovery_episode_success_count}/{recovery_episode_applied_count} episodes)</div><div class="value">{recovery_episode_rate:.1f}%</div></div>
    <div class="metric"><div>Verifier rejection</div><div class="value">{verifier_rejection_rate:.1f}%</div></div>
  </section>

  <h2>Public-Claim Validation</h2>
  {_validation_table(validation)}

  <h2>Per-Task Results</h2>
  {_per_task_table(per_task)}

  <h2>Per-Seed Results</h2>
  {_per_seed_table(per_seed)}

  <h2>Stressor Support</h2>
  {_stressor_table(stressor_support)}

  <h2>Aggregate Metrics</h2>
  {_table(metrics)}

  <h2>Compute Cost</h2>
  {_table(compute)}

  <h2>Failure Clusters</h2>
  {_table(failure_counts)}

  <h2>Top Failure Episodes</h2>
  {_failure_episode_table(self.run_dir)}

  <h2>Raw Summary</h2>
  <pre>{html.escape(json.dumps(self.summary, indent=2))}</pre>
</body>
</html>
"""


def _table(data: dict[str, Any]) -> str:
    rows = "\n".join(
        f"<tr><td>{html.escape(str(key))}</td><td>{html.escape(_format_value(value))}</td></tr>"
        for key, value in sorted(data.items())
    )
    return f"<table><tbody>{rows}</tbody></table>" if rows else "<p>No data.</p>"


def _validation_table(data: Any) -> str:
    if not isinstance(data, dict):
        return "<p>No validation data.</p>"
    checks = data.get("checks", {})
    failures = data.get("failures", [])
    warnings = data.get("warnings", [])
    rows = [
        ("status", data.get("status", "unknown")),
        ("public_claim", data.get("public_claim", False)),
        ("benchmark_tier", data.get("benchmark_tier", "unknown")),
        ("failures", ", ".join(failures) if failures else "none"),
        ("warnings", ", ".join(warnings) if warnings else "none"),
    ]
    if isinstance(checks, dict):
        rows.extend((f"check:{key}", value) for key, value in sorted(checks.items()))
    return _table(dict(rows))


def _stressor_table(data: Any) -> str:
    if not isinstance(data, dict):
        return "<p>No stressor support data.</p>"
    unsupported = data.get("unsupported_by_task", {})
    supported = data.get("supported_by_task", {})
    task_ids = sorted(set(unsupported) | set(supported))
    if not task_ids:
        return "<p>No declared stressors.</p>"
    rows = []
    for task_id in task_ids:
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(task_id))}</td>"
            f"<td>{html.escape(', '.join(supported.get(task_id, [])) or 'none')}</td>"
            f"<td>{html.escape(', '.join(unsupported.get(task_id, [])) or 'none')}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Task</th><th>Supported</th><th>Unsupported</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _failure_episode_table(run_dir: Path | None) -> str:
    if run_dir is None:
        return "<p>No run directory available.</p>"
    episodes_path = Path(run_dir) / "episodes.json"
    if not episodes_path.exists():
        return "<p>No episode artifact available.</p>"
    try:
        episodes = json.loads(episodes_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "<p>Could not read episode artifact.</p>"
    failures = [item for item in episodes if not item.get("success")]
    if not failures:
        return "<p>No failures recorded.</p>"
    failures = sorted(failures, key=lambda item: len(item.get("steps", [])), reverse=True)[:10]
    rows = []
    for item in failures:
        clip = item.get("failure_clip_path") or item.get("replay_path") or ""
        clip_cell = f'<a href="{html.escape(str(clip))}">video</a>' if clip else "none"
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(item.get('task_id')))}</td>"
            f"<td>{int(item.get('episode_index', 0))}</td>"
            f"<td>{html.escape(str(item.get('failure_label') or 'unknown'))}</td>"
            f"<td>{html.escape(str(item.get('failure_label_source') or 'unknown'))}</td>"
            f"<td>{len(item.get('steps', []))}</td>"
            f"<td>{clip_cell}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Task</th><th>Episode</th><th>Failure</th><th>Source</th>"
        "<th>Steps</th><th>Replay</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _format_ci_percent(value: Any) -> str:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return "n/a"
    return f"{float(value[0]) * 100:.1f}-{float(value[1]) * 100:.1f}%"


def _per_task_table(data: dict[str, Any]) -> str:
    if not data:
        return "<p>No per-task data.</p>"
    rows = []
    for task_id, summary in sorted(data.items()):
        task_summary = dict(summary)
        success_rate = float(task_summary.get("success_rate", 0.0)) * 100
        metrics = task_summary.get("metrics", {})
        intervention_rate = float(metrics.get("expert_intervention_rate", 0.0)) * 100
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(task_id))}</td>"
            f"<td>{int(task_summary.get('episodes', 0))}</td>"
            f"<td>{success_rate:.1f}%</td>"
            f"<td>{html.escape(_format_ci_percent(task_summary.get('success_rate_ci95', [])))}</td>"
            f"<td>{intervention_rate:.1f}%</td>"
            f"<td>{html.escape(str(task_summary.get('primary_failure_mode') or 'none'))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Task</th><th>Episodes</th><th>Success</th>"
        "<th>95% CI</th><th>Expert intervention</th><th>Primary failure</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _per_seed_table(data: dict[str, Any]) -> str:
    if not data:
        return "<p>No per-seed data.</p>"
    rows = []
    for seed, summary in sorted(data.items(), key=lambda item: int(item[0]) if str(item[0]).isdigit() else str(item[0])):
        seed_summary = dict(summary)
        success_rate = float(seed_summary.get("success_rate", 0.0)) * 100
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(seed))}</td>"
            f"<td>{int(seed_summary.get('episodes', 0))}</td>"
            f"<td>{success_rate:.1f}%</td>"
            f"<td>{html.escape(_format_ci_percent(seed_summary.get('success_rate_ci95', [])))}</td>"
            f"<td>{html.escape(str(seed_summary.get('primary_failure_mode') or 'none'))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Seed</th><th>Episodes</th><th>Success</th>"
        "<th>95% CI</th><th>Primary failure</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )
