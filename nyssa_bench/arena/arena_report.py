from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any

from nyssa_bench.arena.pairwise_runner import EpisodeKey, PairwiseSummary
from nyssa_bench.arena.preference_schema import PreferenceRecord


def save_pairwise_results(summary: PairwiseSummary, out_dir: str | Path) -> Path:
    """Save outcomes plus machine-readable pairing coverage sidecars."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "pairwise_results.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for outcome in summary.outcomes:
            handle.write(
                json.dumps(outcome.to_dict(), sort_keys=True, allow_nan=False) + "\n"
            )
    save_pairwise_summary(summary, out_dir)
    save_pairwise_coverage(summary, out_dir)
    save_pairwise_metrics(summary, out_dir)
    return path


def save_pairwise_summary(summary: PairwiseSummary, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "pairwise_summary.json"
    path.write_text(
        json.dumps(summary.to_dict(), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return path


def save_pairwise_coverage(summary: PairwiseSummary, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "pairwise_coverage.csv"
    coverage = summary.coverage
    row = {
        "comparison_mode": summary.comparison_mode,
        "pairing_claim_eligible": str(summary.pairing_claim_eligible).lower(),
        "comparison_contract_sha256": summary.comparison_contract_sha256,
        "policy_a_label": coverage.policy_a_label,
        "policy_b_label": coverage.policy_b_label,
        "policy_a_requested_count": coverage.policy_a_requested_count,
        "policy_b_requested_count": coverage.policy_b_requested_count,
        "policy_a_unique_count": coverage.policy_a_unique_count,
        "policy_b_unique_count": coverage.policy_b_unique_count,
        "matched_count": coverage.matched_count,
        "unmatched_a_count": coverage.unmatched_a_count,
        "unmatched_b_count": coverage.unmatched_b_count,
        "duplicate_a_count": coverage.duplicate_a_count,
        "duplicate_b_count": coverage.duplicate_b_count,
        "policy_a_coverage": coverage.policy_a_coverage,
        "policy_b_coverage": coverage.policy_b_coverage,
        "joint_coverage": coverage.joint_coverage,
        "condition_compatible_count": summary.paired_metrics.get(
            "condition_compatible_pairs", 0
        ),
        "condition_incompatible_count": summary.paired_metrics.get(
            "condition_incompatible_pairs", 0
        ),
        "unmatched_a_keys": _keys_json(coverage.unmatched_a_keys),
        "unmatched_b_keys": _keys_json(coverage.unmatched_b_keys),
        "caveats": json.dumps(summary.caveats),
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return path


def save_pairwise_metrics(summary: PairwiseSummary, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "pairwise_metrics.csv"
    metrics = summary.paired_metrics
    rows = [
        _metric_row("success_difference", metrics.get("success_difference", {})),
        _metric_row(
            "time_to_failure_difference",
            metrics.get("time_to_failure_difference", {}),
        ),
    ]
    rows.extend(
        _metric_row(metric_id, value)
        for metric_id, value in sorted(
            dict(metrics.get("numeric_deltas", {})).items()
        )
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def save_preference_table(
    preferences: list[PreferenceRecord], out_dir: str | Path
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "preference_table.csv"
    with path.open("w", encoding="utf-8") as handle:
        handle.write("task_id,seed,episode_index,choice,reason,evaluator_id,blinded\n")
        for item in preferences:
            handle.write(
                ",".join(
                    [
                        _csv(item.task_id),
                        str(item.seed),
                        str(item.episode_index),
                        _csv(item.choice),
                        _csv(item.reason),
                        _csv(item.evaluator_id or ""),
                        str(item.blinded).lower(),
                    ]
                )
                + "\n"
            )
    return path


def save_arena_report(
    summary: PairwiseSummary,
    out_dir: str | Path,
    *,
    title: str = "NyssaBench Arena Report",
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(item.task_id)}</td>"
        f"<td>{item.seed}</td>"
        f"<td>{item.episode_index}</td>"
        f"<td>{html.escape(item.winner)}</td>"
        f"<td>{html.escape(str(item.policy_a_failure or ''))}</td>"
        f"<td>{html.escape(str(item.policy_b_failure or ''))}</td>"
        f"<td>{'yes' if item.condition_compatible else 'no'}</td>"
        f"<td>{html.escape(item.evidence['time_to_failure']['status'])}</td>"
        "</tr>"
        for item in summary.outcomes
    )
    wins = ", ".join(
        f"{html.escape(key)}: {value}" for key, value in sorted(summary.wins.items())
    )
    failure_deltas = ", ".join(
        f"{html.escape(key)}: {value}"
        for key, value in sorted(summary.failure_deltas.items())
    )
    coverage = summary.coverage
    status_class = "complete" if coverage.complete else "partial"
    status_heading = (
        "Complete paired comparison"
        if coverage.complete
        else "Partial exploratory comparison"
    )
    status_detail = (
        "Episode pairing is complete. This satisfies the pairing requirement for benchmark claims."
        if coverage.complete
        else "NON-COMPARABLE PARTIAL OUTPUT: unmatched episodes were excluded. Do not use this report for benchmark claims."
    )
    caveats = "".join(f"<li>{html.escape(caveat)}</li>" for caveat in summary.caveats)
    caveat_section = f"<h2>Caveats</h2><ul>{caveats}</ul>" if caveats else ""
    unmatched_a = _key_table_rows(coverage.unmatched_a_keys)
    unmatched_b = _key_table_rows(coverage.unmatched_b_keys)
    paired_metrics = _paired_metrics_table(summary.paired_metrics)
    condition_mismatches = _condition_mismatch_table(summary.condition_mismatches)
    body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: Inter, Arial, sans-serif; margin: 40px; color: #17202a; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; }}
    th, td {{ border-bottom: 1px solid #d8dee4; padding: 8px; text-align: left; }}
    .status {{ border-left: 5px solid #247a42; padding: 12px 16px; background: #f1f8f3; }}
    .status.partial {{ border-left-color: #b42318; background: #fff2f0; }}
    .coverage {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; }}
    .coverage div {{ border: 1px solid #d8dee4; padding: 12px; }}
    .coverage strong {{ display: block; font-size: 1.35rem; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <section class="status {status_class}">
    <h2>{status_heading}</h2>
    <p>{status_detail}</p>
  </section>
  <h2>Pairing coverage</h2>
  <div class="coverage">
    <div><strong>{coverage.matched_count}</strong>Matched</div>
    <div><strong>{coverage.unmatched_a_count}</strong>Unmatched {html.escape(coverage.policy_a_label)}</div>
    <div><strong>{coverage.unmatched_b_count}</strong>Unmatched {html.escape(coverage.policy_b_label)}</div>
    <div><strong>{_percent(coverage.policy_a_coverage)}</strong>{html.escape(coverage.policy_a_label)} coverage</div>
    <div><strong>{_percent(coverage.policy_b_coverage)}</strong>{html.escape(coverage.policy_b_label)} coverage</div>
    <div><strong>{_percent(coverage.joint_coverage)}</strong>Joint coverage</div>
  </div>
  <p>Requested episodes: {html.escape(coverage.policy_a_label)}={coverage.policy_a_requested_count}, {html.escape(coverage.policy_b_label)}={coverage.policy_b_requested_count}</p>
  <p>Total pairs: {summary.total_pairs}</p>
  <p>Wins: {wins or "none"}</p>
  <p>Failure deltas: {failure_deltas or "none"}</p>
  <p>Comparison contract: <code>{html.escape(summary.comparison_contract_sha256)}</code></p>
  {caveat_section}
  <h2>Paired metrics</h2>
  {paired_metrics}
  <h2>Condition mismatches</h2>
  {condition_mismatches}
  <h2>Unmatched {html.escape(coverage.policy_a_label)} episodes</h2>
  {_key_table(unmatched_a)}
  <h2>Unmatched {html.escape(coverage.policy_b_label)} episodes</h2>
  {_key_table(unmatched_b)}
  <h2>Matched outcomes</h2>
  <table>
    <thead>
      <tr><th>Task</th><th>Seed</th><th>Episode</th><th>Winner</th><th>Policy A failure</th><th>Policy B failure</th><th>Condition matched</th><th>Failure-time comparison</th></tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
</body>
</html>
"""
    path = out_dir / "arena_report.html"
    path.write_text(body, encoding="utf-8")
    return path


def _keys_json(keys: tuple[EpisodeKey, ...]) -> str:
    return json.dumps([key.to_dict() for key in keys], separators=(",", ":"))


def _metric_row(metric_id: str, value: Any) -> dict[str, Any]:
    measurement = dict(value) if isinstance(value, dict) else {}
    interval = measurement.get("ci95")
    return {
        "metric": metric_id,
        "status": measurement.get("status", "unavailable"),
        "value": measurement.get("value"),
        "ci95_low": interval[0]
        if isinstance(interval, (list, tuple)) and len(interval) == 2
        else None,
        "ci95_high": interval[1]
        if isinstance(interval, (list, tuple)) and len(interval) == 2
        else None,
        "sample_size": measurement.get("sample_size", 0),
        "missing_count": measurement.get("missing_count", 0),
    }


def _paired_metrics_table(metrics: dict[str, Any]) -> str:
    values = {
        "success_difference": metrics.get("success_difference", {}),
        "time_to_failure_difference": metrics.get(
            "time_to_failure_difference", {}
        ),
        **dict(metrics.get("numeric_deltas", {})),
    }
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(metric_id)}</td>"
        f"<td>{html.escape(str(value.get('status', 'unavailable')))}</td>"
        f"<td>{html.escape(str(value.get('value', '')))}</td>"
        f"<td>{html.escape(str(value.get('ci95', '')))}</td>"
        f"<td>{int(value.get('sample_size', 0) or 0)}</td>"
        f"<td>{int(value.get('missing_count', 0) or 0)}</td>"
        "</tr>"
        for metric_id, value in sorted(values.items())
        if isinstance(value, dict)
    )
    return (
        "<table><thead><tr><th>Metric</th><th>Status</th><th>A-B</th>"
        "<th>95% CI</th><th>Pairs</th><th>Missing</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _condition_mismatch_table(values: tuple[dict[str, Any], ...]) -> str:
    if not values:
        return "<p>None</p>"
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(item.get('episode_key', {})))}</td>"
        f"<td>{html.escape(', '.join(item.get('differing_fields', [])))}</td>"
        f"<td><code>{html.escape(str(item.get('policy_a_condition_sha256', '')))}</code></td>"
        f"<td><code>{html.escape(str(item.get('policy_b_condition_sha256', '')))}</code></td>"
        "</tr>"
        for item in values
    )
    return (
        "<table><thead><tr><th>Episode</th><th>Differing fields</th>"
        "<th>Policy A condition</th><th>Policy B condition</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _key_table_rows(keys: tuple[EpisodeKey, ...]) -> str:
    return "".join(
        "<tr>"
        f"<td>{html.escape(key.task_id)}</td>"
        f"<td>{key.seed}</td>"
        f"<td>{key.episode_index}</td>"
        "</tr>"
        for key in keys
    )


def _key_table(rows: str) -> str:
    if not rows:
        return "<p>None</p>"
    return (
        "<table><thead><tr><th>Task</th><th>Seed</th><th>Episode</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _percent(value: float) -> str:
    return f"{value:.1%}"


def _csv(value: str) -> str:
    escaped = value.replace('"', '""')
    if any(char in escaped for char in [",", "\n", '"']):
        return f'"{escaped}"'
    return escaped
