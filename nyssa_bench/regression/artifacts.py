from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from nyssa_bench.regression.evaluator import REGRESSION_REPORT_FORMAT
from nyssa_bench.regression.protocol import RegressionStudySpec


def load_regression_study(path: str | Path) -> RegressionStudySpec:
    return RegressionStudySpec.from_dict(_load_mapping(Path(path), "regression study"))


def write_regression_report(
    report: Mapping[str, Any], out_dir: str | Path
) -> dict[str, Path]:
    validate_regression_report(report)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "regression_report.json"
    html_path = out_dir / "regression_report.html"
    json_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    html_path.write_text(_report_html(report), encoding="utf-8")
    return {"json": json_path, "html": html_path}


def load_regression_report(path: str | Path) -> dict[str, Any]:
    report = _load_mapping(Path(path), "regression report")
    validate_regression_report(report)
    return report


def validate_regression_report(report: Mapping[str, Any]) -> None:
    if report.get("format") != REGRESSION_REPORT_FORMAT:
        raise ValueError(f"unsupported regression report format: {report.get('format')}")
    required = {
        "format",
        "schema_version",
        "study_id",
        "study_version",
        "prespecified_at",
        "spec_sha256",
        "decision",
        "exit_code",
        "decision_semantics",
        "summary",
        "policies",
        "evidence_requirements",
        "study_metadata",
        "cells",
        "rules",
        "interpretation",
        "report_sha256",
    }
    unknown = sorted(set(report) - required)
    missing = sorted(required - set(report))
    if unknown:
        raise ValueError("unknown regression report fields: " + ", ".join(unknown))
    if missing:
        raise ValueError("missing regression report fields: " + ", ".join(missing))
    if report.get("schema_version") != 1:
        raise ValueError("unsupported regression report schema version")
    if not str(report.get("study_id", "")).strip() or not str(
        report.get("study_version", "")
    ).strip():
        raise ValueError("regression report study identity must be non-empty")
    _require_sha256(report.get("spec_sha256"), "regression spec_sha256")
    decision = str(report.get("decision", ""))
    expected_exit = {"pass": 0, "fail": 1, "inconclusive": 2, "invalid": 3}.get(
        decision
    )
    if expected_exit is None or report.get("exit_code") != expected_exit:
        raise ValueError("regression report decision and exit code are inconsistent")
    cells = report.get("cells")
    rules = report.get("rules")
    if not isinstance(cells, list) or not all(isinstance(item, Mapping) for item in cells):
        raise ValueError("regression report cells must be mappings")
    if not isinstance(rules, list) or not all(isinstance(item, Mapping) for item in rules):
        raise ValueError("regression report rules must be mappings")
    cell_ids = [str(item.get("cell_id", "")) for item in cells]
    rule_ids = [str(item.get("rule_id", "")) for item in rules]
    if (
        any(not value for value in (*cell_ids, *rule_ids))
        or len(cell_ids) != len(set(cell_ids))
        or len(rule_ids) != len(set(rule_ids))
    ):
        raise ValueError("regression report cell and rule IDs must be non-empty and unique")
    if any(
        item.get("status") not in {"ready", "inconclusive", "invalid"}
        for item in cells
    ):
        raise ValueError("regression report contains an unsupported cell status")
    if any(
        item.get("status") not in {"passed", "failed", "inconclusive", "invalid"}
        for item in rules
    ):
        raise ValueError("regression report contains an unsupported rule status")
    derived_decision = _derive_decision(cells, rules)
    if decision != derived_decision:
        raise ValueError("regression report decision does not match cell and rule states")
    expected_summary = {
        "cell_status_counts": _status_counts(cells),
        "rule_status_counts": _status_counts(rules),
        "cells": len(cells),
        "rules": len(rules),
    }
    if report.get("summary") != expected_summary:
        raise ValueError("regression report summary does not match its records")
    unhashed = {key: value for key, value in report.items() if key != "report_sha256"}
    if report.get("report_sha256") != _sha256(unhashed):
        raise ValueError("regression report hash mismatch")


def _report_html(report: Mapping[str, Any]) -> str:
    decision = str(report.get("decision", "invalid"))
    cell_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(item.get('cell_id', '')))}</td>"
        f"<td>{html.escape(str(item.get('condition_kind', '')))}</td>"
        f"<td>{html.escape(str(item.get('status', '')))}</td>"
        f"<td>{html.escape(str(item.get('pinned_episode_count', '')))}</td>"
        f"<td>{html.escape(str(item.get('reason') or ''))}</td>"
        "</tr>"
        for item in report.get("cells", [])
    )
    rule_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(item.get('rule_id', '')))}</td>"
        f"<td>{html.escape(str(item.get('metric_id', '')))}</td>"
        f"<td>{html.escape(str(item.get('kind', '')))}</td>"
        f"<td>{html.escape(str(item.get('status', '')))}</td>"
        f"<td>{html.escape(str(item.get('reason') or ''))}</td>"
        "</tr>"
        for item in report.get("rules", [])
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>NyssaBench policy regression - {html.escape(str(report.get('study_id', '')))}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 40px; color: #17202a; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; }}
    th, td {{ border-bottom: 1px solid #d8dee4; padding: 8px; text-align: left; }}
    .decision {{ font-size: 28px; font-weight: 700; text-transform: uppercase; }}
    pre {{ background: #f6f8fa; padding: 16px; overflow: auto; }}
  </style>
</head>
<body>
  <h1>Policy checkpoint regression study</h1>
  <p><strong>Study:</strong> {html.escape(str(report.get('study_id', '')))}@
  {html.escape(str(report.get('study_version', '')))}<br>
  <strong>Prespecified:</strong> {html.escape(str(report.get('prespecified_at', '')))}<br>
  <strong>Spec:</strong> <code>{html.escape(str(report.get('spec_sha256', '')))}</code></p>
  <div class="decision">{html.escape(decision)}</div>
  <p>{html.escape(str(report.get('interpretation', '')))}</p>
  <h2>Evaluation cells</h2>
  <table><thead><tr><th>Cell</th><th>Condition</th><th>Status</th><th>Episodes</th><th>Reason</th></tr></thead>
  <tbody>{cell_rows}</tbody></table>
  <h2>Decision rules</h2>
  <table><thead><tr><th>Rule</th><th>Metric</th><th>Kind</th><th>Status</th><th>Reason</th></tr></thead>
  <tbody>{rule_rows}</tbody></table>
  <h2>Machine-readable evidence</h2>
  <pre>{html.escape(json.dumps(report, indent=2))}</pre>
</body>
</html>
"""


def _load_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
        value = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid {label}: {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain a mapping")
    return dict(value)


def _sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: Any, label: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def _derive_decision(
    cells: list[Any], rules: list[Any]
) -> str:
    if any(item.get("status") == "invalid" for item in (*cells, *rules)):
        return "invalid"
    if any(item.get("status") == "failed" for item in rules):
        return "fail"
    if any(item.get("status") != "passed" for item in rules):
        return "inconclusive"
    return "pass"


def _status_counts(values: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        status = str(value.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))
