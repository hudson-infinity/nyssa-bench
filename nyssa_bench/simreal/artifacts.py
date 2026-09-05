from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from nyssa_bench.simreal.protocol import SimRealStudySpec


def load_sim_real_study(path: str | Path) -> SimRealStudySpec:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
        data = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid sim-real study: {path}") from exc
    if not isinstance(data, Mapping):
        raise ValueError("sim-real study must contain a mapping")
    return SimRealStudySpec.model_validate(data)


def write_sim_real_report(
    report: Mapping[str, Any], out_dir: str | Path
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(report)
    payload["report_sha256"] = _sha256(payload)
    json_path = out_dir / "sim_real_study.json"
    html_path = out_dir / "sim_real_study.html"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    metric_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(metric_id))}</td>"
        f"<td>{html.escape(str(value.get('status', '')))}</td>"
        f"<td><pre>{html.escape(json.dumps(value, indent=2))}</pre></td>"
        "</tr>"
        for metric_id, value in payload.get("metrics", {}).items()
    )
    html_path.write_text(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<title>NyssaBench sim-real study</title></head><body>"
        f"<h1>{html.escape(str(payload.get('study_id', '')))}</h1>"
        f"<p>Status: <strong>{html.escape(str(payload.get('status', '')))}</strong></p>"
        f"<p>{html.escape(str(payload.get('claim_boundary', '')))}</p>"
        "<table><thead><tr><th>Metric</th><th>Status</th><th>Evidence</th></tr></thead>"
        f"<tbody>{metric_rows}</tbody></table>"
        f"<pre>{html.escape(json.dumps(payload, indent=2))}</pre></body></html>\n",
        encoding="utf-8",
    )
    return {"json": json_path, "html": html_path}


def load_sim_real_report(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid sim-real report: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("sim-real report must contain a mapping")
    report = dict(payload)
    observed = report.pop("report_sha256", None)
    if observed != _sha256(report):
        raise ValueError("sim-real report hash mismatch")
    if report.get("format") != "nyssa-sim-real-study-report-v1":
        raise ValueError("unsupported sim-real report format")
    report["report_sha256"] = observed
    return report


def _sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
