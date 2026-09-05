from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from nyssa_bench.hardware_study.protocol import HardwareCalibrationStudy


def load_hardware_study(path: str | Path) -> HardwareCalibrationStudy:
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
        value = (
            json.loads(text)
            if source.suffix.lower() == ".json"
            else yaml.safe_load(text)
        )
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid hardware calibration study: {source}") from exc
    if not isinstance(value, Mapping):
        raise ValueError("hardware calibration study must contain a mapping")
    return HardwareCalibrationStudy.model_validate(value)


def write_hardware_study_report(
    report: Mapping[str, Any], out_dir: str | Path
) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "hardware_calibration.json"
    html_path = out / "hardware_calibration.html"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(item.get('check_id', '')))}</td>"
        f"<td>{html.escape(str(item.get('status', '')))}</td>"
        f"<td>{html.escape(str(item.get('reason') or ''))}</td>"
        "</tr>"
        for item in report.get("checks", [])
    )
    html_path.write_text(
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        "<title>NyssaBench hardware calibration</title></head><body>"
        "<h1>Hardware calibration audit</h1>"
        f"<p>Status: <strong>{html.escape(str(report.get('status', '')))}</strong></p>"
        "<table><thead><tr><th>Check</th><th>Status</th><th>Reason</th></tr>"
        f"</thead><tbody>{rows}</tbody></table></body></html>\n",
        encoding="utf-8",
    )
    return {"json": json_path, "html": html_path}
