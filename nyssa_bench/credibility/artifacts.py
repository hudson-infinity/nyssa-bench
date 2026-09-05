from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from nyssa_bench.credibility.protocol import CredibilitySpec


def load_credibility_spec(path: str | Path) -> CredibilitySpec:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
        data = (
            json.loads(text, parse_constant=_reject_json_constant)
            if path.suffix.lower() == ".json"
            else yaml.safe_load(text)
        )
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid credibility spec: {path}") from exc
    if not isinstance(data, Mapping):
        raise ValueError("credibility spec must contain a mapping")
    return CredibilitySpec.model_validate(data)


def write_credibility_report(
    report: Mapping[str, Any], out_dir: str | Path
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(report)
    payload["report_sha256"] = _sha256(payload)
    json_path = out_dir / "phase1_credibility.json"
    html_path = out_dir / "phase1_credibility.html"
    json_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(gate_id))}</td>"
        f"<td>{html.escape(str(gate.get('name', '')))}</td>"
        f"<td>{html.escape(str(gate.get('status', '')))}</td>"
        "</tr>"
        for gate_id, gate in payload.get("gates", {}).items()
    )
    html_path.write_text(
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        "<title>NyssaBench Phase 1 credibility</title></head><body>"
        "<h1>NyssaBench Phase 1 credibility</h1>"
        f"<p>Highest completed gate: {html.escape(str(payload.get('highest_completed_gate', '')))}</p>"
        "<table><thead><tr><th>Gate</th><th>Name</th><th>Status</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
        f"<pre>{html.escape(json.dumps(payload, indent=2))}</pre></body></html>\n",
        encoding="utf-8",
    )
    return {"json": json_path, "html": html_path}


def _sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")
