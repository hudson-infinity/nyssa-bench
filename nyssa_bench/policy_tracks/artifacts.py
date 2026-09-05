from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from nyssa_bench.policy_tracks.protocol import PolicyTrackRegistry


def load_policy_track_registry(path: str | Path) -> PolicyTrackRegistry:
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
        value = (
            json.loads(text)
            if source.suffix.lower() == ".json"
            else yaml.safe_load(text)
        )
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid policy-track registry: {source}") from exc
    if not isinstance(value, Mapping):
        raise ValueError("policy-track registry must contain a mapping")
    return PolicyTrackRegistry.model_validate(value)


def write_policy_track_report(
    report: Mapping[str, Any], out_dir: str | Path
) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "policy_tracks.json"
    html_path = out / "policy_tracks.html"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(track.get('track_id', '')))}</td>"
        f"<td>{html.escape(str(track.get('role', '')))}</td>"
        f"<td>{html.escape(str(track.get('policy_family', '')))}</td>"
        f"<td>{html.escape(str(track.get('validated', False)))}</td>"
        "</tr>"
        for track in report.get("tracks", [])
    )
    html_path.write_text(
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        "<title>NyssaBench policy tracks</title></head><body>"
        "<h1>Policy track audit</h1>"
        f"<p>Status: <strong>{html.escape(str(report.get('status', '')))}</strong></p>"
        "<table><thead><tr><th>Track</th><th>Role</th><th>Family</th>"
        f"<th>Validated</th></tr></thead><tbody>{rows}</tbody></table></body></html>\n",
        encoding="utf-8",
    )
    return {"json": json_path, "html": html_path}
