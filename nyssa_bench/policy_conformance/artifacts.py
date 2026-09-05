from __future__ import annotations

import html
import json
import shutil
from pathlib import Path
from typing import Any, Mapping

import yaml

from nyssa_bench.nep import PolicyContract
from nyssa_bench.package_resources import policy_example_root


def load_policy_contract(path: str | Path) -> PolicyContract:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
        data = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid policy contract: {path}") from exc
    if not isinstance(data, Mapping):
        raise ValueError("policy contract must contain a mapping")
    return PolicyContract.model_validate(data)


def write_policy_conformance_report(
    report: Mapping[str, Any], out_dir: str | Path
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "policy_conformance.json"
    html_path = out_dir / "policy_conformance.html"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(item.get('check_id', '')))}</td>"
        f"<td>{html.escape(str(item.get('phase', '')))}</td>"
        f"<td>{html.escape(str(item.get('status', '')))}</td>"
        f"<td>{html.escape(str(item.get('detail', '')))}</td>"
        "</tr>"
        for item in report.get("checks", [])
    )
    html_path.write_text(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<title>NyssaBench policy conformance</title></head><body>"
        "<h1>Policy conformance</h1>"
        f"<p>Status: <strong>{html.escape(str(report.get('status', '')))}</strong></p>"
        "<p>This report covers adapter and contract conformance, not a validated policy track.</p>"
        "<table><thead><tr><th>Check</th><th>Phase</th><th>Status</th><th>Detail</th>"
        f"</tr></thead><tbody>{rows}</tbody></table>"
        f"<pre>{html.escape(json.dumps(report, indent=2))}</pre></body></html>\n",
        encoding="utf-8",
    )
    return {"json": json_path, "html": html_path}


def write_policy_example(kind: str, out_dir: str | Path) -> dict[str, Path]:
    names = {
        "state": (
            "state_policy.py",
            "state_policy_contract.json",
            "state_policy.json",
        ),
        "image-chunk": (
            "image_chunk_policy.py",
            "image_chunk_policy_contract.json",
            "image_chunk_policy.json",
        ),
    }
    if kind not in names:
        raise ValueError(f"unknown policy example kind: {kind}")
    root = policy_example_root()
    out_dir = Path(out_dir)
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    policy_name, contract_name, checkpoint_name = names[kind]
    outputs = {
        "policy": out_dir / policy_name,
        "contract": out_dir / contract_name,
        "checkpoint": checkpoint_dir / checkpoint_name,
    }
    sources = {
        "policy": root / policy_name,
        "contract": root / contract_name,
        "checkpoint": root / "checkpoints" / checkpoint_name,
    }
    for key, source in sources.items():
        if not source.is_file():
            raise FileNotFoundError(f"packaged policy example is missing: {source}")
        shutil.copyfile(source, outputs[key])
    return outputs
