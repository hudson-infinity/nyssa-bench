from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from .protocol import (
    REAL_EVIDENCE_LEDGER_FORMAT,
    RealEvidencePackage,
)
from .validation import RealEvidenceValidationReport, comparison_pairs


def sanitized_evidence_manifest(
    package: RealEvidencePackage, report: RealEvidenceValidationReport
) -> dict[str, Any]:
    payload = package.model_dump(mode="json")
    payload["real_episode"]["identity"].pop("operator_id", None)
    payload["real_episode"]["identity"]["operator_id_included"] = False
    payload["real_episode"]["failure_events"] = [
        _event_summary(item) for item in package.real_episode.failure_events
    ]
    for index, variant in enumerate(package.reconstructed_variants):
        payload["reconstructed_variants"][index]["failure_events"] = [
            _event_summary(item) for item in variant.failure_events
        ]
    for artifact in payload["artifacts"]:
        artifact.pop("path", None)
        artifact.pop("external_locator", None)
    validation = report.to_dict()
    validation.pop("real_ledger", None)
    validation.pop("variant_ledgers", None)
    payload["validation"] = validation
    payload["comparison_pairs"] = comparison_pairs(package)
    return payload


def _event_summary(event: dict[str, Any]) -> dict[str, Any]:
    provenance = event.get("provenance", {})
    return {
        "format": event.get("format"),
        "event_id": event.get("event_id"),
        "role": event.get("role"),
        "category": event.get("category"),
        "subtype": event.get("subtype"),
        "onset_step": event.get("onset_step"),
        "end_step": event.get("end_step"),
        "confidence": event.get("confidence"),
        "provenance": provenance,
        "evidence_payloads_included": False,
    }


def write_real_evidence_artifacts(
    package: RealEvidencePackage,
    report: RealEvidenceValidationReport,
    out_dir: str | Path,
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "real_evidence_manifest.json"
    ledgers_path = out_dir / "real_evidence_ledgers.json"
    pairs_path = out_dir / "real_sim_pairs.json"
    report_path = out_dir / "real_evidence_report.html"
    manifest_path.write_text(
        json.dumps(
            sanitized_evidence_manifest(package, report), indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    ledgers_path.write_text(
        json.dumps(
            {
                "format": REAL_EVIDENCE_LEDGER_FORMAT,
                "real": report.real_ledger.to_dict()
                if report.real_ledger is not None
                else None,
                "reconstructed": {
                    key: value.to_dict()
                    for key, value in sorted(report.variant_ledgers.items())
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    pairs_path.write_text(
        json.dumps(
            {
                "format": "nyssa-real-sim-pairs-v1",
                "pairs": comparison_pairs(package),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    report_path.write_text(_html_report(package, report), encoding="utf-8")
    return {
        "manifest": manifest_path,
        "ledgers": ledgers_path,
        "pairs": pairs_path,
        "report": report_path,
    }


def _html_report(
    package: RealEvidencePackage, report: RealEvidenceValidationReport
) -> str:
    issue_rows = (
        "".join(
            "<tr>"
            f"<td>{html.escape(item.severity)}</td>"
            f"<td>{html.escape(item.code)}</td>"
            f"<td>{html.escape(item.path)}</td>"
            f"<td>{html.escape(item.message)}</td>"
            "</tr>"
            for item in report.issues
        )
        or '<tr><td colspan="4">No validation issues</td></tr>'
    )
    calibration_rows = "".join(
        "<tr>"
        f"<td>{html.escape(item.calibration_id)}</td>"
        f"<td>{html.escape(item.calibration_type)}</td>"
        f"<td>{html.escape(item.status)}</td>"
        f"<td><code>{html.escape(json.dumps(item.uncertainty, sort_keys=True))}</code></td>"
        f"<td><code>{html.escape(json.dumps(item.fit_quality, sort_keys=True))}</code></td>"
        "</tr>"
        for item in package.calibrations
    )
    mismatch_rows = "".join(
        "<tr>"
        f"<td>{html.escape(variant.variant_id)}</td>"
        f"<td>{html.escape(item.category)}</td>"
        f"<td>{html.escape(item.description)}</td>"
        f"<td>{item.magnitude:g} {html.escape(item.unit)}</td>"
        f"<td>{item.confidence:.3f}</td>"
        "</tr>"
        for variant in package.reconstructed_variants
        for item in variant.mismatches
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>NyssaBench Real Evidence</title>
<style>body{{font-family:Arial,sans-serif;margin:40px;color:#17202a}}table{{border-collapse:collapse;width:100%;margin:16px 0}}th,td{{border-bottom:1px solid #d8dee4;padding:8px;text-align:left;vertical-align:top}}code{{overflow-wrap:anywhere}}</style>
</head><body>
<h1>Real and Reconstructed Evidence</h1>
<p><strong>Package:</strong> <code>{html.escape(package.identity)}</code><br>
<strong>Real episode:</strong> {html.escape(package.real_episode.identity.episode_id)}<br>
<strong>Variants:</strong> {len(package.reconstructed_variants)}</p>
<h2>Readiness</h2>
<table><tbody>
<tr><td>Valid</td><td>{report.valid}</td></tr>
<tr><td>Evidence ready</td><td>{report.evidence_ready}</td></tr>
<tr><td>Calibration ready</td><td>{report.calibration_ready}</td></tr>
<tr><td>Governance ready</td><td>{report.governance_ready}</td></tr>
<tr><td>Comparison ready</td><td>{report.comparison_ready}</td></tr>
<tr><td>Claim ready</td><td>{report.claim_ready}</td></tr>
</tbody></table>
<h2>Calibration and uncertainty</h2>
<table><thead><tr><th>ID</th><th>Type</th><th>Status</th><th>Uncertainty</th><th>Fit quality</th></tr></thead><tbody>{calibration_rows}</tbody></table>
<h2>Real/sim mismatches</h2>
<table><thead><tr><th>Variant</th><th>Category</th><th>Description</th><th>Magnitude</th><th>Confidence</th></tr></thead><tbody>{mismatch_rows}</tbody></table>
<h2>Validation issues</h2>
<table><thead><tr><th>Severity</th><th>Code</th><th>Path</th><th>Message</th></tr></thead><tbody>{issue_rows}</tbody></table>
</body></html>"""
