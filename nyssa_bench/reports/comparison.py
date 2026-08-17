from __future__ import annotations

import hashlib
import html
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from nyssa_bench.metrics.sim_to_real import score_summary


COMPARISON_CONTRACT_FORMAT = "nyssa-comparison-contract-v1"
COMPARISON_SET_FORMAT = "nyssa-comparison-set-v1"
LEADERBOARD_FORMAT = "nyssa-leaderboard-v2"
REQUIRED_SEED_PROTOCOL_FIELDS = (
    "format",
    "episode_seed_stride",
    "formula",
    "shared_across_tasks",
)


class ComparisonMetadataError(ValueError):
    """Raised when a run cannot provide a complete comparison contract."""

    def __init__(self, run_dir: str | Path, missing_fields: list[str]):
        self.run_dir = Path(run_dir)
        self.missing_fields = missing_fields
        fields = ", ".join(missing_fields)
        super().__init__(
            f"Run {self.run_dir.as_posix()} has missing or inconsistent comparison metadata: {fields}"
        )


class IncompatibleRunsError(ValueError):
    """Raised when strict comparison is requested for incompatible runs."""

    def __init__(self, mismatches: list[dict[str, Any]]):
        self.mismatches = mismatches
        details = "; ".join(
            f"{item['field']}: {item['baseline_run']}={_display_value(item['baseline_value'])}, "
            f"{item['run']}={_display_value(item['value'])}"
            for item in mismatches
        )
        super().__init__(
            "Runs are not comparison-compatible. "
            f"{details}. Use --allow-incompatible only for explicitly exploratory output."
        )


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
    score = float(
        summary.get(
            "prototype_reliability_score",
            summary.get("sim_to_real_score", score_summary(summary)),
        )
    )
    summary["prototype_reliability_score"] = score
    return summary


def load_comparison_contract(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    run_metadata = _load_yaml_mapping(run_dir / "run.yaml")
    config = _load_yaml_mapping(run_dir / "config.yaml")
    manifest = _load_json_mapping(run_dir / "dataset_manifest.json")

    suite_config = config.get("suite") if isinstance(config.get("suite"), dict) else {}
    manifest_suite = (
        manifest.get("suite") if isinstance(manifest.get("suite"), dict) else {}
    )
    manifest_run = manifest.get("run") if isinstance(manifest.get("run"), dict) else {}
    metadata_issues: list[str] = []
    suite_id = _select_consistent_value(
        "suite_id",
        [
            ("run.yaml", run_metadata.get("suite_id")),
            ("config.yaml", suite_config.get("suite_id")),
            ("dataset_manifest.json:suite", manifest_suite.get("suite_id")),
            ("dataset_manifest.json:run", manifest_run.get("suite_id")),
        ],
        metadata_issues,
    )
    engine_name = _select_consistent_value(
        "engine_name",
        [
            ("run.yaml", run_metadata.get("engine_name")),
            ("config.yaml", config.get("engine")),
            ("dataset_manifest.json:run", manifest_run.get("engine_name")),
        ],
        metadata_issues,
    )
    episodes_per_task = _select_consistent_value(
        "episodes_per_task",
        [
            ("run.yaml", run_metadata.get("episodes_per_task")),
            ("config.yaml", config.get("episodes_per_task")),
            ("dataset_manifest.json:run", manifest_run.get("episodes_per_task")),
        ],
        metadata_issues,
    )
    seed_protocol = _select_consistent_value(
        "seed_protocol",
        [
            ("run.yaml", run_metadata.get("seed_protocol")),
            ("config.yaml", config.get("seed_protocol")),
            ("dataset_manifest.json:run", manifest_run.get("seed_protocol")),
        ],
        metadata_issues,
    )
    task_ids = _select_consistent_value(
        "task_ids",
        [
            ("run.yaml", run_metadata.get("task_ids")),
            ("config.yaml", suite_config.get("tasks")),
            ("dataset_manifest.json:suite", manifest_suite.get("tasks")),
            ("dataset_manifest.json:run", manifest_run.get("task_ids")),
        ],
        metadata_issues,
        normalize=_normalize_task_ids,
    )
    stressor_sources = [
        ("run.yaml", run_metadata.get("stressor_config")),
        ("config.yaml", config.get("stressor_config")),
        ("dataset_manifest.json:run", manifest_run.get("stressor_config")),
    ]
    stressor_config = (
        _select_consistent_value(
            "stressor_config",
            stressor_sources,
            metadata_issues,
            normalize=_normalize_stressor_config,
        )
        if any(value is not None for _, value in stressor_sources)
        else None
    )
    normalized_task_ids = _normalize_task_ids(task_ids)
    manifest_tasks = manifest.get("tasks")
    task_definitions = (
        {
            str(task.get("task_id")): task
            for task in manifest_tasks
            if isinstance(task, dict) and task.get("task_id") is not None
        }
        if isinstance(manifest_tasks, list)
        else {}
    )

    missing_fields = metadata_issues
    if normalized_task_ids and sorted(task_definitions) != normalized_task_ids:
        missing_fields.append(
            "task_ids (dataset_manifest.json:tasks conflicts with declared task set)"
        )
    for field, value in (
        ("suite_id", suite_id),
        ("engine_name", engine_name),
        ("task_ids", normalized_task_ids),
        ("episodes_per_task", episodes_per_task),
        ("seed_protocol", seed_protocol),
    ):
        if value is None:
            missing_fields.append(field)
    if not normalized_task_ids:
        missing_fields.append("task_ids")
    if not isinstance(seed_protocol, dict) or not seed_protocol:
        missing_fields.append("seed_protocol")
    else:
        for field in REQUIRED_SEED_PROTOCOL_FIELDS:
            if field not in seed_protocol:
                missing_fields.append(f"seed_protocol.{field}")

    tasks: dict[str, Any] = {}
    for task_id in normalized_task_ids or []:
        task = task_definitions.get(task_id)
        if task is None:
            missing_fields.append(f"tasks.{task_id}")
            continue
        if "success" not in task:
            missing_fields.append(f"tasks.{task_id}.success")
        if "randomization" not in task:
            missing_fields.append(f"tasks.{task_id}.randomization")
        if "ood_splits" not in task:
            missing_fields.append(f"tasks.{task_id}.ood_splits")
        tasks[task_id] = {
            "success": task.get("success"),
            "stressors": {
                "randomization": task.get("randomization"),
                "ood_splits": task.get("ood_splits"),
            },
        }

    if missing_fields:
        raise ComparisonMetadataError(run_dir, sorted(set(missing_fields)))

    normalized_seed_protocol = {
        str(key): value
        for key, value in seed_protocol.items()
        if key not in {"run_seed", "seed"}
    }
    if not normalized_seed_protocol:
        raise ComparisonMetadataError(run_dir, ["seed_protocol"])

    return {
        "format": COMPARISON_CONTRACT_FORMAT,
        "suite_id": str(suite_id),
        "engine_name": str(engine_name),
        "task_ids": normalized_task_ids,
        "tasks": tasks,
        "episodes_per_task": int(episodes_per_task),
        "seed_protocol": normalized_seed_protocol,
        "stressor_config": _normalize_stressor_config(stressor_config),
    }


def comparison_contract_hash(contract: dict[str, Any]) -> str:
    encoded = json.dumps(
        contract, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compare_runs(
    run_dirs: list[str | Path], *, allow_incompatible: bool = False
) -> dict[str, Any]:
    if not run_dirs:
        raise ValueError("At least one run directory is required")

    runs = [load_run_summary(path) for path in run_dirs]
    contracts = [load_comparison_contract(path) for path in run_dirs]
    contract_hashes = [comparison_contract_hash(contract) for contract in contracts]
    run_names = [str(item.get("run_dir")) for item in runs]
    mismatches = _comparison_mismatches(contracts, run_names)
    if mismatches and not allow_incompatible:
        raise IncompatibleRunsError(mismatches)

    comparable = not mismatches
    comparison_mode = "strict" if comparable else "exploratory"
    comparison_contract = {
        "format": COMPARISON_SET_FORMAT,
        "contract_format": COMPARISON_CONTRACT_FORMAT,
        "comparable": comparable,
        "mode": comparison_mode,
        "shared_contract": contracts[0] if comparable else None,
        "run_contract_sha256": sorted(set(contract_hashes)),
        "mismatched_fields": sorted({item["field"] for item in mismatches}),
    }
    comparison_sha256 = comparison_contract_hash(comparison_contract)
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
        "comparable": comparable,
        "comparison_mode": comparison_mode,
        "comparison_contract": comparison_contract,
        "comparison_contract_sha256": comparison_sha256,
        "run_contracts": [
            {"run_dir": run_name, "contract": contract, "sha256": contract_hash}
            for run_name, contract, contract_hash in zip(
                run_names, contracts, contract_hashes
            )
        ],
        "mismatches": mismatches,
        "ranking": [
            {
                "rank": index + 1,
                "run_dir": item.get("run_dir"),
                "success_rate": item.get("success_rate", 0.0),
                "success_rate_ci95": item.get("success_rate_ci95", [0.0, 0.0]),
                "prototype_reliability_score": item.get(
                    "prototype_reliability_score", 0.0
                ),
                "benchmark_tier": item.get("benchmark_tier", "unknown"),
                "public_claim": item.get("public_claim", False),
                "public_claim_status": (item.get("public_claim_validation") or {}).get(
                    "status", "unknown"
                ),
                "primary_failure_mode": item.get("primary_failure_mode"),
                "expert_intervention_rate": (item.get("metrics") or {}).get(
                    "expert_intervention_rate", 0.0
                ),
                "recovery_success_rate": (item.get("metrics") or {}).get(
                    "recovery_success_rate", 0.0
                ),
                "recovery_success_count": (item.get("metrics") or {}).get(
                    "recovery_success_count", 0.0
                ),
                "recovery_applied_count": (item.get("metrics") or {}).get(
                    "recovery_applied_count", 0.0
                ),
                "recovery_episode_success_rate": (item.get("metrics") or {}).get(
                    "recovery_episode_success_rate", 0.0
                ),
                "verifier_rejection_rate": (item.get("metrics") or {}).get(
                    "verifier_rejection_rate", 0.0
                ),
                "wall_time_seconds": (item.get("compute") or {}).get(
                    "wall_time_seconds", 0.0
                ),
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
    payload = {
        "format": LEADERBOARD_FORMAT,
        "comparable": comparison["comparable"],
        "comparison_mode": comparison["comparison_mode"],
        "comparison_contract": comparison["comparison_contract"],
        "comparison_contract_sha256": comparison["comparison_contract_sha256"],
        "mismatches": comparison["mismatches"],
        "ranking": comparison["ranking"],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
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
    comparable = bool(comparison.get("comparable"))
    status_class = "comparable" if comparable else "non-comparable"
    status_title = (
        "Comparable strict comparison"
        if comparable
        else "NON-COMPARABLE EXPLORATORY OUTPUT"
    )
    status_detail = (
        "All runs satisfy the same comparison contract."
        if comparable
        else "Runs differ on comparison-critical fields. This ordering must not be used as a benchmark ranking."
    )
    mismatch_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(item['field']))}</td>"
        f"<td>{html.escape(str(item['baseline_run']))}</td>"
        f"<td><code>{html.escape(_display_value(item['baseline_value']))}</code></td>"
        f"<td>{html.escape(str(item['run']))}</td>"
        f"<td><code>{html.escape(_display_value(item['value']))}</code></td>"
        "</tr>"
        for item in comparison.get("mismatches", [])
    )
    mismatch_table = ""
    if mismatch_rows:
        mismatch_table = (
            "<h2>Compatibility mismatches</h2>"
            "<table><thead><tr><th>Field</th><th>Baseline run</th><th>Baseline value</th>"
            "<th>Run</th><th>Value</th></tr></thead>"
            f"<tbody>{mismatch_rows}</tbody></table>"
        )
    contract_hash = html.escape(
        str(comparison.get("comparison_contract_sha256", "unknown"))
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>NyssaBench Policy Comparison</title>
  <style>
    body {{ font-family: Inter, Arial, sans-serif; margin: 40px; color: #17202a; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #d8dee4; padding: 8px; text-align: left; vertical-align: top; }}
    .status {{ border-left: 5px solid #238636; background: #f0fff4; padding: 12px 16px; margin-bottom: 24px; }}
    .status.non-comparable {{ border-color: #cf222e; background: #fff1f0; }}
    .status h2 {{ margin: 0 0 6px; font-size: 1.1rem; }}
    code {{ overflow-wrap: anywhere; }}
  </style>
</head>
<body>
  <h1>Policy Comparison</h1>
  <section class="status {status_class}">
    <h2>{status_title}</h2>
    <p>{status_detail}</p>
    <p>Comparison contract SHA-256: <code>{contract_hash}</code></p>
  </section>
  {mismatch_table}
  <h2>{"Ranking" if comparable else "Exploratory ordering"}</h2>
  <table>
    <thead>
        <tr><th>Rank</th><th>Run</th><th>Success</th><th>95% CI</th><th>Prototype reliability</th><th>Tier</th><th>Claim status</th><th>Expert intervention</th><th>Recovery success (successful/applied)</th><th>Verifier rejection</th><th>Wall time</th><th>Primary failure</th></tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
</body>
</html>
"""


def _comparison_mismatches(
    contracts: list[dict[str, Any]], run_names: list[str]
) -> list[dict[str, Any]]:
    baseline = contracts[0]
    mismatches: list[dict[str, Any]] = []
    for contract, run_name in zip(contracts[1:], run_names[1:]):
        for field, baseline_value, value in _diff_values(baseline, contract):
            mismatches.append(
                {
                    "field": field,
                    "baseline_run": run_names[0],
                    "baseline_value": baseline_value,
                    "run": run_name,
                    "value": value,
                }
            )
    return mismatches


def _diff_values(left: Any, right: Any, path: str = "") -> list[tuple[str, Any, Any]]:
    if isinstance(left, dict) and isinstance(right, dict):
        differences: list[tuple[str, Any, Any]] = []
        for key in sorted(set(left) | set(right)):
            child_path = f"{path}.{key}" if path else str(key)
            if key not in left:
                differences.append((child_path, "<missing>", right[key]))
            elif key not in right:
                differences.append((child_path, left[key], "<missing>"))
            else:
                differences.extend(_diff_values(left[key], right[key], child_path))
        return differences
    if left != right:
        return [(path, left, right)]
    return []


def _select_consistent_value(
    field: str,
    candidates: list[tuple[str, Any]],
    issues: list[str],
    *,
    normalize: Callable[[Any], Any] | None = None,
) -> Any:
    present = [(source, value) for source, value in candidates if value is not None]
    if not present:
        issues.append(field)
        return None

    normalize = normalize or (lambda value: value)
    baseline = normalize(present[0][1])
    conflicting_sources = [
        source for source, value in present[1:] if normalize(value) != baseline
    ]
    if conflicting_sources:
        sources = ", ".join([present[0][0], *conflicting_sources])
        issues.append(f"{field} (conflict across {sources})")
    return present[0][1]


def _normalize_task_ids(value: Any) -> list[str] | None:
    if not isinstance(value, list):
        return None
    return sorted(str(task_id) for task_id in value)


def _normalize_stressor_config(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {
            "format": "nyssa-stressor-config-v1",
            "unsupported_policy": "error",
            "stressors": [],
        }
    return {
        "format": value.get("format", "nyssa-stressor-config-v1"),
        "unsupported_policy": value.get("unsupported_policy", "error"),
        "stressors": value.get("stressors", []),
    }


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return value if isinstance(value, dict) else {}


def _load_json_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _display_value(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True)


def _format_ci_percent(value: Any) -> str:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return "n/a"
    return f"{float(value[0]) * 100:.1f}-{float(value[1]) * 100:.1f}%"
