from __future__ import annotations

import csv
import html
import json
from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROBUSTNESS_SWEEP_FORMAT = "nyssa-robustness-sweep-v1"


def load_robustness_sweep(
    run_dirs: list[str | Path],
    *,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    if len(run_dirs) < 2:
        raise ValueError("A robustness sweep requires at least two run directories")
    contract: dict[str, Any] | None = None
    stressor_id: str | None = None
    episodes_by_severity: dict[float, list[dict[str, Any]]] = {}
    parameters_by_severity: dict[float, list[dict[str, Any]]] = {}
    source_runs: dict[float, list[str]] = {}

    for run_dir_value in run_dirs:
        run_dir = Path(run_dir_value)
        run = yaml.safe_load((run_dir / "run.yaml").read_text(encoding="utf-8")) or {}
        config = run.get("stressor_config")
        if not isinstance(config, dict):
            raise ValueError(f"Run does not declare a stressor_config: {run_dir}")
        specs = config.get("stressors", [])
        if (
            not isinstance(specs, list)
            or len(specs) != 1
            or not isinstance(specs[0], dict)
        ):
            raise ValueError(
                f"Robustness runs must configure exactly one focal stressor: {run_dir}"
            )
        spec = specs[0]
        current_id = str(spec.get("stressor_id", ""))
        if stressor_id is None:
            stressor_id = current_id
        elif current_id != stressor_id:
            raise ValueError(
                f"Robustness sweep mixes stressors '{stressor_id}' and '{current_id}'"
            )
        severity = float(spec.get("severity", -1.0))
        if not 0.0 <= severity <= 1.0:
            raise ValueError(f"Invalid stressor severity in {run_dir}: {severity}")
        current_contract = {
            "suite_id": run.get("suite_id"),
            "task_ids": run.get("task_ids"),
            "policy_name": run.get("policy_name"),
            "engine_name": run.get("engine_name"),
            "episodes_per_task": run.get("episodes_per_task"),
        }
        if contract is None:
            contract = current_contract
        elif current_contract != contract:
            raise ValueError(f"Robustness sweep run contract mismatch: {run_dir}")
        manifest = json.loads(
            (run_dir / "stressor_manifest.json").read_text(encoding="utf-8")
        )
        unsupported = manifest.get("summary", {}).get("unsupported_stressors", [])
        if unsupported:
            raise ValueError(
                f"Robustness sweep includes unsupported stressors in {run_dir}: {unsupported}"
            )
        episodes = json.loads((run_dir / "episodes.json").read_text(encoding="utf-8"))
        if not isinstance(episodes, list):
            raise ValueError(f"episodes.json must contain a list: {run_dir}")
        episodes_by_severity.setdefault(severity, []).extend(
            dict(item) for item in episodes
        )
        parameters_by_severity.setdefault(severity, []).extend(
            _applied_parameters(item) for item in episodes
        )
        source_runs.setdefault(severity, []).append(run_dir.as_posix())

    assert stressor_id is not None
    summary = robustness_sweep_metrics(
        stressor_id=stressor_id,
        episodes_by_severity=episodes_by_severity,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    summary["contract"] = contract
    summary["source_runs"] = {
        str(key): value for key, value in sorted(source_runs.items())
    }
    for point in summary["points"]:
        severity = float(point["severity"])
        point["applied_parameters"] = _unique_mappings(parameters_by_severity[severity])
    return summary


def robustness_sweep_metrics(
    *,
    stressor_id: str,
    episodes_by_severity: dict[float, list[Any]],
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    severities = sorted(float(severity) for severity in episodes_by_severity)
    if len(severities) < 2:
        raise ValueError("A robustness sweep requires at least two distinct severities")
    if severities[0] != 0.0:
        raise ValueError(
            "A robustness sweep requires severity 0.0 as the clean baseline"
        )
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")

    outcomes: dict[float, dict[tuple[str, int, int], bool]] = {}
    for severity in severities:
        keyed: dict[tuple[str, int, int], bool] = {}
        for episode in episodes_by_severity[severity]:
            key = _episode_key(episode)
            if key in keyed:
                raise ValueError(
                    f"Duplicate episode identity at severity {severity}: {key}"
                )
            keyed[key] = _episode_success(episode)
        if not keyed:
            raise ValueError(f"Severity {severity} has no episodes")
        outcomes[severity] = keyed

    baseline_keys = set(outcomes[0.0])
    for severity in severities[1:]:
        if set(outcomes[severity]) != baseline_keys:
            raise ValueError(
                f"Severity {severity} does not have complete matched episode coverage"
            )
    ordered_keys = sorted(baseline_keys)
    rates = {
        severity: sum(outcomes[severity][key] for key in ordered_keys)
        / len(ordered_keys)
        for severity in severities
    }
    clean_rate = rates[0.0]
    auc = _normalized_auc(severities, [rates[severity] for severity in severities])
    bootstrap_aucs = _bootstrap_auc(
        severities,
        outcomes,
        ordered_keys,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    points = []
    for severity in severities:
        successes = sum(outcomes[severity].values())
        total = len(outcomes[severity])
        points.append(
            {
                "severity": severity,
                "episodes": total,
                "success_count": successes,
                "success_rate": rates[severity],
                "success_rate_ci95": _wilson_ci(successes, total),
                "degradation_from_clean": clean_rate - rates[severity],
            }
        )
    return {
        "format": ROBUSTNESS_SWEEP_FORMAT,
        "stressor_id": stressor_id,
        "paired_episode_coverage": len(ordered_keys),
        "severity_domain": [severities[0], severities[-1]],
        "clean_success_rate": clean_rate,
        "max_severity_success_rate": rates[severities[-1]],
        "max_severity_degradation": clean_rate - rates[severities[-1]],
        "robustness_auc": auc,
        "robustness_auc_ci95": [
            float(np.percentile(bootstrap_aucs, 2.5)),
            float(np.percentile(bootstrap_aucs, 97.5)),
        ],
        "auc_convention": "trapezoidal_success_rate_integral_normalized_by_observed_severity_span",
        "uncertainty": {
            "point_intervals": "Wilson 95% binomial confidence intervals",
            "auc_interval": "paired episode bootstrap percentile interval",
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
        },
        "points": points,
    }


def save_robustness_report(
    summary: dict[str, Any], out_dir: str | Path
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "robustness.json"
    csv_path = out_dir / "robustness.csv"
    html_path = out_dir / "robustness.html"
    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "severity",
            "episodes",
            "success_count",
            "success_rate",
            "success_rate_ci95_low",
            "success_rate_ci95_high",
            "degradation_from_clean",
            "robustness_auc",
            "robustness_auc_ci95_low",
            "robustness_auc_ci95_high",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for point in summary["points"]:
            writer.writerow(
                {
                    **{key: point[key] for key in fieldnames[:4]},
                    "success_rate_ci95_low": point["success_rate_ci95"][0],
                    "success_rate_ci95_high": point["success_rate_ci95"][1],
                    "degradation_from_clean": point["degradation_from_clean"],
                    "robustness_auc": summary["robustness_auc"],
                    "robustness_auc_ci95_low": summary["robustness_auc_ci95"][0],
                    "robustness_auc_ci95_high": summary["robustness_auc_ci95"][1],
                }
            )
    rows = "".join(
        "<tr>"
        f"<td>{point['severity']:.3f}</td>"
        f"<td>{point['success_count']}/{point['episodes']}</td>"
        f"<td>{point['success_rate']:.1%}</td>"
        f"<td>{point['success_rate_ci95'][0]:.1%}-{point['success_rate_ci95'][1]:.1%}</td>"
        f"<td>{point['degradation_from_clean']:.1%}</td>"
        "</tr>"
        for point in summary["points"]
    )
    body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>NyssaBench Robustness Sweep</title>
  <style>
    body {{ font-family: Inter, Arial, sans-serif; margin: 40px; color: #17202a; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #d8dee4; padding: 8px; text-align: left; }}
  </style>
</head>
<body>
  <h1>Robustness Sweep: {html.escape(str(summary["stressor_id"]))}</h1>
  <p>Clean performance: {summary["clean_success_rate"]:.1%}</p>
  <p>Maximum-severity performance: {summary["max_severity_success_rate"]:.1%}</p>
  <p>Maximum-severity degradation: {summary["max_severity_degradation"]:.1%}</p>
  <p>Robustness AUC: {summary["robustness_auc"]:.4f} (95% CI {summary["robustness_auc_ci95"][0]:.4f}-{summary["robustness_auc_ci95"][1]:.4f})</p>
  <table>
    <thead><tr><th>Severity</th><th>Successes</th><th>Success rate</th><th>95% CI</th><th>Degradation</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</body>
</html>
"""
    html_path.write_text(body, encoding="utf-8")
    return {
        "robustness_json": json_path,
        "robustness_csv": csv_path,
        "robustness_report": html_path,
    }


def _episode_key(episode: Any) -> tuple[str, int, int]:
    if isinstance(episode, dict):
        return (
            str(episode["task_id"]),
            int(episode["seed"]),
            int(episode["episode_index"]),
        )
    return (str(episode.task_id), int(episode.seed), int(episode.episode_index))


def _episode_success(episode: Any) -> bool:
    return bool(episode["success"] if isinstance(episode, dict) else episode.success)


def _normalized_auc(severities: list[float], rates: list[float]) -> float:
    span = severities[-1] - severities[0]
    if span <= 0.0:
        raise ValueError("Robustness AUC requires a positive severity span")
    area = sum(
        (severities[index] - severities[index - 1])
        * (rates[index] + rates[index - 1])
        / 2.0
        for index in range(1, len(severities))
    )
    return float(area / span)


def _bootstrap_auc(
    severities: list[float],
    outcomes: dict[float, dict[tuple[str, int, int], bool]],
    ordered_keys: list[tuple[str, int, int]],
    *,
    samples: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = np.empty(samples, dtype=float)
    for sample_index in range(samples):
        indices = rng.integers(0, len(ordered_keys), size=len(ordered_keys))
        sampled_keys = [ordered_keys[index] for index in indices]
        rates = [
            sum(outcomes[severity][key] for key in sampled_keys) / len(sampled_keys)
            for severity in severities
        ]
        result[sample_index] = _normalized_auc(severities, rates)
    return result


def _wilson_ci(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    proportion = successes / total
    denominator = 1.0 + z**2 / total
    center = (proportion + z**2 / (2.0 * total)) / denominator
    margin = (
        z
        * sqrt((proportion * (1.0 - proportion) + z**2 / (4.0 * total)) / total)
        / denominator
    )
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _applied_parameters(episode: dict[str, Any]) -> dict[str, Any]:
    context = episode.get("stressor_context", {})
    applications = context.get("applications", []) if isinstance(context, dict) else []
    if not applications or not isinstance(applications[0], dict):
        return {}
    return dict(applications[0].get("applied_parameters", {}))


def _unique_mappings(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique = {
        json.dumps(value, sort_keys=True, separators=(",", ":")): value
        for value in values
    }
    return [unique[key] for key in sorted(unique)]
