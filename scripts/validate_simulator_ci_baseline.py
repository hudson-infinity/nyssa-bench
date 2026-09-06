from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "claims" / "mujoco_ci_baseline.json"
REQUIRED_CONTRACT = {
    "smoke_format": "nyssa-simulator-ci-smoke-v1",
    "task_id": "mujoco_inverted_pendulum",
    "python_version": "3.11.16",
    "mujoco_version": "3.12.0",
    "gymnasium_version": "1.3.0",
    "MUJOCO_GL": "osmesa",
    "PYOPENGL_PLATFORM": "osmesa",
    "state_restore_fidelity": "exact_mujoco_integration_state_and_rng",
    "stressor_id": "action_gaussian_noise",
    "stressor_condition_id": "installed-simulator-smoke",
    "episodes": 2,
    "replay_requested": False,
}


def validate_simulator_ci_baseline(path: str | Path = BASELINE) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid simulator CI baseline: {source}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("simulator CI baseline must contain a mapping")
    if payload.get("format") != "nyssa-simulator-ci-baseline-v1":
        raise ValueError("unsupported simulator CI baseline format")
    if payload.get("engine") != "mujoco" or payload.get("status") != "baseline_passed":
        raise ValueError("baseline engine or status is invalid")
    minimum_runs = _positive_int(payload.get("minimum_runs"), "minimum_runs")
    minimum_rate = _rate(payload.get("minimum_pass_rate"), "minimum_pass_rate")
    runs = payload.get("runs")
    if not isinstance(runs, list) or len(runs) < minimum_runs:
        raise ValueError("baseline has fewer runs than its policy requires")
    run_ids = []
    passes = 0
    for run in runs:
        if not isinstance(run, Mapping):
            raise ValueError("baseline run must be a mapping")
        run_id = _positive_int(run.get("run_id"), "run_id")
        if not re.fullmatch(r"[0-9a-f]{40}", str(run.get("head_sha", ""))):
            raise ValueError(f"run {run_id} has an invalid commit SHA")
        if not re.fullmatch(
            r"[0-9a-f]{64}", str(run.get("simulator_smoke_sha256", ""))
        ):
            raise ValueError(f"run {run_id} has an invalid smoke artifact hash")
        conclusion = run.get("conclusion")
        if conclusion not in {"success", "failure", "cancelled"}:
            raise ValueError(f"run {run_id} has an unsupported conclusion")
        passes += conclusion == "success"
        run_ids.append(run_id)
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("baseline run IDs must be unique")
    observed_rate = passes / len(runs)
    if payload.get("observed_runs") != len(runs):
        raise ValueError("observed_runs does not match run records")
    if payload.get("observed_passes") != passes:
        raise ValueError("observed_passes does not match run records")
    reported_rate = payload.get("observed_pass_rate")
    if not isinstance(reported_rate, (int, float)) or not math.isclose(
        float(reported_rate), observed_rate
    ):
        raise ValueError("observed_pass_rate does not match run records")
    if observed_rate < minimum_rate:
        raise ValueError("simulator baseline is below the required pass rate")
    contract = payload.get("common_contract")
    if not isinstance(contract, Mapping) or any(
        contract.get(key) != value for key, value in REQUIRED_CONTRACT.items()
    ):
        raise ValueError("simulator baseline common contract has drifted")
    if not isinstance(contract.get("platform"), str) or not contract["platform"]:
        raise ValueError("simulator baseline platform is missing")
    if not isinstance(payload.get("claim_boundary"), str) or not payload[
        "claim_boundary"
    ]:
        raise ValueError("simulator baseline requires a claim boundary")
    return {
        "format": "nyssa-simulator-ci-baseline-validation-v1",
        "status": "validated",
        "engine": "mujoco",
        "run_count": len(runs),
        "pass_count": passes,
        "pass_rate": observed_rate,
        "runner_class": payload.get("runner_class"),
    }


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _rate(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a rate")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be between zero and one")
    return result


def main() -> int:
    try:
        report = validate_simulator_ci_baseline()
    except ValueError as exc:
        print(f"simulator CI baseline validation failed: {exc}")
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
