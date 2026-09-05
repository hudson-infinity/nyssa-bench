from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from nyssa_bench.core.registry import ENGINE_SUPPORT_TIER, make_engine
from nyssa_bench.core.suite import Suite
from nyssa_bench.nep import (
    NEPManifest,
    PolicyContract,
    result_pack_pipeline_manifest,
    write_nep_manifest,
)
from nyssa_bench.policies.base import load_policy_from_path
from nyssa_bench.plugins import get_plugin_registry
from nyssa_bench.runner import PolicyRunner


POLICY_CONFORMANCE_FORMAT = "nyssa-policy-conformance-report-v1"


class PolicyConformanceExecutionError(RuntimeError):
    def __init__(self, phase: str, message: str) -> None:
        self.phase = phase
        super().__init__(message)


def evaluate_policy_conformance(
    *,
    policy_path: str | Path,
    contract: PolicyContract,
    suite: Suite,
    engine_name: str,
    out_dir: str | Path,
    episodes: int = 1,
    capture_replay: bool = False,
) -> dict[str, Any]:
    if len(suite.tasks) != 1:
        raise ValueError("policy conformance requires exactly one selected task")
    if episodes <= 0:
        raise ValueError("policy conformance episodes must be positive")
    policy_path = Path(policy_path).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    task = suite.tasks[0]
    checks: list[dict[str, Any]] = []
    probes: list[dict[str, Any]] = []
    policy = None
    engine = None
    try:
        policy = load_policy_from_path(policy_path)
        metadata = _policy_metadata(policy)
        checks.extend(_metadata_checks(metadata, contract, policy_path))
        checks.extend(_lifecycle_checks(policy, contract))
        mappings = task.success.get("engine_env_ids", {})
        factories = task.success.get("engine_factory", {})
        mapped = bool(
            isinstance(mappings, Mapping)
            and mappings.get(engine_name)
            or isinstance(factories, Mapping)
            and factories.get(engine_name)
            or engine_name in get_plugin_registry().engines
        )
        checks.append(
            _check(
                "engine_task_mapping",
                mapped,
                phase="engine_capability",
                expected=f"explicit mapping for {engine_name}",
                observed={"env_id": mappings.get(engine_name) if isinstance(mappings, Mapping) else None, "factory": factories.get(engine_name) if isinstance(factories, Mapping) else None},
                detail=task.task_id,
            )
        )
        engine = make_engine(engine_name)
        engine.load_task(task)
        for seed in (0, 1):
            first = _probe_once(
                policy,
                engine,
                task,
                contract,
                seed=seed,
                repetition=0,
            )
            second = _probe_once(
                policy,
                engine,
                task,
                contract,
                seed=seed,
                repetition=1,
            )
            probes.extend((first, second))
            checks.extend(first["checks"])
            checks.extend(second["checks"])
            if contract.deterministic_seeding:
                equal = first.get("action_sha256") == second.get("action_sha256")
                checks.append(
                    _check(
                        "matched_reset_determinism",
                        equal,
                        phase="matched_reset",
                        expected=first.get("action_sha256"),
                        observed=second.get("action_sha256"),
                        detail=f"seed={seed}",
                    )
                )
            else:
                checks.append(
                    _not_applicable(
                        "matched_reset_determinism",
                        "policy contract does not claim deterministic seeding",
                        phase="matched_reset",
                    )
                )
    except Exception as exc:
        phase = getattr(exc, "phase", "preflight")
        checks.append(
            _check(
                "preflight_execution",
                False,
                phase=str(phase),
                expected="successful policy/engine preflight",
                observed=f"{type(exc).__name__}: {exc}",
                detail="preflight raised before a full experiment",
            )
        )
    finally:
        for component in (engine, policy):
            close = getattr(component, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:
                    checks.append(
                        _check(
                            "component_close",
                            False,
                            phase="cleanup",
                            expected="clean close",
                            observed=f"{type(exc).__name__}: {exc}",
                            detail=type(component).__name__,
                        )
                    )

    preflight_valid = not any(check["status"] == "failed" for check in checks)
    smoke = None
    if preflight_valid:
        smoke = _run_smoke(
            policy_path=policy_path,
            contract=contract,
            suite=suite,
            engine_name=engine_name,
            out_dir=out_dir / "smoke_run",
            episodes=episodes,
            capture_replay=capture_replay,
        )
        if smoke["status"] != "passed":
            checks.append(
                _check(
                    "integration_smoke",
                    False,
                    phase="smoke_run",
                    expected="completed result pack",
                    observed=smoke.get("error"),
                    detail="small integration run failed",
                )
            )
    conformant = not any(check["status"] == "failed" for check in checks)
    return {
        "format": POLICY_CONFORMANCE_FORMAT,
        "status": "conformant" if conformant else "nonconformant",
        "conformant": conformant,
        "integration_only": True,
        "claim_scope": "adapter_and_contract_conformance_not_validated_policy_track",
        "policy_path": policy_path.as_posix(),
        "policy_file_sha256": _file_sha256(policy_path),
        "policy_contract": contract.model_dump(mode="json"),
        "suite_id": suite.suite_id,
        "task_id": task.task_id,
        "engine": engine_name,
        "engine_support_tier": ENGINE_SUPPORT_TIER.get(
            engine_name, "external_plugin"
        ),
        "checks": checks,
        "check_status_counts": _counts(check["status"] for check in checks),
        "probes": probes,
        "smoke": smoke,
    }


def _probe_once(
    policy: Any,
    engine: Any,
    task: Any,
    contract: PolicyContract,
    *,
    seed: int,
    repetition: int,
) -> dict[str, Any]:
    checks = []
    reset = getattr(policy, "reset", None)
    try:
        if callable(reset):
            reset(task=task, seed=seed)
        observation, _ = engine.reset(seed=seed)
    except Exception as exc:
        raise PolicyConformanceExecutionError(
            "reset", f"seed={seed} repetition={repetition}: {exc}"
        ) from exc
    modalities = _observation_modalities(observation)
    checks.append(
        _check(
            "observation_modalities",
            set(contract.observation_modalities) <= modalities,
            phase="observation",
            expected=list(contract.observation_modalities),
            observed=sorted(modalities),
            detail=f"seed={seed} repetition={repetition}",
        )
    )
    started = time.perf_counter()
    try:
        action = policy.act(observation)
    except Exception as exc:
        raise PolicyConformanceExecutionError(
            "policy_action", f"seed={seed} repetition={repetition}: {exc}"
        ) from exc
    latency_ms = (time.perf_counter() - started) * 1000.0
    action_check, normalized = _validate_action(action, observation, contract)
    checks.append(
        _check(
            "action_contract",
            action_check["valid"],
            phase="policy_action",
            expected=action_check["expected"],
            observed=action_check["observed"],
            detail=action_check["detail"],
        )
    )
    return {
        "seed": seed,
        "repetition": repetition,
        "latency_ms": latency_ms,
        "modalities": sorted(modalities),
        "action_shape": list(normalized.shape) if normalized is not None else None,
        "action_sha256": _array_sha256(normalized)
        if normalized is not None
        else None,
        "checks": checks,
    }


def _validate_action(
    action: Any,
    observation: Mapping[str, Any],
    contract: PolicyContract,
) -> tuple[dict[str, Any], np.ndarray | None]:
    try:
        values = np.asarray(action, dtype=float)
    except (TypeError, ValueError) as exc:
        return {
            "valid": False,
            "expected": "numeric action",
            "observed": type(action).__name__,
            "detail": str(exc),
        }, None
    expected_shape = (
        (contract.action_dimension,)
        if contract.prediction_horizon == 1
        else (contract.prediction_horizon, contract.action_dimension)
    )
    live = observation.get("action_space", {})
    live_shape = tuple(live.get("shape", ())) if isinstance(live, Mapping) else ()
    live_low = np.asarray(live.get("low"), dtype=float) if isinstance(live, Mapping) else np.asarray([])
    live_high = np.asarray(live.get("high"), dtype=float) if isinstance(live, Mapping) else np.asarray([])
    low = np.asarray(contract.action_lower_bounds, dtype=float)
    high = np.asarray(contract.action_upper_bounds, dtype=float)
    shape_valid = values.shape == expected_shape
    live_valid = live_shape == (contract.action_dimension,)
    live_bounds_valid = bool(
        live_low.shape == low.shape
        and live_high.shape == high.shape
        and np.allclose(live_low, low, rtol=0.0, atol=1e-8)
        and np.allclose(live_high, high, rtol=0.0, atol=1e-8)
    )
    finite = bool(np.isfinite(values).all())
    bounded = bool(
        finite
        and live_bounds_valid
        and values.shape[-1:] == (contract.action_dimension,)
        and (values >= live_low).all()
        and (values <= live_high).all()
    )
    return {
        "valid": shape_valid and live_valid and live_bounds_valid and finite and bounded,
        "expected": {
            "predicted_shape": list(expected_shape),
            "live_shape": [contract.action_dimension],
            "bounds": [list(low), list(high)],
        },
        "observed": {
            "predicted_shape": list(values.shape),
            "live_shape": list(live_shape),
            "live_bounds": [list(live_low), list(live_high)],
            "contract_matches_live_bounds": live_bounds_valid,
            "finite": finite,
            "bounded": bounded,
        },
        "detail": "actions are validated before engine.step",
    }, values


def _metadata_checks(
    metadata: Mapping[str, Any], contract: PolicyContract, policy_path: Path
) -> list[dict[str, Any]]:
    expected = {
        "policy_id": contract.policy_id,
        "policy_version": contract.policy_version,
        "policy_family": contract.policy_family,
        "checkpoint_id": contract.checkpoint_id,
        "checkpoint_sha256": contract.checkpoint_sha256,
        "preprocessing_sha256": contract.preprocessing_sha256,
        "observation_modalities": list(contract.observation_modalities),
        "action_representation": contract.action_representation,
        "action_dimension": contract.action_dimension,
    }
    checks = [
        _check(
            f"metadata.{key}",
            metadata.get(key) == value,
            phase="metadata",
            expected=value,
            observed=metadata.get(key),
            detail=policy_path.name,
        )
        for key, value in expected.items()
    ]
    declared_horizon = metadata.get("prediction_horizon")
    checks.append(
        _check(
            "metadata.prediction_horizon",
            declared_horizon == contract.prediction_horizon,
            phase="metadata",
            expected=contract.prediction_horizon,
            observed=declared_horizon,
            detail="action sequence contract",
        )
    )
    checks.append(
        _check(
            "metadata.execution_horizon",
            metadata.get("execution_horizon") == contract.execution_horizon,
            phase="metadata",
            expected=contract.execution_horizon,
            observed=metadata.get("execution_horizon"),
            detail="action sequence execution contract",
        )
    )
    device = metadata.get("device")
    checks.append(
        _check(
            "metadata.device",
            isinstance(device, str) and bool(device.strip()),
            phase="dependency_capability",
            expected="declared execution device",
            observed=device,
            detail="device availability is exercised during policy.act",
        )
    )
    return checks


def _lifecycle_checks(policy: Any, contract: PolicyContract) -> list[dict[str, Any]]:
    checks = []
    for method_name in ("reset", "act", "close"):
        checks.append(
            _check(
                f"lifecycle.{method_name}",
                callable(getattr(policy, method_name, None)),
                phase="lifecycle",
                expected="callable",
                observed=type(getattr(policy, method_name, None)).__name__,
                detail=contract.state_semantics,
            )
        )
    return checks


def _run_smoke(
    *,
    policy_path: Path,
    contract: PolicyContract,
    suite: Suite,
    engine_name: str,
    out_dir: Path,
    episodes: int,
    capture_replay: bool,
) -> dict[str, Any]:
    try:
        runner = PolicyRunner(
            policy=policy_path.as_posix(),
            engine=engine_name,
            episodes=episodes,
            seed=23,
            out=out_dir,
            max_steps=max(1, contract.execution_horizon),
            capture_replay=capture_replay,
            policy_action_horizon=contract.prediction_horizon,
            policy_execution_horizon=contract.execution_horizon,
        )
        report = runner.evaluate(suite)
    except Exception as exc:
        return {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "run_dir": out_dir.as_posix(),
        }
    required = ("metrics.json", "episodes.json", "dataset_manifest.json", "report.html")
    missing = [name for name in required if not (out_dir / name).is_file()]
    nep = None
    if not missing:
        base = result_pack_pipeline_manifest(engine_name, out_dir)
        task_contract = base.task.model_copy(
            update={
                "observation_modalities": tuple(
                    sorted(
                        set(base.task.observation_modalities)
                        | set(contract.observation_modalities)
                    )
                ),
                "action_representation": contract.action_representation,
            }
        )
        nep = NEPManifest.create(
            evaluation_id=f"policy-conformance-{contract.policy_id}",
            task=task_contract,
            stressor=base.stressor,
            policy=contract,
            failure_evidence=base.failure_evidence,
            intervention=base.intervention,
            claim=base.claim,
            artifacts=base.artifacts,
        )
        write_nep_manifest(nep, out_dir / "nep_manifest.json")
    return {
        "status": "passed" if not missing else "failed",
        "error": None if not missing else f"missing artifacts: {', '.join(missing)}",
        "run_dir": out_dir.as_posix(),
        "episodes": len(runner.episode_results),
        "success_rate": report.summary.get("success_rate"),
        "benchmark_tier": report.summary.get("benchmark_tier"),
        "public_claim": report.summary.get("public_claim"),
        "replay_requested": capture_replay,
        "artifacts": [
            *required,
            *(("nep_manifest.json",) if nep else ()),
        ],
        "nep_content_sha256": nep.content_sha256 if nep else None,
    }


def _policy_metadata(policy: Any) -> Mapping[str, Any]:
    metadata = getattr(policy, "metadata", None)
    if not callable(metadata):
        return {}
    try:
        value = metadata()
    except Exception as exc:
        raise PolicyConformanceExecutionError("metadata", str(exc)) from exc
    return value if isinstance(value, Mapping) else {}


def _observation_modalities(observation: Mapping[str, Any]) -> set[str]:
    modalities: set[str] = set()

    def visit(value: Any, key: str = "") -> None:
        lowered = key.lower()
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                if child_key == "action_space":
                    continue
                visit(child, str(child_key))
            return
        if isinstance(value, str):
            if any(token in lowered for token in ("instruction", "language", "text")):
                modalities.add("language")
            return
        try:
            array = np.asarray(value)
        except Exception:
            return
        if array.ndim >= 3 and any(token in lowered for token in ("rgb", "image", "camera")):
            modalities.add("rgb")
        elif array.ndim >= 2 and "depth" in lowered:
            modalities.add("depth")
        elif np.issubdtype(array.dtype, np.number):
            modalities.add("state")

    visit(observation)
    return modalities


def _check(
    check_id: str,
    passed: bool,
    *,
    phase: str,
    expected: Any,
    observed: Any,
    detail: str,
) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "status": "passed" if passed else "failed",
        "phase": phase,
        "expected": expected,
        "observed": observed,
        "detail": detail,
    }


def _not_applicable(check_id: str, detail: str, *, phase: str) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "status": "not_applicable",
        "phase": phase,
        "expected": None,
        "observed": None,
        "detail": detail,
    }


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return dict(sorted(counts.items()))


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(value.astype(np.float64).tobytes()).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
