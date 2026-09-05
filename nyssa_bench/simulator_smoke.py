from __future__ import annotations

import argparse
import json
import os
import platform
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from nyssa_bench.core.registry import make_engine
from nyssa_bench.core.suite import Suite
from nyssa_bench.core.task import TaskSpec
from nyssa_bench.recovery.state import state_sha256
from nyssa_bench.nep import result_pack_pipeline_manifest, write_nep_manifest
from nyssa_bench.runner import PolicyRunner
from nyssa_bench.stressors import StressorConfig, StressorSpec
from nyssa_bench.utils.reproducibility import package_versions
from nyssa_bench.version import __version__


TASKS = {
    "mujoco": ("mujoco_control_v0", "mujoco_inverted_pendulum"),
    "maniskill": ("maniskill_smoke_v0", "maniskill_pick_cube"),
}


def run_simulator_smoke(
    engine_name: str,
    out_dir: str | Path,
    *,
    restore_repeats: int = 3,
    capture_replay: bool | None = None,
) -> dict[str, Any]:
    if engine_name not in TASKS:
        raise ValueError(f"unsupported simulator smoke engine: {engine_name}")
    if restore_repeats < 2:
        raise ValueError("simulator smoke requires at least two restore repeats")
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    suite_id, task_id = TASKS[engine_name]
    task = TaskSpec.load(task_id)
    restore_checks = []
    engine = make_engine(engine_name)
    try:
        engine.load_task(task)
        capability = engine.state_restore_capability()
        if not bool(capability.get("supported")):
            raise RuntimeError(
                f"{engine_name} does not provide smoke-test state restoration"
            )
        for attempt in range(restore_repeats):
            observation, reset_info = engine.reset(seed=attempt)
            action = _zero_action(observation)
            snapshot = engine.get_state()
            snapshot_hash = state_sha256(snapshot)
            engine.step(action)
            restored_observation = engine.set_state(snapshot)
            restored_hash = state_sha256(engine.get_state())
            if restored_hash != snapshot_hash:
                raise RuntimeError(
                    f"{engine_name} state hash changed after restore on attempt {attempt}"
                )
            restore_checks.append(
                {
                    "attempt": attempt,
                    "seed": attempt,
                    "state_sha256": snapshot_hash,
                    "reset_info_keys": sorted(str(key) for key in reset_info),
                    "restored_observation": restored_observation is not None,
                    "action_shape": list(np.asarray(action).shape),
                    "action_within_bounds": _action_within_bounds(
                        action, observation
                    ),
                }
            )
    finally:
        engine.close()

    suite = Suite.load(suite_id).filter_tasks([task_id])
    replay = engine_name == "maniskill" if capture_replay is None else capture_replay
    stressor = StressorSpec(
        stressor_id="action_gaussian_noise",
        severity=0.25,
        parameters={"max_std": 0.05},
        seed=17,
    )
    run_dir = out_dir / "result_pack"
    runner = PolicyRunner(
        policy="random",
        engine=engine_name,
        episodes=2,
        seed=7,
        out=run_dir,
        max_steps=2,
        capture_replay=replay,
        stressor_config=StressorConfig(
            condition_id="installed-simulator-smoke",
            stressors=(stressor,),
        ),
    )
    report = runner.evaluate(suite)
    episode_seeds = [episode.seed for episode in runner.episode_results]
    if len(episode_seeds) != len(set(episode_seeds)):
        raise RuntimeError("simulator smoke episode seeds are not disjoint")
    if not all(
        episode.metrics.get("stressor_applied_count", 0.0) >= 1.0
        for episode in runner.episode_results
    ):
        raise RuntimeError("simulator smoke stressor was not applied to every episode")
    required_artifacts = [
        "run.yaml",
        "config.yaml",
        "environment.json",
        "package_versions.json",
        "metrics.json",
        "episodes.json",
        "dataset_manifest.json",
        "stressor_manifest.json",
        "failure_ledger.json",
        "report.html",
    ]
    missing = [name for name in required_artifacts if not (run_dir / name).is_file()]
    if missing:
        raise RuntimeError(
            "simulator smoke result pack is incomplete: " + ", ".join(missing)
        )
    nep_manifest = result_pack_pipeline_manifest(engine_name, run_dir)
    write_nep_manifest(nep_manifest, run_dir / "nep_manifest.json")
    required_artifacts.append("nep_manifest.json")
    replay_count = len(list((run_dir / "videos").glob("*.mp4")))
    if replay and replay_count != len(runner.episode_results):
        raise RuntimeError("ManiSkill smoke did not produce one replay per episode")
    return {
        "format": "nyssa-simulator-ci-smoke-v1",
        "status": "passed",
        "engine": engine_name,
        "task_id": task_id,
        "nyssa_bench_version": __version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "rendering": {
            "MUJOCO_GL": os.getenv("MUJOCO_GL"),
            "PYOPENGL_PLATFORM": os.getenv("PYOPENGL_PLATFORM"),
            "DISPLAY": os.getenv("DISPLAY"),
        },
        "package_versions": package_versions(),
        "state_restore_capability": capability,
        "restore_checks": restore_checks,
        "result_pack": run_dir.as_posix(),
        "episode_seeds": episode_seeds,
        "stressor_id": stressor.stressor_id,
        "stressor_condition_id": "installed-simulator-smoke",
        "episodes": len(runner.episode_results),
        "replay_requested": replay,
        "replay_count": replay_count,
        "benchmark_tier": report.summary.get("benchmark_tier"),
        "public_claim": report.summary.get("public_claim"),
        "validation_status": (
            report.summary.get("public_claim_validation") or {}
        ).get("status"),
        "required_artifacts": required_artifacts,
        "nep": {
            "version": nep_manifest.nep_version,
            "content_sha256": nep_manifest.content_sha256,
            "claim_tier": nep_manifest.claim.requested_tier,
        },
    }


def _zero_action(observation: dict[str, Any]) -> np.ndarray:
    action_space = observation.get("action_space", {})
    shape = action_space.get("shape") if isinstance(action_space, dict) else None
    if not isinstance(shape, (list, tuple)) or not shape:
        raise RuntimeError("simulator smoke observation has no action shape contract")
    return np.zeros(tuple(int(value) for value in shape), dtype=np.float32)


def _action_within_bounds(action: Any, observation: dict[str, Any]) -> bool:
    contract = observation.get("action_space", {})
    if not isinstance(contract, dict):
        return False
    values = np.asarray(action, dtype=float)
    low = np.asarray(contract.get("low"), dtype=float)
    high = np.asarray(contract.get("high"), dtype=float)
    return bool(
        values.shape == low.shape == high.shape
        and np.isfinite(values).all()
        and (values >= low).all()
        and (values <= high).all()
    )


def _write_diagnostic(out_dir: Path, payload: dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "simulator_smoke.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run an installed NyssaBench simulator integration smoke."
    )
    parser.add_argument("--engine", choices=sorted(TASKS), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--restore-repeats", type=int, default=3)
    parser.add_argument("--capture-replay", action="store_true")
    args = parser.parse_args(argv)
    out_dir = Path(args.out).resolve()
    try:
        payload = run_simulator_smoke(
            args.engine,
            out_dir,
            restore_repeats=args.restore_repeats,
            capture_replay=args.capture_replay or None,
        )
    except Exception as exc:
        _write_diagnostic(
            out_dir,
            {
                "format": "nyssa-simulator-ci-smoke-v1",
                "status": "failed",
                "engine": args.engine,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "package_versions": package_versions(),
                "rendering": {
                    "MUJOCO_GL": os.getenv("MUJOCO_GL"),
                    "PYOPENGL_PLATFORM": os.getenv("PYOPENGL_PLATFORM"),
                    "DISPLAY": os.getenv("DISPLAY"),
                },
            },
        )
        raise
    path = _write_diagnostic(out_dir, payload)
    print(f"simulator_smoke: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
