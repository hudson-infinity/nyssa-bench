from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from nyssa_bench.core.suite import Suite, list_suites
from nyssa_bench.engines.base import NyssaEngine
from nyssa_bench.package_resources import config_root, resource_root
from nyssa_bench.plugins import get_plugin_registry
from nyssa_bench.runner import PolicyRunner
from nyssa_bench.stressors import list_stressors
from nyssa_bench.version import __version__


class _PackagingSmokeEngine(NyssaEngine):
    max_steps = 1

    def load_task(self, task_spec: Any) -> None:
        self.task_spec = task_spec

    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        return self._observation(), {"seed": seed}

    def step(
        self, action: Any
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        return self._observation(), 1.0, True, False, {"success": True}

    def render(self) -> Any:
        return None

    def get_state(self) -> dict[str, Any]:
        return {"step": 0}

    def close(self) -> None:
        return None

    @staticmethod
    def _observation() -> dict[str, Any]:
        return {
            "raw": np.asarray([0.0], dtype=np.float32),
            "action_space": {
                "type": "box",
                "shape": [1],
                "low": [-1.0],
                "high": [1.0],
                "dtype": "float32",
            },
        }


class _PackagingSmokePolicy:
    def reset(self, task: Any | None = None, seed: int | None = None) -> None:
        return None

    def act(self, observation: dict[str, Any]) -> list[float]:
        return [0.0]


def run_packaging_smoke(out_dir: str | Path) -> dict[str, Any]:
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    suites = list_suites()
    stressors = list_stressors()
    if "tabletop_manipulation_v0" not in suites or not stressors:
        raise RuntimeError("bundled suite or stressor resources are unavailable")
    config_dir = config_root("suites")
    conformance_dir = resource_root("conformance")
    schema_dir = resource_root("schemas")
    suite = Suite.load("tabletop_manipulation_v0").filter_tasks(["pick_cube"])
    get_plugin_registry().engines["packaging_smoke"] = _PackagingSmokeEngine
    runner = PolicyRunner(
        policy=_PackagingSmokePolicy(),
        engine="packaging_smoke",
        episodes=1,
        seed=0,
        out=out_dir,
        capture_replay=False,
    )
    report = runner.evaluate(suite)
    required = [
        out_dir / "metrics.json",
        out_dir / "episodes.json",
        out_dir / "dataset_manifest.json",
        out_dir / "report.html",
    ]
    missing = [path.name for path in required if not path.is_file()]
    if report.summary.get("success_rate") != 1.0 or missing:
        raise RuntimeError(f"installed artifact smoke failed; missing={missing}")
    payload = {
        "format": "nyssa-installed-artifact-smoke-v1",
        "nyssa_bench_version": __version__,
        "package_root": Path(__file__).resolve().parent.as_posix(),
        "working_directory": Path.cwd().resolve().as_posix(),
        "config_root": config_dir.as_posix(),
        "conformance_root": conformance_dir.as_posix(),
        "schema_root": schema_dir.as_posix(),
        "suite_count": len(suites),
        "stressor_count": len(stressors),
        "suite_id": suite.suite_id,
        "task_ids": [task.task_id for task in suite.tasks],
        "success_rate": report.summary["success_rate"],
        "benchmark_tier": report.summary["benchmark_tier"],
        "public_claim": report.summary["public_claim"],
        "artifacts": {
            path.name: _sha256(path) for path in required
        },
    }
    path = out_dir / "installed_artifact_smoke.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the NyssaBench installed-artifact smoke test."
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    payload = run_packaging_smoke(args.out)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
