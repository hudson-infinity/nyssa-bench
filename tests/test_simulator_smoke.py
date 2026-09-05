from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from nyssa_bench.engines.base import NyssaEngine
from nyssa_bench.plugins import get_plugin_registry
from nyssa_bench.simulator_smoke import TASKS, main, run_simulator_smoke
from nyssa_bench.nep.reference import (
    REFERENCE_ACTION_DIMENSIONS,
    REFERENCE_TASKS,
)


class _SimulatorCiEngine(NyssaEngine):
    max_steps = 2

    def __init__(self) -> None:
        self.position = 0.0
        self.elapsed = 0

    def load_task(self, task_spec: Any) -> None:
        self.task_spec = task_spec

    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        self.position = 0.0
        self.elapsed = 0
        return self._observation(), {"seed": seed}

    def step(
        self, action: Any
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self.position += float(np.asarray(action).reshape(-1)[0])
        self.elapsed += 1
        return self._observation(), 1.0, True, False, {"success": True}

    def render(self) -> Any:
        return None

    def get_state(self) -> dict[str, Any]:
        return {"position": self.position, "elapsed": self.elapsed}

    def set_state(self, state: Any) -> dict[str, Any]:
        self.position = float(state["position"])
        self.elapsed = int(state["elapsed"])
        return self._observation()

    def state_restore_capability(self) -> dict[str, Any]:
        return {
            "supported": True,
            "fidelity": "exact_unit_state",
            "captures_rng": False,
            "exact": True,
            "reason": None,
        }

    def close(self) -> None:
        return None

    def _observation(self) -> dict[str, Any]:
        return {
            "raw": [self.position],
            "action_space": {
                "type": "box",
                "shape": [1],
                "low": [-1.0],
                "high": [1.0],
                "dtype": "float32",
            },
        }


def test_simulator_smoke_covers_restore_stressors_seeds_and_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine_name = "simulator_ci_unit"
    get_plugin_registry().engines[engine_name] = _SimulatorCiEngine
    monkeypatch.setitem(
        TASKS, engine_name, ("tabletop_manipulation_v0", "pick_cube")
    )
    monkeypatch.setitem(REFERENCE_TASKS, engine_name, "pick_cube")
    monkeypatch.setitem(REFERENCE_ACTION_DIMENSIONS, engine_name, 1)

    report = run_simulator_smoke(
        engine_name,
        tmp_path,
        restore_repeats=3,
        capture_replay=False,
    )

    assert report["status"] == "passed"
    assert len(report["restore_checks"]) == 3
    assert all(item["action_within_bounds"] for item in report["restore_checks"])
    assert len(report["episode_seeds"]) == len(set(report["episode_seeds"])) == 2
    assert report["stressor_id"] == "action_gaussian_noise"
    assert report["replay_count"] == 0
    for artifact in report["required_artifacts"]:
        assert (tmp_path / "result_pack" / artifact).is_file()


def test_simulator_smoke_failure_writes_diagnostics(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least two restore repeats"):
        main(["--engine", "mujoco", "--out", str(tmp_path), "--restore-repeats", "1"])

    diagnostic = json.loads(
        (tmp_path / "simulator_smoke.json").read_text(encoding="utf-8")
    )
    assert diagnostic["status"] == "failed"
    assert diagnostic["error_type"] == "ValueError"
