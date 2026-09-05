from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from nyssa_bench.cli import main
from nyssa_bench.core.suite import Suite
from nyssa_bench.engines.base import NyssaEngine
from nyssa_bench.plugins import get_plugin_registry
from nyssa_bench.nep import load_nep_manifest
from nyssa_bench.nep.reference import (
    REFERENCE_ACTION_DIMENSIONS,
    REFERENCE_TASKS,
)
from nyssa_bench.policy_conformance import (
    evaluate_policy_conformance,
    load_policy_contract,
    write_policy_conformance_report,
)


ROOT = Path(__file__).resolve().parents[1]
STATE_POLICY = ROOT / "examples" / "policies" / "state_policy.py"
STATE_CONTRACT = ROOT / "examples" / "policies" / "state_policy_contract.json"


class _ConformanceEngine(NyssaEngine):
    max_steps = 2

    def __init__(self) -> None:
        self.value = 0.0

    def load_task(self, task_spec: Any) -> None:
        self.task_spec = task_spec

    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        self.value = float(seed or 0)
        return self._observation(), {"seed": seed}

    def step(
        self, action: Any
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self.value += float(np.asarray(action).reshape(-1)[0])
        return self._observation(), 1.0, True, False, {"success": True}

    def render(self) -> Any:
        return None

    def get_state(self) -> dict[str, float]:
        return {"value": self.value}

    def close(self) -> None:
        return None

    def _observation(self) -> dict[str, Any]:
        return {
            "raw": {"state": np.asarray([self.value], dtype=np.float32)},
            "action_space": {
                "type": "box",
                "shape": [1],
                "low": [-3.0],
                "high": [3.0],
                "dtype": "float32",
            },
        }


def _suite(engine_name: str) -> Suite:
    suite = Suite.load("tabletop_manipulation_v0").filter_tasks(["pick_cube"])
    suite.tasks[0].success.setdefault("engine_factory", {})[engine_name] = (
        "tests:_ConformanceEngine"
    )
    return suite


def _register() -> str:
    name = "policy_conformance_unit"
    get_plugin_registry().engines[name] = _ConformanceEngine
    REFERENCE_TASKS[name] = "pick_cube"
    REFERENCE_ACTION_DIMENSIONS[name] = 1
    return name


def test_state_policy_conformance_generates_machine_and_html_reports(
    tmp_path: Path,
) -> None:
    engine = _register()
    report = evaluate_policy_conformance(
        policy_path=STATE_POLICY,
        contract=load_policy_contract(STATE_CONTRACT),
        suite=_suite(engine),
        engine_name=engine,
        out_dir=tmp_path,
    )
    paths = write_policy_conformance_report(report, tmp_path)

    assert report["conformant"] is True
    assert report["integration_only"] is True
    assert report["check_status_counts"].get("failed", 0) == 0
    assert len(report["probes"]) == 4
    assert report["smoke"]["status"] == "passed"
    assert report["smoke"]["public_claim"] is False
    smoke_nep = load_nep_manifest(tmp_path / "smoke_run" / "nep_manifest.json")
    assert smoke_nep.policy.policy_id == "nyssa_example_state_policy"
    assert paths["json"].is_file()
    assert "not a validated policy track" in paths["html"].read_text(
        encoding="utf-8"
    )


def test_conformance_detects_state_leakage_before_smoke(tmp_path: Path) -> None:
    contract = load_policy_contract(STATE_CONTRACT)
    policy = tmp_path / "leaky_policy.py"
    metadata = {
        "policy_id": contract.policy_id,
        "policy_version": contract.policy_version,
        "policy_family": contract.policy_family,
        "checkpoint_id": contract.checkpoint_id,
        "checkpoint_sha256": contract.checkpoint_sha256,
        "preprocessing_sha256": contract.preprocessing_sha256,
        "observation_modalities": list(contract.observation_modalities),
        "action_representation": contract.action_representation,
        "action_dimension": contract.action_dimension,
        "prediction_horizon": contract.prediction_horizon,
        "execution_horizon": contract.execution_horizon,
    }
    policy.write_text(
        "import numpy as np\n"
        "class PolicyAdapter:\n"
        "    def __init__(self): self.calls = 0\n"
        "    def reset(self, task=None, seed=None): pass\n"
        "    def act(self, observation):\n"
        "        self.calls += 1\n"
        "        return np.asarray([self.calls / 10.0])\n"
        f"    def metadata(self): return {metadata!r}\n"
        "    def close(self): pass\n",
        encoding="utf-8",
    )

    engine = _register()
    report = evaluate_policy_conformance(
        policy_path=policy,
        contract=contract,
        suite=_suite(engine),
        engine_name=engine,
        out_dir=tmp_path / "out",
    )

    assert report["conformant"] is False
    assert report["smoke"] is None
    assert any(
        check["check_id"] == "matched_reset_determinism"
        and check["status"] == "failed"
        for check in report["checks"]
    )


def test_image_chunk_example_validates_rgb_and_chunk_shape() -> None:
    import importlib.util

    path = ROOT / "examples" / "policies" / "image_chunk_policy.py"
    spec = importlib.util.spec_from_file_location("image_chunk_example", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    policy = module.create_policy()
    action = policy.act(
        {
            "raw": {"rgb": np.zeros((32, 32, 3), dtype=np.uint8)},
            "action_space": {"shape": [7]},
        }
    )

    assert action.shape == (4, 7)


def test_conform_policy_cli_runs_public_example(tmp_path: Path) -> None:
    engine = _register()
    exit_code = main(
        [
            "conform-policy",
            "--policy",
            str(STATE_POLICY),
            "--policy-contract",
            str(STATE_CONTRACT),
            "--suite",
            "tabletop_manipulation_v0",
            "--task",
            "pick_cube",
            "--engine",
            engine,
            "--out",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    report = json.loads(
        (tmp_path / "policy_conformance.json").read_text(encoding="utf-8")
    )
    assert report["conformant"] is True


def test_installed_example_command_writes_self_contained_bundle(
    tmp_path: Path,
) -> None:
    assert (
        main(
            [
                "write-policy-example",
                "--kind",
                "state",
                "--out",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert (tmp_path / "state_policy.py").is_file()
    assert (tmp_path / "state_policy_contract.json").is_file()
    assert (tmp_path / "checkpoints" / "state_policy.json").is_file()
    assert load_policy_contract(tmp_path / "state_policy_contract.json").action_dimension == 1
