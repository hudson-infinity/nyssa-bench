from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.validate_simulator_ci_baseline import validate_simulator_ci_baseline


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "claims" / "mujoco_ci_baseline.json"


def test_committed_mujoco_baseline_passes_policy() -> None:
    report = validate_simulator_ci_baseline(BASELINE)

    assert report == {
        "format": "nyssa-simulator-ci-baseline-validation-v1",
        "status": "validated",
        "engine": "mujoco",
        "run_count": 20,
        "pass_count": 20,
        "pass_rate": 1.0,
        "runner_class": "ubuntu-latest",
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["runs"].pop(), "fewer runs"),
        (
            lambda value: value["runs"][0].update(conclusion="failure"),
            "observed_passes",
        ),
        (
            lambda value: value["common_contract"].update(mujoco_version="0.0.0"),
            "common contract",
        ),
        (
            lambda value: value["runs"][1].update(
                run_id=value["runs"][0]["run_id"]
            ),
            "unique",
        ),
    ],
)
def test_baseline_tampering_is_rejected(
    tmp_path: Path, mutation, message: str
) -> None:
    payload = json.loads(BASELINE.read_text(encoding="utf-8"))
    mutation(payload)
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        validate_simulator_ci_baseline(path)
