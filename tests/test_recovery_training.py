import json
from pathlib import Path

import pytest

from nyssa_bench.datasets.recovery_training import load_recovery_episodes


def _observation() -> dict:
    return {
        "raw": [0.0],
        "action_space": {
            "type": "box",
            "shape": [1],
            "low": [-1.0],
            "high": [1.0],
        },
    }


def _write_episodes(path: Path, steps: list[dict]) -> None:
    path.write_text(
        json.dumps([{"task_id": "unit_task", "steps": steps}]), encoding="utf-8"
    )


def test_recovery_training_filters_negative_context_and_uses_explicit_target(
    tmp_path: Path,
):
    episodes_path = tmp_path / "episodes.json"
    _write_episodes(
        episodes_path,
        [
            {
                "observation": _observation(),
                "executed_action": [0.9],
                "executed_action_source": "policy",
                "target_action": None,
                "target_source": None,
                "target_valid": False,
                "record_type": "negative_context",
            },
            {
                "observation": _observation(),
                "executed_action": [0.25],
                "executed_action_source": "recovery",
                "target_action": [0.25],
                "target_source": "recovery",
                "target_valid": True,
                "record_type": "supervised_target",
            },
        ],
    )

    episodes = load_recovery_episodes([episodes_path])

    assert len(episodes) == 1
    assert len(episodes[0]["steps"]) == 1
    assert episodes[0]["steps"][0]["action"] == [0.25]
    assert episodes[0]["steps"][0]["target_source"] == "recovery"


def test_recovery_training_rejects_policy_action_marked_as_valid_target(tmp_path: Path):
    episodes_path = tmp_path / "episodes.json"
    _write_episodes(
        episodes_path,
        [
            {
                "observation": _observation(),
                "executed_action": [0.9],
                "executed_action_source": "policy",
                "target_action": [0.9],
                "target_source": "policy",
                "target_valid": True,
                "record_type": "supervised_target",
            }
        ],
    )

    with pytest.raises(ValueError, match="Ineligible recovery target source"):
        load_recovery_episodes([episodes_path])


def test_recovery_training_filters_ambiguous_legacy_action(tmp_path: Path):
    episodes_path = tmp_path / "episodes.json"
    _write_episodes(
        episodes_path,
        [
            {
                "observation": _observation(),
                "action": [0.9],
                "info": {"recovery_attempted": True},
            }
        ],
    )

    with pytest.raises(ValueError, match="No eligible expert or recovery targets"):
        load_recovery_episodes([episodes_path])
