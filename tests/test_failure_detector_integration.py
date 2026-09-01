from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from nyssa_bench.failures import FailureEventLedger
from nyssa_bench.failures.detectors import FailureDetectorManager, StallDetector


def _zero_action(space: Any) -> Any:
    sample = space.sample()
    if hasattr(sample, "detach"):
        sample = sample.detach()
    if hasattr(sample, "cpu"):
        sample = sample.cpu()
    if hasattr(sample, "numpy"):
        sample = sample.numpy()
    return np.zeros_like(sample)


def _exercise_reward_detector(
    *,
    engine: Any,
    engine_name: str,
    observation: dict[str, Any],
    reset_info: dict[str, Any],
) -> tuple[list[Any], dict[str, Any]]:
    detector = StallDetector(stall_window=2, reward_tolerance=1_000_000.0, min_steps=2)
    manager = FailureDetectorManager(detectors=(detector,), engine_name=engine_name)
    manager.reset(
        task=None,
        engine=engine,
        observation=observation,
        stressor_context=None,
        reset_info=reset_info,
    )
    ledger = FailureEventLedger(
        task_id=f"{engine_name}-integration",
        episode_index=0,
        episode_seed=0,
        engine_name=engine_name,
    )
    emissions = []
    for step_index in range(3):
        observation, reward, terminated, truncated, info = engine.step(
            _zero_action(engine.env.action_space)
        )
        emissions.extend(
            manager.detect(
                step_index=step_index,
                observation=observation,
                action=None,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info,
                task=None,
                engine=engine,
            )
        )
        if terminated or truncated:
            break
    events = manager.emit(ledger, emissions, default_step=0)
    return events, manager.manifest(events=ledger.events)


def test_real_mujoco_reward_signal_drives_streaming_detector_when_installed():
    gym = pytest.importorskip("gymnasium")
    pytest.importorskip("mujoco")
    from nyssa_bench.engines.mujoco_adapter import MuJoCoEngine

    try:
        env = gym.make("InvertedPendulum-v4")
    except Exception as exc:
        pytest.skip(f"MuJoCo integration environment unavailable: {exc}")
    engine = MuJoCoEngine()
    engine.env = env
    try:
        observation, reset_info = engine.reset(seed=0)
        events, manifest = _exercise_reward_detector(
            engine=engine,
            engine_name="mujoco",
            observation=observation,
            reset_info=reset_info,
        )
    finally:
        engine.close()

    assert events
    assert events[0].subtype == "planner_stuck"
    assert events[0].provenance.component_id == "stall_detector"
    assert manifest["detectors"][0]["support"]["status"] == "supported"


def test_real_maniskill_reward_signal_drives_streaming_detector_when_installed():
    gym = pytest.importorskip("gymnasium")
    pytest.importorskip("mani_skill")
    from nyssa_bench.engines.maniskill_adapter import ManiSkillEngine

    try:
        env = gym.make(
            "PickCube-v1",
            obs_mode="state_dict",
            control_mode="pd_ee_delta_pose",
            sim_backend="cpu",
        )
    except Exception as exc:
        pytest.skip(f"ManiSkill integration environment unavailable: {exc}")
    engine = ManiSkillEngine()
    engine.env = env
    try:
        observation, reset_info = engine.reset(seed=0)
        events, manifest = _exercise_reward_detector(
            engine=engine,
            engine_name="maniskill",
            observation=observation,
            reset_info=reset_info,
        )
    finally:
        engine.close()

    assert events
    assert events[0].subtype == "planner_stuck"
    assert events[0].provenance.component_id == "stall_detector"
    assert manifest["detectors"][0]["support"]["status"] == "supported"
