from __future__ import annotations

import json
import random
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from nyssa_bench.cli import main
from nyssa_bench.core.registry import make_engine
from nyssa_bench.core.suite import Suite
from nyssa_bench.core.task import TaskSpec
from nyssa_bench.engines.base import NyssaEngine
from nyssa_bench.experts.base import ExpertActionScore, ExpertProvider
from nyssa_bench.plugins import get_plugin_registry
from nyssa_bench.policies.base import Policy
from nyssa_bench.policies.bc_policy import BCPolicy
from nyssa_bench.policies.task_bc_policy import TaskBCPolicy
from nyssa_bench.recovery import (
    COUNTERFACTUAL_RECOVERY_MANIFEST_FORMAT,
    CounterfactualBranchRunner,
    load_counterfactual_recovery_manifest,
    summarize_counterfactual_recovery,
)
from nyssa_bench.runner import PolicyRunner
from nyssa_bench.stressors import StressorContext, StressorPipeline


class _RestorableEngine(NyssaEngine):
    max_steps = 3

    def __init__(self, *, stochastic: bool = False) -> None:
        self.position = 0.0
        self.elapsed = 0
        self.stochastic = stochastic
        self.rng = np.random.default_rng(0)

    def load_task(self, task_spec: TaskSpec) -> None:
        self.task_spec = task_spec

    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        self.position = 0.0
        self.elapsed = 0
        self.rng = np.random.default_rng(seed)
        return self._observation(), {"seed": seed}

    def step(
        self, action: Any
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        value = float(np.asarray(action).reshape(-1)[0])
        noise = float(self.rng.uniform(-0.2, 0.2)) if self.stochastic else 0.0
        self.position += value
        self.elapsed += 1
        success = self.position >= 1.0
        terminated = success or self.elapsed >= self.max_steps
        return (
            self._observation(),
            value + noise,
            terminated,
            False,
            {
                "success": success,
                "completion_time": float(self.elapsed),
                "path_efficiency": 1.0 if success else 0.0,
                "grasp_success": success,
            },
        )

    def render(self) -> Any:
        return None

    def get_state(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "elapsed": self.elapsed,
            "rng": deepcopy(self.rng.bit_generator.state),
        }

    def set_state(self, state: Any) -> dict[str, Any]:
        self.position = float(state["position"])
        self.elapsed = int(state["elapsed"])
        self.rng.bit_generator.state = deepcopy(state["rng"])
        return self._observation()

    def state_restore_capability(self) -> dict[str, Any]:
        return {
            "supported": True,
            "fidelity": "exact_unit_engine_and_rng",
            "captures_rng": True,
            "exact": True,
            "reason": None,
        }

    def seed_branch_rng(self, seed: int) -> bool:
        self.rng = np.random.default_rng(seed)
        return True

    def close(self) -> None:
        return None

    def _observation(self) -> dict[str, Any]:
        return {
            "raw": np.asarray([self.position], dtype=float),
            "action_space": {
                "type": "box",
                "shape": [1],
                "low": [-1.0],
                "high": [1.0],
            },
        }


class _StatefulPolicy(Policy):
    def __init__(self, action: float = -1.0) -> None:
        self.action = action
        self.calls = 0
        self.rng = np.random.default_rng(0)

    def act(self, observation: dict[str, Any]) -> Any:
        del observation
        self.calls += 1
        return np.asarray([self.action])

    def get_state(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "rng": deepcopy(self.rng.bit_generator.state),
        }

    def set_state(self, state: Any) -> None:
        self.calls = int(state["calls"])
        self.rng.bit_generator.state = deepcopy(state["rng"])

    def state_restore_capability(self) -> dict[str, Any]:
        return {
            "supported": True,
            "fidelity": "exact_unit_policy_and_rng",
            "captures_rng": True,
            "exact": True,
            "reason": None,
        }

    def seed_branch_rng(self, seed: int) -> bool:
        self.rng = np.random.default_rng(seed)
        return True


class _RecoveryExpert(ExpertProvider):
    provider_id = "unit-recovery"

    def score_action(
        self,
        observation: dict[str, Any],
        action: Any,
        *,
        task: Any,
        engine: Any | None = None,
    ) -> ExpertActionScore:
        del observation, action, task, engine
        return ExpertActionScore(accepted=False, confidence=1.0, reason="unit_risk")

    def recover(
        self,
        *,
        state: dict[str, Any],
        failure: str | None,
        task: Any,
        engine: Any | None = None,
    ) -> list[Any]:
        del state, failure, task, engine
        return [np.asarray([1.0])]

    def act(
        self,
        observation: dict[str, Any],
        *,
        task: Any,
        engine: Any | None = None,
    ) -> Any:
        del observation, task, engine
        return np.asarray([1.0])

    def state_restore_capability(self) -> dict[str, Any]:
        return {
            "supported": True,
            "fidelity": "exact_declared_stateless",
            "captures_rng": False,
            "exact": True,
            "reason": None,
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "capabilities": ["score_action", "recover", "act"],
        }


def _pipeline(engine_name: str = "unit_branch") -> StressorPipeline:
    pipeline = StressorPipeline(
        (),
        context=StressorContext(
            engine_name=engine_name,
            task_id="unit_branch_task",
        ),
        episode_seed=7,
    )
    pipeline.before_reset(SimpleNamespace())
    pipeline.after_reset(SimpleNamespace(), {})
    return pipeline


def _task(engine: str = "unit_branch") -> TaskSpec:
    return TaskSpec(
        task_id="unit_branch_task",
        engine=engine,
        robot="unit",
        scene="unit",
        description="Counterfactual recovery unit task",
        success={
            "engine_factory": {engine: "tests:_RestorableEngine"},
            "success_info_keys": ["success"],
            "max_steps": 3,
        },
        failure_labels=["missed_target"],
    )


def _evaluate_direct(
    *, stochastic: bool = False, repeats: int = 2
) -> tuple[Any, _RestorableEngine, _StatefulPolicy]:
    engine = _RestorableEngine(stochastic=stochastic)
    engine.reset(seed=7)
    policy = _StatefulPolicy()
    expert = _RecoveryExpert()
    pipeline = _pipeline()
    runner = CounterfactualBranchRunner(repeats=repeats, horizon_steps=2)
    record = runner.evaluate_recovery(
        engine=engine,
        policy=policy,
        expert=expert,
        stressors=pipeline,
        task=_task(),
        observation=engine._observation(),
        task_id="unit_branch_task",
        episode_index=0,
        episode_seed=7,
        step_index=0,
        recovery_attempt_id=1,
        continuation_actions=[np.asarray([-1.0])],
        recovery_actions=[np.asarray([1.0])],
        trigger_reason="unit_risk",
        trigger_event_id="failure-event-1",
    )
    return record, engine, policy


def test_branch_runner_restores_live_state_and_emits_exact_matched_pairs() -> None:
    record, engine, policy = _evaluate_direct()

    assert engine.position == 0.0
    assert engine.elapsed == 0
    assert policy.calls == 0
    assert record.branch_point.restoration_grade == "exact"
    assert record.branch_point.matched_randomness is True
    assert record.branch_point.strongest_causal_claim_eligible is True
    assert len(record.outcomes) == 4
    for repeat_index in range(2):
        outcomes = {
            item.branch_kind: item
            for item in record.outcomes
            if item.repeat_index == repeat_index
        }
        assert outcomes["continue"].success is False
        assert outcomes["recovery"].success is True
        assert (
            outcomes["continue"].matched_rng_sha256
            == outcomes["recovery"].matched_rng_sha256
        )
    json.dumps(record.to_dict(), allow_nan=False)


def test_deterministic_continuations_match_after_every_restore() -> None:
    record, _, _ = _evaluate_direct(repeats=3)
    continuation_digests = {
        outcome.trajectory_sha256
        for outcome in record.outcomes
        if outcome.branch_kind == "continue"
    }

    assert len(continuation_digests) == 1


def test_stochastic_repeats_are_distinct_but_paired_within_repeat() -> None:
    record, _, _ = _evaluate_direct(stochastic=True, repeats=3)
    noise_by_repeat = []
    for repeat_index in range(3):
        outcomes = {
            item.branch_kind: item
            for item in record.outcomes
            if item.repeat_index == repeat_index
        }
        continue_reward = outcomes["continue"].steps[0].reward
        recovery_reward = outcomes["recovery"].steps[0].reward
        assert recovery_reward - continue_reward == pytest.approx(2.0)
        noise_by_repeat.append(continue_reward + 1.0)

    assert len(set(noise_by_repeat)) == 3


def test_branch_evaluation_restores_process_rng_streams() -> None:
    random.seed(91)
    np.random.seed(91)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    expected_python = random.random()
    expected_numpy = float(np.random.random())
    random.setstate(python_state)
    np.random.set_state(numpy_state)

    _evaluate_direct(stochastic=True, repeats=2)

    assert random.random() == expected_python
    assert float(np.random.random()) == expected_numpy


def test_counterfactual_summary_uses_branch_points_as_bootstrap_clusters() -> None:
    record, _, _ = _evaluate_direct(repeats=3)
    summary = summarize_counterfactual_recovery(
        [
            {
                "counterfactual_recovery": [record],
                "metrics": {"counterfactual_eligible_branch_point_count": 2.0},
            }
        ]
    )

    assert summary["matched_pairs"] == 3
    assert summary["matched_branch_points"] == 1
    assert summary["claim_tier"] == "exact_counterfactual_partial_coverage"
    assert summary["coverage"] == {
        "numerator": 1,
        "denominator": 2,
        "rate": 0.5,
    }
    assert summary["metrics"]["counterfactual_recovery_gain"] == 1.0
    assert summary["metric_ci95"]["counterfactual_recovery_gain"] == [1.0, 1.0]
    assert summary["interventions"]["helpful"] == 3
    assert summary["interventions"]["false"] == 0
    assert summary["interventions"]["harmful"] == 0
    assert summary["interventions"]["mean_plan_actions"] == 1.0
    serialized_summary = summarize_counterfactual_recovery(
        [
            {
                "counterfactual_recovery": [record.to_dict()],
                "metrics": {"counterfactual_eligible_branch_point_count": 2.0},
            }
        ]
    )
    assert serialized_summary == summary


def test_unsupported_policy_state_records_qualified_absence_without_execution() -> None:
    class DuckPolicy:
        def act(self, observation: dict[str, Any]) -> Any:
            del observation
            return np.asarray([-1.0])

    engine = _RestorableEngine()
    observation, _ = engine.reset(seed=7)
    record = CounterfactualBranchRunner(repeats=1, horizon_steps=1).evaluate_recovery(
        engine=engine,
        policy=DuckPolicy(),
        expert=_RecoveryExpert(),
        stressors=_pipeline(),
        task=_task(),
        observation=observation,
        task_id="unit_branch_task",
        episode_index=0,
        episode_seed=7,
        step_index=0,
        recovery_attempt_id=1,
        continuation_actions=[np.asarray([-1.0])],
        recovery_actions=[np.asarray([1.0])],
        trigger_reason="unit_risk",
        trigger_event_id=None,
    )

    assert record.branch_point.restoration_grade == "unsupported"
    assert "policy" in str(record.branch_point.unsupported_reason)
    assert record.outcomes == ()
    assert engine.position == 0.0


def test_branch_execution_error_is_recorded_and_live_state_is_restored() -> None:
    class FailingPolicy(_StatefulPolicy):
        def act(self, observation: dict[str, Any]) -> Any:
            del observation
            self.calls += 1
            raise RuntimeError("branch prediction failed")

    engine = _RestorableEngine()
    observation, _ = engine.reset(seed=7)
    policy = FailingPolicy()
    record = CounterfactualBranchRunner(
        repeats=1, horizon_steps=2
    ).evaluate_recovery(
        engine=engine,
        policy=policy,
        expert=_RecoveryExpert(),
        stressors=_pipeline(),
        task=_task(),
        observation=observation,
        task_id="unit_branch_task",
        episode_index=0,
        episode_seed=7,
        step_index=0,
        recovery_attempt_id=1,
        continuation_actions=[np.asarray([-1.0])],
        recovery_actions=[np.asarray([1.0])],
        trigger_reason="unit_risk",
        trigger_event_id=None,
    )

    continuation = next(
        outcome for outcome in record.outcomes if outcome.branch_kind == "continue"
    )
    assert continuation.status == "error"
    assert continuation.error_type == "RuntimeError"
    assert continuation.error_message == "branch prediction failed"
    assert record.branch_point.strongest_causal_claim_eligible is False
    assert engine.position == 0.0
    assert engine.elapsed == 0
    assert policy.calls == 0


def test_builtin_bc_policy_state_contracts_are_exact(tmp_path: Path) -> None:
    from nyssa_bench.baselines.simple_bc import (
        LinearBCPolicy,
        TaskRoutedLinearBCPolicy,
    )

    linear = LinearBCPolicy(
        weights=np.zeros((1, 1)),
        bias=np.zeros(1),
        feature_dim=1,
        action_size=1,
    )
    policy = BCPolicy(model=linear)
    state = policy.get_state()
    policy.set_state(state)
    assert policy.state_restore_capability()["exact"] is True

    routed_model = TaskRoutedLinearBCPolicy(tmp_path, missing_task="zero")
    routed = TaskBCPolicy(model=routed_model)
    routed_model.current_task_id = "first"
    routed_state = routed.get_state()
    routed_model.current_task_id = "second"
    routed.set_state(routed_state)
    assert routed_model.current_task_id == "first"
    assert routed.state_restore_capability()["exact"] is True


def test_policy_runner_writes_counterfactual_manifest_and_metric_vector(
    tmp_path: Path,
) -> None:
    class RegisteredEngine(_RestorableEngine):
        pass

    get_plugin_registry().engines["unit_branch"] = RegisteredEngine
    suite = Suite(
        suite_id="unit_counterfactual_v0",
        description="Counterfactual integration fixture",
        tasks=(_task(),),
    )
    runner = PolicyRunner(
        policy=_StatefulPolicy(),
        engine="unit_branch",
        episodes=1,
        seed=7,
        out=tmp_path,
        capture_replay=False,
        expert_provider=_RecoveryExpert(),
        enable_recovery=True,
        counterfactual_repeats=2,
        counterfactual_horizon=2,
        counterfactual_oracle=True,
    )

    report = runner.evaluate(suite)

    evidence = report.summary["counterfactual_recovery"]
    assert evidence["claim_tier"] == "exact_counterfactual"
    assert evidence["matched_pairs"] == 2
    assert report.summary["metrics"]["counterfactual_recovery_gain"] == 1.0
    measurement = report.summary["metric_vector"]["values"][
        "counterfactual_recovery_gain"
    ]
    assert measurement["status"] == "available"
    assert measurement["sample_size"] == 1
    manifest = json.loads(
        (tmp_path / "counterfactual_recovery.json").read_text(encoding="utf-8")
    )
    assert manifest["format"] == COUNTERFACTUAL_RECOVERY_MANIFEST_FORMAT
    assert len(manifest["branch_points"]) == 1
    _, loaded_records = load_counterfactual_recovery_manifest(
        tmp_path / "counterfactual_recovery.json"
    )
    assert loaded_records[0].to_dict() == manifest["branch_points"][0]
    assert main(["validate", str(tmp_path / "counterfactual_recovery.json")]) == 0
    tampered = deepcopy(manifest)
    tampered["summary"]["matched_pairs"] = 99
    tampered_path = tmp_path / "counterfactual_recovery_tampered.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match branch-point evidence"):
        load_counterfactual_recovery_manifest(tampered_path)
    dataset_manifest = json.loads(
        (tmp_path / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    assert "counterfactual_recovery.json" in dataset_manifest["artifacts"]


def test_counterfactual_manifest_rejects_inconsistent_derived_counts(
    tmp_path: Path,
) -> None:
    record, _, _ = _evaluate_direct(repeats=1)
    payload = {
        "format": COUNTERFACTUAL_RECOVERY_MANIFEST_FORMAT,
        "configuration": {"enabled": True},
        "summary": {},
        "branch_points": [record.to_dict()],
    }
    payload["branch_points"][0]["outcomes"][0]["steps_executed"] = 99
    path = tmp_path / "counterfactual_recovery.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="steps_executed"):
        load_counterfactual_recovery_manifest(path)


def test_runner_rejects_counterfactual_configuration_without_recovery() -> None:
    with pytest.raises(ValueError, match="enable_recovery"):
        PolicyRunner(
            policy=_StatefulPolicy(),
            engine="unit_branch",
            counterfactual_repeats=1,
        )


def test_mujoco_adapter_restores_manual_state_and_rejects_wrong_shapes() -> None:
    class Data:
        def __init__(self) -> None:
            self.qpos = np.asarray([0.0, 0.0])
            self.qvel = np.asarray([0.0, 0.0])
            self.time = 0.0

    class Target:
        def __init__(self) -> None:
            self.data = Data()
            self.model = None
            self.np_random = np.random.default_rng(4)

        def _get_obs(self) -> np.ndarray:
            return self.data.qpos.copy()

    class Env:
        def __init__(self) -> None:
            self.unwrapped = Target()
            self.action_space = SimpleNamespace(
                __class__=SimpleNamespace(__name__="Box"),
                shape=(2,),
                low=np.asarray([-1.0, -1.0]),
                high=np.asarray([1.0, 1.0]),
                dtype=np.dtype("float64"),
            )

    engine = make_engine("mujoco")
    engine.env = Env()
    state = engine.get_state()
    engine.env.unwrapped.data.qpos[:] = 3.0

    observation = engine.set_state(state)

    assert engine.env.unwrapped.data.qpos.tolist() == [0.0, 0.0]
    assert observation is not None
    with pytest.raises(ValueError, match="qpos state has shape"):
        engine.set_state({**state, "qpos": np.asarray([1.0])})


def test_maniskill_adapter_restores_state_rng_controller_and_wrapper_counter() -> None:
    class Controller:
        def __init__(self) -> None:
            self.value = 4

        def get_state(self) -> dict[str, int]:
            return {"value": self.value}

        def set_state(self, state: dict[str, int]) -> None:
            self.value = int(state["value"])

    class Target:
        def __init__(self) -> None:
            self.value = np.asarray([1.0])
            self.np_random = np.random.default_rng(3)
            self.agent = SimpleNamespace(controller=Controller())
            self._elapsed_steps = 2

        def get_state_dict(self) -> dict[str, np.ndarray]:
            return {"value": self.value.copy()}

        def set_state_dict(self, state: dict[str, np.ndarray]) -> None:
            self.value = np.asarray(state["value"]).copy()

        def get_obs(self) -> np.ndarray:
            return self.value.copy()

    class Env:
        def __init__(self) -> None:
            self.env = Target()
            self.unwrapped = self.env
            self._elapsed_steps = 2
            self.action_space = SimpleNamespace(
                shape=(1,),
                low=np.asarray([-1.0]),
                high=np.asarray([1.0]),
                dtype=np.dtype("float64"),
            )

    engine = make_engine("maniskill")
    engine.env = Env()
    state = engine.get_state()
    expected_random = float(engine.env.unwrapped.np_random.random())
    engine.env.unwrapped.value[:] = 9.0
    engine.env.unwrapped.agent.controller.value = 9
    engine.env._elapsed_steps = 9
    engine.env.unwrapped._elapsed_steps = 9

    observation = engine.set_state(state)

    assert engine.state_restore_capability()["exact"] is True
    assert engine.env.unwrapped.value.tolist() == [1.0]
    assert engine.env.unwrapped.agent.controller.value == 4
    assert engine.env._elapsed_steps == 2
    assert engine.env.unwrapped._elapsed_steps == 2
    assert float(engine.env.unwrapped.np_random.random()) == expected_random
    assert observation is not None
    assert observation["raw"].tolist() == [1.0]


def test_real_mujoco_continuation_matches_after_restore_when_installed() -> None:
    gym = pytest.importorskip("gymnasium")
    pytest.importorskip("mujoco")
    try:
        env = gym.make("InvertedPendulum-v4")
    except Exception as exc:
        pytest.skip(f"MuJoCo runtime is unavailable: {exc}")
    engine = make_engine("mujoco")
    engine.env = env
    engine.task_spec = TaskSpec(
        task_id="mujoco_restore_fixture",
        engine="mujoco",
        robot="inverted_pendulum",
        scene="unit",
        description="MuJoCo restore fixture",
        success={"reward_threshold": 1e9},
    )
    try:
        engine.reset(seed=11)
        snapshot = engine.get_state()
        first = engine.step(np.asarray([0.1]))
        engine.set_state(snapshot)
        second = engine.step(np.asarray([0.1]))
    finally:
        engine.close()

    np.testing.assert_allclose(first[0]["raw"], second[0]["raw"])
    assert first[1:] == second[1:]
