import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from nyssa_bench import PolicyRunner, Suite
from nyssa_bench.engines.base import NyssaEngine
from nyssa_bench.core.episode import EpisodeResult, StepRecord
from nyssa_bench.experts import ExpertActionScore, ExpertProvider, make_expert_provider
from nyssa_bench.metrics.failure_mapper import FailureMapper
from nyssa_bench.metrics.run_claims import RunClaimValidator
from nyssa_bench.policies.robomimic_adapter import RoboMimicPolicy
from nyssa_bench.plugins import get_plugin_registry
from nyssa_bench.validation.run_claim import (
    RunClaimValidator as StableRunClaimValidator,
)


class UnitEngine(NyssaEngine):
    max_steps = 2

    def load_task(self, task_spec: Any) -> None:
        self.task_spec = task_spec

    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        return _observation(), {"seed": seed}

    def step(
        self, action: Any
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        return (
            _observation(),
            1.0,
            True,
            False,
            {
                "success": True,
                "completion_time": 1.0,
                "path_efficiency": 1.0,
                "grasp_success": True,
            },
        )

    def render(self) -> Any:
        return np.zeros((32, 32, 3), dtype=np.uint8)

    def get_state(self) -> dict[str, Any]:
        return {}

    def close(self) -> None:
        return None


def _observation() -> dict[str, Any]:
    return {
        "raw": [0.0],
        "action_space": {
            "type": "box",
            "shape": [1],
            "low": [-1.0],
            "high": [1.0],
            "dtype": "float32",
        },
    }


def _maniskill_state_observation(extra: dict[str, Any]) -> dict[str, Any]:
    return {
        "raw": {"agent": {"qpos": [[0.0] * 9], "qvel": [[0.0] * 9]}, "extra": extra},
        "action_space": {
            "type": "box",
            "shape": [7],
            "low": [-1.0] * 7,
            "high": [1.0] * 7,
            "dtype": "float32",
        },
    }


def _register_unit_engine() -> None:
    get_plugin_registry().engines["unit_real"] = UnitEngine


def test_suite_loads_tasks():
    suite = Suite.load("tabletop_manipulation_v0")
    assert suite.suite_id == "tabletop_manipulation_v0"
    assert len(suite.tasks) == 5
    assert suite.tasks[0].task_id == "pick_cube"

    articulated = Suite.load("articulated_object_v0")
    assert len(articulated.tasks) == 10

    maniskill = Suite.load("maniskill_smoke_v0")
    assert maniskill.tasks[0].success["engine_env_ids"]["maniskill"] == "PickCube-v1"

    focused = Suite.load("maniskill_manipulation_v0")
    assert [task.task_id for task in focused.tasks] == [
        "maniskill_pick_cube",
        "maniskill_stack_cube",
        "maniskill_push_cube",
    ]
    assert all(
        task.success["control_mode"] == "pd_ee_delta_pose" for task in focused.tasks
    )

    planner_bc = Suite.load("maniskill_planner_bc_v0")
    assert [task.task_id for task in planner_bc.tasks] == [
        "maniskill_pick_cube_joint",
        "maniskill_stack_cube_joint",
        "maniskill_push_cube_joint",
    ]
    assert all(
        task.success["control_mode"] == "pd_joint_pos" for task in planner_bc.tasks
    )

    mujoco = Suite.load("mujoco_control_v0")
    assert mujoco.tasks[0].success["engine_env_ids"]["mujoco"] == "Reacher-v5"


def test_suite_filters_tasks():
    suite = Suite.load("mujoco_control_v0").filter_tasks(["mujoco_pusher"])

    assert suite.suite_id == "mujoco_control_v0"
    assert [task.task_id for task in suite.tasks] == ["mujoco_pusher"]

    with pytest.raises(ValueError, match="does not contain requested task"):
        Suite.load("mujoco_control_v0").filter_tasks(["missing_task"])


def test_mujoco_adapter_falls_back_to_available_gym_version():
    from nyssa_bench.engines.mujoco_adapter import (
        _make_mujoco_env,
        _mujoco_env_id_candidates,
    )

    class VersionNotFound(Exception):
        pass

    class ErrorNamespace:
        pass

    ErrorNamespace.VersionNotFound = VersionNotFound

    class FakeGym:
        error = ErrorNamespace

        def __init__(self) -> None:
            self.requested: list[str] = []

        def make(self, env_id: str, **kwargs: Any) -> dict[str, Any]:
            self.requested.append(env_id)
            if env_id == "Reacher-v5":
                raise VersionNotFound(env_id)
            return {"env_id": env_id, "kwargs": kwargs}

    assert _mujoco_env_id_candidates("Reacher-v5") == [
        "Reacher-v5",
        "Reacher-v4",
        "Reacher-v2",
    ]

    gym = FakeGym()
    env = _make_mujoco_env(gym, "Reacher-v5", {"render_mode": "rgb_array"})

    assert env["env_id"] == "Reacher-v4"
    assert env["kwargs"] == {"render_mode": "rgb_array"}
    assert gym.requested == ["Reacher-v5", "Reacher-v4"]


def test_mujoco_adapter_defaults_to_egl_for_headless_rgb_rendering(monkeypatch):
    from nyssa_bench.engines.mujoco_adapter import _configure_headless_mujoco_rendering

    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.delenv("PYOPENGL_PLATFORM", raising=False)

    _configure_headless_mujoco_rendering("rgb_array")

    if os.name == "nt":
        assert "MUJOCO_GL" not in os.environ
    else:
        assert os.environ["MUJOCO_GL"] == "egl"
        assert os.environ["PYOPENGL_PLATFORM"] == "egl"


def test_mujoco_adapter_respects_explicit_renderer(monkeypatch):
    from nyssa_bench.engines.mujoco_adapter import _configure_headless_mujoco_rendering

    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setenv("MUJOCO_GL", "osmesa")
    monkeypatch.delenv("PYOPENGL_PLATFORM", raising=False)

    _configure_headless_mujoco_rendering("rgb_array")

    assert os.environ["MUJOCO_GL"] == "osmesa"
    assert "PYOPENGL_PLATFORM" not in os.environ


def test_maniskill_adapter_accepts_env_overrides(monkeypatch):
    from nyssa_bench.engines.maniskill_adapter import _maniskill_env_kwargs

    task = Suite.load("maniskill_smoke_v0").tasks[0]
    monkeypatch.setenv("NYSSA_MANISKILL_RENDER_MODE", "rgb_array")
    monkeypatch.setenv("NYSSA_MANISKILL_RENDER_DEVICE", "cpu")
    monkeypatch.setenv("NYSSA_MANISKILL_SIM_BACKEND", "cpu")

    kwargs = _maniskill_env_kwargs(task)

    assert kwargs["render_mode"] == "rgb_array"
    assert kwargs["render_device"] == "cpu"
    assert kwargs["sim_backend"] == "cpu"
    assert kwargs["obs_mode"] == task.success["obs_mode"]
    assert kwargs["control_mode"] == task.success["control_mode"]
    assert kwargs["max_episode_steps"] == task.success["max_steps"]


def test_maniskill_adapter_allows_max_episode_step_override(monkeypatch):
    from nyssa_bench.engines.maniskill_adapter import _maniskill_env_kwargs

    task = (
        Suite.load("maniskill_planner_bc_v0")
        .filter_tasks(["maniskill_stack_cube_joint"])
        .tasks[0]
    )
    monkeypatch.setenv("NYSSA_MANISKILL_MAX_EPISODE_STEPS", "180")

    kwargs = _maniskill_env_kwargs(task)

    assert kwargs["max_episode_steps"] == 180


def test_maniskill_adapter_can_disable_render_mode(monkeypatch):
    from nyssa_bench.engines.maniskill_adapter import _maniskill_env_kwargs

    task = Suite.load("maniskill_smoke_v0").tasks[0]
    monkeypatch.setenv("NYSSA_MANISKILL_RENDER_MODE", "none")

    kwargs = _maniskill_env_kwargs(task)

    assert "render_mode" not in kwargs


def test_maniskill_adapter_restores_structured_and_flat_states():
    from nyssa_bench.engines.maniskill_adapter import ManiSkillEngine

    class FakeActionSpace:
        shape = (1,)
        low = np.asarray([-1.0])
        high = np.asarray([1.0])
        dtype = np.float32

    class FakeUnwrapped:
        def __init__(self) -> None:
            self.state_dict = None
            self.state_vector = None

        def set_state_dict(self, state: Any) -> None:
            self.state_dict = state

        def set_state(self, state: Any) -> None:
            self.state_vector = state

        def get_obs(self) -> list[float]:
            return [0.0]

    class FakeEnv:
        action_space = FakeActionSpace()

        def __init__(self) -> None:
            self.unwrapped = FakeUnwrapped()

    engine = ManiSkillEngine()
    engine.env = FakeEnv()

    structured_observation = engine.set_state(
        {"env_states": {"actors": {"cube": [1.0, 2.0]}}}
    )
    flat_observation = engine.set_state([3.0, 4.0])

    assert engine.env.unwrapped.state_dict["actors"]["cube"].tolist() == [1.0, 2.0]
    assert engine.env.unwrapped.state_vector.tolist() == [3.0, 4.0]
    assert structured_observation["raw"] == [0.0]
    assert flat_observation["action_space"]["shape"] == [1]


def test_runner_restores_policy_initial_state_before_first_step():
    class StateRestoreEngine(NyssaEngine):
        max_steps = 1
        restored_states: list[Any] = []

        def __init__(self) -> None:
            self.state = None

        def load_task(self, task_spec: Any) -> None:
            self.task_spec = task_spec

        def reset(
            self, seed: int | None = None
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            return _observation(), {"seed": seed}

        def set_state(self, state: Any) -> dict[str, Any]:
            self.state = state
            self.restored_states.append(state)
            return _observation()

        def step(
            self, action: Any
        ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
            success = self.state == {"demo_state": [1.0, 2.0]}
            return _observation(), 1.0, True, False, {"success": success}

        def render(self) -> Any:
            return None

        def get_state(self) -> dict[str, Any]:
            return {"raw": self.state}

        def close(self) -> None:
            return None

    class StateReplayPolicy:
        def reset(self, task: Any | None = None, seed: int | None = None) -> None:
            return None

        def initial_state(self, observation: dict[str, Any]) -> dict[str, Any]:
            return {"demo_state": [1.0, 2.0]}

        def act(self, observation: dict[str, Any]) -> list[float]:
            return [0.0]

    StateRestoreEngine.restored_states = []
    get_plugin_registry().engines["state_restore_unit"] = StateRestoreEngine
    suite = Suite.load("tabletop_manipulation_v0").filter_tasks(["pick_cube"])

    report = PolicyRunner(
        policy=StateReplayPolicy(),
        engine="state_restore_unit",
        episodes=1,
        capture_replay=False,
    ).evaluate(suite)

    assert report.summary["success_rate"] == 1.0
    assert StateRestoreEngine.restored_states == [{"demo_state": [1.0, 2.0]}]


def test_runner_writes_artifacts(tmp_path: Path):
    _register_unit_engine()
    suite = Suite.load("tabletop_manipulation_v0")
    runner = PolicyRunner(
        policy="random", engine="unit_real", episodes=2, seed=123, out=tmp_path
    )
    report = runner.evaluate(suite)

    assert report.summary["episodes"] == 10
    assert 0.0 <= report.summary["success_rate"] <= 1.0
    assert len(report.summary["success_rate_ci95"]) == 2
    assert report.summary["benchmark_tier"] == "prototype"
    assert report.summary["public_claim"] is False
    assert (
        "minimum_episodes_per_task"
        in report.summary["public_claim_validation"]["failures"]
    )
    assert "pick_cube" in report.summary["per_task"]
    assert "success_rate_ci95" in report.summary["per_task"]["pick_cube"]
    assert report.summary["per_seed"]
    assert 0.0 <= report.summary["prototype_reliability_score"] <= 1.0
    assert report.summary["metrics"]["expert_intervention_rate"] == 0.0
    assert report.summary["metrics"]["recovery_success_rate"] == 0.0
    assert report.summary["metrics"]["verifier_rejection_rate"] == 0.0
    assert "unsupported_stressors" in report.summary["stressor_support"]
    assert (tmp_path / "config.yaml").exists()
    assert (tmp_path / "run.yaml").exists()
    assert (tmp_path / "environment.json").exists()
    assert (tmp_path / "package_versions.json").exists()
    assert (tmp_path / "git_info.json").exists()
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "episodes.json").exists()
    assert (tmp_path / "episodes.jsonl").exists()
    assert (tmp_path / "dataset_manifest.json").exists()
    assert (tmp_path / "failure_gallery.html").exists()
    assert (tmp_path / "recovery_dataset" / "manifest.json").exists()
    assert (tmp_path / "replay_manifest.json").exists()
    assert (tmp_path / "replay.html").exists()
    assert (tmp_path / "videos").is_dir()
    assert any((tmp_path / "videos").glob("*.mp4"))
    assert (tmp_path / "failures").is_dir()
    assert (tmp_path / "plots").is_dir()
    assert (tmp_path / "report.html").exists()


def test_runner_namespaces_episode_seeds_without_losing_task_pairing(tmp_path: Path):
    _register_unit_engine()
    runner = PolicyRunner(
        policy="random",
        engine="unit_real",
        episodes=2,
        seed=3,
        out=tmp_path,
        capture_replay=False,
    )

    runner.evaluate(Suite.load("maniskill_smoke_v0"))

    seeds_by_task = {
        task_id: [
            episode.seed
            for episode in runner.episode_results
            if episode.task_id == task_id
        ]
        for task_id in {episode.task_id for episode in runner.episode_results}
    }
    assert set(tuple(seeds) for seeds in seeds_by_task.values()) == {
        (3_000_000, 3_000_001)
    }
    assert runner.run_metadata["seed_protocol"]["format"] == "nyssa-episode-seed-v2"


def test_runner_records_expert_verifier_interventions(tmp_path: Path):
    _register_unit_engine()

    class RejectingExpert(ExpertProvider):
        provider_id = "rejecting_unit"

        def score_action(self, observation, action, *, task, engine=None):
            return ExpertActionScore(
                accepted=False, confidence=0.1, reason="unit_reject"
            )

        def act(self, observation, *, task, engine=None):
            return 0.0

        def recover(self, *, state, failure, task, engine=None):
            return [0.0]

    suite = Suite.load("maniskill_smoke_v0")
    runner = PolicyRunner(
        policy="random",
        engine="unit_real",
        episodes=1,
        seed=123,
        out=tmp_path,
        expert_provider=RejectingExpert(),
        enable_verifier=True,
        enable_recovery=True,
        capture_replay=False,
    )
    report = runner.evaluate(suite)

    assert report.summary["metrics"]["expert_intervention_rate"] == 1.0
    assert report.summary["metrics"]["verifier_rejection_rate"] == 1.0
    assert report.summary["metrics"]["expert_intervention_rate"] <= 1.0
    assert report.summary["metrics"]["recovery_success_rate"] == 1.0
    assert runner.run_metadata["expert_provider"]["provider_id"] == "rejecting_unit"
    episodes = (tmp_path / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    assert '"verifier_rejected": true' in episodes[0]


def test_runner_recovery_only_variant_uses_rejection_trigger_without_verifier(
    tmp_path: Path,
):
    class RecoveryOnlyEngine(UnitEngine):
        def step(
            self, action: Any
        ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
            self.last_action = float(np.asarray(action).reshape(-1)[0])
            return super().step(action)

    class RecoveryOnlyExpert(ExpertProvider):
        provider_id = "recovery_only"

        def score_action(self, observation, action, *, task, engine=None):
            return ExpertActionScore(accepted=False, confidence=1.0, reason="recover")

        def act(self, observation, *, task, engine=None):
            raise AssertionError(
                "recovery-only must not apply the verifier fallback action"
            )

        def recover(self, *, state, failure, task, engine=None):
            return [0.5]

    engine = RecoveryOnlyEngine()
    get_plugin_registry().engines["recovery_only_unit"] = lambda: engine
    runner = PolicyRunner(
        policy="random",
        engine="recovery_only_unit",
        episodes=1,
        seed=123,
        out=tmp_path,
        expert_provider=RecoveryOnlyExpert(),
        enable_verifier=False,
        enable_recovery=True,
        capture_replay=False,
    )

    report = runner.evaluate(
        Suite.load("maniskill_smoke_v0").filter_tasks(["maniskill_pick_cube"])
    )

    assert engine.last_action == 0.5
    assert report.summary["metrics"]["recovery_attempt_count"] == 1.0
    assert report.summary["metrics"]["action_rejection_count"] == 1.0
    assert report.summary["metrics"]["verifier_rejection_count"] == 0.0
    episode = (tmp_path / "episodes.json").read_text(encoding="utf-8")
    assert '"action_rejected": true' in episode
    assert '"verifier_rejected": false' in episode


def test_runner_keeps_failed_recovery_as_context_without_bc_target(tmp_path: Path):
    from nyssa_bench.datasets.recovery_training import load_recovery_episodes

    class NoPlanRecoveryExpert(ExpertProvider):
        provider_id = "no_plan_recovery"

        def score_action(self, observation, action, *, task, engine=None):
            return ExpertActionScore(accepted=False, confidence=1.0, reason="no_plan")

        def act(self, observation, *, task, engine=None):
            raise AssertionError(
                "recovery-only must not apply the verifier fallback action"
            )

        def recover(self, *, state, failure, task, engine=None):
            return None

    get_plugin_registry().engines["no_plan_recovery_unit"] = UnitEngine
    runner = PolicyRunner(
        policy="random",
        engine="no_plan_recovery_unit",
        episodes=1,
        seed=123,
        out=tmp_path,
        expert_provider=NoPlanRecoveryExpert(),
        enable_verifier=False,
        enable_recovery=True,
        capture_replay=False,
    )

    report = runner.evaluate(
        Suite.load("maniskill_smoke_v0").filter_tasks(["maniskill_pick_cube"])
    )

    manifest = json.loads(
        (tmp_path / "recovery_dataset" / "manifest.json").read_text(encoding="utf-8")
    )
    episodes_path = tmp_path / "recovery_dataset" / "episodes.json"
    recovery_episodes = json.loads(episodes_path.read_text(encoding="utf-8"))
    recovery_step = recovery_episodes[0]["steps"][0]
    assert manifest["format"] == "nyssa-recovery-dataset-v2"
    assert manifest["target_steps"] == 0
    assert manifest["context_steps"] == 1
    assert manifest["recovery_attempts"] == 1
    assert manifest["recovery_applied_attempts"] == 0
    assert manifest["recovery_successful_attempts"] == 0
    assert manifest["recovery_attempt_outcomes"] == {"not_applied": 1}
    assert recovery_step["record_type"] == "negative_context"
    assert recovery_step["executed_action_source"] == "policy"
    assert recovery_step["target_action"] is None
    assert recovery_step["target_source"] is None
    assert recovery_step["target_valid"] is False
    assert recovery_step["target_invalid_reason"] == "rejected_policy_action"
    assert recovery_step["info"]["recovery_outcome"] == "not_applied"
    assert report.summary["metrics"]["recovery_attempt_count"] == 1.0
    assert report.summary["metrics"]["recovery_applied_count"] == 0.0
    assert report.summary["metrics"]["recovery_not_applied_count"] == 1.0
    assert report.summary["metrics"]["recovery_success_rate"] == 0.0
    with pytest.raises(ValueError, match="No eligible expert or recovery targets"):
        load_recovery_episodes([episodes_path])


def test_runner_executes_recovery_macro_plan(tmp_path: Path):
    from nyssa_bench.datasets.recovery_training import load_recovery_episodes

    class MacroRecoveryEngine(UnitEngine):
        max_steps = 3

        def reset(
            self, seed: int | None = None
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            self.actions = []
            return _observation(), {"seed": seed}

        def step(
            self, action: Any
        ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
            value = float(np.asarray(action, dtype=float).reshape(-1)[0])
            self.actions.append(value)
            success = len(self.actions) >= 2 and self.actions[:2] == [0.25, 0.75]
            return (
                _observation(),
                1.0,
                success,
                False,
                {
                    "success": success,
                    "completion_time": float(len(self.actions)),
                    "path_efficiency": 1.0,
                    "grasp_success": success,
                },
            )

    class MacroRecoveryExpert(ExpertProvider):
        provider_id = "macro_recovery"

        def score_action(self, observation, action, *, task, engine=None):
            return ExpertActionScore(
                accepted=False, confidence=1.0, reason="needs_macro"
            )

        def act(self, observation, *, task, engine=None):
            return 0.0

        def recover(self, *, state, failure, task, engine=None):
            self.last_recovery_details = {"recovery_plan_label": "unit_macro"}
            return [0.25, 0.75]

    get_plugin_registry().engines["macro_recovery_unit"] = MacroRecoveryEngine
    suite = Suite.load("maniskill_smoke_v0").filter_tasks(["maniskill_pick_cube"])
    runner = PolicyRunner(
        policy="random",
        engine="macro_recovery_unit",
        episodes=1,
        seed=123,
        out=tmp_path,
        expert_provider=MacroRecoveryExpert(),
        enable_verifier=True,
        enable_recovery=True,
        recovery_attribution_horizon=1,
        capture_replay=False,
    )

    report = runner.evaluate(suite)

    assert report.summary["success_rate"] == 1.0
    assert report.summary["metrics"]["recovery_cached_action_count"] == 1.0
    assert report.summary["metrics"]["recovery_plan_action_count"] == 2.0
    episode = (tmp_path / "episodes.json").read_text(encoding="utf-8")
    assert '"action_source": "recovery"' in episode
    assert '"recovery_plan_label": "unit_macro"' in episode
    recovery_path = tmp_path / "recovery_dataset" / "episodes.json"
    recovery_manifest = json.loads(
        (tmp_path / "recovery_dataset" / "manifest.json").read_text(encoding="utf-8")
    )
    recovery_dataset = json.loads(recovery_path.read_text(encoding="utf-8"))
    recovery_steps = recovery_dataset[0]["steps"]
    assert len(recovery_steps) == 2
    assert all(step["target_valid"] is True for step in recovery_steps)
    assert all(step["target_source"] == "recovery" for step in recovery_steps)
    assert [step["target_action"] for step in recovery_steps] == [0.25, 0.75]
    assert all(step["info"]["recovery_outcome"] == "success" for step in recovery_steps)
    assert all(step["info"]["recovery_plan_success"] is True for step in recovery_steps)
    assert all(
        step["info"]["recovery_attribution_horizon_steps"] == 2
        for step in recovery_steps
    )
    assert recovery_manifest["recovery_attempts"] == 1
    assert recovery_manifest["recovery_applied_attempts"] == 1
    assert recovery_manifest["recovery_successful_attempts"] == 1
    assert recovery_manifest["recovery_success_rate"] == 1.0
    training_steps = load_recovery_episodes([recovery_path])[0]["steps"]
    assert [step["action"] for step in training_steps] == [0.25, 0.75]


def test_runner_records_zero_recovery_attempt_outcomes(tmp_path: Path):
    class AcceptingExpert(ExpertProvider):
        provider_id = "accepting_unit"

        def score_action(self, observation, action, *, task, engine=None):
            return ExpertActionScore(accepted=True, confidence=1.0, reason="accepted")

    get_plugin_registry().engines["zero_recovery_unit"] = UnitEngine
    runner = PolicyRunner(
        policy="random",
        engine="zero_recovery_unit",
        episodes=1,
        out=tmp_path,
        expert_provider=AcceptingExpert(),
        enable_recovery=True,
        capture_replay=False,
    )

    report = runner.evaluate(
        Suite.load("maniskill_smoke_v0").filter_tasks(["maniskill_pick_cube"])
    )

    metrics = report.summary["metrics"]
    assert metrics["recovery_attempt_count"] == 0.0
    assert metrics["recovery_applied_count"] == 0.0
    assert metrics["recovery_success_count"] == 0.0
    assert metrics["recovery_success_rate"] == 0.0
    assert metrics["recovery_episode_success_rate"] == 0.0


def test_runner_does_not_attribute_success_after_recovery_window(tmp_path: Path):
    class DelayedSuccessEngine(UnitEngine):
        max_steps = 4

        def reset(
            self, seed: int | None = None
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            self.step_count = 0
            return _observation(), {"seed": seed}

        def step(
            self, action: Any
        ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
            self.step_count += 1
            success = self.step_count == 3
            return (
                _observation(),
                1.0,
                success,
                False,
                {
                    "success": success,
                    "completion_time": float(self.step_count),
                },
            )

    class FirstStepRecoveryExpert(ExpertProvider):
        provider_id = "first_step_recovery"

        def score_action(self, observation, action, *, task, engine=None):
            return ExpertActionScore(
                accepted=bool(engine.step_count),
                confidence=1.0,
                reason="first_step_only",
            )

        def recover(self, *, state, failure, task, engine=None):
            return [0.5]

    get_plugin_registry().engines["delayed_success_unit"] = DelayedSuccessEngine
    runner = PolicyRunner(
        policy="random",
        engine="delayed_success_unit",
        episodes=1,
        out=tmp_path,
        expert_provider=FirstStepRecoveryExpert(),
        enable_recovery=True,
        recovery_attribution_horizon=2,
        capture_replay=False,
    )

    report = runner.evaluate(
        Suite.load("maniskill_smoke_v0").filter_tasks(["maniskill_pick_cube"])
    )

    metrics = report.summary["metrics"]
    assert report.summary["success_rate"] == 1.0
    assert metrics["recovery_attempt_count"] == 1.0
    assert metrics["recovery_applied_count"] == 1.0
    assert metrics["recovery_success_count"] == 0.0
    assert metrics["recovery_failure_count"] == 1.0
    assert metrics["recovery_success_rate"] == 0.0
    assert metrics["recovery_episode_success_rate"] == 0.0
    steps = runner.episode_results[0].steps
    assert steps[0].info["recovery_outcome"] == "window_expired"
    assert steps[0].info["recovery_outcome_step"] == 1
    assert steps[0].info["recovery_success"] is False
    assert steps[1].info["recovery_attribution_attempt_id"] == 1
    assert steps[2].info["recovery_attribution_attempt_id"] is None


def test_runner_attributes_only_latest_of_multiple_recovery_attempts(tmp_path: Path):
    class MultipleRecoveryEngine(UnitEngine):
        max_steps = 5

        def reset(
            self, seed: int | None = None
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            self.step_count = 0
            return _observation(), {"seed": seed}

        def step(
            self, action: Any
        ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
            self.step_count += 1
            success = self.step_count == 4
            return (
                _observation(),
                1.0,
                success,
                False,
                {
                    "success": success,
                    "completion_time": float(self.step_count),
                },
            )

    class MultipleRecoveryExpert(ExpertProvider):
        provider_id = "multiple_recovery"

        def score_action(self, observation, action, *, task, engine=None):
            rejected = engine.step_count in {0, 2}
            return ExpertActionScore(
                accepted=not rejected, confidence=1.0, reason="scheduled"
            )

        def recover(self, *, state, failure, task, engine=None):
            return [0.25]

    get_plugin_registry().engines["multiple_recovery_unit"] = MultipleRecoveryEngine
    runner = PolicyRunner(
        policy="random",
        engine="multiple_recovery_unit",
        episodes=1,
        out=tmp_path,
        expert_provider=MultipleRecoveryExpert(),
        enable_recovery=True,
        recovery_attribution_horizon=3,
        capture_replay=False,
    )

    report = runner.evaluate(
        Suite.load("maniskill_smoke_v0").filter_tasks(["maniskill_pick_cube"])
    )

    metrics = report.summary["metrics"]
    assert metrics["recovery_attempt_count"] == 2.0
    assert metrics["recovery_applied_count"] == 2.0
    assert metrics["recovery_success_count"] == 1.0
    assert metrics["recovery_failure_count"] == 1.0
    assert metrics["recovery_success_rate"] == 0.5
    assert metrics["recovery_episode_applied_count"] == 1.0
    assert metrics["recovery_episode_success_count"] == 1.0
    assert metrics["recovery_episode_success_rate"] == 1.0
    steps = runner.episode_results[0].steps
    assert steps[0].info["recovery_attempt_id"] == 1
    assert steps[0].info["recovery_outcome"] == "superseded"
    assert steps[0].info["recovery_success"] is False
    assert steps[2].info["recovery_attempt_id"] == 2
    assert steps[2].info["recovery_outcome"] == "success"
    assert steps[2].info["recovery_outcome_step"] == 3
    assert steps[3].info["recovery_attribution_attempt_id"] == 2
    assert steps[3].info["recovery_success"] is True


def test_builtin_expert_providers_emit_actions():
    task = Suite.load("maniskill_smoke_v0").tasks[0]
    observation = _observation()

    bounds = make_expert_provider("bounds-verifier")
    assert bounds.score_action(observation, [2.0], task=task).accepted is False
    assert bounds.act(observation, task=task) is not None

    mujoco = make_expert_provider("mujoco-heuristic")
    assert mujoco.act(observation, task=task) is not None
    assert "recover" in mujoco.metadata()["capabilities"]

    scripted = make_expert_provider("maniskill-scripted")
    assert scripted.act(observation, task=task) is not None
    assert scripted.metadata()["provider_id"] == "maniskill-scripted"


def test_maniskill_scripted_pick_cube_closes_near_low_grasp_target():
    from nyssa_bench.baselines.scripted_maniskill import ManiSkillScriptedHeuristic

    task = (
        Suite.load("maniskill_smoke_v0").filter_tasks(["maniskill_pick_cube"]).tasks[0]
    )
    controller = ManiSkillScriptedHeuristic()
    controller.reset(task=task)
    observation = _maniskill_state_observation(
        {
            "is_grasped": [False],
            "tcp_pose": [[0.0, 0.0, 0.041, 1.0, 0.0, 0.0, 0.0]],
            "obj_pose": [[0.0, 0.0, 0.02, 1.0, 0.0, 0.0, 0.0]],
            "goal_pos": [[0.1, 0.1, 0.28]],
        }
    )

    action = np.asarray(controller.act(observation), dtype=float).reshape(-1)

    assert action[2] < 0.0
    assert action[-1] == -1.0


def test_maniskill_scripted_push_cube_moves_behind_then_through_object():
    from nyssa_bench.baselines.scripted_maniskill import ManiSkillScriptedHeuristic

    task = (
        Suite.load("maniskill_smoke_v0").filter_tasks(["maniskill_push_cube"]).tasks[0]
    )
    controller = ManiSkillScriptedHeuristic()
    controller.reset(task=task)
    observation = _maniskill_state_observation(
        {
            "tcp_pose": [[-0.075, 0.0, 0.038, 1.0, 0.0, 0.0, 0.0]],
            "obj_pose": [[0.0, 0.0, 0.02, 1.0, 0.0, 0.0, 0.0]],
            "goal_pos": [[0.2, 0.0, 0.001]],
        }
    )

    action = np.asarray(controller.act(observation), dtype=float).reshape(-1)

    assert action[0] > 0.0
    assert abs(action[1]) < 1e-6
    assert action[-1] == 1.0


def test_maniskill_scripted_stack_cube_uses_cube_a_and_cube_b_keys():
    from nyssa_bench.baselines.scripted_maniskill import ManiSkillScriptedHeuristic

    task = (
        Suite.load("maniskill_smoke_v0").filter_tasks(["maniskill_stack_cube"]).tasks[0]
    )
    controller = ManiSkillScriptedHeuristic()
    controller.reset(task=task)
    observation = _maniskill_state_observation(
        {
            "is_cubeA_grasped": [True],
            "tcp_pose": [[0.0, 0.0, 0.12, 1.0, 0.0, 0.0, 0.0]],
            "cubeA_pose": [[0.0, 0.0, 0.08, 1.0, 0.0, 0.0, 0.0]],
            "cubeB_pose": [[0.1, 0.0, 0.02, 1.0, 0.0, 0.0, 0.0]],
        }
    )

    action = np.asarray(controller.act(observation), dtype=float).reshape(-1)

    assert action[0] > 0.0
    assert action[2] < 0.0
    assert action[-1] == -1.0


def test_mujoco_expert_uses_rollout_scoring_and_restores_state():
    task = Suite.load("mujoco_control_v0").tasks[0]
    expert = make_expert_provider("mujoco-heuristic")
    engine = _FakeMuJoCoEngine()
    observation = {
        "raw": [1.0, -1.0],
        "action_space": {
            "type": "box",
            "shape": [2],
            "low": [-1.0, -1.0],
            "high": [1.0, 1.0],
        },
    }

    score = expert.score_action(observation, [1.0, 1.0], task=task, engine=engine)
    recovery = expert.recover(
        state={"observation": observation},
        failure="bad_action",
        task=task,
        engine=engine,
    )

    assert score.accepted is False
    assert score.reason == "lower_than_candidate_reward"
    assert score.details is not None
    assert score.details["rollout_horizon"] == 3
    assert score.details["score_gap"] > score.details["rollout_margin"]
    assert recovery is None
    assert expert.last_recovery_details is not None
    assert expert.last_recovery_details["recovery_disabled"] is True
    assert expert.last_recovery_details["task_id"] == "mujoco_reacher"
    assert engine.env.unwrapped.data.qpos.tolist() == [0.0, 0.0]
    assert engine.env.unwrapped.data.qvel.tolist() == [0.0, 0.0]
    assert engine.elapsed_steps == 0
    assert engine.episode_return == 0.0


def test_mujoco_recovery_defaults_to_pusher_only():
    from nyssa_bench.experts.base import MuJoCoHeuristicExpertProvider

    class Plan:
        label = "pusher_approach_then_push"
        sequence = [np.asarray([0.1]), np.asarray([0.2]), np.asarray([0.3])]
        details = {"recovery_plan_label": label}

    suite = Suite.load("mujoco_control_v0")
    reacher = suite.filter_tasks(["mujoco_reacher"]).tasks[0]
    pusher = suite.filter_tasks(["mujoco_pusher"]).tasks[0]
    expert = MuJoCoHeuristicExpertProvider(rollout_horizon=3)
    observation = {
        "raw": [0.0],
        "action_space": {
            "type": "box",
            "shape": [1],
            "low": [-1.0],
            "high": [1.0],
        },
    }
    expert._best_rollout_plan = lambda *args, **kwargs: Plan()  # type: ignore[method-assign]

    reacher_recovery = expert.recover(
        state={"observation": observation},
        failure="bad_action",
        task=reacher,
        engine=object(),
    )
    pusher_recovery = expert.recover(
        state={"observation": observation},
        failure="bad_action",
        task=pusher,
        engine=object(),
    )

    assert reacher_recovery is None
    assert pusher_recovery is not None
    assert len(pusher_recovery) == 3
    assert expert.metadata()["recovery_task_ids"] == ["mujoco_pusher"]


def test_mujoco_recovery_task_allowlist_can_enable_all(monkeypatch):
    from nyssa_bench.experts.base import MuJoCoHeuristicExpertProvider

    class Plan:
        label = "single_action_0"
        sequence = [np.asarray([0.1]), np.asarray([0.2])]
        details = {"recovery_plan_label": label}

    monkeypatch.setenv("NYSSA_MUJOCO_RECOVERY_TASKS", "all")
    task = Suite.load("mujoco_control_v0").filter_tasks(["mujoco_reacher"]).tasks[0]
    expert = MuJoCoHeuristicExpertProvider(rollout_horizon=2)
    observation = {
        "raw": [0.0],
        "action_space": {
            "type": "box",
            "shape": [1],
            "low": [-1.0],
            "high": [1.0],
        },
    }
    expert._best_rollout_plan = lambda *args, **kwargs: Plan()  # type: ignore[method-assign]

    recovery = expert.recover(
        state={"observation": observation},
        failure="bad_action",
        task=task,
        engine=object(),
    )

    assert recovery is not None
    assert len(recovery) == 2
    assert expert.metadata()["recovery_task_ids"] == ["all"]


def test_mujoco_expert_accepts_near_expert_action_inside_rollout_margin():
    from nyssa_bench.experts.base import MuJoCoHeuristicExpertProvider

    task = Suite.load("mujoco_control_v0").tasks[0]
    expert = MuJoCoHeuristicExpertProvider(rollout_margin=0.25, rollout_horizon=3)
    engine = _FakeMuJoCoEngine()
    observation = {
        "raw": [1.0, -1.0],
        "action_space": {
            "type": "box",
            "shape": [2],
            "low": [-1.0, -1.0],
            "high": [1.0, 1.0],
        },
    }

    score = expert.score_action(observation, [-0.95, 1.0], task=task, engine=engine)

    assert score.accepted is True
    assert score.reason is None
    assert score.details is not None
    assert 0.0 < score.details["score_gap"] <= score.details["rollout_margin"]
    assert engine.env.unwrapped.data.qpos.tolist() == [0.0, 0.0]
    assert engine.env.unwrapped.data.qvel.tolist() == [0.0, 0.0]


def test_mujoco_expert_samples_random_action_sequences():
    from nyssa_bench.experts.base import MuJoCoHeuristicExpertProvider

    task = Suite.load("mujoco_control_v0").tasks[0]
    expert = MuJoCoHeuristicExpertProvider(
        rollout_margin=0.25, rollout_horizon=2, candidate_count=4, random_seed=123
    )
    engine = _FakeMuJoCoEngine()
    observation = {
        "raw": [1.0, -1.0],
        "action_space": {
            "type": "box",
            "shape": [2],
            "low": [-1.0, -1.0],
            "high": [1.0, 1.0],
        },
    }

    expert.reset(task=task, seed=5, engine=engine)
    score = expert.score_action(observation, [1.0, 1.0], task=task, engine=engine)

    assert score.details is not None
    assert score.details["rollout_horizon"] == 2
    assert score.details["candidate_count"] >= 4
    assert expert.metadata()["candidate_count"] == 4
    assert engine.env.unwrapped.data.qpos.tolist() == [0.0, 0.0]


def test_mujoco_pusher_expert_uses_task_shaped_rollout_score():
    from nyssa_bench.experts.base import MuJoCoHeuristicExpertProvider

    task = Suite.load("mujoco_control_v0").filter_tasks(["mujoco_pusher"]).tasks[0]
    expert = MuJoCoHeuristicExpertProvider(
        rollout_margin=0.25,
        rollout_horizon=2,
        candidate_count=0,
        pusher_shaping_scale=7.0,
    )
    engine = _FakeMuJoCoEngine()
    observation = {
        "raw": [1.0, -1.0],
        "action_space": {
            "type": "box",
            "shape": [2],
            "low": [-1.0, -1.0],
            "high": [1.0, 1.0],
        },
    }

    score = expert.score_action(observation, [1.0, 1.0], task=task, engine=engine)

    assert score.accepted is False
    assert score.details is not None
    assert score.details["rollout_score_kind"] == "task_shaped_return"
    assert score.details["pusher_shaping_scale"] == 7.0
    assert score.details["adaptive_margin_enabled"] is True
    assert score.details["effective_rollout_margin"] <= score.details["rollout_margin"]
    assert score.details["candidate_return_spread"] is not None
    assert score.details["candidate_top_return_spread"] is not None
    assert score.details["margin_top_count"] is not None
    assert score.details["margin_top_k"] == 2
    assert expert.metadata()["pusher_shaping_scale"] == 7.0
    assert expert.metadata()["adaptive_margin"] == "auto"
    assert expert.metadata()["margin_top_k"] == 2
    assert engine.env.unwrapped.data.qpos.tolist() == [0.0, 0.0]


def test_mujoco_rollout_score_prioritizes_success_threshold():
    from nyssa_bench.experts.base import _evaluate_mujoco_action_sequence

    task = Suite.load("mujoco_control_v0").filter_tasks(["mujoco_pusher"]).tasks[0]
    engine = _FakeMuJoCoEngine()

    success_score = _evaluate_mujoco_action_sequence(engine, [[-1.0, 1.0]], task=task)
    miss_score = _evaluate_mujoco_action_sequence(engine, [[-0.4, 0.4]], task=task)

    assert success_score is not None
    assert miss_score is not None
    assert success_score > miss_score + 50.0
    assert engine.env.unwrapped.data.qpos.tolist() == [0.0, 0.0]
    assert engine.env.unwrapped.data.qvel.tolist() == [0.0, 0.0]


def test_mujoco_pusher_adaptive_margin_tracks_candidate_spread():
    from nyssa_bench.experts.base import MuJoCoHeuristicExpertProvider

    pusher = Suite.load("mujoco_control_v0").filter_tasks(["mujoco_pusher"]).tasks[0]
    reacher = Suite.load("mujoco_control_v0").filter_tasks(["mujoco_reacher"]).tasks[0]
    expert = MuJoCoHeuristicExpertProvider(
        rollout_margin=0.25,
        margin_fraction=0.25,
        min_margin=1e-6,
        adaptive_margin="auto",
    )

    margin, details = expert._effective_rollout_margin(
        pusher,
        [-7.539147, -7.539038, -10.0, -15.0, -21.0, -7.539070, -7.539080, -30.0],
    )
    fixed_margin, fixed_details = expert._effective_rollout_margin(
        reacher, [-7.539147, -7.539038]
    )

    assert details["adaptive_margin_enabled"] is True
    assert details["candidate_return_spread"] > 20.0
    assert details["candidate_top_return_spread"] == pytest.approx(0.000032)
    assert details["margin_top_count"] == 2
    assert details["margin_top_k"] == 2
    assert margin == pytest.approx(0.000008)
    assert fixed_details["adaptive_margin_enabled"] is False
    assert fixed_details["candidate_return_spread"] is None
    assert fixed_margin == 0.25


def test_mujoco_pusher_adaptive_margin_can_use_top_fraction_when_top_k_disabled():
    from nyssa_bench.experts.base import MuJoCoHeuristicExpertProvider

    pusher = Suite.load("mujoco_control_v0").filter_tasks(["mujoco_pusher"]).tasks[0]
    expert = MuJoCoHeuristicExpertProvider(
        rollout_margin=0.25,
        margin_fraction=0.25,
        margin_top_k=0,
        margin_top_fraction=0.25,
        min_margin=1e-6,
        adaptive_margin="auto",
    )

    margin, details = expert._effective_rollout_margin(
        pusher,
        [-7.539147, -7.539038, -10.0, -15.0, -21.0, -7.539070, -7.539080, -30.0],
    )

    assert details["margin_top_k"] == 0
    assert details["margin_top_count"] == 2
    assert margin == pytest.approx(0.000008)


def test_mujoco_pusher_guided_sequences_use_body_geometry_and_restore_state():
    from nyssa_bench.experts.base import MuJoCoHeuristicExpertProvider

    task = Suite.load("mujoco_control_v0").filter_tasks(["mujoco_pusher"]).tasks[0]
    expert = MuJoCoHeuristicExpertProvider(rollout_horizon=5, candidate_count=0)
    engine = _FakeMuJoCoEngine()
    observation = {
        "raw": [1.0, -1.0],
        "action_space": {
            "type": "box",
            "shape": [2],
            "low": [-1.0, -1.0],
            "high": [1.0, 1.0],
        },
    }

    sequences = expert._pusher_guided_action_sequences(
        observation, task=task, engine=engine
    )

    assert len(sequences) >= 3
    assert all(len(sequence) == 5 for sequence in sequences)
    assert any(np.linalg.norm(np.asarray(sequence[0])) > 0.0 for sequence in sequences)
    assert engine.env.unwrapped.data.qpos.tolist() == [0.0, 0.0]
    assert engine.env.unwrapped.data.qvel.tolist() == [0.0, 0.0]


def test_mujoco_pusher_scaled_guided_plans_emit_stable_labels():
    from nyssa_bench.experts.base import MuJoCoHeuristicExpertProvider

    task = Suite.load("mujoco_control_v0").filter_tasks(["mujoco_pusher"]).tasks[0]
    expert = MuJoCoHeuristicExpertProvider(rollout_horizon=5, candidate_count=0)
    engine = _FakeMuJoCoEngine()
    observation = {
        "raw": [1.0, -1.0],
        "action_space": {
            "type": "box",
            "shape": [2],
            "low": [-1.0, -1.0],
            "high": [1.0, 1.0],
        },
    }

    plans = expert._pusher_guided_rollout_plans(observation, task=task, engine=engine)
    labels = {plan.label for plan in plans}

    assert "pusher_push_s1" in labels
    assert "pusher_approach_then_push_s1_split2" in labels
    assert "pusher_alternating_approach_push_s2" in labels
    assert "pusher_finish_settle" in labels
    assert "pusher_finish_push_settle_s0p1_pulse1" in labels
    assert all(len(plan.sequence) == 5 for plan in plans)
    assert expert.metadata()["pusher_action_scales"] == [0.5, 1.0, 1.5, 2.0]
    assert expert.metadata()["pusher_finish_scales"] == [0.05, 0.1, 0.2, 0.35]
    assert expert.metadata()["pusher_planning_horizon"] == 15


def test_mujoco_pusher_candidate_plans_use_longer_planning_horizon():
    from nyssa_bench.experts.base import MuJoCoHeuristicExpertProvider

    task = Suite.load("mujoco_control_v0").filter_tasks(["mujoco_pusher"]).tasks[0]
    expert = MuJoCoHeuristicExpertProvider(rollout_horizon=5, candidate_count=0)
    engine = _FakeMuJoCoEngine()
    observation = {
        "raw": [1.0, -1.0],
        "action_space": {
            "type": "box",
            "shape": [2],
            "low": [-1.0, -1.0],
            "high": [1.0, 1.0],
        },
    }

    plans = expert._candidate_rollout_plans(
        observation, include_zero=True, task=task, engine=engine
    )

    assert any(plan.label.startswith("pusher_push") for plan in plans)
    assert {len(plan.sequence) for plan in plans} == {15}


def test_mujoco_pusher_recovery_commits_only_sequential_macros():
    from nyssa_bench.experts.base import MuJoCoHeuristicExpertProvider

    class Plan:
        def __init__(self, label: str) -> None:
            self.label = label
            self.sequence = [np.asarray([0.1]), np.asarray([0.2]), np.asarray([0.3])]
            self.details = {"recovery_plan_label": label}

    task = Suite.load("mujoco_control_v0").filter_tasks(["mujoco_pusher"]).tasks[0]
    expert = MuJoCoHeuristicExpertProvider(rollout_horizon=3)
    observation = {
        "raw": [0.0],
        "action_space": {
            "type": "box",
            "shape": [1],
            "low": [-1.0],
            "high": [1.0],
        },
    }

    expert._best_rollout_plan = lambda *args, **kwargs: Plan("pusher_push")  # type: ignore[method-assign]
    push_recovery = expert.recover(
        state={"observation": observation},
        failure="bad_action",
        task=task,
        engine=object(),
    )

    assert push_recovery is not None
    assert len(push_recovery) == 1
    assert expert.last_recovery_details is not None
    assert expert.last_recovery_details["recovery_plan_committed"] is False
    assert expert.last_recovery_details["recovery_plan_candidate_length"] == 3

    expert._best_rollout_plan = lambda *args, **kwargs: Plan(
        "pusher_approach_then_push"
    )  # type: ignore[method-assign]
    mixed_recovery = expert.recover(
        state={"observation": observation},
        failure="bad_action",
        task=task,
        engine=object(),
    )

    assert mixed_recovery is not None
    assert len(mixed_recovery) == 3
    assert expert.last_recovery_details is not None
    assert expert.last_recovery_details["recovery_plan_committed"] is True


def test_runner_executes_action_chunks(tmp_path: Path):
    class TwoStepEngine(UnitEngine):
        max_steps = 2

        def reset(
            self, seed: int | None = None
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            self.elapsed = 0
            return _observation(), {"seed": seed}

        def step(
            self, action: Any
        ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
            self.elapsed += 1
            return (
                _observation(),
                1.0,
                self.elapsed >= 2,
                False,
                {
                    "success": self.elapsed >= 2,
                    "completion_time": float(self.elapsed),
                    "path_efficiency": 1.0,
                    "grasp_success": True,
                },
            )

    get_plugin_registry().engines["chunk_unit"] = TwoStepEngine

    class ChunkPolicy:
        def __init__(self) -> None:
            self.calls = 0

        def act(self, observation):
            self.calls += 1
            return np.asarray([[0.1], [0.2], [0.3]], dtype=float)

    policy = ChunkPolicy()
    suite = Suite.load("maniskill_smoke_v0")
    runner = PolicyRunner(
        policy=policy,
        engine="chunk_unit",
        episodes=1,
        seed=123,
        out=tmp_path,
        capture_replay=False,
        policy_action_horizon=3,
        policy_execution_horizon=2,
    )
    report = runner.evaluate(suite)

    assert policy.calls == 3
    assert report.summary["metrics"]["policy_action_chunk_count"] == 1.0
    assert report.summary["metrics"]["policy_cached_action_count"] == 1.0
    assert runner.episode_results[0].steps[0].info["policy_action_chunk_size"] == 2
    assert runner.episode_results[0].steps[1].info["policy_cached_action"] is True


class _FakeMuJoCoData:
    def __init__(self) -> None:
        self.qpos = np.asarray([0.0, 0.0], dtype=float)
        self.qvel = np.asarray([0.0, 0.0], dtype=float)
        self.time = 0.0


class _FakeMuJoCoUnwrapped:
    def __init__(self) -> None:
        self.data = _FakeMuJoCoData()
        self.model = None

    def set_state(self, qpos, qvel) -> None:
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel

    def get_body_com(self, name: str):
        if name in {"object", "obj", "puck", "object0"}:
            return np.asarray([self.data.qpos[0], self.data.qpos[1], 0.0], dtype=float)
        if name in {"goal", "target"}:
            return np.asarray([-1.0, 1.0, 0.0], dtype=float)
        if name in {"tips_arm", "fingertip", "tip", "end_effector"}:
            return np.asarray(
                [self.data.qpos[0] + 0.1, self.data.qpos[1] - 0.1, 0.0], dtype=float
            )
        raise KeyError(name)


class _FakeMuJoCoEnv:
    def __init__(self) -> None:
        self.unwrapped = _FakeMuJoCoUnwrapped()
        self._elapsed_steps = 0

    def step(self, action):
        action_array = np.asarray(action, dtype=float).reshape(-1)
        self.unwrapped.data.qpos += action_array
        self.unwrapped.data.qvel[:] = action_array
        self.unwrapped.data.time += 1.0
        self._elapsed_steps += 1
        reward = -float(np.linalg.norm(action_array - np.asarray([-1.0, 1.0])))
        return {}, reward, False, False, {}


class _FakeMuJoCoEngine:
    def __init__(self) -> None:
        self.env = _FakeMuJoCoEnv()
        self.episode_return = 0.0
        self.elapsed_steps = 0


def test_public_claim_requires_replay_video_evidence(tmp_path: Path):
    suite = Suite.load("mujoco_control_v0")
    episodes = []
    for task in suite.tasks:
        for index in range(100):
            episodes.append(
                EpisodeResult(
                    task_id=task.task_id,
                    episode_index=index,
                    seed=index,
                    success=False,
                    failure_label="timeout",
                    failure_label_source="mapper",
                    metrics={},
                    replay_path="videos/missing.mp4",
                    steps=[
                        StepRecord(
                            observation={},
                            action=0.0,
                            reward=0.0,
                            terminated=False,
                            truncated=True,
                            info={},
                        )
                    ],
                )
            )

    validation = RunClaimValidator().validate(
        suite=suite,
        engine_name="mujoco",
        episodes_per_task=100,
        episodes=episodes,
        out_dir=tmp_path,
        package_versions={"mujoco": "test"},
        git_info={"commit": "test"},
    )

    assert validation.public_claim is False
    assert "replay_video_evidence" in validation.failures
    assert "git_worktree_clean" in validation.failures


def test_stable_validation_import_path():
    assert StableRunClaimValidator is RunClaimValidator


def test_runner_loads_duck_typed_policy_file(tmp_path: Path):
    _register_unit_engine()
    policy_path = tmp_path / "user_policy.py"
    policy_path.write_text(
        """
class PolicyAdapter:
    def act(self, observation):
        return 0.2
""".strip(),
        encoding="utf-8",
    )

    suite = Suite.load("tabletop_manipulation_v0")
    runner = PolicyRunner(
        policy=str(policy_path), engine="unit_real", episodes=1, seed=123
    )
    report = runner.evaluate(suite)

    assert report.summary["episodes"] == 5


def test_external_policy_requires_real_model():
    from nyssa_bench.policies.openvla_adapter import OpenVLAPolicy

    with pytest.raises(RuntimeError, match="NYSSA_OPENVLA_POLICY"):
        OpenVLAPolicy()

    from nyssa_bench.policies.bc_policy import BCPolicy
    from nyssa_bench.policies.scripted_oracle_policy import ScriptedOraclePolicy

    scripted = ScriptedOraclePolicy()
    action = scripted.act(_observation())
    assert action is not None

    with pytest.raises(RuntimeError, match="BC checkpoint not found"):
        BCPolicy()

    with pytest.raises(RuntimeError, match="RoboMimic checkpoint not found"):
        RoboMimicPolicy()


def test_failure_mapper_does_not_default_to_first_label():
    task = Suite.load("maniskill_smoke_v0").tasks[0]
    classification = FailureMapper().classify({}, task_spec=task, step_count=0)

    assert classification.label == "missed_target"
    assert classification.source == "mapper"

    unknown_task = task.__class__(
        task_id="unknown_failure_task",
        engine=task.engine,
        robot=task.robot,
        scene=task.scene,
        description=task.description,
        success=task.success,
        failure_labels=["bad_grasp"],
    )
    unknown = FailureMapper().classify({}, task_spec=unknown_task, step_count=0)
    assert unknown.label == "unknown_failure"


def test_failure_mapper_labels_terminal_mujoco_instability():
    task = (
        Suite.load("mujoco_control_v0")
        .filter_tasks(["mujoco_inverted_pendulum"])
        .tasks[0]
    )

    classification = FailureMapper().classify(
        {}, task_spec=task, step_count=12, terminated=True
    )

    assert classification.label == "unstable_contact"
    assert classification.source == "mapper"
