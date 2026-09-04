from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from nyssa_bench.core.episode import EpisodeResult
from nyssa_bench.core.suite import Suite
from nyssa_bench.core.task import TaskSpec
from nyssa_bench.engines.base import NyssaEngine
from nyssa_bench.experts.base import ExpertProvider
from nyssa_bench.plugins import get_plugin_registry
from nyssa_bench.recovery import CounterfactualBranchRunner
from nyssa_bench.runner import PolicyRunner
from nyssa_bench.runners import (
    EpisodeComponents,
    EpisodeRequest,
    EpisodeRunner,
    ExperimentCell,
    ExperimentRunner,
    LifecycleContext,
    LifecycleExecutionError,
    TransitionLifecycle,
    TransitionResult,
    ablation_cells,
    policy_seed_cells,
)


class _Hook:
    component_id = "unit_hook"

    def __init__(self, events: list[str], *, fail_phase: str | None = None) -> None:
        self.events = events
        self.fail_phase = fail_phase

    def on_lifecycle(self, context: LifecycleContext, payload: Any) -> None:
        del payload
        self.events.append(f"hook:{context.phase}:{context.branch_kind}")
        if context.phase == self.fail_phase:
            raise RuntimeError(f"failed in {context.phase}")


class _Stressor:
    def __init__(self, events: list[str], *, fail_transform: bool = False) -> None:
        self.events = events
        self.fail_transform = fail_transform

    def before_step(self, engine: Any, *, step_index: int) -> None:
        del engine, step_index
        self.events.append("stressor:before_step")

    def transform_action(
        self, action: Any, *, observation: Any, step_index: int
    ) -> Any:
        del observation, step_index
        self.events.append("stressor:transform_action")
        if self.fail_transform:
            raise ValueError("bad action transform")
        return action + 1

    def after_step(
        self, engine: Any, info: dict[str, Any], *, step_index: int
    ) -> None:
        del engine, info, step_index
        self.events.append("stressor:after_step")

    def transform_observation(self, observation: Any, *, step_index: int) -> Any:
        del step_index
        self.events.append("stressor:transform_observation")
        return {**observation, "transformed": True}


class _StepEngine:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def step(self, action: Any):
        self.events.append("engine:step")
        return {"raw": action}, 1.0, False, False, {"success": False}


def test_transition_lifecycle_has_stable_order_and_context() -> None:
    events: list[str] = []
    lifecycle = TransitionLifecycle([_Hook(events)])

    result = lifecycle.execute(
        engine=_StepEngine(events),
        stressors=_Stressor(events),
        observation={"raw": 0},
        action=2,
        task_id="unit_task",
        episode_index=0,
        episode_seed=4,
        step_index=3,
        branch_kind="continue",
        after_engine_step=lambda info: events.append("callback:engine"),
        after_stressors=lambda transition: events.append("callback:stressors"),
    )

    assert events == [
        "hook:before_step:continue",
        "stressor:before_step",
        "stressor:transform_action",
        "engine:step",
        "hook:after_engine_step:continue",
        "callback:engine",
        "stressor:after_step",
        "stressor:transform_observation",
        "callback:stressors",
        "hook:after_step:continue",
    ]
    assert result.action_before_stressors == 2
    assert result.action == 3
    assert result.observation == {"raw": 3, "transformed": True}


def test_transition_failure_reports_component_phase_task_and_step() -> None:
    lifecycle = TransitionLifecycle()

    with pytest.raises(LifecycleExecutionError) as caught:
        lifecycle.execute(
            engine=_StepEngine([]),
            stressors=_Stressor([], fail_transform=True),
            observation={"raw": 0},
            action=2,
            task_id="unit_task",
            episode_index=0,
            episode_seed=4,
            step_index=3,
        )

    error = caught.value
    assert error.component_id == "stressor_pipeline"
    assert error.phase == "before_step"
    assert error.task_id == "unit_task"
    assert error.step_index == 3
    assert "support=supported" in str(error)


def test_episode_finalization_does_not_mask_primary_failure() -> None:
    events: list[str] = []
    hook = _Hook(events, fail_phase="episode_finalize")

    def fail(request: EpisodeRequest, components: EpisodeComponents) -> EpisodeResult:
        del request, components
        raise ValueError("episode failed")

    runner = EpisodeRunner(fail, hooks=[hook])
    request = EpisodeRequest(SimpleNamespace(task_id="unit_task"), 0, 1)
    components = EpisodeComponents(
        engine=SimpleNamespace(),
        policy=SimpleNamespace(),
        expert=SimpleNamespace(),
        stressor_factory=lambda request: None,
        detector_factory=lambda request: None,
    )

    with pytest.raises(LifecycleExecutionError) as caught:
        runner.run(request, components)

    assert caught.value.component_id == "episode_executor"
    finalize_error = getattr(caught.value, "lifecycle_finalize_error")
    assert isinstance(finalize_error, LifecycleExecutionError)
    assert finalize_error.phase == "episode_finalize"
    assert events == ["hook:episode_start:None", "hook:episode_finalize:None"]


class _LifecycleEngine(NyssaEngine):
    max_steps = 1

    def load_task(self, task_spec: TaskSpec) -> None:
        self.task = task_spec

    def reset(self, seed: int | None = None):
        return {"raw": [0.0]}, {"seed": seed}

    def step(self, action: Any):
        return {"raw": [action]}, 1.0, True, False, {"success": True}

    def render(self) -> Any:
        return None

    def get_state(self) -> dict[str, Any]:
        return {}

    def close(self) -> None:
        return None


class _Policy:
    def reset(self, task: Any = None, seed: int | None = None) -> None:
        del task, seed

    def act(self, observation: dict[str, Any]) -> float:
        del observation
        return 0.25

    def close(self) -> None:
        return None


class _Recorder:
    recorder_id = "unit_recorder"

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def reset(self, context: LifecycleContext) -> None:
        self.events.append(f"recorder:reset:{context.phase}")

    def record_transition(
        self, context: LifecycleContext, transition: TransitionResult
    ) -> None:
        self.events.append(f"recorder:step:{context.step_index}:{transition.reward}")

    def finalize(
        self, context: LifecycleContext, episode: EpisodeResult
    ) -> dict[str, float]:
        self.events.append(f"recorder:finalize:{context.phase}")
        return {"unit_recorded_metric": float(len(episode.steps))}


def test_policy_runner_injects_hooks_and_metric_recorders(tmp_path: Path) -> None:
    get_plugin_registry().engines["lifecycle_unit"] = _LifecycleEngine
    task = TaskSpec(
        task_id="lifecycle_task",
        engine="lifecycle_unit",
        robot="unit",
        scene="unit",
        description="Lifecycle test",
        success={
            "engine_factory": {"lifecycle_unit": "tests:_LifecycleEngine"},
            "success_info_keys": ["success"],
            "max_steps": 1,
        },
    )
    suite = Suite("lifecycle_suite", "Lifecycle suite", (task,))
    hook_events: list[str] = []
    recorder_events: list[str] = []
    runner = PolicyRunner(
        policy=_Policy(),
        engine="lifecycle_unit",
        episodes=1,
        out=tmp_path,
        capture_replay=False,
        lifecycle_hooks=(_Hook(hook_events),),
        metric_recorders=(_Recorder(recorder_events),),
    )

    report = runner.evaluate(suite)

    assert report.summary["metrics"]["unit_recorded_metric"] == 1.0
    assert recorder_events == [
        "recorder:reset:episode_start",
        "recorder:step:0:1.0",
        "recorder:finalize:episode_finalize",
    ]
    phases = [value.split(":")[1] for value in hook_events]
    assert phases == [
        "task_load",
        "component_reset",
        "episode_start",
        "before_reset",
        "after_reset",
        "before_policy",
        "after_policy",
        "before_step",
        "after_engine_step",
        "after_step",
        "episode_finalize",
        "resource_cleanup",
    ]


def test_policy_runner_attempts_all_cleanup_after_one_close_fails() -> None:
    close_events: list[str] = []

    class FailingCloseEngine(_LifecycleEngine):
        def close(self) -> None:
            close_events.append("engine")
            raise RuntimeError("engine close failed")

    class ClosingPolicy(_Policy):
        def close(self) -> None:
            close_events.append("policy")

    class ClosingExpert(ExpertProvider):
        def close(self) -> None:
            close_events.append("expert")

    get_plugin_registry().engines["cleanup_unit"] = FailingCloseEngine
    task = TaskSpec(
        task_id="cleanup_task",
        engine="cleanup_unit",
        robot="unit",
        scene="unit",
        description="Cleanup test",
        success={
            "engine_factory": {"cleanup_unit": "tests:FailingCloseEngine"},
            "success_info_keys": ["success"],
            "max_steps": 1,
        },
    )
    runner = PolicyRunner(
        policy=ClosingPolicy(),
        engine="cleanup_unit",
        episodes=1,
        capture_replay=False,
        expert_provider=ClosingExpert(),
    )

    with pytest.raises(LifecycleExecutionError, match="engine close failed") as caught:
        runner.evaluate(Suite("cleanup_suite", "Cleanup suite", (task,)))

    assert caught.value.phase == "resource_cleanup"
    assert close_events == ["engine", "policy", "expert"]


def test_experiment_matrix_expansion_and_duplicate_protection(tmp_path: Path) -> None:
    cells = policy_seed_cells(
        policies=["a", "b"],
        seeds=[0, 1],
        out_dir=tmp_path,
        enable_verifier=True,
        enable_recovery=False,
    )
    assert [(cell.policy, cell.seed) for cell in cells] == [
        ("a", 0),
        ("a", 1),
        ("b", 0),
        ("b", 1),
    ]
    ablations = ablation_cells(
        policy="a",
        variants=["base", "verifier_recovery"],
        seeds=[0],
        out_dir=tmp_path,
    )
    assert ablations[0].enable_recovery is False
    assert ablations[1].enable_verifier is True
    assert ablations[1].enable_recovery is True

    executed: list[Path] = []

    class Run:
        def __init__(self, cell: ExperimentCell) -> None:
            self.cell = cell

        def evaluate(self, suite: Any) -> None:
            del suite
            executed.append(self.cell.run_dir)

    runner = ExperimentRunner(Run)
    assert runner.execute(object(), cells) == executed
    before_duplicate = list(executed)
    with pytest.raises(ValueError, match="duplicate experiment run directory"):
        runner.execute(object(), [cells[0], cells[0]])
    assert executed == before_duplicate


def test_counterfactual_branches_use_shared_transition_lifecycle() -> None:
    calls: list[str | None] = []

    class RecordingLifecycle:
        def execute(self, **kwargs: Any) -> TransitionResult:
            calls.append(kwargs.get("branch_kind"))
            return TransitionResult(
                observation=kwargs["observation"],
                action_before_stressors=kwargs["action"],
                action=kwargs["action"],
                reward=0.0,
                terminated=True,
                truncated=False,
                info={"success": False},
            )

    class Restorable:
        def get_state(self) -> dict[str, Any]:
            return {}

        def set_state(self, state: Any, **kwargs: Any) -> dict[str, Any]:
            del state, kwargs
            return {"raw": [0.0]}

        def state_restore_capability(self) -> dict[str, Any]:
            return {
                "supported": True,
                "fidelity": "exact_unit",
                "captures_rng": True,
                "exact": True,
                "reason": None,
            }

        def seed_branch_rng(self, seed: int) -> bool:
            del seed
            return True

        def act(self, observation: dict[str, Any], **kwargs: Any) -> float:
            del observation, kwargs
            return 0.0

        def drain_failure_events(self) -> list[Any]:
            return []

    component = Restorable()
    runner = CounterfactualBranchRunner(
        repeats=1,
        horizon_steps=1,
        transition_lifecycle=RecordingLifecycle(),  # type: ignore[arg-type]
    )
    record = runner.evaluate_recovery(
        engine=component,
        policy=component,
        expert=component,
        stressors=component,
        task=SimpleNamespace(task_id="unit_task"),
        observation={"raw": [0.0]},
        task_id="unit_task",
        episode_index=0,
        episode_seed=0,
        step_index=0,
        recovery_attempt_id=1,
        continuation_actions=[0.0],
        recovery_actions=[1.0],
        trigger_reason="test",
        trigger_event_id=None,
    )

    assert calls == ["continue", "recovery"]
    assert len(record.outcomes) == 2
