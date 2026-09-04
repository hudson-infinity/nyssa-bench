from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Protocol, Sequence


LifecyclePhase = Literal[
    "task_load",
    "component_reset",
    "episode_start",
    "before_reset",
    "after_reset",
    "before_policy",
    "after_policy",
    "before_verifier",
    "after_verifier",
    "before_recovery",
    "after_recovery",
    "before_step",
    "after_engine_step",
    "after_step",
    "episode_finalize",
    "resource_cleanup",
]


@dataclass(frozen=True)
class LifecycleContext:
    task_id: str
    episode_index: int
    episode_seed: int
    phase: LifecyclePhase
    step_index: int | None = None
    branch_kind: str | None = None


class LifecycleHook(Protocol):
    component_id: str

    def on_lifecycle(self, context: LifecycleContext, payload: Any) -> None: ...


class MetricRecorder(Protocol):
    recorder_id: str

    def reset(self, context: LifecycleContext) -> None: ...

    def record_transition(
        self, context: LifecycleContext, transition: "TransitionResult"
    ) -> None: ...

    def finalize(self, context: LifecycleContext, episode: Any) -> dict[str, float]: ...


class EngineStep(Protocol):
    def step(
        self, action: Any
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]: ...


class StressorStep(Protocol):
    def before_step(self, engine: Any, *, step_index: int) -> None: ...

    def transform_action(
        self, action: Any, *, observation: Any, step_index: int
    ) -> Any: ...

    def after_step(
        self, engine: Any, info: dict[str, Any], *, step_index: int
    ) -> None: ...

    def transform_observation(self, observation: Any, *, step_index: int) -> Any: ...


@dataclass(frozen=True)
class TransitionResult:
    observation: dict[str, Any]
    action_before_stressors: Any
    action: Any
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class LifecycleExecutionError(RuntimeError):
    def __init__(
        self,
        *,
        component_id: str,
        phase: LifecyclePhase,
        task_id: str,
        step_index: int | None,
        cause: Exception,
        support_status: str = "supported",
    ) -> None:
        location = f" step={step_index}" if step_index is not None else ""
        super().__init__(
            f"Lifecycle component '{component_id}' failed during {phase} "
            f"for task '{task_id}'{location} (support={support_status}): {cause}"
        )
        self.component_id = component_id
        self.phase = phase
        self.task_id = task_id
        self.step_index = step_index
        self.support_status = support_status
        self.__cause__ = cause


class LifecycleDispatcher:
    def __init__(self, hooks: Sequence[LifecycleHook] = ()) -> None:
        self.hooks = tuple(hooks)

    def emit(self, context: LifecycleContext, payload: Any = None) -> None:
        for hook in self.hooks:
            invoke_component(
                hook.component_id,
                context,
                hook.on_lifecycle,
                context,
                payload,
            )


class TransitionLifecycle:
    """Shared engine/stressor transition ordering for episodes and branches."""

    def __init__(self, hooks: Sequence[LifecycleHook] = ()) -> None:
        self.dispatcher = LifecycleDispatcher(hooks)

    def execute(
        self,
        *,
        engine: EngineStep,
        stressors: StressorStep,
        observation: dict[str, Any],
        action: Any,
        task_id: str,
        episode_index: int,
        episode_seed: int,
        step_index: int,
        branch_kind: str | None = None,
        after_engine_step: Callable[[dict[str, Any]], None] | None = None,
        after_stressors: Callable[[TransitionResult], None] | None = None,
    ) -> TransitionResult:
        context = LifecycleContext(
            task_id=task_id,
            episode_index=episode_index,
            episode_seed=episode_seed,
            phase="before_step",
            step_index=step_index,
            branch_kind=branch_kind,
        )
        self.dispatcher.emit(context, {"observation": observation, "action": action})
        invoke_component(
            "stressor_pipeline",
            context,
            stressors.before_step,
            engine,
            step_index=step_index,
        )
        action_before_stressors = action
        action = invoke_component(
            "stressor_pipeline",
            context,
            stressors.transform_action,
            action,
            observation=observation,
            step_index=step_index,
        )
        transition = invoke_component(
            engine.__class__.__name__,
            _phase(context, "after_engine_step"),
            engine.step,
            action,
        )
        next_observation, reward, terminated, truncated, info = transition
        info = dict(info)
        self.dispatcher.emit(
            _phase(context, "after_engine_step"),
            {
                "observation": next_observation,
                "action": action,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "info": info,
            },
        )
        if after_engine_step is not None:
            invoke_component(
                "after_engine_step_hook",
                _phase(context, "after_engine_step"),
                after_engine_step,
                info,
            )
        invoke_component(
            "stressor_pipeline",
            _phase(context, "after_step"),
            stressors.after_step,
            engine,
            info,
            step_index=step_index,
        )
        next_observation = invoke_component(
            "stressor_pipeline",
            _phase(context, "after_step"),
            stressors.transform_observation,
            next_observation,
            step_index=step_index + 1,
        )
        result = TransitionResult(
            observation=next_observation,
            action_before_stressors=action_before_stressors,
            action=action,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=info,
        )
        if after_stressors is not None:
            invoke_component(
                "after_stressors_hook",
                _phase(context, "after_step"),
                after_stressors,
                result,
            )
        self.dispatcher.emit(_phase(context, "after_step"), result)
        return result


def invoke_component(
    component_id: str,
    context: LifecycleContext,
    method: Callable[..., Any],
    *args: Any,
    support_status: str = "supported",
    **kwargs: Any,
) -> Any:
    try:
        return method(*args, **kwargs)
    except LifecycleExecutionError:
        raise
    except Exception as exc:
        raise LifecycleExecutionError(
            component_id=component_id,
            phase=context.phase,
            task_id=context.task_id,
            step_index=context.step_index,
            cause=exc,
            support_status=support_status,
        ) from exc


def _phase(context: LifecycleContext, phase: LifecyclePhase) -> LifecycleContext:
    return LifecycleContext(
        task_id=context.task_id,
        episode_index=context.episode_index,
        episode_seed=context.episode_seed,
        phase=phase,
        step_index=context.step_index,
        branch_kind=context.branch_kind,
    )
