from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

from nyssa_bench.core.episode import EpisodeResult
from nyssa_bench.runners.lifecycle import (
    LifecycleContext,
    LifecycleDispatcher,
    LifecycleHook,
    LifecyclePhase,
    MetricRecorder,
    invoke_component,
)


@dataclass(frozen=True)
class EpisodeRequest:
    task: Any
    episode_index: int
    episode_seed: int

    @property
    def task_id(self) -> str:
        return str(getattr(self.task, "task_id", "unknown"))


class EpisodeEngine(Protocol):
    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]: ...

    def step(
        self, action: Any
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]: ...


class EpisodePolicy(Protocol):
    def act(self, observation: dict[str, Any]) -> Any: ...


class EpisodeExpert(Protocol):
    def score_action(self, observation: dict[str, Any], action: Any, **kwargs: Any) -> Any: ...

    def recover(self, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class EpisodeComponents:
    engine: EpisodeEngine
    policy: EpisodePolicy
    expert: EpisodeExpert
    stressor_factory: Callable[[EpisodeRequest], Any]
    detector_factory: Callable[[EpisodeRequest], Any]
    branch_factory: Callable[[EpisodeRequest], Any] | None = None
    monitor_manager: Any | None = None
    metric_recorders: tuple[MetricRecorder, ...] = ()


class EpisodeRunner:
    """Own one episode's outer lifecycle and guaranteed finalization."""

    def __init__(
        self,
        executor: Callable[[EpisodeRequest, EpisodeComponents], EpisodeResult],
        *,
        hooks: Sequence[LifecycleHook] = (),
    ) -> None:
        self.executor = executor
        self.dispatcher = LifecycleDispatcher(hooks)

    def run(
        self, request: EpisodeRequest, components: EpisodeComponents
    ) -> EpisodeResult:
        start = self._context(request, "episode_start")
        episode: EpisodeResult | None = None
        primary_error: BaseException | None = None
        try:
            self.dispatcher.emit(start, components)
            for recorder in components.metric_recorders:
                invoke_component(recorder.recorder_id, start, recorder.reset, start)
            episode = invoke_component(
                "episode_executor",
                start,
                self.executor,
                request,
                components,
            )
            finalize = self._context(request, "episode_finalize")
            for recorder in components.metric_recorders:
                values = invoke_component(
                    recorder.recorder_id,
                    finalize,
                    recorder.finalize,
                    finalize,
                    episode,
                )
                if values:
                    episode.metrics.update(values)
            return episode
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            try:
                self.dispatcher.emit(
                    self._context(request, "episode_finalize"), episode
                )
            except Exception as finalize_error:
                if primary_error is None:
                    raise
                setattr(primary_error, "lifecycle_finalize_error", finalize_error)

    @staticmethod
    def _context(
        request: EpisodeRequest, phase: LifecyclePhase
    ) -> LifecycleContext:
        return LifecycleContext(
            task_id=request.task_id,
            episode_index=request.episode_index,
            episode_seed=request.episode_seed,
            phase=phase,
        )
