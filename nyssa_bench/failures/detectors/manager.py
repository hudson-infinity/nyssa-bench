from __future__ import annotations

from collections.abc import Iterable
from typing import Any, cast

from nyssa_bench.failures.protocol import FailureEventDraft

from .protocol import FailureDetector


class FailureDetectorManager:
    """Coordinates per-step detector execution and event emission payload collection."""

    def __init__(self, detectors: Iterable[FailureDetector] | None = None) -> None:
        self.detectors = tuple(detectors or ())

    def reset(
        self,
        *,
        task: Any,
        engine: Any,
        observation: dict[str, Any] | None,
        stressor_context: dict[str, Any] | None = None,
    ) -> None:
        for detector in self.detectors:
            detector.reset(
                task=task,
                engine=engine,
                observation=observation,
                stressor_context=stressor_context,
            )

    def observe_before_action(
        self,
        *,
        step_index: int,
        observation: dict[str, Any] | None,
        action: Any,
        task: Any,
        engine: Any,
        stressor_context: dict[str, Any] | None = None,
    ) -> None:
        for detector in self.detectors:
            detector.observe_before_action(
                step_index=step_index,
                observation=observation,
                action=action,
                task=task,
                engine=engine,
                stressor_context=stressor_context,
            )

    def observe_after_action(
        self,
        *,
        step_index: int,
        pre_observation: dict[str, Any] | None,
        post_observation: dict[str, Any] | None,
        action: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
        task: Any,
        engine: Any,
        stressor_context: dict[str, Any] | None = None,
    ) -> list[FailureEventDraft | dict[str, Any]]:
        payloads: list[FailureEventDraft | dict[str, Any]] = []
        for detector in self.detectors:
            payloads.extend(
                _coerce_payloads(
                    detector.observe_after_action(
                        step_index=step_index,
                        pre_observation=pre_observation,
                        post_observation=post_observation,
                        action=action,
                        reward=reward,
                        terminated=terminated,
                        truncated=truncated,
                        info=info,
                        task=task,
                        engine=engine,
                        stressor_context=stressor_context,
                    )
                )
            )
        return payloads

    def detect(
        self,
        *,
        step_index: int,
        observation: dict[str, Any] | None,
        action: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
        task: Any,
        engine: Any,
        stressor_context: dict[str, Any] | None = None,
    ) -> list[FailureEventDraft | dict[str, Any]]:
        payloads: list[FailureEventDraft | dict[str, Any]] = []
        for detector in self.detectors:
            payloads.extend(
                _coerce_payloads(
                    detector.detect(
                        step_index=step_index,
                        observation=observation,
                        action=action,
                        reward=reward,
                        terminated=terminated,
                        truncated=truncated,
                        info=info,
                        task=task,
                        engine=engine,
                        stressor_context=stressor_context,
                    )
                )
            )
        return payloads

    def finalize(
        self,
        *,
        step_index: int,
        final_observation: dict[str, Any] | None,
        reward: float,
        terminated: bool,
        truncated: bool,
        success: bool,
        info: dict[str, Any],
        task: Any,
        engine: Any,
        stressor_context: dict[str, Any] | None = None,
    ) -> list[FailureEventDraft | dict[str, Any]]:
        payloads: list[FailureEventDraft | dict[str, Any]] = []
        for detector in self.detectors:
            payloads.extend(
                _coerce_payloads(
                    detector.finalize(
                        step_index=step_index,
                        final_observation=final_observation,
                        reward=reward,
                        terminated=terminated,
                        truncated=truncated,
                        success=success,
                        info=info,
                        task=task,
                        engine=engine,
                        stressor_context=stressor_context,
                    )
                )
            )
        return payloads

    def drain_draft_payloads(
        self,
        payloads: Iterable[FailureEventDraft | dict[str, Any]],
    ) -> list[FailureEventDraft | dict[str, Any]]:
        result: list[FailureEventDraft | dict[str, Any]] = []
        for payload in payloads:
            result.extend(_coerce_payloads(payload))
        return result


def _coerce_payloads(
    payload: list | tuple | FailureEventDraft | dict | None,
) -> list[FailureEventDraft | dict[str, Any]]:
    if payload is None:
        return []
    if isinstance(payload, (FailureEventDraft, dict)):
        return [cast(FailureEventDraft | dict[str, Any], payload)]
    if isinstance(payload, (list, tuple)):
        result: list[FailureEventDraft | dict[str, Any]] = []
        for item in payload:
            result.extend(_coerce_payloads(item))
        return result
    raise TypeError(
        "Failure detector methods must return None, list, tuple, FailureEventDraft, or dict."
    )
