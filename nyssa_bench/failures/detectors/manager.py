from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

from nyssa_bench.failures.ledger import FailureEventLedger
from nyssa_bench.failures.protocol import FailureEvent, FailureEventDraft

from .protocol import DetectorSupport, FailureDetector, FailureDetectorContract


FAILURE_DETECTOR_MANIFEST_FORMAT = "nyssa-failure-detector-manifest-v1"


class FailureDetectorRuntimeError(RuntimeError):
    """Identifies the detector and lifecycle phase that failed."""


@dataclass(frozen=True)
class DetectorEmission:
    detector_id: str
    detector_version: str
    payload: FailureEventDraft | dict[str, Any]


class FailureDetectorManager:
    """Capability-check and execute an isolated detector set for one episode."""

    def __init__(
        self,
        detectors: Iterable[FailureDetector] | None = None,
        *,
        engine_name: str | None = None,
    ) -> None:
        self.detectors = tuple(detectors or ())
        self.engine_name = engine_name
        self._contracts = {
            detector.detector_id: detector.contract() for detector in self.detectors
        }
        if len(self._contracts) != len(self.detectors):
            raise ValueError("failure detector IDs must be unique within a manager")
        self._supports: dict[str, DetectorSupport] = {}
        self._capabilities: set[str] = set()
        self._task_id = "unknown"
        self._initialized: set[str] = set()
        self._finalized = False

    def reset(
        self,
        *,
        task: Any,
        engine: Any,
        observation: Mapping[str, Any] | None,
        stressor_context: Mapping[str, Any] | None = None,
        reset_info: Mapping[str, Any] | None = None,
    ) -> None:
        self._task_id = str(getattr(task, "task_id", "unknown"))
        if self.engine_name is None:
            self.engine_name = _inferred_engine_name(engine)
        self._capabilities = _engine_capabilities(engine, reset_info)
        self._supports.clear()
        self._initialized.clear()
        self._finalized = False

        for detector in self.detectors:
            contract = self._contracts[detector.detector_id]
            if contract.mode == "instrumented":
                added = self._call(
                    detector,
                    "request_instrumentation",
                    None,
                    detector.request_instrumentation,
                    task=task,
                    engine=engine,
                )
                if added:
                    self._capabilities.update(str(item) for item in added)
            support = self._evaluate_support(contract)
            self._supports[detector.detector_id] = support
            if support.status == "unsupported":
                continue
            self._call(
                detector,
                "reset",
                None,
                detector.reset,
                task=task,
                engine=engine,
                observation=_read_only(observation),
                stressor_context=_read_only(stressor_context),
            )
            self._initialized.add(detector.detector_id)

    def observe_before_action(
        self,
        *,
        step_index: int,
        observation: Mapping[str, Any] | None,
        action: Any,
        task: Any,
        engine: Any,
        stressor_context: Mapping[str, Any] | None = None,
    ) -> list[DetectorEmission]:
        return self._invoke_active(
            "observe_before_action",
            step_index,
            task=task,
            engine=engine,
            observation=_read_only(observation),
            action=action,
            stressor_context=_read_only(stressor_context),
        )

    def observe_after_action(
        self,
        *,
        step_index: int,
        pre_observation: Mapping[str, Any] | None,
        post_observation: Mapping[str, Any] | None,
        action: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Mapping[str, Any],
        task: Any,
        engine: Any,
        stressor_context: Mapping[str, Any] | None = None,
    ) -> list[DetectorEmission]:
        self._refresh_support(engine, info)
        return self._invoke_active(
            "observe_after_action",
            step_index,
            task=task,
            engine=engine,
            pre_observation=_read_only(pre_observation),
            post_observation=_read_only(post_observation),
            action=action,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=_read_only(info),
            stressor_context=_read_only(stressor_context),
        )

    def detect(
        self,
        *,
        step_index: int,
        observation: Mapping[str, Any] | None,
        action: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Mapping[str, Any],
        task: Any,
        engine: Any,
        stressor_context: Mapping[str, Any] | None = None,
    ) -> list[DetectorEmission]:
        self._refresh_support(engine, info)
        return self._invoke_active(
            "detect",
            step_index,
            task=task,
            engine=engine,
            observation=_read_only(observation),
            action=action,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=_read_only(info),
            stressor_context=_read_only(stressor_context),
        )

    def finalize(
        self,
        *,
        step_index: int,
        final_observation: Mapping[str, Any] | None,
        reward: float,
        terminated: bool,
        truncated: bool,
        success: bool,
        info: Mapping[str, Any],
        task: Any,
        engine: Any,
        stressor_context: Mapping[str, Any] | None = None,
    ) -> list[DetectorEmission]:
        if self._finalized:
            raise RuntimeError("failure detector manager has already been finalized")
        self._refresh_support(engine, info)
        emissions = self._invoke_active(
            "finalize",
            step_index,
            task=task,
            engine=engine,
            final_observation=_read_only(final_observation),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            success=success,
            info=_read_only(info),
            stressor_context=_read_only(stressor_context),
        )
        for detector_id, support in tuple(self._supports.items()):
            if support.status == "pending":
                self._supports[detector_id] = DetectorSupport(
                    status="unsupported",
                    available_signals=support.available_signals,
                    missing_requirements=support.missing_requirements,
                    reason="required runtime signals were not observed during the episode",
                )
        self._finalized = True
        return emissions

    def emit(
        self,
        ledger: FailureEventLedger,
        emissions: Iterable[DetectorEmission],
        *,
        default_step: int,
    ) -> list[FailureEvent]:
        emitted: list[FailureEvent] = []
        emitters: dict[tuple[str, str], Any] = {}
        for emission in emissions:
            key = (emission.detector_id, emission.detector_version)
            emitter = emitters.get(key)
            if emitter is None:
                emitter = ledger.emitter(
                    "external_monitor",
                    emission.detector_id,
                    annotation_source=f"detector@{emission.detector_version}",
                )
                emitters[key] = emitter
            emitted.append(
                emitter.emit_payload(emission.payload, default_step=default_step)
            )
        return emitted

    def manifest(self, *, events: Iterable[FailureEvent] = ()) -> dict[str, Any]:
        event_counts: dict[str, int] = {}
        for event in events:
            if event.provenance.source == "external_monitor":
                detector_id = event.provenance.component_id
                event_counts[detector_id] = event_counts.get(detector_id, 0) + 1
        return {
            "format": FAILURE_DETECTOR_MANIFEST_FORMAT,
            "engine_name": self.engine_name or "unknown",
            "task_id": self._task_id,
            "finalized": self._finalized,
            "detectors": [
                {
                    "contract": self._contracts[detector.detector_id].to_dict(),
                    "support": self._supports.get(
                        detector.detector_id,
                        DetectorSupport(
                            status="unsupported", reason="manager was not reset"
                        ),
                    ).to_dict(),
                    "emitted_event_count": event_counts.get(detector.detector_id, 0),
                }
                for detector in self.detectors
            ],
        }

    def _refresh_support(self, engine: Any, info: Mapping[str, Any] | None) -> None:
        self._capabilities.update(_engine_capabilities(engine, info))
        for detector_id, support in tuple(self._supports.items()):
            if support.status != "pending":
                continue
            self._supports[detector_id] = self._evaluate_support(
                self._contracts[detector_id]
            )

    def _evaluate_support(self, contract: FailureDetectorContract) -> DetectorSupport:
        return contract.support(
            engine_name=self.engine_name or "unknown",
            task_id=self._task_id,
            capabilities=self._capabilities,
        )

    def _invoke_active(
        self,
        phase: str,
        step_index: int,
        **kwargs: Any,
    ) -> list[DetectorEmission]:
        if self._finalized:
            raise RuntimeError("failure detector manager has already been finalized")
        emissions: list[DetectorEmission] = []
        for detector in self.detectors:
            if (
                self._supports.get(
                    detector.detector_id, DetectorSupport("pending")
                ).status
                != "supported"
            ):
                continue
            method = getattr(detector, phase)
            payloads = _coerce_payloads(
                self._call(
                    detector, phase, step_index, method, step_index=step_index, **kwargs
                )
            )
            emissions.extend(
                DetectorEmission(
                    detector_id=detector.detector_id,
                    detector_version=detector.detector_version,
                    payload=payload,
                )
                for payload in payloads
            )
        return emissions

    def _call(
        self,
        detector: FailureDetector,
        phase: str,
        lifecycle_step: int | None,
        method: Any,
        **kwargs: Any,
    ) -> Any:
        try:
            return method(**kwargs)
        except Exception as exc:
            location = f" step={lifecycle_step}" if lifecycle_step is not None else ""
            raise FailureDetectorRuntimeError(
                f"Failure detector '{detector.detector_id}' failed during {phase} "
                f"for task '{self._task_id}'{location}: {exc}"
            ) from exc


def _engine_capabilities(engine: Any, info: Mapping[str, Any] | None) -> set[str]:
    capabilities = {"reward"}
    if info:
        capabilities.update(f"info.{key}" for key in info)
    method = getattr(engine, "failure_signal_capabilities", None)
    if callable(method):
        provided = method(info=dict(info) if info is not None else None)
        if provided:
            capabilities.update(str(item) for item in provided)
    return capabilities


def _inferred_engine_name(engine: Any) -> str:
    name = engine.__class__.__name__
    for suffix in ("Engine", "Adapter"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name.lower() or "unknown"


def _read_only(value: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if value is None:
        return None
    return MappingProxyType(dict(value))


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
