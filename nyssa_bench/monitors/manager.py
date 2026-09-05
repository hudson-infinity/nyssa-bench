from __future__ import annotations

import copy
import time
from dataclasses import replace
from typing import Any, Mapping, Sequence

from nyssa_bench.failures.protocol import FailureEvent, FailureLedgerRecord
from nyssa_bench.monitors.base import FailureMonitor
from nyssa_bench.monitors.protocol import (
    FailureMonitorContract,
    MonitorInput,
    MonitorOutcome,
    MonitorPrediction,
    MonitorPredictionRecord,
    contract_sha256,
    prediction_id,
)


class FailureMonitorRuntimeError(RuntimeError):
    pass


class FailureMonitorManager:
    def __init__(self, monitors: Sequence[FailureMonitor] = ()) -> None:
        self.monitors = tuple(monitors)
        contracts = [monitor.contract() for monitor in self.monitors]
        self._ordered_contracts = tuple(contracts)
        self.contracts = {contract.monitor_id: contract for contract in contracts}
        if len(self.contracts) != len(self.monitors):
            raise ValueError("failure monitor IDs must be unique")
        for monitor, contract in zip(
            self.monitors, self._ordered_contracts, strict=True
        ):
            _validate_monitor_implementation(monitor, contract)
        self.predictions: list[MonitorPrediction] = []
        self.support: dict[str, dict[str, Any]] = {}
        self._task_id = "unknown"
        self._episode_index = 0
        self._episode_seed = 0

    def reset(
        self,
        *,
        task: Any,
        episode_index: int,
        seed: int,
        policy: Any,
        engine: Any,
    ) -> None:
        self.predictions = []
        self.support = {}
        self._task_id = str(getattr(task, "task_id", "unknown"))
        self._episode_index = int(episode_index)
        self._episode_seed = int(seed)
        for monitor, contract in zip(
            self.monitors, self._ordered_contracts, strict=True
        ):
            missing = _unavailable_required_inputs(contract, policy, engine)
            if missing:
                self.support[contract.monitor_id] = {
                    "status": "unsupported",
                    "missing_inputs": missing,
                    "reason": "required runtime inputs are unavailable",
                }
                continue
            try:
                monitor.reset(task=task, episode_index=episode_index, seed=seed)
            except Exception as exc:
                raise FailureMonitorRuntimeError(
                    f"Failure monitor '{contract.monitor_id}' failed during reset "
                    f"for task '{self._task_id}': {exc}"
                ) from exc
            self.support[contract.monitor_id] = {
                "status": "supported",
                "missing_inputs": [],
                "reason": None,
            }

    def predict(
        self,
        *,
        step_index: int,
        observation: Mapping[str, Any] | None,
        proposed_action: Any,
        policy: Any,
        engine: Any,
        policy_action_timestamp: int | None = None,
        failure_event_ids: Sequence[str] = (),
    ) -> tuple[MonitorPrediction, ...]:
        emitted = []
        known_event_ids = tuple(failure_event_ids)
        supported = [
            contract
            for contract in self._ordered_contracts
            if self.support.get(contract.monitor_id, {}).get("status") == "supported"
        ]
        required_sources = {
            item.source for contract in supported for item in contract.inputs
        }
        try:
            policy_internal = (
                policy.monitor_state()
                if "policy_internal" in required_sources
                and callable(getattr(policy, "monitor_state", None))
                else None
            )
            privileged_state = (
                engine.get_state()
                if "privileged_state" in required_sources
                and callable(getattr(engine, "get_state", None))
                else None
            )
        except Exception as exc:
            raise FailureMonitorRuntimeError(
                f"Failure monitor input collection failed for task '{self._task_id}' "
                f"step={step_index}: {exc}"
            ) from exc
        for monitor, contract in zip(
            self.monitors, self._ordered_contracts, strict=True
        ):
            if self.support.get(contract.monitor_id, {}).get("status") != "supported":
                continue
            started = time.perf_counter()
            try:
                monitor_input = _build_input(
                    contract,
                    task_id=self._task_id,
                    episode_index=self._episode_index,
                    episode_seed=self._episode_seed,
                    step_index=step_index,
                    policy_action_timestamp=(
                        step_index
                        if policy_action_timestamp is None
                        else policy_action_timestamp
                    ),
                    observation=observation,
                    proposed_action=proposed_action,
                    policy_internal=policy_internal,
                    privileged_state=privileged_state,
                    failure_event_ids=known_event_ids,
                )
            except Exception as exc:
                raise FailureMonitorRuntimeError(
                    f"Failure monitor '{contract.monitor_id}' could not collect inputs "
                    f"for task '{self._task_id}' step={step_index}: {exc}"
                ) from exc
            missing_runtime = _missing_runtime_values(contract, monitor_input)
            if missing_runtime:
                self.support[contract.monitor_id] = {
                    "status": "runtime_unsupported",
                    "missing_inputs": missing_runtime,
                    "reason": "required runtime input values are unavailable",
                    "first_unsupported_step": step_index,
                    "predictions_before_unsupported": sum(
                        item.monitor_id == contract.monitor_id
                        for item in self.predictions
                    ),
                }
                continue
            try:
                prediction = monitor.predict(monitor_input)
            except Exception as exc:
                raise FailureMonitorRuntimeError(
                    f"Failure monitor '{contract.monitor_id}' failed during predict "
                    f"for task '{self._task_id}' step={step_index}: {exc}"
                ) from exc
            if (
                prediction.failure_event_ids_before_prediction
                != monitor_input.failure_event_ids_before_prediction
            ):
                raise ValueError(
                    "monitor prediction failure-event inputs do not match its contract"
                )
            latency_ms = (time.perf_counter() - started) * 1000.0
            prediction = replace(
                prediction,
                latency_ms=latency_ms,
                failure_event_ids_before_prediction=known_event_ids,
            )
            _validate_prediction(
                prediction,
                monitor_input,
                contract,
                known_event_ids=known_event_ids,
            )
            self.predictions.append(prediction)
            emitted.append(prediction)
        return tuple(emitted)

    def finalize(
        self,
        *,
        success: bool,
        truncated: bool,
        episode_steps: int,
        failure_ledger: FailureLedgerRecord,
        intervention_links: Mapping[str, str] | None = None,
    ) -> tuple[MonitorPredictionRecord, ...]:
        links = dict(intervention_links or {})
        events = tuple(
            event
            for event in failure_ledger.events
            if event.role in {"symptom", "mechanism", "consequence"}
        )
        return tuple(
            MonitorPredictionRecord(
                prediction=prediction,
                outcome=_label_prediction(
                    prediction,
                    self.contracts[prediction.monitor_id],
                    success=success,
                    truncated=truncated,
                    episode_steps=episode_steps,
                    events=events,
                ),
                intervention_branch_point_id=links.get(prediction.prediction_id),
            )
            for prediction in self.predictions
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "contracts": [
                self.contracts[key].to_dict() for key in sorted(self.contracts)
            ],
            "support": {key: self.support[key] for key in sorted(self.support)},
            "prediction_count": len(self.predictions),
        }

    def close(self) -> None:
        errors = []
        for monitor, contract in zip(
            self.monitors, self._ordered_contracts, strict=True
        ):
            try:
                monitor.close()
            except Exception as exc:
                errors.append((contract.monitor_id, exc))
        if errors:
            monitor_id, error = errors[0]
            failure = FailureMonitorRuntimeError(
                f"Failure monitor '{monitor_id}' failed during close: {error}"
            )
            setattr(failure, "additional_close_errors", tuple(errors[1:]))
            raise failure from error


def _unavailable_required_inputs(
    contract: FailureMonitorContract, policy: Any, engine: Any
) -> list[str]:
    missing = []
    for item in contract.inputs:
        if not item.required:
            continue
        if item.source == "policy_internal" and not callable(
            getattr(policy, "monitor_state", None)
        ):
            missing.append(item.input_id)
        elif item.source == "privileged_state" and not callable(
            getattr(engine, "get_state", None)
        ):
            missing.append(item.input_id)
        elif item.source not in {
            "observation",
            "proposed_action",
            "policy_internal",
            "privileged_state",
            "failure_event_ids",
        }:
            missing.append(item.input_id)
    return missing


def _validate_monitor_implementation(
    monitor: FailureMonitor, contract: FailureMonitorContract
) -> None:
    monitor_type = type(monitor)
    if contract.state_semantics in {"resettable", "restorable"} and (
        monitor_type.reset is FailureMonitor.reset
    ):
        raise ValueError(
            f"monitor '{contract.monitor_id}' declares {contract.state_semantics} "
            "state but does not implement reset()"
        )
    if contract.state_semantics == "restorable" and (
        monitor_type.get_state is FailureMonitor.get_state
        or monitor_type.set_state is FailureMonitor.set_state
    ):
        raise ValueError(
            f"monitor '{contract.monitor_id}' declares restorable state but does not "
            "implement get_state() and set_state()"
        )


def _build_input(
    contract: FailureMonitorContract,
    *,
    task_id: str,
    episode_index: int,
    episode_seed: int,
    step_index: int,
    policy_action_timestamp: int,
    observation: Mapping[str, Any] | None,
    proposed_action: Any,
    policy_internal: Any,
    privileged_state: Any,
    failure_event_ids: tuple[str, ...],
) -> MonitorInput:
    sources = {item.source for item in contract.inputs}
    return MonitorInput(
        task_id=task_id,
        episode_index=episode_index,
        episode_seed=episode_seed,
        environment_step=step_index,
        observation_timestamp=step_index,
        policy_action_timestamp=policy_action_timestamp,
        observation=copy.deepcopy(observation)
        if "observation" in sources
        else None,
        proposed_action=copy.deepcopy(proposed_action)
        if "proposed_action" in sources
        else None,
        policy_internal=copy.deepcopy(policy_internal)
        if "policy_internal" in sources
        else None,
        privileged_state=copy.deepcopy(privileged_state)
        if "privileged_state" in sources
        else None,
        failure_event_ids_before_prediction=failure_event_ids
        if "failure_event_ids" in sources
        else (),
    )


def _missing_runtime_values(
    contract: FailureMonitorContract, monitor_input: MonitorInput
) -> list[str]:
    values = {
        "observation": monitor_input.observation,
        "proposed_action": monitor_input.proposed_action,
        "policy_internal": monitor_input.policy_internal,
        "privileged_state": monitor_input.privileged_state,
        "failure_event_ids": monitor_input.failure_event_ids_before_prediction,
    }
    return [
        item.input_id
        for item in contract.inputs
        if item.required and values.get(item.source) is None
    ]


def _validate_prediction(
    prediction: MonitorPrediction,
    monitor_input: MonitorInput,
    contract: FailureMonitorContract,
    *,
    known_event_ids: tuple[str, ...],
) -> None:
    expected_id = prediction_id(
        contract,
        task_id=monitor_input.task_id,
        episode_seed=monitor_input.episode_seed,
        episode_index=monitor_input.episode_index,
        step_index=monitor_input.environment_step,
    )
    expected = {
        "prediction_id": expected_id,
        "monitor_id": contract.monitor_id,
        "contract_sha256": contract_sha256(contract),
        "task_id": monitor_input.task_id,
        "episode_index": monitor_input.episode_index,
        "episode_seed": monitor_input.episode_seed,
        "environment_step": monitor_input.environment_step,
        "observation_timestamp": monitor_input.observation_timestamp,
        "policy_action_timestamp": monitor_input.policy_action_timestamp,
    }
    mismatches = [
        key for key, value in expected.items() if getattr(prediction, key) != value
    ]
    if mismatches:
        raise ValueError(
            "monitor prediction identity/timestamp mismatch: "
            + ", ".join(mismatches)
        )
    if (
        prediction.failure_event_ids_before_prediction
        != known_event_ids
    ):
        raise ValueError(
            "monitor prediction failure-event alignment metadata is inconsistent"
        )
    fields = {
        "failure_risk": prediction.failure_risk,
        "success_probability": prediction.success_probability,
        "failure_category": prediction.failure_category,
        "failure_mechanism": prediction.failure_mechanism,
        "expected_time_to_failure": prediction.expected_time_to_failure,
        "recovery_eligibility": prediction.recovery_eligibility
        if prediction.recovery_eligibility != "unknown"
        else None,
        "uncertainty": prediction.uncertainty or None,
        "evidence_references": prediction.evidence_references or None,
    }
    undeclared = [
        key for key, value in fields.items() if value is not None and key not in contract.outputs
    ]
    if prediction.intervention_recommended and (
        not contract.intervention_recommendations
        or "intervention_recommendation" not in contract.outputs
    ):
        undeclared.append("intervention_recommendation")
    missing_required_outputs = []
    if "failure_risk" in contract.outputs and prediction.failure_risk is None:
        missing_required_outputs.append("failure_risk")
    if (
        "success_probability" in contract.outputs
        and prediction.success_probability is None
    ):
        missing_required_outputs.append("success_probability")
    if missing_required_outputs:
        raise ValueError(
            "monitor omitted declared probability outputs: "
            + ", ".join(sorted(missing_required_outputs))
        )
    if undeclared:
        raise ValueError(
            "monitor emitted undeclared outputs: " + ", ".join(sorted(undeclared))
        )


def _label_prediction(
    prediction: MonitorPrediction,
    contract: FailureMonitorContract,
    *,
    success: bool,
    truncated: bool,
    episode_steps: int,
    events: Sequence[FailureEvent],
) -> MonitorOutcome:
    future_events = [
        event
        for event in events
        if event.onset_step >= prediction.environment_step
        and event.event_id not in prediction.failure_event_ids_before_prediction
    ]
    horizon_end = (
        prediction.environment_step + contract.prediction_horizon_steps
        if contract.prediction_horizon_steps is not None
        else None
    )
    within = [
        event
        for event in future_events
        if horizon_end is None or event.onset_step <= horizon_end
    ]
    first = min(within, key=lambda event: (event.onset_step, event.event_id)) if within else None
    observation_complete = (
        success
        or not truncated
        or horizon_end is None
        or episode_steps - 1 >= horizon_end
        or first is not None
    )
    if not observation_complete:
        return MonitorOutcome(
            status="censored",
            failure_within_horizon=None,
            eventual_episode_failure=None,
            failure_onset_step=None,
            failure_category=None,
            failure_mechanism=None,
            recovery_eligible=None,
        )
    mechanism = next(
        (
            event.subtype
            for event in within
            if event.role == "mechanism"
            and (first is None or event.onset_step == first.onset_step)
        ),
        None,
    )
    return MonitorOutcome(
        status="observed",
        failure_within_horizon=first is not None,
        eventual_episode_failure=not success,
        failure_onset_step=first.onset_step if first else None,
        failure_category=first.category if first else None,
        failure_mechanism=mechanism,
        recovery_eligible=(
            True
            if first.recovery_eligibility == "eligible"
            else False
            if first.recovery_eligibility == "ineligible"
            else None
        )
        if first is not None
        else None,
    )
