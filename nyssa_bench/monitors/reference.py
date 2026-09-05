from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from nyssa_bench.monitors.base import FailureMonitor
from nyssa_bench.monitors.protocol import (
    FailureMonitorContract,
    MonitorInput,
    MonitorInputSpec,
    MonitorPrediction,
    contract_sha256,
    prediction_id,
)


class ActionMagnitudeFailureMonitor(FailureMonitor):
    """Integration baseline based only on proposed action magnitude."""

    def __init__(self, *, alert_threshold: float = 0.8) -> None:
        self.alert_threshold = float(alert_threshold)
        self._contract = FailureMonitorContract(
            monitor_id="action_magnitude_reference",
            monitor_version="1.0.0",
            inputs=(
                MonitorInputSpec(
                    input_id="proposed_action",
                    source="proposed_action",
                    visibility="policy_observable",
                    description="proposed environment-space action",
                ),
            ),
            outputs=(
                "failure_risk",
                "failure_category",
                "intervention_recommendation",
            ),
            checkpoint_id="builtin_action_magnitude_reference_v1",
            checkpoint_sha256=hashlib.sha256(
                b"nyssa-action-magnitude-reference-v1"
            ).hexdigest(),
            preprocessing_sha256=hashlib.sha256(
                b"flatten_abs_max_clip_0_1"
            ).hexdigest(),
            state_semantics="stateless",
            deterministic=True,
            prediction_horizon_steps=1,
            alert_threshold=self.alert_threshold,
            calibration_bins=10,
            intervention_recommendations=True,
            declared_compute={
                "kind": "analytic_reference",
                "device": "cpu",
                "learned_parameters": 0,
            },
        )

    def contract(self) -> FailureMonitorContract:
        return self._contract

    def predict(self, monitor_input: MonitorInput) -> MonitorPrediction:
        action = np.asarray(monitor_input.proposed_action, dtype=float).reshape(-1)
        finite = action[np.isfinite(action)]
        risk = min(1.0, float(np.max(np.abs(finite)))) if finite.size else 1.0
        return MonitorPrediction(
            prediction_id=prediction_id(
                self._contract,
                task_id=monitor_input.task_id,
                episode_seed=monitor_input.episode_seed,
                episode_index=monitor_input.episode_index,
                step_index=monitor_input.environment_step,
            ),
            monitor_id=self._contract.monitor_id,
            contract_sha256=contract_sha256(self._contract),
            task_id=monitor_input.task_id,
            episode_index=monitor_input.episode_index,
            episode_seed=monitor_input.episode_seed,
            environment_step=monitor_input.environment_step,
            observation_timestamp=monitor_input.observation_timestamp,
            policy_action_timestamp=monitor_input.policy_action_timestamp,
            failure_risk=risk,
            failure_category="control" if risk >= self.alert_threshold else None,
            intervention_recommended=risk >= self.alert_threshold,
            compute={"action_values": int(action.size)},
            failure_event_ids_before_prediction=(
                monitor_input.failure_event_ids_before_prediction
            ),
        )

    def get_state(self) -> Any:
        return None

    def set_state(self, state: Any) -> None:
        if state is not None:
            raise ValueError("stateless reference monitor requires null state")
