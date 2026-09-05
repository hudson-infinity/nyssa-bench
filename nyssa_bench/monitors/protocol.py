from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping


MONITOR_CONTRACT_FORMAT = "nyssa-failure-monitor-contract-v1"
MONITOR_PREDICTION_FORMAT = "nyssa-failure-monitor-prediction-v1"
MONITOR_RECORD_FORMAT = "nyssa-failure-monitor-record-v1"
MONITOR_MANIFEST_FORMAT = "nyssa-failure-monitor-manifest-v1"

InputVisibility = Literal["policy_observable", "policy_internal", "privileged"]
MonitorStateSemantics = Literal["stateless", "resettable", "restorable"]
RecoveryPrediction = Literal["eligible", "ineligible", "unknown"]
OutcomeStatus = Literal["observed", "censored", "invalid"]


@dataclass(frozen=True)
class MonitorInputSpec:
    input_id: str
    source: str
    visibility: InputVisibility
    required: bool = True
    description: str = ""

    def __post_init__(self) -> None:
        if not self.input_id.strip() or not self.source.strip():
            raise ValueError("monitor input identity and source are required")
        if self.visibility not in {
            "policy_observable",
            "policy_internal",
            "privileged",
        }:
            raise ValueError(f"unsupported monitor input visibility: {self.visibility}")
        required_visibility = {
            "observation": "policy_observable",
            "proposed_action": "policy_observable",
            "policy_internal": "policy_internal",
            "privileged_state": "privileged",
            "failure_event_ids": "privileged",
        }.get(self.source)
        if required_visibility is None:
            raise ValueError(f"unsupported monitor input source: {self.source}")
        if self.visibility != required_visibility:
            raise ValueError(
                f"monitor input {self.source} must use {required_visibility} visibility"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_id": self.input_id,
            "source": self.source,
            "visibility": self.visibility,
            "required": self.required,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MonitorInputSpec":
        _reject_unknown(
            data,
            {
                "input_id",
                "source",
                "visibility",
                "required",
                "description",
            },
            "monitor input",
        )
        return cls(
            input_id=str(data.get("input_id", "")),
            source=str(data.get("source", "")),
            visibility=str(data.get("visibility", "")),  # type: ignore[arg-type]
            required=_boolean(data.get("required"), "monitor input required"),
            description=str(data.get("description", "")),
        )


@dataclass(frozen=True)
class FailureMonitorContract:
    monitor_id: str
    monitor_version: str
    inputs: tuple[MonitorInputSpec, ...]
    outputs: tuple[str, ...]
    checkpoint_id: str
    checkpoint_sha256: str
    preprocessing_sha256: str
    state_semantics: MonitorStateSemantics
    deterministic: bool
    prediction_horizon_steps: int | None
    alert_threshold: float
    calibration_bins: int = 10
    intervention_recommendations: bool = False
    declared_compute: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.monitor_id.strip() or not self.checkpoint_id.strip():
            raise ValueError("monitor and checkpoint identities are required")
        if not re.fullmatch(
            r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:[-+][0-9A-Za-z.-]+)?",
            self.monitor_version,
        ):
            raise ValueError("monitor_version must use semantic versioning")
        _require_sha256(self.checkpoint_sha256, "checkpoint_sha256")
        _require_sha256(self.preprocessing_sha256, "preprocessing_sha256")
        if not self.inputs:
            raise ValueError("monitor contract requires at least one input")
        input_ids = [item.input_id for item in self.inputs]
        if len(input_ids) != len(set(input_ids)):
            raise ValueError("monitor input IDs must be unique")
        allowed_outputs = {
            "failure_risk",
            "success_probability",
            "failure_category",
            "failure_mechanism",
            "expected_time_to_failure",
            "recovery_eligibility",
            "intervention_recommendation",
            "uncertainty",
            "evidence_references",
        }
        if not self.outputs or not set(self.outputs) <= allowed_outputs:
            raise ValueError("monitor contract declares unsupported outputs")
        if len(self.outputs) != len(set(self.outputs)):
            raise ValueError("monitor outputs must be unique")
        if not {"failure_risk", "success_probability"}.intersection(self.outputs):
            raise ValueError("monitor must emit failure risk or success probability")
        if self.state_semantics not in {"stateless", "resettable", "restorable"}:
            raise ValueError(f"unsupported monitor state semantics: {self.state_semantics}")
        if not isinstance(self.deterministic, bool):
            raise ValueError("monitor deterministic must be a boolean")
        if self.prediction_horizon_steps is not None and self.prediction_horizon_steps <= 0:
            raise ValueError("prediction horizon must be positive or null")
        if not math.isfinite(self.alert_threshold) or not 0.0 <= self.alert_threshold <= 1.0:
            raise ValueError("alert threshold must be finite and within [0, 1]")
        if self.calibration_bins < 2:
            raise ValueError("calibration_bins must be at least 2")
        if (
            self.intervention_recommendations
            and "intervention_recommendation" not in self.outputs
        ):
            raise ValueError(
                "intervention recommendation capability must be declared as an output"
            )
        if not isinstance(self.intervention_recommendations, bool):
            raise ValueError("intervention_recommendations must be a boolean")
        if self.schema_version != 1:
            raise ValueError(f"unsupported monitor contract version: {self.schema_version}")
        _canonical_json(self.declared_compute)

    @property
    def observability_tier(self) -> str:
        visibilities = {item.visibility for item in self.inputs}
        if "privileged" in visibilities:
            return "privileged_monitor"
        return "deployable_monitor"

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "format": MONITOR_CONTRACT_FORMAT,
            "schema_version": self.schema_version,
            "monitor_id": self.monitor_id,
            "monitor_version": self.monitor_version,
            "inputs": [item.to_dict() for item in self.inputs],
            "outputs": list(self.outputs),
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "preprocessing_sha256": self.preprocessing_sha256,
            "state_semantics": self.state_semantics,
            "deterministic": self.deterministic,
            "prediction_horizon_steps": self.prediction_horizon_steps,
            "alert_threshold": self.alert_threshold,
            "calibration_bins": self.calibration_bins,
            "intervention_recommendations": self.intervention_recommendations,
            "declared_compute": self.declared_compute,
            "observability_tier": self.observability_tier,
        }
        payload["contract_sha256"] = _sha256(payload)
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FailureMonitorContract":
        _check_format(data, MONITOR_CONTRACT_FORMAT, "monitor contract")
        _reject_unknown(
            data,
            {
                "format",
                "schema_version",
                "monitor_id",
                "monitor_version",
                "inputs",
                "outputs",
                "checkpoint_id",
                "checkpoint_sha256",
                "preprocessing_sha256",
                "state_semantics",
                "deterministic",
                "prediction_horizon_steps",
                "alert_threshold",
                "calibration_bins",
                "intervention_recommendations",
                "declared_compute",
                "observability_tier",
                "contract_sha256",
            },
            "monitor contract",
        )
        raw_inputs = data.get("inputs")
        outputs = data.get("outputs")
        compute = data.get("declared_compute", {})
        if not isinstance(raw_inputs, list) or not all(
            isinstance(item, Mapping) for item in raw_inputs
        ):
            raise ValueError("monitor inputs must be a list of mappings")
        if not isinstance(outputs, list) or not all(isinstance(item, str) for item in outputs):
            raise ValueError("monitor outputs must be a list of strings")
        if not isinstance(compute, Mapping):
            raise ValueError("declared_compute must be a mapping")
        contract = cls(
            monitor_id=str(data.get("monitor_id", "")),
            monitor_version=str(data.get("monitor_version", "")),
            inputs=tuple(MonitorInputSpec.from_dict(item) for item in raw_inputs),
            outputs=tuple(outputs),
            checkpoint_id=str(data.get("checkpoint_id", "")),
            checkpoint_sha256=str(data.get("checkpoint_sha256", "")),
            preprocessing_sha256=str(data.get("preprocessing_sha256", "")),
            state_semantics=str(data.get("state_semantics", "")),  # type: ignore[arg-type]
            deterministic=_boolean(data.get("deterministic"), "monitor deterministic"),
            prediction_horizon_steps=int(data["prediction_horizon_steps"])
            if data.get("prediction_horizon_steps") is not None
            else None,
            alert_threshold=float(data.get("alert_threshold", float("nan"))),
            calibration_bins=int(data.get("calibration_bins", 0)),
            intervention_recommendations=_boolean(
                data.get("intervention_recommendations"),
                "intervention_recommendations",
            ),
            declared_compute=dict(compute),
            schema_version=int(data.get("schema_version", 1)),
        )
        expected = contract.to_dict()
        for key in ("contract_sha256", "observability_tier"):
            if data.get(key) != expected[key]:
                raise ValueError(f"monitor contract derived field mismatch: {key}")
        return contract


@dataclass(frozen=True)
class MonitorInput:
    task_id: str
    episode_index: int
    episode_seed: int
    environment_step: int
    observation_timestamp: int
    policy_action_timestamp: int
    observation: Mapping[str, Any] | None
    proposed_action: Any
    policy_internal: Any = None
    privileged_state: Any = None
    failure_event_ids_before_prediction: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("monitor input task_id must be non-empty")
        if min(
            self.episode_index,
            self.episode_seed,
            self.environment_step,
            self.observation_timestamp,
            self.policy_action_timestamp,
        ) < 0:
            raise ValueError("monitor input indices and timestamps must be non-negative")
        if self.observation_timestamp != self.environment_step:
            raise ValueError("observation timestamp must align to environment step")
        if self.policy_action_timestamp > self.environment_step:
            raise ValueError("policy-action timestamp cannot be in the future")
        _validate_unique_text(
            self.failure_event_ids_before_prediction,
            "monitor input failure-event IDs",
        )

    @property
    def policy_action_age_steps(self) -> int:
        return self.environment_step - self.policy_action_timestamp


@dataclass(frozen=True)
class MonitorPrediction:
    prediction_id: str
    monitor_id: str
    contract_sha256: str
    task_id: str
    episode_index: int
    episode_seed: int
    environment_step: int
    observation_timestamp: int
    policy_action_timestamp: int
    failure_risk: float | None = None
    success_probability: float | None = None
    failure_category: str | None = None
    failure_mechanism: str | None = None
    expected_time_to_failure: float | None = None
    recovery_eligibility: RecoveryPrediction = "unknown"
    intervention_recommended: bool = False
    uncertainty: dict[str, Any] = field(default_factory=dict)
    evidence_references: tuple[dict[str, Any], ...] = ()
    latency_ms: float = 0.0
    compute: dict[str, Any] = field(default_factory=dict)
    failure_event_ids_before_prediction: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for text, label in (
            (self.prediction_id, "prediction_id"),
            (self.monitor_id, "monitor_id"),
            (self.task_id, "task_id"),
        ):
            if not text.strip():
                raise ValueError(f"{label} must be non-empty")
        _require_sha256(self.contract_sha256, "monitor contract_sha256")
        if min(
            self.episode_index,
            self.episode_seed,
            self.environment_step,
            self.observation_timestamp,
            self.policy_action_timestamp,
        ) < 0:
            raise ValueError("prediction indices and timestamps must be non-negative")
        if self.observation_timestamp != self.environment_step:
            raise ValueError("prediction observation timestamp is not aligned")
        if self.policy_action_timestamp > self.environment_step:
            raise ValueError("prediction policy-action timestamp is in the future")
        if self.failure_risk is None and self.success_probability is None:
            raise ValueError("prediction requires failure risk or success probability")
        for value, label in (
            (self.failure_risk, "failure_risk"),
            (self.success_probability, "success_probability"),
        ):
            if value is not None and (
                not math.isfinite(value) or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{label} must be finite and within [0, 1]")
        if self.failure_risk is not None and self.success_probability is not None and not math.isclose(
            self.failure_risk + self.success_probability, 1.0, abs_tol=1e-6
        ):
            raise ValueError("failure risk and success probability must be complementary")
        if self.expected_time_to_failure is not None and (
            not math.isfinite(self.expected_time_to_failure)
            or self.expected_time_to_failure < 0.0
        ):
            raise ValueError("expected time to failure must be finite and non-negative")
        if self.recovery_eligibility not in {"eligible", "ineligible", "unknown"}:
            raise ValueError("invalid predicted recovery eligibility")
        if not isinstance(self.intervention_recommended, bool):
            raise ValueError("intervention_recommended must be a boolean")
        for text_value, label in (
            (self.failure_category, "failure_category"),
            (self.failure_mechanism, "failure_mechanism"),
        ):
            if text_value is not None and (
                not isinstance(text_value, str) or not text_value.strip()
            ):
                raise ValueError(f"{label} must be non-empty when provided")
        if not math.isfinite(self.latency_ms) or self.latency_ms < 0.0:
            raise ValueError("monitor latency must be finite and non-negative")
        _canonical_json(self.uncertainty)
        _canonical_json(list(self.evidence_references))
        _canonical_json(self.compute)
        _validate_unique_text(
            self.failure_event_ids_before_prediction,
            "prediction failure-event IDs",
        )

    @property
    def risk(self) -> float:
        if self.failure_risk is not None:
            return self.failure_risk
        assert self.success_probability is not None
        return 1.0 - self.success_probability

    @property
    def policy_action_age_steps(self) -> int:
        return self.environment_step - self.policy_action_timestamp

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": MONITOR_PREDICTION_FORMAT,
            "prediction_id": self.prediction_id,
            "monitor_id": self.monitor_id,
            "contract_sha256": self.contract_sha256,
            "task_id": self.task_id,
            "episode_index": self.episode_index,
            "episode_seed": self.episode_seed,
            "environment_step": self.environment_step,
            "observation_timestamp": self.observation_timestamp,
            "policy_action_timestamp": self.policy_action_timestamp,
            "policy_action_age_steps": self.policy_action_age_steps,
            "failure_risk": self.failure_risk,
            "success_probability": self.success_probability,
            "failure_category": self.failure_category,
            "failure_mechanism": self.failure_mechanism,
            "expected_time_to_failure": self.expected_time_to_failure,
            "recovery_eligibility": self.recovery_eligibility,
            "intervention_recommended": self.intervention_recommended,
            "uncertainty": self.uncertainty,
            "evidence_references": list(self.evidence_references),
            "latency_ms": self.latency_ms,
            "compute": self.compute,
            "failure_event_ids_before_prediction": list(
                self.failure_event_ids_before_prediction
            ),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MonitorPrediction":
        _check_format(data, MONITOR_PREDICTION_FORMAT, "monitor prediction")
        allowed = {
            "format",
            "prediction_id",
            "monitor_id",
            "contract_sha256",
            "task_id",
            "episode_index",
            "episode_seed",
            "environment_step",
            "observation_timestamp",
            "policy_action_timestamp",
            "policy_action_age_steps",
            "failure_risk",
            "success_probability",
            "failure_category",
            "failure_mechanism",
            "expected_time_to_failure",
            "recovery_eligibility",
            "intervention_recommended",
            "uncertainty",
            "evidence_references",
            "latency_ms",
            "compute",
            "failure_event_ids_before_prediction",
        }
        _reject_unknown(data, allowed, "monitor prediction")
        uncertainty = data.get("uncertainty", {})
        evidence = data.get("evidence_references", [])
        compute = data.get("compute", {})
        event_ids = data.get("failure_event_ids_before_prediction", [])
        if not isinstance(uncertainty, Mapping) or not isinstance(compute, Mapping):
            raise ValueError("prediction uncertainty and compute must be mappings")
        if not isinstance(evidence, list) or not all(
            isinstance(item, Mapping) for item in evidence
        ):
            raise ValueError("prediction evidence references must be mappings")
        if not isinstance(event_ids, list) or not all(
            isinstance(item, str) for item in event_ids
        ):
            raise ValueError("prediction failure event IDs must be strings")
        prediction = cls(
            prediction_id=str(data.get("prediction_id", "")),
            monitor_id=str(data.get("monitor_id", "")),
            contract_sha256=str(data.get("contract_sha256", "")),
            task_id=str(data.get("task_id", "")),
            episode_index=int(data.get("episode_index", -1)),
            episode_seed=int(data.get("episode_seed", -1)),
            environment_step=int(data.get("environment_step", -1)),
            observation_timestamp=int(data.get("observation_timestamp", -1)),
            policy_action_timestamp=int(data.get("policy_action_timestamp", -1)),
            failure_risk=float(data["failure_risk"])
            if data.get("failure_risk") is not None
            else None,
            success_probability=float(data["success_probability"])
            if data.get("success_probability") is not None
            else None,
            failure_category=str(data["failure_category"])
            if data.get("failure_category") is not None
            else None,
            failure_mechanism=str(data["failure_mechanism"])
            if data.get("failure_mechanism") is not None
            else None,
            expected_time_to_failure=float(data["expected_time_to_failure"])
            if data.get("expected_time_to_failure") is not None
            else None,
            recovery_eligibility=str(  # type: ignore[arg-type]
                data.get("recovery_eligibility", "unknown")
            ),
            intervention_recommended=_boolean(
                data.get("intervention_recommended"),
                "intervention_recommended",
            ),
            uncertainty=dict(uncertainty),
            evidence_references=tuple(dict(item) for item in evidence),
            latency_ms=float(data.get("latency_ms", float("nan"))),
            compute=dict(compute),
            failure_event_ids_before_prediction=tuple(event_ids),
        )
        if data.get("policy_action_age_steps") != prediction.policy_action_age_steps:
            raise ValueError("prediction policy_action_age_steps mismatch")
        return prediction


@dataclass(frozen=True)
class MonitorOutcome:
    status: OutcomeStatus
    failure_within_horizon: bool | None
    eventual_episode_failure: bool | None
    failure_onset_step: int | None
    failure_category: str | None
    failure_mechanism: str | None
    recovery_eligible: bool | None

    def __post_init__(self) -> None:
        if self.status not in {"observed", "censored", "invalid"}:
            raise ValueError(f"unsupported monitor outcome status: {self.status}")
        for value, label in (
            (self.failure_within_horizon, "failure_within_horizon"),
            (self.eventual_episode_failure, "eventual_episode_failure"),
            (self.recovery_eligible, "recovery_eligible"),
        ):
            if value is not None and not isinstance(value, bool):
                raise ValueError(f"{label} must be a boolean or null")
        if self.status == "observed" and self.failure_within_horizon is None:
            raise ValueError("observed outcomes require a binary horizon label")
        if self.status == "observed" and self.eventual_episode_failure is None:
            raise ValueError("observed outcomes require an eventual episode label")
        if self.status != "observed" and self.failure_within_horizon is not None:
            raise ValueError("censored/invalid outcomes cannot carry a horizon label")
        if self.status != "observed" and any(
            value is not None
            for value in (
                self.eventual_episode_failure,
                self.failure_onset_step,
                self.failure_category,
                self.failure_mechanism,
                self.recovery_eligible,
            )
        ):
            raise ValueError("censored/invalid outcomes cannot carry observed labels")
        if self.failure_onset_step is not None and self.failure_onset_step < 0:
            raise ValueError("failure onset must be non-negative")
        if self.failure_within_horizon is True and self.failure_onset_step is None:
            raise ValueError("positive outcomes require a failure onset")
        if self.failure_within_horizon is False and self.failure_onset_step is not None:
            raise ValueError("negative outcomes cannot contain a failure onset")
        if self.failure_within_horizon is False and any(
            value is not None
            for value in (
                self.failure_category,
                self.failure_mechanism,
                self.recovery_eligible,
            )
        ):
            raise ValueError("negative outcomes cannot contain failure attributes")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "failure_within_horizon": self.failure_within_horizon,
            "eventual_episode_failure": self.eventual_episode_failure,
            "failure_onset_step": self.failure_onset_step,
            "failure_category": self.failure_category,
            "failure_mechanism": self.failure_mechanism,
            "recovery_eligible": self.recovery_eligible,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MonitorOutcome":
        _reject_unknown(
            data,
            {
                "status",
                "failure_within_horizon",
                "eventual_episode_failure",
                "failure_onset_step",
                "failure_category",
                "failure_mechanism",
                "recovery_eligible",
            },
            "monitor outcome",
        )
        for key in (
            "failure_within_horizon",
            "eventual_episode_failure",
            "recovery_eligible",
        ):
            if data.get(key) is not None:
                _boolean(data.get(key), key)
        return cls(
            status=str(data.get("status", "")),  # type: ignore[arg-type]
            failure_within_horizon=data.get("failure_within_horizon"),
            eventual_episode_failure=data.get("eventual_episode_failure"),
            failure_onset_step=int(data["failure_onset_step"])
            if data.get("failure_onset_step") is not None
            else None,
            failure_category=str(data["failure_category"])
            if data.get("failure_category") is not None
            else None,
            failure_mechanism=str(data["failure_mechanism"])
            if data.get("failure_mechanism") is not None
            else None,
            recovery_eligible=data.get("recovery_eligible"),
        )


@dataclass(frozen=True)
class MonitorPredictionRecord:
    prediction: MonitorPrediction
    outcome: MonitorOutcome
    intervention_branch_point_id: str | None = None

    def __post_init__(self) -> None:
        if (
            self.intervention_branch_point_id is not None
            and not self.prediction.intervention_recommended
        ):
            raise ValueError(
                "branch links require an intervention recommendation"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": MONITOR_RECORD_FORMAT,
            "prediction": self.prediction.to_dict(),
            "outcome": self.outcome.to_dict(),
            "intervention_branch_point_id": self.intervention_branch_point_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MonitorPredictionRecord":
        _check_format(data, MONITOR_RECORD_FORMAT, "monitor record")
        _reject_unknown(
            data,
            {
                "format",
                "prediction",
                "outcome",
                "intervention_branch_point_id",
            },
            "monitor record",
        )
        prediction = data.get("prediction")
        outcome = data.get("outcome")
        if not isinstance(prediction, Mapping) or not isinstance(outcome, Mapping):
            raise ValueError("monitor record prediction and outcome must be mappings")
        return cls(
            prediction=MonitorPrediction.from_dict(prediction),
            outcome=MonitorOutcome.from_dict(outcome),
            intervention_branch_point_id=str(data["intervention_branch_point_id"])
            if data.get("intervention_branch_point_id") is not None
            else None,
        )


def contract_sha256(contract: FailureMonitorContract) -> str:
    return str(contract.to_dict()["contract_sha256"])


def prediction_id(
    contract: FailureMonitorContract,
    *,
    task_id: str,
    episode_seed: int,
    episode_index: int,
    step_index: int,
) -> str:
    return "monitor-prediction-" + _sha256(
        {
            "contract_sha256": contract_sha256(contract),
            "task_id": task_id,
            "episode_seed": episode_seed,
            "episode_index": episode_index,
            "step_index": step_index,
        }
    )[:24]


def _check_format(data: Mapping[str, Any], expected: str, label: str) -> None:
    if data.get("format") != expected:
        raise ValueError(f"unsupported {label} format: {data.get('format')}")


def _reject_unknown(data: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"unknown {label} fields: {', '.join(unknown)}")


def _require_sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def _validate_unique_text(values: tuple[str, ...], label: str) -> None:
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{label} must contain non-empty strings")
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique")


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
