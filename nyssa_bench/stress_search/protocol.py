from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from nyssa_bench.failures.protocol import FailureEvent
from nyssa_bench.metrics.vector import validate_metric_vector
from nyssa_bench.stressors import StressorConfig, StressorContext, StressorSpec
from nyssa_bench.stressors.registry import make_stressor


STRESS_SEARCH_SPACE_FORMAT = "nyssa-stress-search-space-v1"
STRESS_SEARCH_STUDY_FORMAT = "nyssa-stress-search-study-v1"
STRESS_PROPOSAL_FORMAT = "nyssa-stress-proposal-v1"
STRESS_OBSERVATION_FORMAT = "nyssa-stress-observation-v1"

VariableKind = Literal["continuous", "integer", "categorical"]
ObservationStatus = Literal[
    "success",
    "policy_failure",
    "unsupported",
    "censored",
    "application_error",
    "invalid",
]


@dataclass(frozen=True)
class SearchVariable:
    variable_id: str
    stressor_id: str
    field: str
    kind: VariableKind
    lower: float | int | None = None
    upper: float | int | None = None
    choices: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        if not self.variable_id.strip() or not self.stressor_id.strip():
            raise ValueError("search variable identity must be non-empty")
        if self.field != "severity" and not self.field.startswith("parameters."):
            raise ValueError("search variable field must be severity or parameters.<name>")
        if self.field == "parameters.":
            raise ValueError("searched parameter name must be non-empty")
        if self.kind not in {"continuous", "integer", "categorical"}:
            raise ValueError(f"unsupported search variable kind: {self.kind}")
        if self.kind == "categorical":
            if len(self.choices) < 2:
                raise ValueError("categorical variables require at least two choices")
            serialized = [_canonical_json(choice) for choice in self.choices]
            if len(serialized) != len(set(serialized)):
                raise ValueError("categorical choices must be unique")
            _canonical_json(list(self.choices))
        else:
            lower = _finite(self.lower)
            upper = _finite(self.upper)
            if lower is None or upper is None or lower >= upper:
                raise ValueError("numeric variables require finite increasing bounds")
            if self.kind == "integer" and (
                not float(lower).is_integer() or not float(upper).is_integer()
            ):
                raise ValueError("integer variable bounds must be integral")
        if self.field == "severity" and (
            self.kind != "continuous"
            or float(self.lower or 0.0) < 0.0
            or float(self.upper or 0.0) > 1.0
        ):
            raise ValueError("severity variables must be continuous within [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "variable_id": self.variable_id,
            "stressor_id": self.stressor_id,
            "field": self.field,
            "kind": self.kind,
            "lower": self.lower,
            "upper": self.upper,
            "choices": list(self.choices),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SearchVariable":
        _reject_unknown(
            data,
            {"variable_id", "stressor_id", "field", "kind", "lower", "upper", "choices"},
            "search variable",
        )
        choices = data.get("choices", [])
        if not isinstance(choices, list):
            raise ValueError("search variable choices must be a list")
        return cls(
            variable_id=str(data.get("variable_id", "")),
            stressor_id=str(data.get("stressor_id", "")),
            field=str(data.get("field", "")),
            kind=str(data.get("kind", "")),  # type: ignore[arg-type]
            lower=data.get("lower"),
            upper=data.get("upper"),
            choices=tuple(choices),
        )


@dataclass(frozen=True)
class SearchConstraint:
    constraint_id: str
    kind: Literal["sum_le", "sum_ge", "forbidden_combination"]
    variables: tuple[str, ...]
    bound: float | None = None
    values: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.constraint_id.strip() or not self.variables:
            raise ValueError("constraint identity and variables are required")
        if len(self.variables) != len(set(self.variables)):
            raise ValueError("constraint variables must be unique")
        if self.kind not in {"sum_le", "sum_ge", "forbidden_combination"}:
            raise ValueError(f"unsupported constraint kind: {self.kind}")
        if self.kind in {"sum_le", "sum_ge"} and _finite(self.bound) is None:
            raise ValueError("sum constraints require a finite bound")
        if self.kind == "forbidden_combination" and set(self.values) != set(
            self.variables
        ):
            raise ValueError("forbidden combinations require one value per variable")
        _canonical_json(self.values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "constraint_id": self.constraint_id,
            "kind": self.kind,
            "variables": list(self.variables),
            "bound": self.bound,
            "values": self.values,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SearchConstraint":
        _reject_unknown(
            data,
            {"constraint_id", "kind", "variables", "bound", "values"},
            "search constraint",
        )
        variables = data.get("variables")
        values = data.get("values", {})
        if not isinstance(variables, list) or not all(
            isinstance(item, str) for item in variables
        ):
            raise ValueError("constraint variables must be a list of strings")
        if not isinstance(values, Mapping):
            raise ValueError("constraint values must be a mapping")
        return cls(
            constraint_id=str(data.get("constraint_id", "")),
            kind=str(data.get("kind", "")),  # type: ignore[arg-type]
            variables=tuple(variables),
            bound=float(data["bound"]) if data.get("bound") is not None else None,
            values=dict(values),
        )


@dataclass(frozen=True)
class StressSearchSpace:
    space_id: str
    engine_name: str
    task_id: str
    variables: tuple[SearchVariable, ...]
    observation_mode: str | None = None
    action_mode: str | None = None
    constraints: tuple[SearchConstraint, ...] = ()
    fixed_parameters: dict[str, dict[str, Any]] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"unsupported search-space version: {self.schema_version}")
        if not self.space_id.strip() or not self.engine_name.strip() or not self.task_id.strip():
            raise ValueError("search-space identity, engine, and task are required")
        if not self.variables:
            raise ValueError("search space requires at least one variable")
        variable_ids = [variable.variable_id for variable in self.variables]
        if len(variable_ids) != len(set(variable_ids)):
            raise ValueError("search variable IDs must be unique")
        known = set(variable_ids)
        variables_by_id = {
            variable.variable_id: variable for variable in self.variables
        }
        for constraint in self.constraints:
            unknown = set(constraint.variables) - known
            if unknown:
                raise ValueError(
                    f"constraint {constraint.constraint_id} references unknown variables: "
                    + ", ".join(sorted(unknown))
                )
            if constraint.kind in {"sum_le", "sum_ge"} and any(
                variables_by_id[variable_id].kind == "categorical"
                for variable_id in constraint.variables
            ):
                raise ValueError("sum constraints cannot include categorical variables")
            if constraint.kind == "forbidden_combination":
                for variable_id, value in constraint.values.items():
                    _normalize_value(variables_by_id[variable_id], value)
        stressor_ids = {variable.stressor_id for variable in self.variables}
        unknown_fixed = set(self.fixed_parameters) - stressor_ids
        if unknown_fixed:
            raise ValueError(
                "fixed parameters reference stressors outside the search variables: "
                + ", ".join(sorted(unknown_fixed))
            )
        searched_fields: set[tuple[str, str]] = set()
        for variable in self.variables:
            identity = (variable.stressor_id, variable.field)
            if identity in searched_fields:
                raise ValueError(
                    f"search field is declared more than once: {variable.stressor_id}.{variable.field}"
                )
            searched_fields.add(identity)
            if (
                variable.field.startswith("parameters.")
                and variable.field.split(".", 1)[1]
                in self.fixed_parameters.get(variable.stressor_id, {})
            ):
                raise ValueError(
                    f"searched parameter is also fixed: {variable.stressor_id}.{variable.field}"
                )
        for stressor_id in stressor_ids:
            severity_count = sum(
                variable.stressor_id == stressor_id and variable.field == "severity"
                for variable in self.variables
            )
            if severity_count != 1:
                raise ValueError(
                    f"search space requires exactly one severity variable for {stressor_id}"
                )
        context = StressorContext(
            engine_name=self.engine_name,
            task_id=self.task_id,
            observation_mode=self.observation_mode,
            action_mode=self.action_mode,
        )
        representative = {
            variable.variable_id: _representative_value(variable)
            for variable in self.variables
        }
        representative_specs = {
            spec.stressor_id: spec
            for spec in self._stressor_specs_from_normalized(representative, seed=0)
        }
        for stressor_id in sorted(stressor_ids):
            stressor = make_stressor(stressor_id)
            stressor.reset(representative_specs[stressor_id], seed=0)
            reason = stressor.support_reason(context)
            if reason is not None:
                raise ValueError(
                    f"stressor {stressor_id!r} is not executable in this search space: {reason}"
                )
        _canonical_json(self.fixed_parameters)

    def validate_point(self, point: Mapping[str, Any]) -> dict[str, Any]:
        if set(point) != {variable.variable_id for variable in self.variables}:
            raise ValueError("search point must provide every variable exactly once")
        normalized = {
            variable.variable_id: _normalize_value(variable, point[variable.variable_id])
            for variable in self.variables
        }
        for constraint in self.constraints:
            if not _constraint_satisfied(constraint, normalized):
                raise ValueError(
                    f"search point violates constraint {constraint.constraint_id}"
                )
        return normalized

    def stressor_specs(self, point: Mapping[str, Any], *, seed: int) -> tuple[StressorSpec, ...]:
        normalized = self.validate_point(point)
        return self._stressor_specs_from_normalized(normalized, seed=seed)

    def stressor_config(
        self,
        proposal: "StressProposal",
        *,
        unsupported_policy: Literal["error", "record"] = "error",
    ) -> StressorConfig:
        if proposal.phase not in {"discovery", "confirmation"}:
            raise ValueError(f"unsupported proposal phase: {proposal.phase}")
        return StressorConfig(
            condition_id=f"stress-search:{proposal.proposal_id}",
            stressors=self.stressor_specs(
                proposal.point, seed=proposal.discovery_seed
            ),
            unsupported_policy=unsupported_policy,
        )

    def _stressor_specs_from_normalized(
        self, normalized: Mapping[str, Any], *, seed: int
    ) -> tuple[StressorSpec, ...]:
        grouped: dict[str, dict[str, Any]] = {
            variable.stressor_id: {
                "severity": None,
                "parameters": dict(self.fixed_parameters.get(variable.stressor_id, {})),
            }
            for variable in self.variables
        }
        for variable in self.variables:
            value = normalized[variable.variable_id]
            if variable.field == "severity":
                grouped[variable.stressor_id]["severity"] = value
            else:
                grouped[variable.stressor_id]["parameters"][
                    variable.field.split(".", 1)[1]
                ] = value
        specs = []
        for index, (stressor_id, values) in enumerate(sorted(grouped.items())):
            severity = values["severity"]
            if severity is None:
                raise ValueError(
                    f"search space must include a severity variable for {stressor_id}"
                )
            specs.append(
                StressorSpec(
                    stressor_id=stressor_id,
                    severity=float(severity),
                    parameters=values["parameters"],
                    seed=_derived_seed(seed, index, stressor_id),
                )
            )
        return tuple(specs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": STRESS_SEARCH_SPACE_FORMAT,
            "schema_version": self.schema_version,
            "space_id": self.space_id,
            "engine_name": self.engine_name,
            "task_id": self.task_id,
            "observation_mode": self.observation_mode,
            "action_mode": self.action_mode,
            "variables": [variable.to_dict() for variable in self.variables],
            "constraints": [constraint.to_dict() for constraint in self.constraints],
            "fixed_parameters": self.fixed_parameters,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StressSearchSpace":
        if data.get("format") != STRESS_SEARCH_SPACE_FORMAT:
            raise ValueError(f"unsupported search-space format: {data.get('format')}")
        _reject_unknown(
            data,
            {
                "format",
                "schema_version",
                "space_id",
                "engine_name",
                "task_id",
                "observation_mode",
                "action_mode",
                "variables",
                "constraints",
                "fixed_parameters",
            },
            "search space",
        )
        variables = data.get("variables")
        constraints = data.get("constraints", [])
        fixed = data.get("fixed_parameters", {})
        if (
            not isinstance(variables, list)
            or not all(isinstance(item, Mapping) for item in variables)
            or not isinstance(constraints, list)
            or not all(isinstance(item, Mapping) for item in constraints)
        ):
            raise ValueError("search-space variables and constraints must be lists")
        if not isinstance(fixed, Mapping) or not all(
            isinstance(value, Mapping) for value in fixed.values()
        ):
            raise ValueError("fixed_parameters must map stressors to parameter mappings")
        return cls(
            space_id=str(data.get("space_id", "")),
            engine_name=str(data.get("engine_name", "")),
            task_id=str(data.get("task_id", "")),
            observation_mode=str(data["observation_mode"])
            if data.get("observation_mode") is not None
            else None,
            action_mode=str(data["action_mode"])
            if data.get("action_mode") is not None
            else None,
            variables=tuple(SearchVariable.from_dict(item) for item in variables),
            constraints=tuple(SearchConstraint.from_dict(item) for item in constraints),
            fixed_parameters={str(key): dict(value) for key, value in fixed.items()},
            schema_version=int(data.get("schema_version", 1)),
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict()).encode()).hexdigest()


@dataclass(frozen=True)
class StressProposal:
    proposal_id: str
    proposal_index: int
    point: dict[str, Any]
    discovery_seed: int
    phase: Literal["discovery", "confirmation"] = "discovery"
    parent_proposal_ids: tuple[str, ...] = ()
    acquisition: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.proposal_id.strip() or self.proposal_index < 0 or self.discovery_seed < 0:
            raise ValueError("proposal identity, index, and seed are invalid")
        if self.phase not in {"discovery", "confirmation"}:
            raise ValueError(f"unsupported proposal phase: {self.phase}")
        if len(self.parent_proposal_ids) != len(set(self.parent_proposal_ids)):
            raise ValueError("proposal parent IDs must be unique")
        if self.proposal_id in self.parent_proposal_ids:
            raise ValueError("proposal cannot be its own parent")
        _canonical_json(self.point)
        _canonical_json(self.acquisition)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": STRESS_PROPOSAL_FORMAT,
            "proposal_id": self.proposal_id,
            "proposal_index": self.proposal_index,
            "point": self.point,
            "discovery_seed": self.discovery_seed,
            "phase": self.phase,
            "parent_proposal_ids": list(self.parent_proposal_ids),
            "acquisition": self.acquisition,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StressProposal":
        if data.get("format") != STRESS_PROPOSAL_FORMAT:
            raise ValueError(f"unsupported proposal format: {data.get('format')}")
        _reject_unknown(
            data,
            {
                "format",
                "proposal_id",
                "proposal_index",
                "point",
                "discovery_seed",
                "phase",
                "parent_proposal_ids",
                "acquisition",
            },
            "stress proposal",
        )
        point = data.get("point")
        parents = data.get("parent_proposal_ids", [])
        acquisition = data.get("acquisition", {})
        if not isinstance(point, Mapping) or not isinstance(acquisition, Mapping):
            raise ValueError("proposal point and acquisition must be mappings")
        if not isinstance(parents, list) or not all(
            isinstance(item, str) for item in parents
        ):
            raise ValueError("proposal parents must be a list of strings")
        return cls(
            proposal_id=str(data.get("proposal_id", "")),
            proposal_index=int(data.get("proposal_index", -1)),
            point=dict(point),
            discovery_seed=int(data.get("discovery_seed", -1)),
            phase=str(data.get("phase", "discovery")),  # type: ignore[arg-type]
            parent_proposal_ids=tuple(parents),
            acquisition=dict(acquisition),
        )


@dataclass(frozen=True)
class StressObservation:
    proposal_id: str
    status: ObservationStatus
    success: bool | None
    metric_vector: dict[str, Any] | None
    failure_events: tuple[dict[str, Any], ...] = ()
    safety_events: tuple[dict[str, Any], ...] = ()
    latency_ms: float | None = None
    recovery_gain: float | None = None
    reason: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    application_evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.proposal_id.strip():
            raise ValueError("observation proposal_id must be non-empty")
        if self.status not in {
            "success",
            "policy_failure",
            "unsupported",
            "censored",
            "application_error",
            "invalid",
        }:
            raise ValueError(f"unsupported observation status: {self.status}")
        if self.status in {"success", "policy_failure"}:
            if self.success is not (self.status == "success"):
                raise ValueError("observation status and success disagree")
            if not isinstance(self.metric_vector, dict):
                raise ValueError("policy outcomes require a metric vector")
            if self.metric_vector.get("format") != "nyssa-metric-vector-v1":
                raise ValueError("policy outcomes require nyssa-metric-vector-v1 evidence")
            validate_metric_vector(self.metric_vector)
            if not self.provenance.get("source") or not self.provenance.get(
                "source_id"
            ):
                raise ValueError(
                    "policy outcomes require source and source_id provenance"
                )
            if not self.application_evidence:
                raise ValueError("policy outcomes require stressor application evidence")
        elif self.success is not None:
            raise ValueError("non-policy outcomes cannot set success")
        elif self.metric_vector is not None:
            raise ValueError("non-policy outcomes cannot attach policy metric vectors")
        if self.status == "policy_failure" and not self.failure_events:
            raise ValueError("policy failures require temporal failure-event evidence")
        for event in self.failure_events:
            FailureEvent.from_dict(event)
        if any(not isinstance(event, dict) for event in self.safety_events):
            raise ValueError("safety events must be mappings")
        if self.status not in {"success", "policy_failure"} and not self.reason:
            raise ValueError("non-policy outcomes require a reason")
        if self.latency_ms is not None and (
            not math.isfinite(float(self.latency_ms)) or self.latency_ms < 0
        ):
            raise ValueError("latency must be finite and non-negative")
        if self.recovery_gain is not None and (
            not math.isfinite(float(self.recovery_gain))
            or not -1.0 <= self.recovery_gain <= 1.0
        ):
            raise ValueError("recovery gain must be finite and within [-1, 1]")
        _canonical_json(self.metric_vector)
        _canonical_json(list(self.failure_events))
        _canonical_json(list(self.safety_events))
        _canonical_json(self.provenance)
        _canonical_json(self.application_evidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": STRESS_OBSERVATION_FORMAT,
            "proposal_id": self.proposal_id,
            "status": self.status,
            "success": self.success,
            "metric_vector": self.metric_vector,
            "failure_events": list(self.failure_events),
            "safety_events": list(self.safety_events),
            "latency_ms": self.latency_ms,
            "recovery_gain": self.recovery_gain,
            "reason": self.reason,
            "provenance": self.provenance,
            "application_evidence": self.application_evidence,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StressObservation":
        if data.get("format") != STRESS_OBSERVATION_FORMAT:
            raise ValueError(f"unsupported observation format: {data.get('format')}")
        _reject_unknown(
            data,
            {
                "format",
                "proposal_id",
                "status",
                "success",
                "metric_vector",
                "failure_events",
                "safety_events",
                "latency_ms",
                "recovery_gain",
                "reason",
                "provenance",
                "application_evidence",
            },
            "stress observation",
        )
        failure_events = data.get("failure_events", [])
        safety_events = data.get("safety_events", [])
        metric_vector = data.get("metric_vector")
        provenance = data.get("provenance", {})
        application_evidence = data.get("application_evidence", {})
        if not isinstance(failure_events, list) or not all(
            isinstance(item, dict) for item in failure_events
        ):
            raise ValueError("failure_events must be a list of mappings")
        if not isinstance(safety_events, list) or not all(
            isinstance(item, dict) for item in safety_events
        ):
            raise ValueError("safety_events must be a list of mappings")
        if metric_vector is not None and not isinstance(metric_vector, dict):
            raise ValueError("metric_vector must be a mapping or null")
        if not isinstance(provenance, Mapping):
            raise ValueError("observation provenance must be a mapping")
        if not isinstance(application_evidence, Mapping):
            raise ValueError("application_evidence must be a mapping")
        return cls(
            proposal_id=str(data.get("proposal_id", "")),
            status=str(data.get("status", "")),  # type: ignore[arg-type]
            success=data.get("success"),
            metric_vector=metric_vector,
            failure_events=tuple(failure_events),
            safety_events=tuple(safety_events),
            latency_ms=float(data["latency_ms"])
            if data.get("latency_ms") is not None
            else None,
            recovery_gain=float(data["recovery_gain"])
            if data.get("recovery_gain") is not None
            else None,
            reason=str(data["reason"]) if data.get("reason") is not None else None,
            provenance=dict(provenance),
            application_evidence=dict(application_evidence),
        )


def _normalize_value(variable: SearchVariable, value: Any) -> Any:
    if variable.kind == "categorical":
        if value not in variable.choices:
            raise ValueError(f"{variable.variable_id} is outside its choices")
        return value
    numeric = _finite(value)
    lower, upper = _numeric_bounds(variable)
    if numeric is None or numeric < lower or numeric > upper:
        raise ValueError(f"{variable.variable_id} is outside its bounds")
    if variable.kind == "integer":
        if not float(numeric).is_integer():
            raise ValueError(f"{variable.variable_id} must be integral")
        return int(numeric)
    return float(numeric)


def _representative_value(variable: SearchVariable) -> Any:
    if variable.kind == "categorical":
        return variable.choices[0]
    lower, upper = _numeric_bounds(variable)
    midpoint = (lower + upper) / 2.0
    return int(round(midpoint)) if variable.kind == "integer" else midpoint


def _constraint_satisfied(constraint: SearchConstraint, point: Mapping[str, Any]) -> bool:
    if constraint.kind == "forbidden_combination":
        return not all(point[key] == constraint.values[key] for key in constraint.variables)
    values = [_finite(point[key]) for key in constraint.variables]
    if any(value is None for value in values):
        return False
    total = sum(float(value) for value in values if value is not None)
    assert constraint.bound is not None
    return total <= constraint.bound if constraint.kind == "sum_le" else total >= constraint.bound


def _numeric_bounds(variable: SearchVariable) -> tuple[float, float]:
    lower = _finite(variable.lower)
    upper = _finite(variable.upper)
    if lower is None or upper is None:
        raise ValueError(f"{variable.variable_id} does not have numeric bounds")
    return lower, upper


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _derived_seed(seed: int, index: int, value: str) -> int:
    payload = f"nyssa-stress-search-seed-v1:{seed}:{index}:{value}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _reject_unknown(data: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"unknown {label} fields: {', '.join(unknown)}")
