from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, cast


BRANCH_POINT_FORMAT = "nyssa-counterfactual-branch-point-v1"
BRANCH_OUTCOME_FORMAT = "nyssa-counterfactual-branch-outcome-v1"
COUNTERFACTUAL_RECOVERY_FORMAT = "nyssa-counterfactual-recovery-v1"
COUNTERFACTUAL_RECOVERY_MANIFEST_FORMAT = "nyssa-counterfactual-recovery-manifest-v1"

BranchKind = Literal["continue", "recovery", "oracle"]
BranchStatus = Literal["completed", "error"]
RestorationGrade = Literal["exact", "qualified", "unsupported"]


@dataclass(frozen=True)
class RestoreCapability:
    component: str
    component_id: str
    required: bool
    supported: bool
    fidelity: str
    captures_rng: bool
    exact: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        if not self.component.strip():
            raise ValueError("restore capability component must be non-empty")
        if not self.component_id.strip():
            raise ValueError("restore capability component_id must be non-empty")
        if not self.fidelity.strip():
            raise ValueError("restore capability fidelity must be non-empty")
        if self.exact and not self.supported:
            raise ValueError("an exact restore capability must be supported")
        if self.required and not self.supported and not self.reason:
            raise ValueError("unsupported required components must explain why")

    def to_dict(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "component_id": self.component_id,
            "required": self.required,
            "supported": self.supported,
            "fidelity": self.fidelity,
            "captures_rng": self.captures_rng,
            "exact": self.exact,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RestoreCapability":
        _require_fields(
            data,
            {
                "component",
                "component_id",
                "required",
                "supported",
                "fidelity",
                "captures_rng",
                "exact",
            },
            "restore capability",
        )
        _reject_unknown(
            data,
            {
                "component",
                "component_id",
                "required",
                "supported",
                "fidelity",
                "captures_rng",
                "exact",
                "reason",
            },
            "restore capability",
        )
        return cls(
            component=str(data.get("component", "")),
            component_id=str(data.get("component_id", "")),
            required=_boolean(data, "required", "restore capability"),
            supported=_boolean(data, "supported", "restore capability"),
            fidelity=str(data.get("fidelity", "")),
            captures_rng=_boolean(data, "captures_rng", "restore capability"),
            exact=_boolean(data, "exact", "restore capability"),
            reason=str(data["reason"]) if data.get("reason") is not None else None,
        )


@dataclass(frozen=True)
class BranchPoint:
    branch_point_id: str
    task_id: str
    episode_index: int
    episode_seed: int
    step_index: int
    recovery_attempt_id: int
    requested_repeats: int
    requested_branches: tuple[BranchKind, ...]
    trigger_kind: str
    trigger_reason: str | None
    trigger_event_id: str | None
    snapshot_sha256: str | None
    restoration_grade: RestorationGrade
    restore_capabilities: tuple[RestoreCapability, ...]
    matched_randomness: bool
    repeat_seed_strategy: str
    reseeded_components: tuple[str, ...]
    strongest_causal_claim_eligible: bool
    unsupported_reason: str | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        for name, value in (
            ("episode_index", self.episode_index),
            ("episode_seed", self.episode_seed),
            ("step_index", self.step_index),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.recovery_attempt_id <= 0:
            raise ValueError("recovery_attempt_id must be positive")
        if self.requested_repeats <= 0:
            raise ValueError("requested_repeats must be positive")
        if len(self.requested_branches) != len(set(self.requested_branches)):
            raise ValueError("requested_branches must be unique")
        if not {"continue", "recovery"} <= set(self.requested_branches):
            raise ValueError("requested_branches must include continue and recovery")
        if not self.branch_point_id.strip() or not self.task_id.strip():
            raise ValueError("branch_point_id and task_id must be non-empty")
        if not self.trigger_kind.strip():
            raise ValueError("trigger_kind must be non-empty")
        if not self.repeat_seed_strategy.strip():
            raise ValueError("repeat_seed_strategy must be non-empty")
        if self.schema_version != 1:
            raise ValueError(
                f"Unsupported branch-point schema version: {self.schema_version}"
            )
        if self.restoration_grade == "unsupported" and not self.unsupported_reason:
            raise ValueError("unsupported branch points must include a reason")
        if self.restoration_grade != "unsupported" and not self.snapshot_sha256:
            raise ValueError("executable branch points require a snapshot fingerprint")
        if self.snapshot_sha256 is not None:
            _validate_sha256(self.snapshot_sha256, "branch snapshot")
        components = [item.component for item in self.restore_capabilities]
        if len(components) != len(set(components)):
            raise ValueError("restore capabilities must have unique components")
        required_components = {"engine", "policy", "stressors", "process_rng"}
        if not required_components <= set(components):
            missing = sorted(required_components - set(components))
            raise ValueError(
                "branch point is missing required restore capabilities: "
                f"{', '.join(missing)}"
            )
        required = [item for item in self.restore_capabilities if item.required]
        if self.restoration_grade == "unsupported" and all(
            item.supported for item in required
        ):
            raise ValueError(
                "unsupported restoration grade requires an unsupported component"
            )
        if self.restoration_grade != "unsupported" and not all(
            item.supported for item in required
        ):
            raise ValueError(
                "executable branch points require every required component"
            )
        if self.restoration_grade == "exact" and not all(
            item.exact for item in required
        ):
            raise ValueError("exact restoration requires exact component contracts")
        if self.restoration_grade == "unsupported" and self.matched_randomness:
            raise ValueError("unsupported branch points cannot claim matched randomness")
        if self.strongest_causal_claim_eligible and (
            self.restoration_grade != "exact" or not self.matched_randomness
        ):
            raise ValueError(
                "strongest causal claim eligibility requires exact matched restoration"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": BRANCH_POINT_FORMAT,
            "schema_version": self.schema_version,
            "branch_point_id": self.branch_point_id,
            "task_id": self.task_id,
            "episode_index": self.episode_index,
            "episode_seed": self.episode_seed,
            "step_index": self.step_index,
            "recovery_attempt_id": self.recovery_attempt_id,
            "requested_repeats": self.requested_repeats,
            "requested_branches": list(self.requested_branches),
            "trigger": {
                "kind": self.trigger_kind,
                "reason": self.trigger_reason,
                "failure_event_id": self.trigger_event_id,
            },
            "snapshot_sha256": self.snapshot_sha256,
            "restoration_grade": self.restoration_grade,
            "restore_capabilities": [
                item.to_dict() for item in self.restore_capabilities
            ],
            "matched_randomness": self.matched_randomness,
            "repeat_seed_strategy": self.repeat_seed_strategy,
            "reseeded_components": list(self.reseeded_components),
            "strongest_causal_claim_eligible": self.strongest_causal_claim_eligible,
            "unsupported_reason": self.unsupported_reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BranchPoint":
        _validate_format(data, BRANCH_POINT_FORMAT, "branch point")
        _require_fields(
            data,
            {
                "schema_version",
                "branch_point_id",
                "task_id",
                "episode_index",
                "episode_seed",
                "step_index",
                "recovery_attempt_id",
                "requested_repeats",
                "requested_branches",
                "trigger",
                "snapshot_sha256",
                "restoration_grade",
                "restore_capabilities",
                "matched_randomness",
                "repeat_seed_strategy",
                "reseeded_components",
                "strongest_causal_claim_eligible",
                "unsupported_reason",
            },
            "branch point",
        )
        _reject_unknown(
            data,
            {
                "format",
                "schema_version",
                "branch_point_id",
                "task_id",
                "episode_index",
                "episode_seed",
                "step_index",
                "recovery_attempt_id",
                "requested_repeats",
                "requested_branches",
                "trigger",
                "snapshot_sha256",
                "restoration_grade",
                "restore_capabilities",
                "matched_randomness",
                "repeat_seed_strategy",
                "reseeded_components",
                "strongest_causal_claim_eligible",
                "unsupported_reason",
            },
            "branch point",
        )
        trigger = _mapping(data.get("trigger"), "branch-point trigger")
        _require_fields(
            trigger,
            {"kind", "reason", "failure_event_id"},
            "branch-point trigger",
        )
        _reject_unknown(
            trigger,
            {"kind", "reason", "failure_event_id"},
            "branch-point trigger",
        )
        raw_capabilities = data.get("restore_capabilities", [])
        if not isinstance(raw_capabilities, list):
            raise ValueError("branch-point restore_capabilities must be a list")
        restoration_grade = str(data.get("restoration_grade", ""))
        if restoration_grade not in {"exact", "qualified", "unsupported"}:
            raise ValueError(f"unsupported restoration grade: {restoration_grade}")
        reseeded = data.get("reseeded_components", [])
        if not isinstance(reseeded, list) or not all(
            isinstance(item, str) for item in reseeded
        ):
            raise ValueError("reseeded_components must be a list of strings")
        requested_branches = data.get("requested_branches")
        if not isinstance(requested_branches, list) or not all(
            item in {"continue", "recovery", "oracle"}
            for item in requested_branches
        ):
            raise ValueError("requested_branches contains an unsupported branch")
        return cls(
            branch_point_id=str(data.get("branch_point_id", "")),
            task_id=str(data.get("task_id", "")),
            episode_index=int(data.get("episode_index", -1)),
            episode_seed=int(data.get("episode_seed", -1)),
            step_index=int(data.get("step_index", -1)),
            recovery_attempt_id=int(data.get("recovery_attempt_id", 0)),
            requested_repeats=int(data.get("requested_repeats", 0)),
            requested_branches=tuple(
                cast(BranchKind, item) for item in requested_branches
            ),
            trigger_kind=str(trigger.get("kind", "")),
            trigger_reason=str(trigger["reason"])
            if trigger.get("reason") is not None
            else None,
            trigger_event_id=str(trigger["failure_event_id"])
            if trigger.get("failure_event_id") is not None
            else None,
            snapshot_sha256=str(data["snapshot_sha256"])
            if data.get("snapshot_sha256") is not None
            else None,
            restoration_grade=cast(RestorationGrade, restoration_grade),
            restore_capabilities=tuple(
                RestoreCapability.from_dict(_mapping(item, "restore capability"))
                for item in raw_capabilities
            ),
            matched_randomness=_boolean(data, "matched_randomness", "branch point"),
            repeat_seed_strategy=str(data.get("repeat_seed_strategy", "")),
            reseeded_components=tuple(reseeded),
            strongest_causal_claim_eligible=_boolean(
                data, "strongest_causal_claim_eligible", "branch point"
            ),
            unsupported_reason=str(data["unsupported_reason"])
            if data.get("unsupported_reason") is not None
            else None,
            schema_version=int(data.get("schema_version", 1)),
        )


@dataclass(frozen=True)
class BranchStep:
    offset: int
    action: Any
    reward: float
    terminated: bool
    truncated: bool
    success: bool
    safety_violation: bool
    damage_event_count: float

    def __post_init__(self) -> None:
        if self.offset < 0:
            raise ValueError("branch-step offset must be non-negative")
        if not math.isfinite(float(self.reward)):
            raise ValueError("branch-step reward must be finite")
        if not math.isfinite(float(self.damage_event_count)):
            raise ValueError("branch-step damage count must be finite")
        if self.damage_event_count < 0:
            raise ValueError("branch-step damage count must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "offset": self.offset,
            "action": _jsonable(self.action),
            "reward": float(self.reward),
            "terminated": self.terminated,
            "truncated": self.truncated,
            "success": self.success,
            "safety_violation": self.safety_violation,
            "damage_event_count": float(self.damage_event_count),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BranchStep":
        _require_fields(
            data,
            {
                "offset",
                "action",
                "reward",
                "terminated",
                "truncated",
                "success",
                "safety_violation",
                "damage_event_count",
            },
            "branch step",
        )
        _reject_unknown(
            data,
            {
                "offset",
                "action",
                "reward",
                "terminated",
                "truncated",
                "success",
                "safety_violation",
                "damage_event_count",
            },
            "branch step",
        )
        return cls(
            offset=int(data.get("offset", -1)),
            action=data.get("action"),
            reward=float(data.get("reward", float("nan"))),
            terminated=_boolean(data, "terminated", "branch step"),
            truncated=_boolean(data, "truncated", "branch step"),
            success=_boolean(data, "success", "branch step"),
            safety_violation=_boolean(data, "safety_violation", "branch step"),
            damage_event_count=float(data.get("damage_event_count", 0.0)),
        )


@dataclass(frozen=True)
class BranchOutcome:
    branch_point_id: str
    branch_kind: BranchKind
    repeat_index: int
    branch_seed: int
    status: BranchStatus
    success: bool
    terminated: bool
    truncated: bool
    total_reward: float
    terminal_reason: str
    initial_action_count: int
    trajectory_sha256: str | None
    matched_rng_sha256: str
    steps: tuple[BranchStep, ...] = field(default_factory=tuple)
    error_type: str | None = None
    error_message: str | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.branch_point_id.strip():
            raise ValueError("branch outcome branch_point_id must be non-empty")
        if self.branch_kind not in {"continue", "recovery", "oracle"}:
            raise ValueError(f"unsupported branch kind: {self.branch_kind}")
        if self.status not in {"completed", "error"}:
            raise ValueError(f"unsupported branch status: {self.status}")
        if self.repeat_index < 0 or self.branch_seed < 0:
            raise ValueError("repeat_index and branch_seed must be non-negative")
        if self.initial_action_count < 0:
            raise ValueError("initial_action_count must be non-negative")
        if not math.isfinite(float(self.total_reward)):
            raise ValueError("branch total_reward must be finite")
        if not self.terminal_reason.strip() or not self.matched_rng_sha256.strip():
            raise ValueError(
                "terminal_reason and matched_rng_sha256 must be non-empty"
            )
        if self.status == "completed" and self.error_type is not None:
            raise ValueError("completed outcomes cannot include an error")
        if self.status == "error" and not self.error_type:
            raise ValueError("error outcomes must include error_type")
        if self.status == "completed" and not self.trajectory_sha256:
            raise ValueError("completed outcomes require a trajectory fingerprint")
        if self.trajectory_sha256 is not None:
            _validate_sha256(self.trajectory_sha256, "branch trajectory")
        _validate_sha256(self.matched_rng_sha256, "matched RNG state")
        if self.schema_version != 1:
            raise ValueError(
                f"Unsupported branch-outcome schema version: {self.schema_version}"
            )

    @property
    def safety_event_count(self) -> int:
        return sum(step.safety_violation for step in self.steps)

    @property
    def damage_event_count(self) -> float:
        return sum(step.damage_event_count for step in self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": BRANCH_OUTCOME_FORMAT,
            "schema_version": self.schema_version,
            "branch_point_id": self.branch_point_id,
            "branch_kind": self.branch_kind,
            "repeat_index": self.repeat_index,
            "branch_seed": self.branch_seed,
            "status": self.status,
            "success": self.success,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "steps_executed": len(self.steps),
            "total_reward": float(self.total_reward),
            "terminal_reason": self.terminal_reason,
            "initial_action_count": self.initial_action_count,
            "safety_event_count": self.safety_event_count,
            "damage_event_count": self.damage_event_count,
            "trajectory_sha256": self.trajectory_sha256,
            "matched_rng_sha256": self.matched_rng_sha256,
            "steps": [step.to_dict() for step in self.steps],
            "error": {
                "type": self.error_type,
                "message": self.error_message,
            }
            if self.error_type
            else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BranchOutcome":
        _validate_format(data, BRANCH_OUTCOME_FORMAT, "branch outcome")
        _require_fields(
            data,
            {
                "schema_version",
                "branch_point_id",
                "branch_kind",
                "repeat_index",
                "branch_seed",
                "status",
                "success",
                "terminated",
                "truncated",
                "steps_executed",
                "total_reward",
                "terminal_reason",
                "initial_action_count",
                "safety_event_count",
                "damage_event_count",
                "trajectory_sha256",
                "matched_rng_sha256",
                "steps",
                "error",
            },
            "branch outcome",
        )
        _reject_unknown(
            data,
            {
                "format",
                "schema_version",
                "branch_point_id",
                "branch_kind",
                "repeat_index",
                "branch_seed",
                "status",
                "success",
                "terminated",
                "truncated",
                "steps_executed",
                "total_reward",
                "terminal_reason",
                "initial_action_count",
                "safety_event_count",
                "damage_event_count",
                "trajectory_sha256",
                "matched_rng_sha256",
                "steps",
                "error",
            },
            "branch outcome",
        )
        branch_kind = str(data.get("branch_kind", ""))
        status = str(data.get("status", ""))
        if branch_kind not in {"continue", "recovery", "oracle"}:
            raise ValueError(f"unsupported branch kind: {branch_kind}")
        if status not in {"completed", "error"}:
            raise ValueError(f"unsupported branch status: {status}")
        raw_steps = data.get("steps", [])
        if not isinstance(raw_steps, list):
            raise ValueError("branch outcome steps must be a list")
        error = data.get("error")
        if error is not None:
            error = _mapping(error, "branch outcome error")
        outcome = cls(
            branch_point_id=str(data.get("branch_point_id", "")),
            branch_kind=cast(BranchKind, branch_kind),
            repeat_index=int(data.get("repeat_index", -1)),
            branch_seed=int(data.get("branch_seed", -1)),
            status=cast(BranchStatus, status),
            success=_boolean(data, "success", "branch outcome"),
            terminated=_boolean(data, "terminated", "branch outcome"),
            truncated=_boolean(data, "truncated", "branch outcome"),
            total_reward=float(data.get("total_reward", float("nan"))),
            terminal_reason=str(data.get("terminal_reason", "")),
            initial_action_count=int(data.get("initial_action_count", -1)),
            trajectory_sha256=str(data["trajectory_sha256"])
            if data.get("trajectory_sha256") is not None
            else None,
            matched_rng_sha256=str(data.get("matched_rng_sha256", "")),
            steps=tuple(
                BranchStep.from_dict(_mapping(item, "branch step"))
                for item in raw_steps
            ),
            error_type=str(error["type"])
            if error is not None and error.get("type") is not None
            else None,
            error_message=str(error["message"])
            if error is not None and error.get("message") is not None
            else None,
            schema_version=int(data.get("schema_version", 1)),
        )
        if int(data.get("steps_executed", len(outcome.steps))) != len(outcome.steps):
            raise ValueError("branch outcome steps_executed does not match steps")
        if int(data.get("safety_event_count", outcome.safety_event_count)) != outcome.safety_event_count:
            raise ValueError("branch outcome safety_event_count does not match steps")
        if not math.isclose(
            float(data.get("damage_event_count", outcome.damage_event_count)),
            outcome.damage_event_count,
        ):
            raise ValueError("branch outcome damage_event_count does not match steps")
        return outcome


@dataclass(frozen=True)
class CounterfactualRecoveryRecord:
    branch_point: BranchPoint
    outcomes: tuple[BranchOutcome, ...] = field(default_factory=tuple)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(
                f"Unsupported counterfactual recovery schema version: {self.schema_version}"
            )
        identities: set[tuple[str, int]] = set()
        for outcome in self.outcomes:
            if outcome.branch_point_id != self.branch_point.branch_point_id:
                raise ValueError("branch outcomes must reference their branch point")
            identity = (outcome.branch_kind, outcome.repeat_index)
            if identity in identities:
                raise ValueError(
                    "branch outcomes must be unique by branch kind and repeat index"
                )
            identities.add(identity)
            if outcome.repeat_index >= self.branch_point.requested_repeats:
                raise ValueError("branch outcome repeat exceeds requested repeats")
            if outcome.branch_kind not in self.branch_point.requested_branches:
                raise ValueError("branch outcome kind was not requested")
        if self.branch_point.restoration_grade == "unsupported" and self.outcomes:
            raise ValueError("unsupported branch points cannot contain outcomes")

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": COUNTERFACTUAL_RECOVERY_FORMAT,
            "schema_version": self.schema_version,
            "branch_point": self.branch_point.to_dict(),
            "outcomes": [outcome.to_dict() for outcome in self.outcomes],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CounterfactualRecoveryRecord":
        _validate_format(data, COUNTERFACTUAL_RECOVERY_FORMAT, "recovery record")
        _require_fields(
            data,
            {"schema_version", "branch_point", "outcomes"},
            "recovery record",
        )
        _reject_unknown(
            data,
            {"format", "schema_version", "branch_point", "outcomes"},
            "recovery record",
        )
        raw_outcomes = data.get("outcomes", [])
        if not isinstance(raw_outcomes, list):
            raise ValueError("recovery record outcomes must be a list")
        return cls(
            branch_point=BranchPoint.from_dict(
                _mapping(data.get("branch_point"), "branch point")
            ),
            outcomes=tuple(
                BranchOutcome.from_dict(_mapping(item, "branch outcome"))
                for item in raw_outcomes
            ),
            schema_version=int(data.get("schema_version", 1)),
        )


def _jsonable(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return dict(value)


def _validate_format(data: dict[str, Any], expected: str, label: str) -> None:
    if data.get("format") != expected:
        raise ValueError(f"Unsupported {label} format: {data.get('format')}")


def _reject_unknown(
    data: dict[str, Any], allowed: set[str], label: str
) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"Unknown {label} fields: {', '.join(unknown)}")


def _require_fields(data: dict[str, Any], required: set[str], label: str) -> None:
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"Missing {label} fields: {', '.join(missing)}")


def _boolean(data: dict[str, Any], key: str, label: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{label} {key} must be a boolean")
    return value


def _validate_sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} fingerprint must be a lowercase SHA-256 digest")
