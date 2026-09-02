from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, cast


FAILURE_EVIDENCE_FORMAT = "nyssa-failure-evidence-v1"
CAUSAL_HYPOTHESIS_FORMAT = "nyssa-causal-hypothesis-v1"
FAILURE_EVENT_FORMAT = "nyssa-failure-event-v1"
FAILURE_LEDGER_FORMAT = "nyssa-failure-ledger-v1"

FailureRole = Literal[
    "symptom",
    "mechanism",
    "candidate_cause",
    "contributing_condition",
    "consequence",
]
TemporalPrecision = Literal["exact_step", "step_interval", "terminal_only", "unknown"]
EvidenceVisibility = Literal["policy_observable", "privileged", "external"]
ProvenanceSource = Literal[
    "simulator_state",
    "task_logic",
    "verifier_output",
    "policy_output",
    "stressor",
    "recovery",
    "human_annotation",
    "legacy_mapper",
    "external_monitor",
    "real_robot",
    "reconstructed_simulation",
]
RecoveryEligibility = Literal["eligible", "ineligible", "unknown"]

FAILURE_ROLES = frozenset(
    {
        "symptom",
        "mechanism",
        "candidate_cause",
        "contributing_condition",
        "consequence",
    }
)
TEMPORAL_PRECISIONS = frozenset(
    {"exact_step", "step_interval", "terminal_only", "unknown"}
)
EVIDENCE_VISIBILITIES = frozenset({"policy_observable", "privileged", "external"})
PROVENANCE_SOURCES = frozenset(
    {
        "simulator_state",
        "task_logic",
        "verifier_output",
        "policy_output",
        "stressor",
        "recovery",
        "human_annotation",
        "legacy_mapper",
        "external_monitor",
        "real_robot",
        "reconstructed_simulation",
    }
)
RECOVERY_ELIGIBILITIES = frozenset({"eligible", "ineligible", "unknown"})


@dataclass(frozen=True)
class FailureEvidence:
    evidence_id: str
    evidence_type: str
    payload: dict[str, Any]
    source: str
    annotation_source: str
    confidence: float
    visibility: EvidenceVisibility
    captured_step: int | None = None

    def __post_init__(self) -> None:
        _require_text(self.evidence_id, "evidence_id")
        _require_text(self.evidence_type, "evidence_type")
        _require_text(self.source, "evidence source")
        _require_text(self.annotation_source, "annotation_source")
        _validate_confidence(self.confidence, "evidence confidence")
        if self.visibility not in EVIDENCE_VISIBILITIES:
            raise ValueError(f"Unsupported evidence visibility: {self.visibility}")
        if self.captured_step is not None and self.captured_step < 0:
            raise ValueError("captured_step must be non-negative")

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        default_id: str | None = None,
        default_step: int | None = None,
        default_visibility: EvidenceVisibility | None = None,
    ) -> "FailureEvidence":
        _check_format(data, FAILURE_EVIDENCE_FORMAT, "failure evidence")
        _reject_unknown(
            data,
            {
                "format",
                "evidence_id",
                "evidence_type",
                "payload",
                "source",
                "annotation_source",
                "confidence",
                "visibility",
                "captured_step",
            },
            "failure evidence",
        )
        payload = data.get("payload", {})
        if not isinstance(payload, dict):
            raise ValueError("failure evidence payload must be a mapping")
        evidence_id = data.get("evidence_id", default_id)
        if evidence_id is None:
            raise ValueError("failure evidence requires evidence_id")
        visibility = data.get("visibility", default_visibility)
        if visibility is None:
            raise ValueError("failure evidence requires visibility")
        return cls(
            evidence_id=str(evidence_id),
            evidence_type=str(data.get("evidence_type", "diagnostic")),
            payload=dict(payload),
            source=str(data.get("source", "unknown")),
            annotation_source=str(data.get("annotation_source", "automatic")),
            confidence=float(data.get("confidence", 1.0)),
            visibility=str(visibility),  # type: ignore[arg-type]
            captured_step=int(data["captured_step"])
            if data.get("captured_step") is not None
            else default_step,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": FAILURE_EVIDENCE_FORMAT,
            "evidence_id": self.evidence_id,
            "evidence_type": self.evidence_type,
            "payload": _jsonable(self.payload),
            "source": self.source,
            "annotation_source": self.annotation_source,
            "confidence": self.confidence,
            "visibility": self.visibility,
            "captured_step": self.captured_step,
        }


@dataclass(frozen=True)
class CausalHypothesis:
    parent_event_id: str
    relationship: str = "candidate_parent"
    confidence: float = 0.5
    evidence_ids: tuple[str, ...] = ()
    rationale: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.parent_event_id, "parent_event_id")
        _require_text(self.relationship, "causal relationship")
        _validate_confidence(self.confidence, "relationship confidence")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("causal hypothesis evidence_ids must be unique")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CausalHypothesis":
        _check_format(data, CAUSAL_HYPOTHESIS_FORMAT, "causal hypothesis")
        _reject_unknown(
            data,
            {
                "format",
                "parent_event_id",
                "relationship",
                "confidence",
                "evidence_ids",
                "rationale",
                "semantics",
            },
            "causal hypothesis",
        )
        evidence_ids = data.get("evidence_ids", [])
        if not isinstance(evidence_ids, list):
            raise ValueError("causal hypothesis evidence_ids must be a list")
        semantics = data.get("semantics")
        if semantics not in {None, "hypothesis_only_not_established_causality"}:
            raise ValueError(f"Unsupported causal hypothesis semantics: {semantics}")
        return cls(
            parent_event_id=str(data.get("parent_event_id", "")),
            relationship=str(data.get("relationship", "candidate_parent")),
            confidence=float(data.get("confidence", 0.5)),
            evidence_ids=tuple(str(item) for item in evidence_ids),
            rationale=str(data["rationale"])
            if data.get("rationale") is not None
            else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": CAUSAL_HYPOTHESIS_FORMAT,
            "parent_event_id": self.parent_event_id,
            "relationship": self.relationship,
            "confidence": self.confidence,
            "evidence_ids": list(self.evidence_ids),
            "rationale": self.rationale,
            "semantics": "hypothesis_only_not_established_causality",
        }


@dataclass(frozen=True)
class EventProvenance:
    source: ProvenanceSource
    component_id: str
    annotation_source: str

    def __post_init__(self) -> None:
        if self.source not in PROVENANCE_SOURCES:
            raise ValueError(f"Unsupported failure provenance source: {self.source}")
        _require_text(self.component_id, "provenance component_id")
        _require_text(self.annotation_source, "provenance annotation_source")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EventProvenance":
        _reject_unknown(
            data,
            {"source", "component_id", "annotation_source"},
            "event provenance",
        )
        return cls(
            source=str(data.get("source", "legacy_mapper")),  # type: ignore[arg-type]
            component_id=str(data.get("component_id", "unknown")),
            annotation_source=str(data.get("annotation_source", "automatic")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "component_id": self.component_id,
            "annotation_source": self.annotation_source,
        }


@dataclass(frozen=True)
class FailureEventDraft:
    role: FailureRole
    category: str
    subtype: str
    onset_step: int
    end_step: int | None = None
    temporal_precision: TemporalPrecision = "exact_step"
    confidence: float = 1.0
    evidence: tuple[FailureEvidence, ...] = ()
    causal_hypotheses: tuple[CausalHypothesis, ...] = ()
    consequences: tuple[str, ...] = ()
    recovery_eligibility: RecoveryEligibility = "unknown"
    recovery_reason: str | None = None
    summary_label: str | None = None
    deduplication_key: str | None = None
    event_id: str | None = None

    def __post_init__(self) -> None:
        _validate_event_fields(
            role=self.role,
            category=self.category,
            subtype=self.subtype,
            onset_step=self.onset_step,
            end_step=self.end_step,
            temporal_precision=self.temporal_precision,
            confidence=self.confidence,
            evidence=self.evidence,
            causal_hypotheses=self.causal_hypotheses,
            recovery_eligibility=self.recovery_eligibility,
            event_id=self.event_id,
        )

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        default_step: int,
        default_event_id: str,
    ) -> "FailureEventDraft":
        allowed = {
            "event_id",
            "role",
            "category",
            "subtype",
            "onset_step",
            "end_step",
            "temporal_precision",
            "confidence",
            "evidence",
            "causal_hypotheses",
            "consequences",
            "recovery_eligibility",
            "recovery_reason",
            "summary_label",
            "deduplication_key",
        }
        _reject_unknown(data, allowed, "failure event draft")
        event_id = str(data.get("event_id", default_event_id))
        onset_step = int(data.get("onset_step", default_step))
        return cls(
            event_id=event_id,
            role=str(data.get("role", "symptom")),  # type: ignore[arg-type]
            category=str(data.get("category", "unknown")),
            subtype=str(data.get("subtype", "unknown_failure")),
            onset_step=onset_step,
            end_step=int(data["end_step"])
            if data.get("end_step") is not None
            else None,
            temporal_precision=str(  # type: ignore[arg-type]
                data.get("temporal_precision", "exact_step")
            ),
            confidence=float(data.get("confidence", 1.0)),
            evidence=_parse_evidence(
                data.get("evidence", []),
                event_id=event_id,
                default_step=onset_step,
            ),
            causal_hypotheses=_parse_hypotheses(data.get("causal_hypotheses", [])),
            consequences=_string_tuple(data.get("consequences", []), "consequences"),
            recovery_eligibility=str(  # type: ignore[arg-type]
                data.get("recovery_eligibility", "unknown")
            ),
            recovery_reason=str(data["recovery_reason"])
            if data.get("recovery_reason") is not None
            else None,
            summary_label=str(data["summary_label"])
            if data.get("summary_label") is not None
            else None,
            deduplication_key=str(data["deduplication_key"])
            if data.get("deduplication_key") is not None
            else None,
        )


@dataclass(frozen=True)
class FailureEvent:
    event_id: str
    role: FailureRole
    category: str
    subtype: str
    onset_step: int
    end_step: int | None
    temporal_precision: TemporalPrecision
    confidence: float
    evidence: tuple[FailureEvidence, ...]
    provenance: EventProvenance
    active_stressor_context: dict[str, Any] = field(default_factory=dict)
    causal_hypotheses: tuple[CausalHypothesis, ...] = ()
    consequences: tuple[str, ...] = ()
    recovery_eligibility: RecoveryEligibility = "unknown"
    recovery_reason: str | None = None
    summary_label: str | None = None
    deduplication_key: str | None = None

    def __post_init__(self) -> None:
        _validate_event_fields(
            role=self.role,
            category=self.category,
            subtype=self.subtype,
            onset_step=self.onset_step,
            end_step=self.end_step,
            temporal_precision=self.temporal_precision,
            confidence=self.confidence,
            evidence=self.evidence,
            causal_hypotheses=self.causal_hypotheses,
            recovery_eligibility=self.recovery_eligibility,
            event_id=self.event_id,
        )
        if any(
            item.parent_event_id == self.event_id for item in self.causal_hypotheses
        ):
            raise ValueError("failure events cannot name themselves as causal parents")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FailureEvent":
        _check_format(data, FAILURE_EVENT_FORMAT, "failure event")
        _reject_unknown(
            data,
            {
                "format",
                "event_id",
                "role",
                "category",
                "subtype",
                "onset_step",
                "end_step",
                "temporal_precision",
                "confidence",
                "evidence",
                "provenance",
                "active_stressor_context",
                "causal_hypotheses",
                "consequences",
                "recovery_eligibility",
                "recovery_reason",
                "summary_label",
                "deduplication_key",
            },
            "failure event",
        )
        event_id = str(data.get("event_id", ""))
        onset_step = int(data.get("onset_step", 0))
        provenance = data.get("provenance", {})
        stressors = data.get("active_stressor_context", {})
        if not isinstance(provenance, dict) or not isinstance(stressors, dict):
            raise ValueError("event provenance and stressor context must be mappings")
        return cls(
            event_id=event_id,
            role=str(data.get("role", "symptom")),  # type: ignore[arg-type]
            category=str(data.get("category", "unknown")),
            subtype=str(data.get("subtype", "unknown_failure")),
            onset_step=onset_step,
            end_step=int(data["end_step"])
            if data.get("end_step") is not None
            else None,
            temporal_precision=str(  # type: ignore[arg-type]
                data.get("temporal_precision", "unknown")
            ),
            confidence=float(data.get("confidence", 1.0)),
            evidence=_parse_evidence(
                data.get("evidence", {}),
                event_id=event_id,
                default_step=onset_step,
            ),
            provenance=EventProvenance.from_dict(provenance),
            active_stressor_context=dict(stressors),
            causal_hypotheses=_parse_hypotheses(data.get("causal_hypotheses", [])),
            consequences=_string_tuple(data.get("consequences", []), "consequences"),
            recovery_eligibility=str(  # type: ignore[arg-type]
                data.get("recovery_eligibility", "unknown")
            ),
            recovery_reason=str(data["recovery_reason"])
            if data.get("recovery_reason") is not None
            else None,
            summary_label=str(data["summary_label"])
            if data.get("summary_label") is not None
            else None,
            deduplication_key=str(data["deduplication_key"])
            if data.get("deduplication_key") is not None
            else None,
        )

    def to_dict(self) -> dict[str, Any]:
        evidence = {
            visibility: [
                item.to_dict()
                for item in self.evidence
                if item.visibility == visibility
            ]
            for visibility in (
                "policy_observable",
                "privileged",
                "external",
            )
        }
        return {
            "format": FAILURE_EVENT_FORMAT,
            "event_id": self.event_id,
            "role": self.role,
            "category": self.category,
            "subtype": self.subtype,
            "onset_step": self.onset_step,
            "end_step": self.end_step,
            "temporal_precision": self.temporal_precision,
            "confidence": self.confidence,
            "evidence": evidence,
            "provenance": self.provenance.to_dict(),
            "active_stressor_context": _jsonable(self.active_stressor_context),
            "causal_hypotheses": [item.to_dict() for item in self.causal_hypotheses],
            "consequences": list(self.consequences),
            "recovery_eligibility": self.recovery_eligibility,
            "recovery_reason": self.recovery_reason,
            "summary_label": self.summary_label,
            "deduplication_key": self.deduplication_key,
        }


@dataclass(frozen=True)
class FailureLedgerRecord:
    task_id: str
    episode_index: int
    episode_seed: int
    engine_name: str
    events: tuple[FailureEvent, ...]

    def __post_init__(self) -> None:
        _require_text(self.task_id, "ledger task_id")
        _require_text(self.engine_name, "ledger engine_name")
        if self.episode_index < 0 or self.episode_seed < 0:
            raise ValueError("ledger episode index and seed must be non-negative")
        event_ids = [event.event_id for event in self.events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("failure ledger event IDs must be unique")
        if tuple(sorted(self.events, key=failure_event_sort_key)) != self.events:
            raise ValueError(
                "failure ledger events must use canonical temporal ordering"
            )
        known_ids = set(event_ids)
        known_evidence_ids = {
            evidence.evidence_id for event in self.events for evidence in event.evidence
        }
        for event in self.events:
            unknown = {
                item.parent_event_id for item in event.causal_hypotheses
            } - known_ids
            if unknown:
                raise ValueError(
                    f"event '{event.event_id}' references unknown candidate parents: "
                    f"{', '.join(sorted(unknown))}"
                )
            unknown_evidence = {
                evidence_id
                for hypothesis in event.causal_hypotheses
                for evidence_id in hypothesis.evidence_ids
                if evidence_id not in known_evidence_ids
            }
            if unknown_evidence:
                raise ValueError(
                    f"event '{event.event_id}' causal hypotheses reference unknown "
                    f"evidence: {', '.join(sorted(unknown_evidence))}"
                )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FailureLedgerRecord":
        _check_format(data, FAILURE_LEDGER_FORMAT, "failure ledger")
        _reject_unknown(
            data,
            {
                "format",
                "task_id",
                "episode_index",
                "episode_seed",
                "engine_name",
                "ordering",
                "overlap_semantics",
                "deduplication_semantics",
                "causal_semantics",
                "events",
            },
            "failure ledger",
        )
        raw_events = data.get("events", [])
        if not isinstance(raw_events, list):
            raise ValueError("failure ledger events must be a list")
        semantics = {
            "ordering": "onset_step_then_end_step_then_event_id",
            "overlap_semantics": "allowed_without_implied_causality",
            "deduplication_semantics": "explicit_key_same_semantics_overlapping_interval",
            "causal_semantics": "candidate_hypotheses_only",
        }
        for key, expected in semantics.items():
            if data.get(key) != expected:
                raise ValueError(f"Unsupported failure ledger {key}: {data.get(key)}")
        return cls(
            task_id=str(data.get("task_id", "unknown")),
            episode_index=int(data.get("episode_index", 0)),
            episode_seed=int(data.get("episode_seed", 0)),
            engine_name=str(data.get("engine_name", "unknown")),
            events=tuple(FailureEvent.from_dict(dict(item)) for item in raw_events),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": FAILURE_LEDGER_FORMAT,
            "task_id": self.task_id,
            "episode_index": self.episode_index,
            "episode_seed": self.episode_seed,
            "engine_name": self.engine_name,
            "ordering": "onset_step_then_end_step_then_event_id",
            "overlap_semantics": "allowed_without_implied_causality",
            "deduplication_semantics": "explicit_key_same_semantics_overlapping_interval",
            "causal_semantics": "candidate_hypotheses_only",
            "events": [event.to_dict() for event in self.events],
        }


def failure_event_sort_key(event: FailureEvent) -> tuple[int, int, str]:
    end_step = event.end_step if event.end_step is not None else event.onset_step
    return (event.onset_step, end_step, event.event_id)


def _validate_event_fields(
    *,
    role: str,
    category: str,
    subtype: str,
    onset_step: int,
    end_step: int | None,
    temporal_precision: str,
    confidence: float,
    evidence: tuple[FailureEvidence, ...],
    causal_hypotheses: tuple[CausalHypothesis, ...],
    recovery_eligibility: str,
    event_id: str | None,
) -> None:
    if event_id is not None:
        _require_text(event_id, "event_id")
    if role not in FAILURE_ROLES:
        raise ValueError(f"Unsupported failure event role: {role}")
    _require_text(category, "failure event category")
    _require_text(subtype, "failure event subtype")
    if onset_step < 0:
        raise ValueError("failure event onset_step must be non-negative")
    if end_step is not None and end_step < onset_step:
        raise ValueError("failure event end_step cannot precede onset_step")
    if temporal_precision not in TEMPORAL_PRECISIONS:
        raise ValueError(f"Unsupported temporal precision: {temporal_precision}")
    if temporal_precision == "step_interval" and end_step is None:
        raise ValueError("step_interval events require end_step")
    _validate_confidence(confidence, "failure event confidence")
    evidence_ids = [item.evidence_id for item in evidence]
    if len(evidence_ids) != len(set(evidence_ids)):
        raise ValueError("failure event evidence IDs must be unique")
    hypothesis_keys = [
        (item.parent_event_id, item.relationship) for item in causal_hypotheses
    ]
    if len(hypothesis_keys) != len(set(hypothesis_keys)):
        raise ValueError("failure event causal hypotheses must be unique")
    if recovery_eligibility not in RECOVERY_ELIGIBILITIES:
        raise ValueError(f"Unsupported recovery eligibility: {recovery_eligibility}")


def _parse_evidence(
    value: Any, *, event_id: str, default_step: int
) -> tuple[FailureEvidence, ...]:
    flattened: list[tuple[dict[str, Any], EvidenceVisibility | None]] = []
    if isinstance(value, list):
        flattened.extend((dict(item), None) for item in value)
    elif isinstance(value, dict):
        unknown = set(value) - EVIDENCE_VISIBILITIES
        if unknown:
            raise ValueError(
                f"Unknown failure evidence visibility groups: {', '.join(sorted(unknown))}"
            )
        for visibility, items in value.items():
            if not isinstance(items, list):
                raise ValueError(f"evidence group '{visibility}' must be a list")
            flattened.extend(
                (dict(item), cast(EvidenceVisibility, visibility)) for item in items
            )
    else:
        raise ValueError("failure event evidence must be a list or visibility mapping")
    return tuple(
        FailureEvidence.from_dict(
            item,
            default_id=f"{event_id}:evidence:{index:03d}",
            default_step=default_step,
            default_visibility=visibility,
        )
        for index, (item, visibility) in enumerate(flattened)
    )


def _parse_hypotheses(value: Any) -> tuple[CausalHypothesis, ...]:
    if not isinstance(value, list):
        raise ValueError("causal_hypotheses must be a list")
    return tuple(CausalHypothesis.from_dict(dict(item)) for item in value)


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list")
    return tuple(str(item) for item in value)


def _check_format(data: dict[str, Any], expected: str, name: str) -> None:
    actual = data.get("format")
    if actual != expected:
        raise ValueError(f"Unsupported {name} format: {actual}")


def _reject_unknown(data: dict[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"Unknown {name} fields: {', '.join(unknown)}")


def _validate_confidence(value: float, name: str) -> None:
    if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be finite and within [0, 1]")


def _require_text(value: str, name: str) -> None:
    if not str(value).strip():
        raise ValueError(f"{name} must be non-empty")


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
