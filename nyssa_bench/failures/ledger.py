from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any, Iterable, cast

from nyssa_bench.failures.protocol import (
    CausalHypothesis,
    EventProvenance,
    FailureEvent,
    FailureEventDraft,
    FailureEvidence,
    FailureLedgerRecord,
    ProvenanceSource,
    RecoveryEligibility,
    failure_event_sort_key,
)


class FailureEventLedger:
    """Mutable episode ledger with deterministic ordering and explicit deduplication."""

    def __init__(
        self,
        *,
        task_id: str,
        episode_index: int,
        episode_seed: int,
        engine_name: str,
        stressor_context: dict[str, Any] | None = None,
    ) -> None:
        self.task_id = task_id
        self.episode_index = int(episode_index)
        self.episode_seed = int(episode_seed)
        self.engine_name = engine_name
        self._stressor_context = deepcopy(stressor_context or {})
        self._events: list[FailureEvent] = []
        self._event_sequence = 0

    @property
    def events(self) -> tuple[FailureEvent, ...]:
        return tuple(sorted(self._events, key=failure_event_sort_key))

    def set_stressor_context(self, context: dict[str, Any]) -> None:
        self._stressor_context = deepcopy(context)

    def emitter(
        self,
        source: ProvenanceSource,
        component_id: str,
        *,
        annotation_source: str = "automatic",
    ) -> "FailureEventEmitter":
        return FailureEventEmitter(
            self,
            EventProvenance(
                source=source,
                component_id=component_id,
                annotation_source=annotation_source,
            ),
        )

    def reserve_event_id(self) -> str:
        known_ids = {event.event_id for event in self._events}
        while True:
            self._event_sequence += 1
            candidate = f"event-{self._event_sequence:06d}"
            if candidate not in known_ids:
                return candidate

    def emit(self, event: FailureEvent) -> FailureEvent:
        existing_by_id = next(
            (item for item in self._events if item.event_id == event.event_id), None
        )
        if existing_by_id is not None:
            if existing_by_id == event:
                return existing_by_id
            raise ValueError(f"Conflicting failure event ID: {event.event_id}")

        duplicate = next(
            (item for item in self._events if _deduplication_match(item, event)),
            None,
        )
        if duplicate is None:
            self._events.append(event)
            return event

        merged = _merge_events(duplicate, event)
        self._events[self._events.index(duplicate)] = merged
        return merged

    def snapshot(self) -> FailureLedgerRecord:
        return FailureLedgerRecord(
            task_id=self.task_id,
            episode_index=self.episode_index,
            episode_seed=self.episode_seed,
            engine_name=self.engine_name,
            events=self.events,
        )


class FailureEventEmitter:
    """Provenance-scoped API used by runner components to emit event drafts."""

    def __init__(self, ledger: FailureEventLedger, provenance: EventProvenance) -> None:
        self.ledger = ledger
        self.provenance = provenance

    def emit(self, draft: FailureEventDraft) -> FailureEvent:
        event_id = draft.event_id or self.ledger.reserve_event_id()
        event = FailureEvent(
            event_id=event_id,
            role=draft.role,
            category=draft.category,
            subtype=draft.subtype,
            onset_step=draft.onset_step,
            end_step=draft.end_step,
            temporal_precision=draft.temporal_precision,
            confidence=draft.confidence,
            evidence=draft.evidence,
            provenance=self.provenance,
            active_stressor_context=deepcopy(self.ledger._stressor_context),
            causal_hypotheses=draft.causal_hypotheses,
            consequences=draft.consequences,
            recovery_eligibility=draft.recovery_eligibility,
            recovery_reason=draft.recovery_reason,
            summary_label=draft.summary_label,
            deduplication_key=draft.deduplication_key,
        )
        return self.ledger.emit(event)

    def emit_payload(
        self, payload: FailureEventDraft | dict[str, Any], *, default_step: int
    ) -> FailureEvent:
        if isinstance(payload, FailureEventDraft):
            return self.emit(payload)
        event_id = str(payload.get("event_id") or self.ledger.reserve_event_id())
        draft = FailureEventDraft.from_dict(
            payload,
            default_step=default_step,
            default_event_id=event_id,
        )
        return self.emit(draft)

    def emit_many(
        self,
        payloads: Iterable[FailureEventDraft | dict[str, Any]],
        *,
        default_step: int,
    ) -> list[FailureEvent]:
        return [
            self.emit_payload(payload, default_step=default_step)
            for payload in payloads
        ]


def _deduplication_match(left: FailureEvent, right: FailureEvent) -> bool:
    if not left.deduplication_key or left.deduplication_key != right.deduplication_key:
        return False
    same_semantics = (
        left.role,
        left.category,
        left.subtype,
        left.provenance,
        left.active_stressor_context,
    ) == (
        right.role,
        right.category,
        right.subtype,
        right.provenance,
        right.active_stressor_context,
    )
    return same_semantics and _events_overlap(left, right)


def _events_overlap(left: FailureEvent, right: FailureEvent) -> bool:
    left_end = left.end_step if left.end_step is not None else left.onset_step
    right_end = right.end_step if right.end_step is not None else right.onset_step
    return left.onset_step <= right_end and right.onset_step <= left_end


def _merge_events(left: FailureEvent, right: FailureEvent) -> FailureEvent:
    if (
        left.summary_label
        and right.summary_label
        and left.summary_label != right.summary_label
    ):
        raise ValueError("Cannot deduplicate events with conflicting summary labels")
    onset = min(left.onset_step, right.onset_step)
    end = max(
        left.end_step if left.end_step is not None else left.onset_step,
        right.end_step if right.end_step is not None else right.onset_step,
    )
    precision = (
        left.temporal_precision
        if onset == end and left.temporal_precision == right.temporal_precision
        else "step_interval"
    )
    return replace(
        left,
        onset_step=onset,
        end_step=end,
        temporal_precision=precision,
        confidence=max(left.confidence, right.confidence),
        evidence=_merge_evidence(left, right),
        causal_hypotheses=_merge_hypotheses(left, right),
        consequences=tuple(dict.fromkeys((*left.consequences, *right.consequences))),
        recovery_eligibility=cast(
            RecoveryEligibility, _merge_recovery_eligibility(left, right)
        ),
        recovery_reason=left.recovery_reason or right.recovery_reason,
        summary_label=left.summary_label or right.summary_label,
    )


def _merge_evidence(
    left: FailureEvent, right: FailureEvent
) -> tuple[FailureEvidence, ...]:
    evidence = {item.evidence_id: item for item in left.evidence}
    for item in right.evidence:
        existing = evidence.get(item.evidence_id)
        if existing is not None and existing != item:
            raise ValueError(f"Conflicting failure evidence ID: {item.evidence_id}")
        evidence[item.evidence_id] = item
    return tuple(evidence[key] for key in sorted(evidence))


def _merge_hypotheses(
    left: FailureEvent, right: FailureEvent
) -> tuple[CausalHypothesis, ...]:
    hypotheses = {
        (item.parent_event_id, item.relationship): item
        for item in left.causal_hypotheses
    }
    for item in right.causal_hypotheses:
        key = (item.parent_event_id, item.relationship)
        existing = hypotheses.get(key)
        if existing is None or item.confidence > existing.confidence:
            hypotheses[key] = item
    return tuple(hypotheses[key] for key in sorted(hypotheses))


def _merge_recovery_eligibility(left: FailureEvent, right: FailureEvent) -> str:
    if left.recovery_eligibility == right.recovery_eligibility:
        return left.recovery_eligibility
    if left.recovery_eligibility == "unknown":
        return right.recovery_eligibility
    if right.recovery_eligibility == "unknown":
        return left.recovery_eligibility
    return "unknown"
