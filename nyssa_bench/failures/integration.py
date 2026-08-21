from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable

from nyssa_bench.failures.ledger import FailureEventEmitter
from nyssa_bench.failures.protocol import (
    FailureEventDraft,
    FailureEvidence,
    FailureLedgerRecord,
    FailureRole,
    RecoveryEligibility,
)


@runtime_checkable
class FailureEventSource(Protocol):
    """Optional component API for queued temporal failure event drafts."""

    def drain_failure_events(
        self,
    ) -> Iterable[FailureEventDraft | dict[str, Any]]:
        raise NotImplementedError


def drain_component_failure_events(
    component: Any,
    emitter: FailureEventEmitter,
    *,
    default_step: int,
) -> list[Any]:
    drain = getattr(component, "drain_failure_events", None)
    if not callable(drain):
        return []
    payloads = drain()
    if payloads is None:
        return []
    if isinstance(payloads, (FailureEventDraft, dict)):
        payloads = [payloads]
    if isinstance(payloads, (str, bytes)) or not isinstance(payloads, Iterable):
        raise TypeError(
            "drain_failure_events() must return an iterable of event drafts"
        )
    return emitter.emit_many(payloads, default_step=default_step)


def emit_info_failure_events(
    info: dict[str, Any],
    emitter: FailureEventEmitter,
    *,
    default_step: int,
) -> list[Any]:
    payloads = info.get("failure_events", [])
    if payloads is None:
        return []
    if isinstance(payloads, dict):
        payloads = [payloads]
    if not isinstance(payloads, list):
        raise TypeError("engine info.failure_events must be a list of event drafts")
    return emitter.emit_many(payloads, default_step=default_step)


def compact_stressor_context(manifest: dict[str, Any]) -> dict[str, Any]:
    applications = []
    for item in manifest.get("applications", []):
        if not isinstance(item, dict):
            continue
        applications.append(
            {
                "stressor_id": item.get("stressor_id"),
                "category": item.get("category"),
                "status": item.get("status"),
                "severity": (item.get("requested") or {}).get("severity"),
                "applied_parameters": item.get("applied_parameters", {}),
                "seed": item.get("seed"),
            }
        )
    return {
        "format": "nyssa-active-stressor-context-v1",
        "condition_id": manifest.get("condition_id", "clean"),
        "composition_order": list(manifest.get("composition_order", [])),
        "applications": applications,
    }


def stressor_condition_drafts(
    context: dict[str, Any], *, onset_step: int = 0
) -> list[FailureEventDraft]:
    drafts = []
    for index, item in enumerate(context.get("applications", [])):
        if not isinstance(item, dict) or item.get("status") != "applied":
            continue
        stressor_id = str(item.get("stressor_id", "unknown_stressor"))
        drafts.append(
            FailureEventDraft(
                role="contributing_condition",
                category="distribution_shift",
                subtype=stressor_id,
                onset_step=onset_step,
                temporal_precision="exact_step",
                confidence=1.0,
                evidence=(
                    FailureEvidence(
                        evidence_id=f"stressor:{index}:application",
                        evidence_type="applied_stressor_parameters",
                        payload={
                            "condition_id": context.get("condition_id", "clean"),
                            "severity": item.get("severity"),
                            "applied_parameters": item.get("applied_parameters", {}),
                            "seed": item.get("seed"),
                        },
                        source="stressor_manifest",
                        annotation_source="stressor_pipeline",
                        confidence=1.0,
                        visibility="privileged",
                        captured_step=onset_step,
                    ),
                ),
                consequences=("controlled_distribution_shift_active",),
                deduplication_key=f"stressor:{stressor_id}",
            )
        )
    return drafts


def verifier_rejection_draft(
    score: dict[str, Any], *, step_index: int, recovery_enabled: bool
) -> FailureEventDraft:
    confidence = score.get("confidence")
    return FailureEventDraft(
        role="contributing_condition",
        category="verification",
        subtype=str(score.get("reason") or "action_rejected"),
        onset_step=step_index,
        temporal_precision="exact_step",
        confidence=float(confidence) if confidence is not None else 1.0,
        evidence=(
            FailureEvidence(
                evidence_id=f"verifier:{step_index}:assessment",
                evidence_type="action_assessment",
                payload=dict(score),
                source="verifier_output",
                annotation_source="automatic_verifier",
                confidence=float(confidence) if confidence is not None else 1.0,
                visibility="privileged",
                captured_step=step_index,
            ),
        ),
        consequences=("proposed_action_rejected",),
        recovery_eligibility="eligible" if recovery_enabled else "unknown",
        recovery_reason=str(score.get("reason"))
        if score.get("reason") is not None
        else None,
        deduplication_key=f"verifier:{score.get('reason') or 'rejected'}",
    )


def recovery_attempt_draft(
    *,
    step_index: int,
    attempt_id: int,
    applied: bool,
    plan_length: int,
    reason: str | None,
    verifier_event_id: str | None,
) -> FailureEventDraft:
    hypotheses = []
    if verifier_event_id:
        from nyssa_bench.failures.protocol import CausalHypothesis

        hypotheses.append(
            CausalHypothesis(
                parent_event_id=verifier_event_id,
                relationship="triggered_by_detection",
                confidence=1.0,
                rationale="The verifier rejection directly triggered this recovery attempt.",
            )
        )
    return FailureEventDraft(
        role="contributing_condition",
        category="recovery",
        subtype="recovery_applied" if applied else "recovery_not_applied",
        onset_step=step_index,
        temporal_precision="exact_step",
        confidence=1.0,
        evidence=(
            FailureEvidence(
                evidence_id=f"recovery:{attempt_id}:plan",
                evidence_type="recovery_plan",
                payload={
                    "attempt_id": attempt_id,
                    "applied": applied,
                    "plan_length": plan_length,
                    "trigger_reason": reason,
                },
                source="recovery_component",
                annotation_source="recovery_runner",
                confidence=1.0,
                visibility="privileged",
                captured_step=step_index,
            ),
        ),
        causal_hypotheses=tuple(hypotheses),
        consequences=(
            "executed_recovery_plan" if applied else "no_recovery_plan_available",
        ),
        recovery_eligibility="eligible" if applied else "unknown",
        recovery_reason=reason,
        deduplication_key=f"recovery:{attempt_id}",
    )


def terminal_failure_draft(
    *,
    label: str,
    label_source: str,
    reason: str,
    info: dict[str, Any],
    step_index: int,
) -> FailureEventDraft:
    diagnostic_keys = (
        "failure_label",
        "failure_label_source",
        "collision",
        "collision_count",
        "wrong_object",
        "object_slip",
        "object_dropped",
        "grasp_failed",
        "grasp_success",
        "joint_limit",
        "planner_stuck",
        "latency_failure",
        "latency_ms",
        "max_latency_ms",
        "out_of_distribution_layout",
        "TimeLimit.truncated",
    )
    diagnostics = {key: info[key] for key in diagnostic_keys if key in info}
    confidence = 1.0 if label_source == "env" else 0.75
    if label == "unknown_failure":
        confidence = 0.25
    return FailureEventDraft(
        role=_terminal_role(label),
        category=_terminal_category(label),
        subtype=label,
        onset_step=step_index,
        end_step=step_index,
        temporal_precision="terminal_only",
        confidence=confidence,
        evidence=(
            FailureEvidence(
                evidence_id=f"terminal:{step_index}:{label}",
                evidence_type="terminal_failure_classification",
                payload={"reason": reason, "diagnostics": diagnostics},
                source="task_info" if label_source == "env" else "failure_mapper",
                annotation_source="environment"
                if label_source == "env"
                else "automatic_mapper",
                confidence=confidence,
                visibility="privileged",
                captured_step=step_index,
            ),
        ),
        consequences=("episode_failed",),
        recovery_eligibility=_terminal_recovery_eligibility(label),
        recovery_reason=reason,
        summary_label=label,
        deduplication_key=f"terminal:{label}",
    )


def failure_ledger_from_episode_dict(
    episode: dict[str, Any], *, engine_name: str = "unknown"
) -> FailureLedgerRecord | None:
    ledger = episode.get("failure_ledger")
    if isinstance(ledger, dict):
        return FailureLedgerRecord.from_dict(ledger)
    if bool(episode.get("success", False)) or not episode.get("failure_label"):
        return None

    from nyssa_bench.failures.ledger import FailureEventLedger

    label = str(episode["failure_label"])
    source = str(episode.get("failure_label_source") or "mapper")
    mutable = FailureEventLedger(
        task_id=str(episode.get("task_id", "unknown")),
        episode_index=int(episode.get("episode_index", 0)),
        episode_seed=int(episode.get("seed", 0)),
        engine_name=engine_name,
        stressor_context=dict(episode.get("stressor_context", {})),
    )
    emitter = mutable.emitter(
        "task_logic" if source == "env" else "legacy_mapper",
        "legacy_result_pack_migration",
        annotation_source="legacy_terminal_label",
    )
    step_index = max(0, len(episode.get("steps", [])) - 1)
    emitter.emit(
        terminal_failure_draft(
            label=label,
            label_source=source,
            reason="migrated from legacy episode-level failure label",
            info={},
            step_index=step_index,
        )
    )
    return mutable.snapshot()


def derive_failure_label(
    ledger: FailureLedgerRecord, *, fallback: str | None
) -> str | None:
    candidates = [event for event in ledger.events if event.summary_label]
    if not candidates:
        return fallback
    role_priority = {
        "consequence": 5,
        "symptom": 4,
        "mechanism": 3,
        "candidate_cause": 2,
        "contributing_condition": 1,
    }
    chosen = max(
        candidates,
        key=lambda event: (
            event.onset_step,
            role_priority[event.role],
            event.confidence,
            event.event_id,
        ),
    )
    return chosen.summary_label


def _terminal_role(label: str) -> FailureRole:
    if label in {"timeout", "missed_target", "unknown_failure"}:
        return "consequence"
    if label in {"bad_grasp", "wrong_object", "planner_stuck", "latency_failure"}:
        return "mechanism"
    return "symptom"


def _terminal_category(label: str) -> str:
    if label in {"wrong_object", "occlusion_failure", "out_of_distribution_layout"}:
        return "perception"
    if label in {"missed_target", "planner_stuck"}:
        return "planning"
    if label in {"joint_limit_failure", "latency_failure"}:
        return "control"
    if label in {"bad_grasp", "object_slip", "collision", "unstable_contact"}:
        return "interaction"
    if label == "timeout":
        return "system"
    return "unknown"


def _terminal_recovery_eligibility(label: str) -> RecoveryEligibility:
    if label in {"bad_grasp", "object_slip", "missed_target", "planner_stuck"}:
        return "eligible"
    if label == "timeout":
        return "ineligible"
    return "unknown"
