# Failure Event Protocol

NyssaBench uses `nyssa-failure-event-v1` to represent how a failure develops
through an episode. The temporal ledger complements the backward-compatible
episode-level `failure_label`; it does not replace task success or claim that
correlated events are causal.

## Event Contract

Every event records:

- a stable event ID, onset step, optional end step, and temporal precision;
- a role: `symptom`, `mechanism`, `candidate_cause`,
  `contributing_condition`, or `consequence`;
- a shared category and engine/task-specific subtype;
- confidence-scored evidence with source and annotation provenance;
- evidence visibility: `policy_observable`, `privileged`, or `external`;
- active stressor condition and applied parameters;
- candidate causal parents, downstream consequences, and recovery eligibility;
- an optional `summary_label` used to derive the legacy flat label.

The supported temporal precisions are `exact_step`, `step_interval`,
`terminal_only`, and `unknown`. Confidence is finite and normalized to
`[0, 1]`.

```json
{
  "format": "nyssa-failure-event-v1",
  "event_id": "slip",
  "role": "symptom",
  "category": "interaction",
  "subtype": "object_slip",
  "onset_step": 29,
  "end_step": 29,
  "temporal_precision": "exact_step",
  "confidence": 0.99,
  "evidence": {
    "policy_observable": [],
    "privileged": [
      {
        "format": "nyssa-failure-evidence-v1",
        "evidence_id": "slip:relative-motion",
        "evidence_type": "object_gripper_relative_motion",
        "payload": {"displacement_m": 0.083},
        "source": "simulator_state",
        "annotation_source": "automatic_slip_check",
        "confidence": 0.99,
        "visibility": "privileged",
        "captured_step": 29
      }
    ],
    "external": []
  },
  "provenance": {
    "source": "simulator_state",
    "component_id": "ManiSkillEngine",
    "annotation_source": "engine_adapter"
  },
  "active_stressor_context": {
    "format": "nyssa-active-stressor-context-v1",
    "condition_id": "friction_s05",
    "composition_order": ["friction_scale"],
    "applications": []
  },
  "causal_hypotheses": [],
  "consequences": ["object_released"],
  "recovery_eligibility": "eligible",
  "recovery_reason": "object remains reachable",
  "summary_label": "object_slip",
  "deduplication_key": "slip:object-0"
}
```

## Evidence Visibility

Evidence available in the evaluated policy input is stored under
`policy_observable`. Simulator-only state, task predicates, verifier scores,
and hidden labels are stored under `privileged`. Human notes and measurements
outside the policy/simulator interface are `external`. Evidence cannot occupy
more than one partition within an event.

Visibility describes access, not quality. Each evidence record separately
declares its source, annotation source, and confidence.

## Provenance

The ledger accepts these provenance sources:

- `simulator_state`
- `task_logic`
- `verifier_output`
- `policy_output`
- `stressor`
- `recovery`
- `human_annotation`
- `legacy_mapper`
- `external_monitor`

Each event also records the emitting component ID and annotation source.
Engine, policy, expert/verifier, and stressor adapters can implement the
optional stable hook:

```python
def drain_failure_events(self) -> list[FailureEventDraft | dict]:
    ...
```

The runner drains this queue after the corresponding component operation.
Simulator adapters can also place draft mappings in `info["failure_events"]`.
The runner assigns event IDs, component provenance, and the current stressor
context, so component payloads cannot silently forge those fields.

## Ordering, Overlap, And Deduplication

Ledger order is deterministic: onset step, end step, then event ID. Overlap is
allowed and has no causal meaning by itself.

Events are deduplicated only when all of these conditions hold:

1. both events provide the same non-empty `deduplication_key`;
2. role, category, subtype, provenance, and stressor context match;
3. their temporal intervals overlap.

The merged interval spans both events, confidence uses the maximum observation,
and evidence is unioned by evidence ID. A recurring event outside the first
interval remains a separate event. Conflicting event, evidence, or summary IDs
are rejected.

## Causal Hypotheses

`nyssa-causal-hypothesis-v1` links a child event to a candidate parent with a
relationship name, confidence, rationale, and supporting evidence IDs. Every
serialized relation states `hypothesis_only_not_established_causality`.

Candidate parents must exist in the same episode ledger. Temporal precedence
and correlation do not create links automatically. Detectors and annotators
must emit a hypothesis explicitly.

## Compatibility And Artifacts

New runs write the ledger to:

- `episodes.json` and `episodes.jsonl` under `failure_ledger`;
- `failure_ledger.json` as a run-level manifest and summary;
- `replay_manifest.json` for timeline rendering;
- `report.html` with evidence, provenance, candidate parents, and recovery
  eligibility.

Old result packs remain loadable. When an old failed episode has only
`failure_label` and `failure_label_source`, NyssaBench creates one
`terminal_only` migration event while preserving both top-level fields. A
successful legacy episode does not receive a synthetic failure.

The flat label for a new episode is derived from ledger events carrying
`summary_label` where possible, with the existing `FailureMapper` result as the
fallback. This keeps existing metrics, comparisons, and datasets compatible.
