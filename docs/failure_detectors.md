# Streaming failure detectors

NyssaBench runs failure detectors alongside each episode. Detectors consume
observations, actions, rewards, task information, and adapter-declared signals;
they emit temporal `FailureEvent` drafts into the episode ledger. The terminal
`FailureMapper` remains a compatibility fallback for failures that have no
streaming evidence.

## Lifecycle

The detector manager uses this order for each episode:

1. Collect adapter and reset-info capabilities.
2. Request instrumentation only for detectors whose contract declares
   `mode: instrumented`.
3. Reset compatible detectors.
4. Run `observe_before_action`, `observe_after_action`, and `detect` in declared
   detector order.
5. Run `finalize` for successful and failed episodes.
6. Record contracts, support decisions, configuration, and emitted event counts.

A detector with missing signals starts in `pending` state. The manager activates
it if a required signal appears in a later `info` payload. If the signal never
appears, the final manifest marks the detector `unsupported`; an empty event
list is therefore distinguishable from a supported detector that observed no
failure.

Detector exceptions identify the detector, lifecycle phase, task, and step.
Passive detectors receive read-only mappings so they cannot modify runner-owned
observations, info payloads, or stressor context through the detector API.

## Contract

`FailureDetector.contract()` returns a
`nyssa-failure-detector-v1` contract containing:

- detector identity and semantic version;
- passive or instrumented mode;
- supported engines and tasks;
- alternative signal requirements and evidence visibility;
- temporal precision;
- detector configuration.

Adapters expose guaranteed signals through
`NyssaEngine.failure_signal_capabilities()`. The manager also discovers
`info.<key>` capabilities as the episode runs. Signal IDs are namespaced, such
as `reward`, `info.collision_count`, or `info.is_grasped`.

## Built-in detectors

| Detector | Required signal | Events |
| --- | --- | --- |
| `contact_detector` | Explicit collision, contact-violation, or safety signal | Collision onset edges |
| `grasp_detector` | Grasp, slip, contact-loss, or object-identity signal | Wrong object, bad grasp, object slip |
| `stall_detector` | Reward | Reward-stagnation interval |

Each emitted event uses the detector ID as its provenance component and records
the detector version in the annotation source. Overlapping events from different
detectors remain separate because the ledger only deduplicates events with the
same semantics and provenance.

## Artifacts

Every serialized episode includes `failure_detector_context`. Result packs also
contain `failure_detector_manifest.json`, and `metrics.json` exposes a
`failure_detector_summary`. These records preserve supported, unsupported, and
missing-signal outcomes per episode along with the exact detector contracts.

The optional MuJoCo and ManiSkill integration tests exercise the reward-based
detector against real simulator transitions. They skip when the corresponding
project extra or backend is unavailable.
