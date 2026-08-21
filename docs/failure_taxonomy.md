# Failure Taxonomy

NyssaBench reports failure modes instead of reducing evaluation to pass or fail.

These labels are backward-compatible episode summaries. New runs also record a
versioned temporal ledger that separates symptoms, mechanisms, candidate
causes, contributing conditions, and consequences. See
[Failure Event Protocol](failure_event_protocol.md).

Core labels:

- bad_grasp
- object_slip
- collision
- missed_target
- wrong_object
- occlusion_failure
- planner_stuck
- joint_limit_failure
- timeout
- latency_failure
- unstable_contact
- unknown_failure
- out_of_distribution_layout

Simulator adapters should map their native events into these labels first, then add engine-specific labels only when the shared taxonomy is insufficient. If no diagnostic event matches, use `unknown_failure`; do not default to the first configured failure label.

## Failure Sources

Each failed episode should record where the label came from:

- `env`: simulator or task emitted a failure label directly.
- `mapper`: `FailureMapper` inferred the label from terminal info and events.
- `none`: successful episode.

Public results should not rely on silent placeholder labels. If a task produces
many `unknown_failure` cases, the mapper or adapter needs better diagnostics
before the result is treated as strong.

## Mapper Heuristics

The current `FailureMapper` uses simple event-based rules:

- collision or safety event -> `collision`
- wrong object event -> `wrong_object`
- dropped/slipped object event -> `object_slip`
- failed grasp event -> `bad_grasp`
- joint-limit event -> `joint_limit_failure` when configured
- planner stuck event -> `planner_stuck`
- latency threshold event -> `latency_failure`
- out-of-distribution layout event -> `out_of_distribution_layout`
- failed early termination on tasks that configure contact instability -> `unstable_contact`
- time limit or max-step truncation -> `timeout`
- no success event with a configured target failure -> `missed_target`
- no diagnostic event -> `unknown_failure`

These heuristics are intentionally conservative. Stronger result packs should
prefer environment-native events or task-specific diagnostics when available.
The mapper emits a `terminal_only` ledger event but does not infer causal links
to earlier events.
