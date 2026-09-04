# Failure and Recovery Metrics

NyssaBench should explain how policies fail, whether they recover, and which
failures are sensitive to stressors.

## Core Failure Fields

Each episode should record:

- `success`
- `failure_label`
- `failure_label_source`
- `failure_ledger` (`nyssa-failure-ledger-v1`)
- `steps`
- `replay_path`
- `failure_clip_path`
- task id
- seed
- stressor settings
- policy id and checkpoint metadata

`failure_label` remains the episode summary. The ledger records temporal roles,
evidence visibility, provenance, candidate causal hypotheses, consequences,
and recovery eligibility. See [Failure Event Protocol](failure_event_protocol.md).

## Recovery Fields

Recovery-aware steps record:

- `recovery_attempted`
- `recovery_applied`
- `recovery_attempt_id`
- `recovery_attribution_attempt_id`
- `recovery_outcome`
- `recovery_outcome_step`
- `recovery_success`
- `recovery_plan_outcome`
- `recovery_plan_success`
- `recovery_attribution_start_step`
- `recovery_attribution_end_step`
- `recovery_attribution_horizon_steps`
- `recovery_attribution_criterion`

The `nyssa-recovery-outcomes-v1` criterion attributes success when the task's
success predicate becomes true before a newer attempt and inside a bounded step
window. The window starts at the first recovery action and spans the larger of
the configured attribution horizon or the full recovery-plan length. The
default configured horizon is five transitions and can be changed with
`--recovery-attribution-horizon`.

Attempt outcomes are `success`, `not_applied`, `superseded`, `window_expired`,
`episode_terminated`, `episode_truncated`, or `episode_ended`. Eventual episode
success outside the window does not relabel an earlier recovery as successful.

These bounded outcomes are temporal attribution diagnostics, not causal
effects. Use the matched state-fork protocol in
[Counterfactual Recovery](counterfactual_recovery.md) to estimate recovery gain
against continuation from the same simulator state.

## Aggregate Metrics

Reports should include:

- failure mode distribution
- primary failure mode
- recovery attempt, applied, successful, failed, and not-applied counts
- recovery success rate over applied attempts
- recovery episode success rate over episodes with applied recovery
- counterfactual recovery gain with matched-pair and branch-point denominators
- exact, qualified, unsupported, and incomplete branch coverage
- false, helpful, harmful, and safety-harmful intervention rates
- recovery-minus-continuation step and reward costs
- mean steps before failure
- mean steps to recovery
- drop rate
- collision rate
- timeout rate
- failure by stressor
- failure by seed
- failure by task

## Interpretation

Failure/recovery metrics should not hide low task success. A policy with low
success and high recovery attempts may still be unreliable. Reports should show
success, failure, and recovery together rather than replacing success rate with
a single composite score.

Attempt counts are aggregated before computing rates within a run, task, seed,
or multi-run result pack. Averaging per-episode or per-run recovery rates is not
valid when the number of applied attempts differs.
