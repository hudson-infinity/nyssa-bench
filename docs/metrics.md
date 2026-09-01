# Metrics

New runs write `nyssa-run-metrics-v2` and include a
`nyssa-metric-vector-v1` object. The vector has no composite reliability score.
Each measurement remains separate so success, robustness, recovery, safety, and
cost tradeoffs stay visible.

Every metric definition records:

- the evaluated population and denominator;
- aggregation and missing-data behavior;
- whether higher or lower values are preferred, or whether the metric is only
  descriptive;
- units and the uncertainty convention.

Every measurement records an availability status, value, 95% interval where
defined, sample size, numerator, denominator, source, and reason for missingness.
The allowed statuses are `available`, `unavailable`, `not_applicable`, and
`incompatible`. Missing evidence is never replaced with zero.

## Metric vector

The first schema includes:

- clean and shifted success rates with Wilson intervals;
- matched robustness degradation;
- robustness AUC over severity;
- mean time to failure with censored-episode counts;
- failure-event distributions;
- failure-prediction calibration when a calibrated monitor is present;
- counterfactual recovery gain when matched branches are present;
- intervention and false-intervention rates;
- safety and physical-damage event rates;
- wall time and inference latency;
- sim-real rank and failure-distribution correspondence when a validated
  hardware study is attached.

Robustness AUC uses piecewise-linear trapezoidal interpolation of success rate
and divides by the observed severity span. A sweep must include severity zero,
at least one positive severity, complete matched episode identities, and a
paired bootstrap interval. Runs that do not meet those conditions report the
metric as unavailable or incompatible.

Sim-real measurements require `hardware_calibration.validated: true`, a study
ID, and a contract SHA-256. Simulator-only runs keep those measurements
unavailable, and claim validation rejects available sim-real labels without the
required evidence.

## Legacy migration

Readers accept old unversioned packs through `migrate_metric_summary`. Historical
`prototype_reliability_score`, `score_kind`, and `sim_to_real_score` fields move
under `legacy_metrics`. A `nyssa-metric-migration-v1` record lists every migrated
field and states that the old value is audit-only. It is not converted into any
new metric and is never used for ranking.

The old Python helper module remains importable for compatibility and emits a
`DeprecationWarning`. Current writers and comparison tools do not call it.

## Recovery rates

`recovery_success_rate` is the count of successful applied recovery attempts
divided by all applied recovery attempts. Requests for which no recovery plan is
available remain `not_applied` outcomes and do not enter that denominator.
`recovery_episode_success_rate` is separately computed over episodes containing
at least one applied recovery. Counts are summed before either rate is computed.
See [Failure and Recovery Metrics](failure_and_recovery_metrics.md) for the
bounded attribution rule.

Failure labels include bad grasp, object slip, collision, missed target, wrong
object, occlusion failure, planner stuck, joint limit failure, timeout, latency
failure, unstable contact, unknown failure, and out-of-distribution layout.
