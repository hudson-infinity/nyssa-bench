# Metrics

NyssaBench reports more than success rate. v0.1 includes:

- success_rate
- completion_time
- collision_count
- path_efficiency
- grasp_success_rate
- drop_rate
- object_slip_rate
- wrong_object_rate
- recovery_attempt_count
- recovery_applied_count
- recovery_success_count
- recovery_failure_count
- recovery_not_applied_count
- recovery_success_rate
- recovery_episode_success_rate
- safety_violation_rate
- out_of_distribution_failure_rate
- prototype_reliability_score

`prototype_reliability_score` is a heuristic over simulator success, safety, and robustness. It is not a calibrated sim-to-real score.

`recovery_success_rate` is the count of successful applied recovery attempts
divided by all applied recovery attempts. Requests for which no recovery plan is
available remain `not_applied` outcomes and do not enter that denominator.
`recovery_episode_success_rate` is separately computed over episodes containing
at least one applied recovery. Counts are summed before either rate is computed.
See [Failure and Recovery Metrics](failure_and_recovery_metrics.md) for the
bounded attribution rule.

Failure labels include bad grasp, object slip, collision, missed target, wrong object, occlusion failure, planner stuck, joint limit failure, timeout, latency failure, unstable contact, unknown failure, and out-of-distribution layout.
