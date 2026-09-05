# Paired sim-real studies

`nyssa sim-real-study` executes a prespecified comparison between NyssaBench
simulation result packs and validated real-evidence packages. It extends
success/rank correspondence to failure distributions, shift response, failure
timing, recovery evidence, and held-out predictive value.

The module implements analysis. It does not supply hardware trials or make a
sim-real claim by itself.

## Study contract

A `nyssa-sim-real-study-v1` file declares:

- study identity, semantic version, timezone-aware prespecification time, and
  `policy_task_shift_trial` as the unit of analysis;
- every simulation run ID, core artifact hash, policy/checkpoint/preprocessing
  identity, task, seed, and episode index;
- every #27 real package identity, episode, trial, and reconstructed variant;
- policy, task, shift, severity, simulator step duration, and real event-step
  duration for each pair;
- prespecified exclusions with reasons;
- primary metrics, held-out shifts, bootstrap count/seed, and cluster fields;
- whether recovery comparison is disabled, based on matched real trials, or
  simulation-only.

Included simulation and real identities must be one-to-one. Duplicate,
ambiguous, missing, or many-to-one mappings are invalid. The evaluator reloads
all content-pinned simulation files, verifies policy/checkpoint identity, runs
the full real-evidence validator, and requires the real package to be
claim-ready before pairing.

## Analyses

Policy ranking reports Pearson, Spearman, Kendall tau-b, and Mean Maximum Rank
Violation over paired policy success rates. Ties are handled explicitly.

Failure correspondence reports category counts and one minus Jensen-Shannon
divergence. Shift response compares clean-to-shifted degradation per policy and
task. Both include pair counts and cluster-preserving uncertainty where the
study has enough independent clusters.

Failure-time analysis converts simulator and real event indices to seconds with
the declared units. It reports observed pairs separately from simulator
success, real success, truncation, and missing failure-time censoring. It never
replaces a censored time with zero.

Recovery comparison is unavailable by default. `counterfactual_sim_only`
remains explicitly non-comparable to hardware. `matched_real_trials` requires
both a simulator recovery gain and a real matched recovery gain supplied under
the validated real package's `metadata.matched_recovery_gain_by_trial` mapping.
An operator intervention plus eventual success is not treated as recovery gain.

Incremental predictive analysis fits the prespecified clean-success baseline
and enhanced failure/recovery feature model on non-held-out shifts, then reports
their Brier errors on held-out shifts. Positive improvement favors the enhanced
features. Zero or negative results remain valid outputs and must be reported.

## Run a study

```bash
uv run nyssa sim-real-study studies/policy_failure_calibration.yaml \
  --out benchmark_results/policy_failure_calibration
```

Exit status is 0 for a complete primary analysis, 2 when a primary metric lacks
enough evidence, and 3 for invalid identity or contract evidence. The command
writes `sim_real_study.json` and `sim_real_study.html`; the JSON includes a
content hash, all pair records, exclusions, errors, estimates, intervals,
sample sizes, censoring, assumptions, and held-out analysis details.

## Claim boundary

A complete report means the requested correspondence calculations executed on
valid inputs. Predictive sim-real wording still requires #20's preregistered
hardware study, applicable BenchmarkValidity audits, sensitivity analysis, and
an immutable claim artifact. Synthetic fixtures and reconstructed variants are
tests of the analysis code, not evidence that simulation predicts hardware.
