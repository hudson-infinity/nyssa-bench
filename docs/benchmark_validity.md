# Benchmark validity

Run validity and benchmark validity answer different questions.

- `RunClaimValidator` checks whether a run is complete, reproducible, and
  supported by its artifacts.
- `BenchmarkValidityEvaluator` checks whether the benchmark design supports the
  stated scientific conclusion.

A public benchmark claim requires both. Missing benchmark-validity evidence is
not treated as a pass.

## Protocol

A `nyssa-benchmark-validity-spec-v1` document declares the benchmark identity,
claim tier, required audits, and audit inputs. Evaluation produces a
content-hashed `nyssa-benchmark-validity-report-v1` containing one
`nyssa-benchmark-audit-v1` result per required audit.

Every audit records:

- normalized inputs;
- status: `passed`, `failed`, `missing`, or `not_applicable`;
- severity and claim impact;
- machine-readable evidence;
- a summary and concrete remediation.

Claim impacts are `none`, `warn`, `downgrade`, or `block`. A required blocking
audit that is failed or missing makes `claim_ready` false. Hardware evidence is
`not_applicable` only when no sim-real claim was requested.

## Audits

`shortcut_solvability` checks trivial-policy success against a prespecified
threshold and requires episode denominators.

`train_evaluation_leakage` compares training and evaluation seeds, assets,
tasks, demonstrations, and language identities. Every dimension must be
declared, even when empty. Allowed overlap must be explicit.

`language_observation_ablations` checks how much full-policy performance remains
after language or observation removal. Zero clean performance fails because an
ablation cannot establish construct validity when the reference policy does not
solve the benchmark.

`statistical_precision` enforces prespecified sample-size and 95% interval-width
requirements for each primary estimate.

`paired_design` consumes comparison compatibility from the run comparison layer
and duplicate-free complete coverage from the arena pairing layer. Mismatched
contracts, unmatched episodes, or duplicate keys fail the audit.

`rank_stability` compares complete policy rankings across seeds, tasks,
stressors, or aggregation choices using pair-order agreement.

`hidden_test_integrity` requires protected hidden splits, unpublished contents,
clean contamination status, a SHA-256 commitment, and an evaluator distinct
from the data producer. The commitment can be verified without publishing the
protected members.

`sim_real_predictive_validity` requires a validated, content-addressed hardware
study only for claim tiers that request sim-real evidence. Hardware studies
must report bounded rank correlation, failure-distribution similarity, and
held-out incremental predictive value with intervals and sample sizes.

## Commands

```bash
uv run nyssa audit-benchmark configs/validity/my_benchmark.yaml \
  --out benchmark_results/my_run/benchmark_validity.json

uv run nyssa validate benchmark_results/my_run/benchmark_validity.json
```

`audit-benchmark` exits with `0` when the report is claim-ready and `2` when an
audit blocks or downgrades the requested claim. It writes the report in both
cases so failures remain inspectable.

Attach a validated report to evaluation commands:

```bash
uv run nyssa run \
  --suite mujoco_control_v0 \
  --engine mujoco \
  --policy task_bc_policy \
  --episodes 100 \
  --benchmark-validity benchmark_validity.json \
  --out benchmark_results/mujoco_validated \
  --capture-replay
```

The same option is available on `run-scenario`, `experiment`, and `ablate`.
NyssaBench copies the validated report into the run, embeds it in run metadata,
shows each audit in `report.html`, and hashes `benchmark_validity.json` in
`dataset_manifest.json`.

## Interpretation

These audits enforce declared evidence. They do not manufacture evidence or
prove that a chosen threshold is scientifically appropriate. Thresholds,
exclusions, primary metrics, and claim tiers should be fixed before inspecting
the evaluated policy result. A passing report supports only its declared claim
tier and benchmark version.
