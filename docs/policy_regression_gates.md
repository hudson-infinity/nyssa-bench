# Policy regression gates

`nyssa regression-gate` compares a candidate policy checkpoint with a pinned
baseline under prespecified conditions. It is intended to answer whether a
candidate can replace a baseline and proceed to scarce hardware evaluation. It
does not certify deployment safety, create a public benchmark ranking, or
collapse reliability into one score.

## Decisions

Every study returns one decision and a stable process exit code:

| Decision | Exit code | Meaning |
| --- | ---: | --- |
| `pass` | 0 | Every prespecified rule passed with complete required evidence. |
| `fail` | 1 | At least one performance or blocking safety rule failed. |
| `inconclusive` | 2 | No rule failed, but evidence, coverage, power, or uncertainty was insufficient. |
| `invalid` | 3 | A run identity, artifact hash, condition, episode matrix, or comparison contract was invalid. |

Missing, unsupported, censored, or underpowered evidence never produces a
pass.

## Study contract

A `nyssa-policy-regression-study-v1` document pins:

- baseline and candidate policy names, checkpoint IDs, checkpoint SHA-256s, and
  preprocessing SHA-256s;
- the study version and timezone-aware prespecification timestamp;
- each baseline/candidate run ID, content-pinned baseline artifacts, and the
  candidate artifact binding that the evaluator must hash and retain;
- the comparison-contract hash, condition ID, condition class, and exact
  stressor severities for each cell;
- the complete task, seed, and episode-index matrix;
- required metric-vector fields and minimum pair coverage;
- required RunValidity, BenchmarkValidity, failure-ledger, detector, and replay
  evidence;
- non-inferiority margins, safety limits, minimum independent pairs, and metric
direction before evaluation;
- content-addressed confirmed stress-boundary evidence where a discovered
  condition is promoted to a permanent regression case.

The candidate run must record `started_at` at or after `prespecified_at`.
`run.yaml` must also contain checkpoint and preprocessing identities under
`policy_metadata`; a path or policy display name is not accepted as checkpoint
provenance.

## Rules

Rules select one or more study cells and one source:

| Source | Measurement |
| --- | --- |
| `paired_success` | Candidate and baseline success on exact episode pairs. |
| `episode_metric` | Any finite per-episode metric, including safety, damage, intervention, recovery, or latency fields. |
| `metric_vector` | A registered metric-vector value such as clean/shifted success, degradation, robustness AUC, recovery gain, or compute. |
| `failure_category_rate` | Per-episode presence of the named FailureEvent category. |
| `failure_onset_steps` | First outcome-event onset, with absent events treated as missing rather than zero. |
| `failure_duration_steps` | Duration of the first temporally ordered outcome event. |

For higher-is-better rules, the oriented effect is candidate minus baseline.
For lower-is-better rules, it is baseline minus candidate. Episode rules use a
paired bootstrap clustered by cell, episode seed, and episode index, so tasks
that share a seed are not counted as independent trials. A single metric-vector
cell uses the conservative difference of the two run intervals; repeated cells
use a paired cell bootstrap.

`non_inferiority` passes only when the complete 95% interval stays above the
negative prespecified margin. It fails only when the complete interval is below
that boundary. An interval crossing the boundary is inconclusive.

`safety_block` requires a lower-is-better metric and an absolute candidate
limit. A point estimate above the limit fails immediately. Passing requires the
candidate's 95% upper bound to stay within the limit; otherwise the result is
inconclusive.

## Minimal example

The hashes below are placeholders. Fingerprint completed baseline runs:

```bash
uv run nyssa regression-fingerprint runs/policy_105/clean_seed0 \
  --out regression_inputs/policy_105_clean_seed0.json
```

The fingerprint contains the policy identity, all required core artifact
hashes, replay hashes when present, comparison-contract hash, episode keys,
stressor configuration, and validity evidence needed to construct the study.
Create and version the study contract before starting the candidate runs. A
candidate reference uses `observe_and_record`: it pins the expected run ID and
location before execution, then the evaluator records the candidate artifact
hashes in the signed report. A baseline reference always uses `pinned`.

```yaml
format: nyssa-policy-regression-study-v1
schema_version: 1
study_id: policy_106_release
study_version: 1.0.0
prespecified_at: "2026-09-05T12:00:00Z"
baseline_policy:
  format: nyssa-regression-policy-identity-v1
  policy_name: policy_105
  checkpoint_id: policy_105_final
  checkpoint_sha256: "<64 lowercase hex characters>"
  preprocessing_sha256: "<64 lowercase hex characters>"
candidate_policy:
  format: nyssa-regression-policy-identity-v1
  policy_name: policy_106
  checkpoint_id: policy_106_candidate
  checkpoint_sha256: "<64 lowercase hex characters>"
  preprocessing_sha256: "<64 lowercase hex characters>"
cells:
  - format: nyssa-regression-cell-v1
    cell_id: clean_seed0
    condition_kind: clean
    condition_id: clean
    severity_levels: {}
    comparison_contract_sha256: "<64 lowercase hex characters>"
    baseline_run:
      format: nyssa-regression-run-reference-v1
      run_dir: runs/policy_105/clean_seed0
      run_id: policy_105_clean_seed0
      artifact_binding: pinned
      artifacts_sha256:
        run.yaml: "<64 lowercase hex characters>"
        dataset_manifest.json: "<64 lowercase hex characters>"
        metrics.json: "<64 lowercase hex characters>"
        episodes.json: "<64 lowercase hex characters>"
    candidate_run:
      format: nyssa-regression-run-reference-v1
      run_dir: runs/policy_106/clean_seed0
      run_id: policy_106_clean_seed0
      artifact_binding: observe_and_record
      artifacts_sha256: {}
    episode_keys:
      - {task_id: mujoco_pusher, seed: 0, episode_index: 0}
      - {task_id: mujoco_pusher, seed: 1, episode_index: 1}
    boundary_references: []
rules:
  - format: nyssa-regression-rule-v1
    rule_id: clean_success_non_inferiority
    source: paired_success
    metric_id: success
    cell_ids: [clean_seed0]
    kind: non_inferiority
    direction: higher
    non_inferiority_margin: 0.05
    minimum_pairs: 2
    candidate_limit: null
evidence_requirements:
  format: nyssa-regression-evidence-requirements-v1
  minimum_pair_coverage: 1.0
  require_failure_ledger: true
  require_detector_evidence: true
  require_replays: false
  require_run_validity: true
  require_benchmark_validity: true
  required_metric_vector: [clean_success_rate, safety_violation_rate]
metadata:
  owner: policy-release-team
```

Run the gate from the directory containing the study file:

```bash
uv run nyssa regression-gate policy_106_release.yaml \
  --out benchmark_results/policy_106_release_gate
```

The command writes `regression_report.json` and `regression_report.html`. The
JSON artifact includes the full paired evidence, coverage, rule estimates,
uncertainty, missingness, pinned input hashes, and a report hash. The loader
recomputes its decision and summary before accepting it.

When replay evidence is required, every episode's relative MP4 path must also
appear in that run reference's `artifacts_sha256` mapping. Additional result-pack
artifacts may be pinned there; absolute paths, parent traversal, and
non-canonical separators are rejected.

## Confirmed boundary cases

A cell with `condition_kind: confirmed_boundary` must include at least one
`nyssa-regression-boundary-reference-v1`. The reference pins a complete stress
search study by file hash and names its normalized point. NyssaBench reloads and
validates the stress-search artifact, verifies that held-out confirmation marked
the point as a boundary, derives the stressor configuration from the search
space, and requires both run packs to use the same severities and parameters.

This promotion path preserves discovery and confirmation provenance. It does
not turn an exploratory, unconfirmed condition into a permanent regression
case.
