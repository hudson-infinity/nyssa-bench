# Validation Protocol

This document expands `VALIDATION.md` with the research rationale behind the
code-level gate.

## Public Result Requirements

A public NyssaBench benchmark result must include:

- supported real simulator backend
- non-experimental backend
- explicit task-to-engine environment mapping
- mapped success predicate
- at least 100 episodes per task
- at least 3 seeds per policy in the result pack
- complete task-by-episode matrix with unique seeds paired across tasks
- episode artifacts
- MP4 replay files that exist for every episode
- replay paths that resolve safely inside the assembled run directory
- matching episode denominators in `run.yaml`, `metrics.json`, and
  `episodes.json`
- a replay manifest consistent with episode replay and failure-clip paths
- pack-level replay revalidation after files are copied, pruned, or archived
- selected simulator and policy package versions
- environment metadata
- a recorded git commit from a clean worktree
- diagnosed failure labels from the environment or `FailureMapper`
- unsupported stressors reported honestly
- a claim-ready, content-hashed benchmark-validity report

## Non-Public Runs

The following are useful but not public benchmark results:

- local smoke runs
- `--no-replay` runs
- adapter-contract runs
- runs with missing video artifacts
- runs from a dirty or unidentified git revision
- runs with placeholder policies
- runs with too few episodes
- runs with unsupported stressors silently listed as active

## Current Implementation

The code-level gate lives in:

```txt
nyssa_bench.metrics.run_claims.RunClaimValidator
```

Assembled result packs are independently revalidated by:

```txt
nyssa_bench.reports.replay_validation.validate_result_pack_replays
```

This second gate derives coverage from declared per-episode MP4 paths. Failure
clips, duplicate-content media, galleries, and unreferenced media are counted
separately and cannot inflate episode replay coverage. Its versioned output is
embedded in `manifest.json` and scorecards and summarized in `RESULTS.md`.

For compatibility with the intended public API, it is also re-exported from:

```txt
nyssa_bench.validation.run_claim
```

Benchmark design is evaluated separately by:

```txt
nyssa_bench.validity.BenchmarkValidityEvaluator
```

See [Benchmark validity](benchmark_validity.md) for audit inputs, statuses,
claim impacts, CLI commands, and result-pack integration.

## Audit Risks

Validation should defend against:

- shortcut solvability
- weak statistics
- data leakage
- comparing policies under different randomization ranges
- treating a heuristic reliability score as real sim-to-real validation
- calling video-less artifacts replay-first reports
- treating adapter hooks as evaluated baselines
