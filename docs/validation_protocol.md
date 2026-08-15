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
- selected simulator and policy package versions
- environment metadata
- a recorded git commit from a clean worktree
- diagnosed failure labels from the environment or `FailureMapper`
- unsupported stressors reported honestly

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

For compatibility with the intended public API, it is also re-exported from:

```txt
nyssa_bench.validation.run_claim
```

## Audit Risks

Validation should defend against:

- shortcut solvability
- weak statistics
- data leakage
- comparing policies under different randomization ranges
- treating a heuristic reliability score as real sim-to-real validation
- calling video-less artifacts replay-first reports
- treating adapter hooks as evaluated baselines
