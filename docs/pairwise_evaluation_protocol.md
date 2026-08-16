# Pairwise Evaluation Protocol

Pairwise evaluation is a planned NyssaBench mode inspired by real-world
preference-style robot-policy evaluation. It is not the v0.1 priority, but the
protocol should shape future APIs.

## Goal

Evaluate two policies under matched conditions:

- same suite
- same tasks
- same seeds
- same stressors
- same episode budget
- same success predicates
- same replay evidence requirements

The question is not only "which policy has higher success rate?" but:

- Which policy wins on the same initial conditions?
- Which failures are unique to one policy?
- Which stressors flip the winner?
- Are differences statistically meaningful?

## Episode Identity And Coverage

Episodes are paired only when `(task_id, seed, episode_index)` matches exactly.
The comparison API validates both inputs before computing outcomes:

- duplicate identities in either input are always rejected
- unmatched identities are rejected by default
- empty inputs do not count as complete pairing
- outcomes are ordered by episode identity for deterministic artifacts

Strict matching is the default because silently comparing only an intersection
can bias the result toward a small or easier subset. Complete pairing is a
necessary requirement for a benchmark claim, but it does not replace the other
NyssaBench run, evidence, provenance, or claim-validity gates.

Use partial mode only to diagnose incomplete runs:

```python
from nyssa_bench.arena import compare_episode_pairs

summary = compare_episode_pairs(policy_a, policy_b, allow_partial=True)
```

Partial summaries use the `partial_exploratory` mode, set
`pairing_claim_eligible` to `false`, and carry a visible caveat through every
report. Duplicate identities remain errors in partial mode because selecting
one duplicate would make the comparison ambiguous.

Coverage is reported separately for each input and for their union:

```text
policy A coverage = matched unique keys / policy A unique keys
policy B coverage = matched unique keys / policy B unique keys
joint coverage    = matched unique keys / union of both key sets
```

## Planned Command Shape

```bash
nyssa arena-run \
  --suite maniskill_manipulation_v0 \
  --policy-a bc_policy \
  --policy-b scripted_oracle \
  --episodes 100 \
  --seeds 0 1 2 \
  --blind \
  --out arena_results/bc_vs_scripted
```

## Required Artifacts

- paired manifest
- per-policy run artifacts
- per-seed paired outcomes
- replay videos for both policies
- failure-delta table
- preference or win-rate summary
- significance note
- pairing summary JSON with matched, unmatched, duplicate, and coverage data
- pairing coverage CSV with claim eligibility and unmatched identities

The current arena helpers write:

- `pairwise_results.jsonl` for matched outcomes
- `pairwise_summary.json` for outcomes, coverage, mode, and caveats
- `pairwise_coverage.csv` for tabular coverage auditing
- `arena_report.html` for a human-readable status and unmatched-key report

## Blinding

When `--blind` is used, generated reports should hide policy names behind
stable labels such as `policy_a` and `policy_b` until the comparison is frozen.
This does not make simulation evaluation identical to real-world blinded
human-evaluator studies, but it reduces report bias and prepares the data model
for later real-world evaluation.
