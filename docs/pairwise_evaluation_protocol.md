# Pairwise Evaluation Protocol

Pairwise evaluation compares two policy result sets under matched conditions.
The winner/tie view remains available for compatibility, while paired evidence
records explain differences in failure timing, mechanisms, safety,
intervention, and recovery.

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
- initial observation, stressor execution, and detector contracts are hashed
  and checked for every matched key

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

Condition mismatches are rejected by default. Use
`allow_condition_mismatch=True` only for diagnostics. In that mode the pair is
retained in the compatibility winner view, but scientific deltas exclude it,
the mismatch fields and both condition hashes are recorded, and the summary is
not claim eligible.

Each comparison has a `nyssa-pairwise-comparison-contract-v1` document and a
canonical SHA-256 identity. Callers can supply task and success contract hashes;
the default contract records when those stronger identities are unavailable
instead of inferring them from task names.

Coverage is reported separately for each input and for their union:

```text
policy A coverage = matched unique keys / policy A unique keys
policy B coverage = matched unique keys / policy B unique keys
joint coverage    = matched unique keys / union of both key sets
```

## Planned command shape

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
- paired metric CSV with status, uncertainty, denominator, and missing count

The current arena helpers write:

- `pairwise_results.jsonl` for matched outcomes
- `pairwise_summary.json` for outcomes, coverage, mode, and caveats
- `pairwise_coverage.csv` for tabular coverage auditing
- `pairwise_metrics.csv` for paired scientific measurements
- `arena_report.html` for a human-readable status and unmatched-key report

`pairwise_results.jsonl` uses `nyssa-pairwise-outcome-v2`. Every row carries the
comparison contract hash, condition hashes, condition compatibility, both
episode profiles, temporal failure evidence, censoring status, and metric
deltas. `pairwise_summary.json` uses `nyssa-pairwise-summary-v2`.

## Paired measurements

The aggregate report includes paired success difference with uncertainty,
failure category/role/mechanism differences, and observed time-to-failure
deltas. Successful and truncated episodes remain right censored instead of
receiving an invented failure time.

Safety, damage, collision, intervention, false/harmful intervention,
counterfactual recovery gain, and branch coverage are compared when both sides
provide the metric. Every measurement reports available and missing pair
counts. Both-succeeded and both-failed episodes remain in the paired evidence;
equal terminal success does not erase a difference in mechanism, onset, safety,
or recoverability.

## Blinding

When `--blind` is used, generated reports should hide policy names behind
stable labels such as `policy_a` and `policy_b` until the comparison is frozen.
This does not make simulation evaluation identical to real-world blinded
human-evaluator studies, but it reduces report bias and prepares the data model
for later real-world evaluation.
