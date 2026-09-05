# Learning evidence export

NyssaBench can package measured failures, interventions, recovery branches, and
stress-boundary cases for external learning or data-selection systems. The
export is a handoff contract. It does not train a policy and does not establish
that any learning method is effective.

## Package contents

`nyssa-learning-evidence-manifest-v1` identifies source runs, licenses, privacy
rules, embedded data files, facet indexes, and the mandatory evaluation-reuse
policy. The package contains:

- `episodes.jsonl` with `nyssa-learning-evidence-episode-v1` records;
- `evaluation_exclusions.json` with one exclusion per source episode;
- `facets.json` for deterministic filtering;
- `manifest.json` with SHA-256 identities for every embedded file and the
  manifest itself.

Loading a package revalidates file sizes and hashes, parses every nested schema,
checks episode/exclusion equality, recomputes facets, and rejects unknown or
inconsistent fields.

## Action semantics

Each `nyssa-learning-evidence-step-v1` keeps these values separate:

- policy, cached policy, or recovery proposal before verification;
- rejected policy action, when rejection occurred;
- action selected by the verifier, oracle, or recovery system;
- selected action before stressor transformation;
- action actually sent to the engine;
- explicit oracle and recovery action fields.

The runner writes these fields at execution time. For older result packs, an
accepted policy action may be reconstructed from the executed action. An old
intervention step is rejected when the original policy proposal is missing;
the exporter does not relabel the recovery action as a supervised policy target.

## Failure and recovery evidence

Episode records preserve the validated temporal FailureLedger, terminal summary
label, stressor context, recovery/intervention metrics, and complete
counterfactual branch records. A failed source episode must contain both a
failure label and temporal ledger.

When a source run appears in a stress-search study, the export includes the
study and proposal identities, study hash, observation status, and whether the
condition was a held-out confirmation. Boundary metadata is linked through the
source run ID recorded by `stress-search-ingest-run`.

## Evaluation exclusion

Every exported episode receives a `nyssa-evaluation-exclusion-v1` record with
source benchmark, suite, split, run, episode identity, and source episode hash.
Each record sets `excluded_from_evaluation: true` and explains why the episode
cannot silently re-enter held-out evaluation.

The manifest fixes `evaluation_reuse_policy` to `excluded`. Existing Nyssa
training loaders do not discover this package as an ordinary `episodes.json`
source, so using it requires an explicit downstream integration that consumes
the exclusion records.

Downstream integrations should call
`validate_learning_evidence_use(package, purpose="training")` or use
`purpose="data_selection"`. The same gate rejects `purpose="evaluation"` and
requires every episode exclusion to remain active.

Hidden-test evidence cannot be exported with public privacy classification.
Restricted and private packages require explicit privacy restrictions.

## Content-addressed artifacts

Replay and failure videos remain in their source run. The export records a
`nyssa-run://` URI, media type, byte count, and SHA-256 hash instead of copying
large files. `load_learning_evidence(..., verify_external_artifacts=True)` can
verify those files while the source runs remain available.

Observations larger than `--max-inline-observation-bytes` are replaced by a
content hash and JSON pointer into the source `episodes.json`. The source file
is itself referenced by hash. Small state observations remain inline.

## Export command

```bash
uv run nyssa export-learning-evidence \
  benchmark_results/policy_a/seed_0 \
  benchmark_results/policy_a/seed_1 \
  --out datasets/policy_a_failure_evidence \
  --benchmark-id mujoco_control_v0 \
  --split-id public_test_v1 \
  --split-partition public_test \
  --split-sha256 <64-character-content-hash> \
  --policy-family task_bc_policy=behavior_cloning \
  --license Apache-2.0 \
  --boundary-study benchmark_results/action_noise_search/study.json \
  --failures-only \
  --verify-external-artifacts
```

Use `*=family_name` only as an explicit fallback. Multiple `--license`,
`--privacy-restriction`, and `--boundary-study` options are accepted.

Validate either the package directory or manifest:

```bash
uv run nyssa validate datasets/policy_a_failure_evidence
uv run nyssa validate datasets/policy_a_failure_evidence/manifest.json
```

## Filtering

`query_learning_evidence` intersects one or more indexed facets:

```python
from nyssa_bench.learning_export import load_learning_evidence, query_learning_evidence

package = load_learning_evidence("datasets/policy_a_failure_evidence")
episodes = query_learning_evidence(
    package,
    task="mujoco_pusher",
    failure_type="missed_target",
    stressor="action_gaussian_noise",
    recoverability="counterfactually_evaluated",
)
```

Available facets are task, policy family, failure type, stressor, severity,
recoverability, and boundary membership.

## Scope

NyssaBench exports observations and measured evidence. Demonstration selection,
relabeling, replay-buffer policy, reinforcement learning, imitation learning,
continual updates, and transfer algorithms remain outside this repository. A
downstream training run needs its own held-out evaluation and cannot reuse the
exported episodes as benchmark evidence.
