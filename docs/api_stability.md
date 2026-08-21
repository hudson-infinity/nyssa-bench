# API Stability

NyssaBench is pre-1.0. Breaking changes remain possible, but contributors must
treat public imports, adapter interfaces, CLI commands, and serialized result
artifacts as compatibility contracts rather than internal implementation
details.

## Stability Levels

### Public Stable Candidates

The following imports are stable candidates and should change additively:

- `nyssa_bench.Suite`
- `nyssa_bench.PolicyRunner`
- `nyssa_bench.Report`
- `nyssa_bench.FailureEvent`
- `nyssa_bench.FailureEventDraft`
- `nyssa_bench.FailureEventLedger`
- `nyssa_bench.FailureEvidence`
- `nyssa_bench.core.task.TaskSpec`
- `nyssa_bench.engines.base.NyssaEngine`
- `nyssa_bench.policies.base.PolicyLike`

Adapter methods with no-op defaults, such as `drain_failure_events()`, are
extension points. Add optional methods with backward-compatible defaults rather
than adding a new required abstract method without a migration period.

### Serialized Contracts

Any payload with a `format` or schema version is a public data contract. Current
examples include:

- stressor specs, configs, execution context, and robustness sweeps;
- temporal failure evidence, events, causal hypotheses, and episode ledgers;
- episode JSON/JSONL, run and dataset manifests, replay manifests, and recovery
  datasets;
- comparison contracts, pairwise coverage, scorecards, and leaderboards.

Task and suite YAML are also compatibility-sensitive even where they do not yet
carry one shared format field. Task IDs, suite IDs, environment mappings,
success predicates, seed protocols, and action/observation contracts must not be
silently reinterpreted.

### Internal APIs

Private helpers prefixed with `_` and modules not imported by documented public
entry points may change without a deprecation cycle. They still require tests
when they affect observable behavior or artifacts.

## Compatibility Rules

For a compatible change:

1. Prefer adding optional fields with explicit defaults.
2. Preserve existing field meaning, units, denominator, and evidence source.
3. Keep readers for result packs written by the previous schema version.
4. Add round-trip tests for the current version and migration fixtures for old
   versions.
5. Reject unknown or inconsistent data where silent coercion could strengthen a
   benchmark claim.
6. Preserve top-level compatibility fields when a richer representation is
   introduced. For example, `failure_label` remains available alongside the
   temporal failure ledger.
7. Update exports, type annotations, docs, examples, and generated artifact
   manifests in the same pull request.

Do not reuse an existing format identifier after a breaking schema change.
Create a new version, document the difference, and provide a deterministic
migration when one is scientifically valid. If old evidence cannot support the
new claim, preserve the artifact but lower or invalidate the claim tier.

## Deprecation And Removal

Before removing a public candidate or serialized field:

- mark it deprecated in code and documentation;
- identify its replacement and conversion rules;
- retain it for at least one documented release unless it creates a security or
  scientific-validity defect;
- test both the deprecated read path and the replacement;
- describe the removal timeline in `CHANGELOG.md` and the pull request.

Environment variables, policy factory strings, CLI flags, and output filenames
follow the same process because automation depends on them.

## Contributor Review

Call out compatibility impact explicitly in every pull request. Reviewers should
be able to answer:

- Can old task, suite, run, and episode artifacts still be loaded?
- Are comparison and claim semantics unchanged or versioned?
- Can existing engine and policy adapters continue to run?
- Are newly privileged evidence fields separated from policy-visible data?
- Does the change preserve deterministic seeds and provenance?
- Is any break documented with migration tests and release notes?

Before v1.0, breaking changes may still occur. The v1.0 target is to freeze:

- task spec schema
- suite spec schema
- engine adapter API
- policy adapter API
- metrics summary shape
- run artifact layout
- plugin registration API

See [CONTRIBUTING.md](../CONTRIBUTING.md) for branch, test, simulator, artifact,
and pull-request requirements.
