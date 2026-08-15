# Leaderboard Spec

`nyssa leaderboard` writes a `nyssa-leaderboard-v2` JSON object. It does not rank runs until their comparison-critical metadata has been validated.

Top-level fields:

- `format`: `nyssa-leaderboard-v2`
- `comparable`: whether every run has the same comparison contract
- `comparison_mode`: `strict` or `exploratory`
- `comparison_contract`: the versioned shared/set contract
- `comparison_contract_sha256`: deterministic SHA-256 of that contract
- `mismatches`: field-level differences, empty for a comparable ranking
- `ranking`: entries sorted by success rate and prototype reliability score

Each ranking entry contains:

- `rank`
- `run_dir`
- `success_rate`
- `success_rate_ci95`
- `prototype_reliability_score`
- `benchmark_tier`
- `public_claim`
- `public_claim_status`
- `primary_failure_mode`
- intervention, recovery, verifier, and wall-time metrics

The comparison contract covers suite, engine, task set, per-task success predicate, per-task randomization and OOD stressor declarations, episodes per task, and seed-protocol semantics. Policy identity and the concrete run seed are excluded because those are expected comparison dimensions.

Strict mode rejects mismatched or incomplete contracts. `--allow-incompatible` writes an exploratory payload with `comparable: false`, preserves every mismatch, and must not be presented as a benchmark ranking.
