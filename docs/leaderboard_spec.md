# Leaderboard spec

`nyssa leaderboard` writes a `nyssa-leaderboard-v3` JSON object. It does not
compare runs until their comparison-critical metadata has been validated.

Top-level fields are:

- `format`: `nyssa-leaderboard-v3`;
- `comparable`: whether every run has the same comparison contract;
- `comparison_mode`: `strict` or `exploratory`;
- `comparison_contract` and its deterministic SHA-256;
- `mismatches`: field-level differences;
- `ordering`: the declared display-order rule;
- `ranking`: run entries with success, metric vectors, claim status, and run
  identity.

The display order uses task success rate in descending order and the run path as
a deterministic tie-breaker. It is not a universal reliability ranking. Each
entry retains the complete metric vector so clean and shifted success,
robustness, failure timing, recovery, safety, intervention, and compute costs can
be compared separately. No weighted scalar is used as a primary metric or
tie-breaker.

The comparison contract covers suite, engine, task set, per-task success
predicate, per-task randomization and OOD stressor declarations, episodes per
task, and seed-protocol semantics. Policy identity and the concrete run seed are
excluded because those are expected comparison dimensions.

Strict mode rejects mismatched or incomplete contracts. `--allow-incompatible`
writes an exploratory payload with `comparable: false`, preserves every
mismatch, and must not be presented as a benchmark ranking.
