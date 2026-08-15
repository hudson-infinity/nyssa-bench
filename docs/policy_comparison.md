# Policy Comparison Reports

Use `nyssa compare` to rank compatible run directories by success rate and prototype reliability score. Runs must share suite, engine, task set, success predicates, stressors, episode count, and seed protocol. Policy identity and concrete run seeds may differ.

```bash
nyssa compare runs/mujoco_policy_a_seed0 runs/mujoco_policy_b_seed1 --out reports/compare.html
```

Use `nyssa leaderboard` to write a versioned JSON ranking with its comparison contract and SHA-256 fingerprint.

```bash
nyssa leaderboard runs/mujoco_policy_a_seed0 runs/mujoco_policy_b_seed1 --out reports/leaderboard.json
```

By default, a mismatch raises an error that names every incompatible field and the values from each run. Missing comparison metadata is also rejected, preventing legacy or partial artifacts from being treated as valid rankings.

For diagnostics only, use the explicit override:

```bash
nyssa compare \
  runs/maniskill_policy_a \
  runs/mujoco_policy_b \
  --allow-incompatible \
  --out reports/exploratory.html
```

The resulting report is labeled `NON-COMPARABLE EXPLORATORY OUTPUT`, lists the mismatches, and calls its sorted table an exploratory ordering rather than a ranking. The same override is available on `nyssa leaderboard` and `nyssa scorecard`.
