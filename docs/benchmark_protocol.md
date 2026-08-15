# Benchmark Protocol

Report the suite ID, task IDs, policy adapter, engine adapter, simulator version, seed range, number of episodes, task YAML revision, aggregate metrics, per-task metrics, 95% confidence intervals, and failure counts. Do not compare policies across different task specs, engines, success criteria, or randomization ranges unless the report states the difference explicitly.

Only runs that pass `public_claim_validation` should be published as benchmark claims. A public claim requires a supported real simulator adapter, explicit task-to-environment mappings, mapped success predicates, enough episodes, MP4 replay files that exist on disk, diagnosed failure labels, the selected simulator package version, and a recorded commit from a clean git worktree.

Learned-policy runs must also preserve the observation and action contracts used
during training. For bounded RoboMimic policies, demonstrations are normalized
to `[-1, 1]`, the source bounds are recorded, and inference must verify and
invert that transform before stepping the simulator. A completed rollout with a
missing or mismatched training contract is a pipeline diagnostic, not benchmark
evidence.

NyssaBench uses `nyssa-episode-seed-v2`: simulator seed equals
`run_seed * 1_000_000 + episode_index`, and each task receives the same episode
seed sequence. Policies and ablation variants using the same run seed are paired;
different run seeds have disjoint episode namespaces. Packs produced by the
older additive protocol can have almost complete overlap between adjacent run
seeds and must not be described as independent multi-seed replication.
