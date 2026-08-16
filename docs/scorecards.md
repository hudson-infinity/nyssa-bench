# Scorecards

Public scorecards should be generated from committed task specs, fixed seeds,
published run artifacts, and runs that pass `public_claim_validation`. The
assembled artifacts must also pass
`nyssa-result-pack-replay-validation-v1`; a cached run-level status does not
override missing episode videos, stale replay manifests, or unsafe paths.

`benchmark_results/baselines_v0.json` is generated from real simulator run directories. Do not hand-edit score values. Random-only scorecards are sanity baselines and are not public learned-policy benchmark claims.

Generate scripted demos, train the repo-local BC checkpoint, then run the focused baseline matrix:

```bash
uv run nyssa experiment \
  --suite maniskill_manipulation_v0 \
  --engine maniskill \
  --policies scripted_oracle \
  --seeds 0 1 2 \
  --episodes 100 \
  --out benchmark_results/maniskill_manipulation_v0_demos

uv run nyssa train-bc \
  benchmark_results/maniskill_manipulation_v0_demos/scripted_oracle/seed_0/episodes.json \
  benchmark_results/maniskill_manipulation_v0_demos/scripted_oracle/seed_1/episodes.json \
  benchmark_results/maniskill_manipulation_v0_demos/scripted_oracle/seed_2/episodes.json \
  --out checkpoints/bc_policy.json

uv run nyssa export \
  --run benchmark_results/maniskill_manipulation_v0_demos/scripted_oracle/seed_0 \
  --format robomimic \
  --out benchmark_results/maniskill_manipulation_v0_demos/scripted_oracle/seed_0/robomimic.hdf5

uv run nyssa train-robomimic \
  --config configs/policies/robomimic_bc_flat.json

NYSSA_BC_CHECKPOINT=checkpoints/bc_policy.json \
uv run nyssa experiment \
  --suite maniskill_manipulation_v0 \
  --engine maniskill \
  --policies random scripted_oracle bc_policy \
  --seeds 0 1 2 \
  --episodes 100 \
  --out benchmark_results/maniskill_manipulation_v0
```

Every scorecard should publish `metrics.json`, `episodes.json`,
`episodes.jsonl`, `replay_manifest.json`, `report.html`, environment metadata,
package versions, git info, per-task success rates, confidence intervals,
failure counts, run-level public-claim validation, assembled replay validation,
and unsupported-stressor reporting.

For a learned baseline outside the experiment helper, run a real checkpoint through the adapter factory and include that run directory in the same scorecard command:

```bash
NYSSA_ROBOMIMIC_POLICY=my_project.policies:create_robomimic_policy \
uv run nyssa run --suite maniskill_smoke_v0 --engine maniskill --policy robomimic --episodes 100 --seed 42 --out runs/maniskill_robomimic_v0

uv run nyssa scorecard \
  runs/maniskill_random_v0 \
  runs/mujoco_random_v0 \
  runs/maniskill_robomimic_v0 \
  --out benchmark_results/baselines_v0.json \
  --comparison-out reports/real_baselines_v0.html \
  --leaderboard-out site/leaderboard/leaderboard.json
```
