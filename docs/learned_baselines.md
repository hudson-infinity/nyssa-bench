# Learned Baselines

NyssaBench currently includes a repo-local linear behavior cloning baseline for
the focused ManiSkill result pack.

## Train BC

Generate or import demonstrations first. The repo-local `scripted_oracle` is a
lightweight heuristic and should not be used as a strong demo source unless it
clearly solves the target suite. For stronger ManiSkill demos, generate official
motion-planning trajectories and import them:

```bash
uv run nyssa import-maniskill-demos \
  --input demos/maniskill_motionplanning \
  --out benchmark_results/maniskill_manipulation_v0_planner_demos
```

For smoke testing only, you can generate repo-local scripted demonstrations:

```bash
uv run nyssa experiment \
  --suite maniskill_manipulation_v0 \
  --engine maniskill \
  --policies scripted_oracle \
  --seeds 0 1 2 \
  --episodes 100 \
  --out benchmark_results/maniskill_manipulation_v0_demos
```

Then train the checkpoint:

```bash
uv run nyssa train-bc \
  benchmark_results/maniskill_manipulation_v0_demos/scripted_oracle/seed_0/episodes.json \
  benchmark_results/maniskill_manipulation_v0_demos/scripted_oracle/seed_1/episodes.json \
  benchmark_results/maniskill_manipulation_v0_demos/scripted_oracle/seed_2/episodes.json \
  --out checkpoints/bc_policy.json
```

## Evaluate BC

```bash
NYSSA_BC_CHECKPOINT=checkpoints/bc_policy.json \
uv run nyssa experiment \
  --suite maniskill_manipulation_v0 \
  --engine maniskill \
  --policies random scripted_oracle bc_policy \
  --seeds 0 1 2 \
  --episodes 100 \
  --out benchmark_results/maniskill_manipulation_v0
```

## Interpretation

The linear BC baseline is intentionally simple. It is useful for checking that
NyssaBench can train and evaluate a learned policy from run artifacts. It should
not be described as a strong learned robot policy unless it clearly improves on
random and scripted baselines in validated result artifacts.

## RoboMimic BC

RoboMimic is the next learned baseline after the linear smoke test. Export the
state-aligned planner rollouts to one RoboMimic HDF5 and config per task. The
source must contain live observations paired with actions; action-only planner
imports are rejected by observation coverage and feature variance checks.
NyssaBench normalizes bounded environment actions to `[-1, 1]` in the exported
HDF5, stores the original bounds in both HDF5 and the task manifest, and
denormalizes policy outputs against the live action space during evaluation.
This is required for ManiSkill `pd_joint_pos`, whose absolute joint targets are
not naturally confined to `[-1, 1]`.

Exports created before `nyssa-task-robomimic-export-v3` used raw actions. Delete
their generated HDF5/config/checkpoint tree, re-export from the aligned source,
and retrain; the old checkpoint cannot be corrected only at inference time.

Use held-out simulator seeds for evaluation. A result pack can provide aligned
training observations, but evaluating on seeds already present in that pack is
data leakage and must not be reported as policy generalization.

Export one source directory or result ZIP:

```bash
uv run nyssa export-task-robomimic \
  benchmark_results/maniskill_manipulation_v0_planner_state_demos \
  --out-dir datasets/maniskill_robomimic_by_task \
  --config-dir configs/generated/maniskill_robomimic_by_task \
  --feature-dim 512 \
  --epochs 50 \
  --batch-size 64
```

Train each generated config:

```bash
uv run nyssa train-robomimic \
  --config configs/generated/maniskill_robomimic_by_task/maniskill_pick_cube_bc.json

uv run nyssa train-robomimic \
  --config configs/generated/maniskill_robomimic_by_task/maniskill_push_cube_bc.json

uv run nyssa train-robomimic \
  --config configs/generated/maniskill_robomimic_by_task/maniskill_stack_cube_bc.json
```

After training, evaluate directly from the task export directory. The
`task_robomimic` policy discovers the latest `model_epoch_*.pth` checkpoint for
each task under the generated RoboMimic output tree and derives the live
observation feature dimension from RoboMimic checkpoint metadata. Use
`NYSSA_ROBOMIMIC_FEATURE_DIM` only to override missing or incorrect metadata:

```bash
NYSSA_TASK_ROBOMIMIC_DIR=datasets/maniskill_robomimic_by_task \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
uv run nyssa ablate \
  --suite maniskill_planner_bc_v0 \
  --engine maniskill \
  --policy task_robomimic \
  --seeds 10000 \
  --episodes 20 \
  --variants base \
  --expert-provider maniskill-scripted \
  --out benchmark_results/maniskill_task_robomimic_smoke \
  --capture-replay
```

The export manifest records training episode seeds. Evaluation fails on a
recorded training seed by default. Set `NYSSA_ALLOW_TRAINING_SEED_EVAL=1` only
for an intentional pipeline diagnostic; such a run is not held-out evidence.

Direct files still work if you prefer a curated checkpoint folder:

```txt
checkpoints/robomimic_by_task/maniskill_pick_cube.pth
checkpoints/robomimic_by_task/maniskill_stack_cube.pth
checkpoints/robomimic_by_task/maniskill_push_cube.pth
```

RoboMimic policy outputs are always interpreted as normalized `[-1, 1]`
actions. Keep `task_robomimic_manifest.json` beside curated task checkpoints so
the adapter can verify that training and live action bounds agree. Without the
manifest, the adapter still denormalizes against the live bounds but cannot
detect a controller or robot mismatch.

Use the curated folder by changing only the environment variable:

```bash
NYSSA_TASK_ROBOMIMIC_DIR=checkpoints/robomimic_by_task \
uv run nyssa run \
  --suite maniskill_planner_bc_v0 \
  --engine maniskill \
  --policy task_robomimic \
  --episodes 30 \
  --seed 0 \
  --out runs/task_robomimic_planner_smoke \
  --capture-replay
```

## Stronger Oracle Baselines

ManiSkill ships motion-planning examples for Panda tasks, but they require
native planning dependencies such as `mplib` and Pinocchio. On Windows/Python
3.13 these dependencies may need a separate Linux or conda environment. Use a
planner-backed oracle for publishable upper-bound numbers when those dependencies
are available; otherwise label the repo-local `scripted_oracle` as a lightweight
heuristic baseline.

After generating ManiSkill motion-planning HDF5 files, convert them into Nyssa
episode artifacts:

```bash
uv run nyssa import-maniskill-demos \
  --input demos/maniskill_motionplanning \
  --out benchmark_results/maniskill_manipulation_v0_planner_demos
```

This writes `episodes.json`, `episodes.jsonl`, `manifest.json`, and per-task
episode files under the output directory.

When evaluating BC trained from official ManiSkill Panda motion-planning demos,
use `maniskill_planner_bc_v0`. The official demo generator records actions in
`pd_joint_pos`; this suite uses the same control mode. Do not compare those
checkpoints against `maniskill_manipulation_v0`, which uses end-effector delta
control for the repo-local heuristic baseline.

For the repo-local linear BC baseline, train one checkpoint per task and use the
task-routed policy:

```bash
mkdir -p checkpoints/bc_by_task

uv run nyssa train-bc \
  benchmark_results/maniskill_manipulation_v0_planner_state_demos/maniskill_pick_cube/episodes.json \
  --out checkpoints/bc_by_task/maniskill_pick_cube.json

uv run nyssa train-bc \
  benchmark_results/maniskill_manipulation_v0_planner_state_demos/maniskill_stack_cube/episodes.json \
  --out checkpoints/bc_by_task/maniskill_stack_cube.json

uv run nyssa train-bc \
  benchmark_results/maniskill_manipulation_v0_planner_state_demos/maniskill_push_cube/episodes.json \
  --out checkpoints/bc_by_task/maniskill_push_cube.json

NYSSA_TASK_BC_DIR=checkpoints/bc_by_task \
uv run nyssa run \
  --suite maniskill_planner_bc_v0 \
  --engine maniskill \
  --policy task_bc_policy \
  --episodes 10 \
  --seed 0 \
  --out runs/task_bc_planner_smoke \
  --capture-replay
```
