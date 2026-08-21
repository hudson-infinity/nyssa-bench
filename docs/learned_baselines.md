# Learned Baselines

NyssaBench includes lightweight repository-local behavior-cloning models and
RoboMimic integration. These baselines validate data, checkpoint, routing, and
evaluation contracts. They are not automatically strong robot policies.

## Baseline Matrix

| Model | Checkpoint format | Observation | Action behavior | Intended use |
| --- | --- | --- | --- | --- |
| Linear BC | `nyssa-linear-bc-v1` | Deterministic fixed-length numeric flattening | Environment-space output padded/truncated and clipped | Fast pipeline smoke |
| KNN BC | `nyssa-knn-bc-v1` | Same flattening with standardized features | Weighted neighbor action, padded/truncated and clipped | Non-parametric smoke/baseline |
| Sequence KNN BC | `nyssa-sequence-knn-bc-v1` | Same flattening | Returns an environment-space action chunk | Action-sequence pipeline validation |
| RoboMimic BC | upstream `.pth` plus Nyssa task manifest | Validated fixed-length flat observation | Strict normalized `[-1, 1]` output denormalized to live bounds | Stronger external IL baseline |

Flat preprocessing recursively collects numeric values from `observation.raw`
in sorted key order, removes top-level simulator-state payloads, then truncates
or zero-pads to `feature_dim`. This representation is deterministic but loses
structure and is not a substitute for image/language preprocessing.

Repo-local BC uses permissive action fitting. A checkpoint action is padded or
truncated to the live action size and clipped. Treat action-contract mismatches
as invalid even when this compatibility behavior lets a smoke run continue.

## Data Requirements

Training episodes must contain task IDs, live observations, actions, and action
space contracts. Keep controller/control mode, robot, observation mode, and
action units identical between collection and evaluation.

By default `train-task-bc` and `export-task-robomimic` use successful episodes
only. `--include-failures` changes the training distribution and must be
justified; rejected or failed actions are not automatically corrective targets.

Official ManiSkill Panda motion-planning demonstrations use `pd_joint_pos`.
Evaluate checkpoints trained from them on `maniskill_planner_bc_v0`, not the
end-effector-delta `maniskill_manipulation_v0` suite.

## End-To-End Task-Routed Sequence BC

Assume state-aligned planner demonstrations have been imported under:

```text
benchmark_results/maniskill_manipulation_v0_planner_state_demos
```

Train one sequence checkpoint per task:

```bash
uv run nyssa train-task-bc \
  benchmark_results/maniskill_manipulation_v0_planner_state_demos \
  --out-dir checkpoints/maniskill_sequence_bc_by_task \
  --model sequence-knn \
  --feature-dim 512 \
  --knn-k 1 \
  --action-horizon 16
```

Use a held-out run seed for a one-episode pipeline smoke. The action horizon
declares the returned chunk; the execution horizon commits four actions before
the next model call:

```bash
NYSSA_TASK_BC_DIR=checkpoints/maniskill_sequence_bc_by_task \
uv run nyssa run \
  --suite maniskill_planner_bc_v0 \
  --engine maniskill \
  --policy task_bc_policy \
  --episodes 1 \
  --seed 10000 \
  --policy-action-horizon 16 \
  --policy-execution-horizon 4 \
  --out runs/maniskill_sequence_bc_smoke \
  --no-replay
```

After the smoke succeeds, run the held-out evaluation with replay evidence:

```bash
NYSSA_TASK_BC_DIR=checkpoints/maniskill_sequence_bc_by_task \
uv run nyssa experiment \
  --suite maniskill_planner_bc_v0 \
  --engine maniskill \
  --policies task_bc_policy \
  --seeds 10000 10001 10002 \
  --episodes 100 \
  --policy-action-horizon 16 \
  --policy-execution-horizon 4 \
  --out benchmark_results/maniskill_sequence_bc_heldout \
  --capture-replay
```

Verify that no resulting episode seed appears in the training source. Repo-local
JSON BC does not enforce this automatically. Record the source manifest,
training seeds, feature dimension, checkpoint hashes, and evaluation seeds in
the experiment notes/policy metadata.

Missing task checkpoints fail by default. `NYSSA_TASK_BC_MISSING=zero` is only a
diagnostic pipeline fallback and cannot support a learned-policy claim.

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
  --seeds 10000 10001 10002 \
  --episodes 100 \
  --out benchmark_results/maniskill_manipulation_v0 \
  --capture-replay
```

## Interpretation

The training source above uses run seeds 0, 1, and 2; the evaluation deliberately
uses different run seeds. The linear BC baseline is intentionally simple. It is
useful for checking that NyssaBench can train and evaluate a learned policy from
run artifacts. It should
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
detect a controller/robot mismatch or training-seed overlap. Treat direct
checkpoint runs without reconstructed provenance as diagnostics, not held-out
evidence.

Use the curated folder by changing only the environment variable:

```bash
NYSSA_TASK_ROBOMIMIC_DIR=checkpoints/robomimic_by_task \
uv run nyssa run \
  --suite maniskill_planner_bc_v0 \
  --engine maniskill \
  --policy task_robomimic \
  --episodes 30 \
  --seed 10000 \
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
  --seed 10000 \
  --out runs/task_bc_planner_smoke \
  --capture-replay
```
