# Policy Adapter Reference

Policies implement `act(observation)`. Production adapters also implement the
episode lifecycle, metadata, strict observation/action contracts, and explicit
failure behavior described in [Adding New Policies](adding_new_policies.md).

List installed registry entries and their support tiers:

```bash
uv run nyssa list-policies
```

## Built-In Matrix

| Policy ID | Support tier | Loading path | Observation/action behavior |
| --- | --- | --- | --- |
| `random` | sanity baseline | no checkpoint | Samples the live Box/discrete action contract and seeds its local RNG per episode. |
| `scripted_oracle` | oracle adapter | repo heuristic or `NYSSA_SCRIPTED_ORACLE_POLICY` | Passes structured observation to a scripted controller; repo default is a lightweight heuristic, not a guaranteed oracle. |
| `bc_policy` | learned baseline adapter | `NYSSA_BC_POLICY`, then `NYSSA_BC_CHECKPOINT` | Flat fixed-dimensional repo BC; environment-space output is padded/truncated and clipped. |
| `demo_replay_policy` | teacher replay baseline | `NYSSA_DEMO_REPLAY_DIR` | Task-routed successful demonstration replay with optional privileged initial-state restore; not learned. |
| `task_bc_policy` | task-routed learned baseline | `NYSSA_TASK_BC_POLICY`, then `NYSSA_TASK_BC_DIR` | One repo BC JSON per canonical task key; missing task errors by default. |
| `lerobot` | adapter hook | `NYSSA_LEROBOT_POLICY`, then `NYSSA_LEROBOT_POLICY_PATH` | Passes the wrapped observation to the external model; no universal preprocessing or action validation. |
| `robomimic` | adapter hook | `NYSSA_ROBOMIMIC_POLICY`, then `NYSSA_ROBOMIMIC_CHECKPOINT` | Flat observation; normalized `[-1, 1]` output is strictly sized and denormalized to live bounds. |
| `task_robomimic` | task-routed RoboMimic adapter | `NYSSA_TASK_ROBOMIMIC_DIR` | Per-task discovery plus action-contract validation and seed guard when the task manifest is present. |
| `diffusion` | adapter hook | `NYSSA_DIFFUSION_POLICY` | External structured model; supports action chunks when runner horizons are configured. |
| `openvla` | adapter hook | `NYSSA_OPENVLA_POLICY` | External structured model; instruction/preprocessing must be implemented by the loaded model/adapter. |

Support tier describes integration maturity, not policy quality. An adapter hook
without a validated checkpoint and held-out result is not a baseline result.

## Environment Loading

The named adapters use these variables:

| Variable | Default/fallback |
| --- | --- |
| `NYSSA_SCRIPTED_ORACLE_POLICY` | repo-local ManiSkill scripted heuristic |
| `NYSSA_BC_POLICY` | `NYSSA_BC_CHECKPOINT`, then `checkpoints/bc_policy.json` |
| `NYSSA_TASK_BC_POLICY` | `NYSSA_TASK_BC_DIR`, then `checkpoints/bc_by_task` |
| `NYSSA_DEMO_REPLAY_DIR` | `benchmark_results/maniskill_official_demos_import_v2` |
| `NYSSA_LEROBOT_POLICY` | `NYSSA_LEROBOT_POLICY_PATH`, then `checkpoints/lerobot_policy` |
| `NYSSA_ROBOMIMIC_POLICY` | `NYSSA_ROBOMIMIC_CHECKPOINT`, then `checkpoints/robomimic_policy.pth` |
| `NYSSA_TASK_ROBOMIMIC_DIR` | `checkpoints/robomimic_by_task` |
| `NYSSA_DIFFUSION_POLICY` | required, no fallback model |
| `NYSSA_OPENVLA_POLICY` | required, no fallback model |

Auxiliary controls include `NYSSA_TASK_BC_MISSING`,
`NYSSA_DEMO_REPLAY_FEATURE_DIM`, `NYSSA_ROBOMIMIC_FEATURE_DIM`, and
`NYSSA_ALLOW_TRAINING_SEED_EVAL`. Their diagnostic/override semantics are
documented in the production guide; do not use them to conceal missing
checkpoints or training/evaluation overlap.

Environment model values use `module:attribute` or `file.py:attribute`. A class
is instantiated with no arguments; a function/object is used directly. Use a
class factory for model construction rather than pointing at a zero-argument
function and expecting it to be invoked.

```bash
NYSSA_OPENVLA_POLICY=my_project.policies:OpenVLAPolicyFactory \
uv run nyssa run --suite <suite> --engine maniskill --policy openvla ...

NYSSA_DIFFUSION_POLICY=my_project.policies:DiffusionPolicyFactory \
uv run nyssa run \
  --suite <suite> \
  --engine maniskill \
  --policy diffusion \
  --policy-action-horizon 16 \
  --policy-execution-horizon 4 \
  ...
```

A direct Python file passed to `--policy` is a different loader: it invokes
`create_policy()` or instantiates `PolicyAdapter` from that file.

## Model Call Resolution

Generic adapters call the first available method from an adapter-specific list
containing `predict_action`, `select_action`, `get_action`, and/or `act`, then
fall back to calling the model object. Dictionary outputs are unwrapped under
`action`, `actions`, `pred_action`, or `pred_actions`; tensors are detached and
converted to CPU/NumPy where possible.

This normalization does not prove correct shape, units, controller, or bounds.
External models must enforce those contracts before simulator execution.

## Lifecycle Forwarding

`scripted_oracle`, `bc_policy`, and `task_bc_policy` forward `reset` and `close`
to their controller/model where implemented. RoboMimic calls its episode reset
methods. The current LeRobot, diffusion, and OpenVLA hooks do not forward model
reset/close. Use a direct custom `Policy` when the external model needs strict
lifecycle management or richer metadata.

## Optional Dependencies

Install all stable stacks:

```bash
uv sync --extra all --extra dev
```

Or add policy stacks to an existing lean simulator environment:

```bash
uv sync --inexact --extra lerobot --extra robomimic --extra vla --extra diffusion
```

The `vla` and `diffusion` extras provide common PyTorch/Transformers libraries,
not every upstream model repository or checkpoint. Install model code and
weights according to its upstream license and instructions, then record exact
versions and checkpoint hashes in policy metadata.

## Claim Boundaries

- `random` verifies the evaluation pipeline, not learned performance.
- `scripted_oracle` is an oracle only when its controller is shown to solve the
  target task; the repo heuristic must be labeled accordingly.
- `demo_replay_policy` is a privileged teacher replay, not policy generalization.
- Repo-local linear/KNN models are smoke baselines unless validated strongly.
- `NYSSA_TASK_BC_MISSING=zero` and `NYSSA_ALLOW_TRAINING_SEED_EVAL=1` are
  diagnostic overrides and invalidate learned held-out claims.
- A loaded LeRobot, RoboMimic, diffusion, or VLA hook becomes evidence only with
  checkpoint provenance, compatible contracts, held-out seeds, and complete
  result artifacts.
