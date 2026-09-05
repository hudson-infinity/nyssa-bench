# Adding New Policies

Policy adapters are responsible for more than calling a model. A production
adapter must make task routing, observation preprocessing, action units and
shape, sequence execution, checkpoint identity, random seeds, and failure
behavior explicit enough to reproduce and audit a result.

This guide covers direct Python policy files, environment-loaded model classes,
repository-integrated policies, and learned checkpoint baselines. See
[Policy Adapters](policy_adapters.md) for the built-in matrix and
[Learned Baselines](learned_baselines.md) for training workflows.

## Runner Lifecycle

`PolicyRunner` creates one policy object for an evaluation and uses this order:

1. Construct/load the policy once.
2. For each task and episode, call `policy.reset(task=task_spec,
   seed=episode_seed)` when available.
3. Reset the engine.
4. If the policy defines `initial_state`, request and restore that simulator
   state before the first action.
5. Call `act(observation)` when no cached action from a previous chunk remains.
6. Drain optional policy failure-event drafts after each new policy action.
7. Apply verifier/recovery and stressor transforms, then step the engine.
8. Call `close()` once after all tasks, including when evaluation raises.
9. Read `metadata()` for the run manifest. Metadata must remain available after
   `close()`.

The only method required by `PolicyLike` is:

```python
def act(observation: dict) -> object:
    ...
```

A production policy should implement the full lifecycle where relevant:

```python
def reset(task=None, seed=None): ...
def initial_state(observation=None): ...  # only for state-aligned replay/evaluation
def act(observation): ...
def drain_failure_events(): ...
def metadata(): ...
def close(): ...
```

`reset` receives the loaded `TaskSpec` and the episode seed, not just the run
seed. NyssaBench episode seeds use:

```text
run_seed * 1_000_000 + episode_index
```

Clear recurrent state, action queues, task routing, and all framework RNG state
inside `reset`. Do not let one task or episode influence the next.

`initial_state` is optional and privileged. Use it only when the policy's
evaluation contract requires restoring a recorded simulator state. The selected
engine must support `set_state`; otherwise evaluation fails.

## Policy Metadata

The runner stores `policy.metadata()` under `run.yaml:policy_metadata`. If the
method is absent or does not return a mapping, it records only `policy_class`.
Production adapters should provide a
[`nyssa-nep-policy-contract-v0.1`](nyssa_evaluation_protocol.md) and make
`metadata()` report the same identity. Run `nyssa conform-policy` before a full
experiment. See the [external policy quickstart](external_policy_quickstart.md).
The metadata should include at least:

```python
{
    "policy_id": "my_policy_v1",
    "policy_class": "MyPolicyAdapter",
    "policy_family": "diffusion",
    "checkpoint_path": "checkpoints/my_policy/model.safetensors",
    "checkpoint_sha256": "...",
    "checkpoint_format": "my-policy-v1",
    "framework": "torch",
    "framework_version": "2.x",
    "device": "cuda:0",
    "dtype": "float32",
    "observation_contract": {"mode": "state_dict", "feature_dim": 512},
    "action_output_space": "environment",
    "action_shape": [7],
    "action_horizon": 1,
    "training_sources": ["dataset-manifest-sha256"],
    "training_episode_seeds": [0, 1, 2],
    "deterministic_eval": True,
}
```

Use repository-relative paths or stable artifact IDs. Do not record secrets,
temporary download URLs, or unverifiable labels such as `latest`. Record the
actual checkpoint hash after loading. If metadata is unavailable, the run may
be useful as a smoke test but is weak policy evidence.

## Direct Python Policy File

Pass a file directly to `--policy`:

```bash
uv run nyssa run \
  --suite maniskill_planner_bc_v0 \
  --engine maniskill \
  --policy policies/strict_linear_policy.py \
  --episodes 1 \
  --seed 10000 \
  --out runs/strict_linear_policy_smoke \
  --no-replay
```

The file must expose `create_policy()` or a `PolicyAdapter` class. The resulting
object must have a callable `act(observation)`.

This example loads a small JSON checkpoint and rejects action-contract
mismatches instead of silently padding, truncating, or clipping them:

```python
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from nyssa_bench.baselines.features import action_space_contract, flatten_observation
from nyssa_bench.policies.base import Policy


def policy_visible_observation(observation: dict[str, Any]) -> dict[str, Any]:
    raw = observation.get("raw", observation)
    if not isinstance(raw, dict):
        return observation
    filtered = {
        key: value
        for key, value in raw.items()
        if key not in {"env_states", "states", "state"}
    }
    return {**observation, "raw": filtered}


class PolicyAdapter(Policy):
    def __init__(self) -> None:
        self.path = Path(os.environ["MY_POLICY_CHECKPOINT"])
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("format") != "strict-linear-policy-v1":
            raise RuntimeError(f"Unsupported checkpoint format: {payload.get('format')}")
        self.feature_dim = int(payload["feature_dim"])
        self.action_shape = tuple(int(value) for value in payload["action_shape"])
        self.weights = np.asarray(payload["weights"], dtype=float)
        self.bias = np.asarray(payload["bias"], dtype=float)
        action_size = int(np.prod(self.action_shape))
        if self.weights.shape != (self.feature_dim, action_size):
            raise ValueError("Checkpoint weights do not match feature/action dimensions")
        if self.bias.shape != (action_size,):
            raise ValueError("Checkpoint bias does not match action dimension")
        self.task_id: str | None = None
        self.episode_seed: int | None = None

    def reset(self, task: Any | None = None, seed: int | None = None) -> None:
        self.task_id = str(getattr(task, "task_id", "")) or None
        self.episode_seed = seed

    def act(self, observation: dict[str, Any]) -> np.ndarray:
        contract = action_space_contract(observation)
        live_shape = tuple(int(value) for value in contract["shape"])
        if live_shape != self.action_shape:
            raise ValueError(
                f"Live action shape {live_shape} does not match checkpoint {self.action_shape}"
            )
        features = flatten_observation(
            policy_visible_observation(observation), self.feature_dim
        )
        action = (features @ self.weights + self.bias).reshape(self.action_shape)
        if not np.all(np.isfinite(action)):
            raise ValueError("Policy produced a non-finite action")
        low = np.asarray(contract["low"], dtype=float).reshape(self.action_shape)
        high = np.asarray(contract["high"], dtype=float).reshape(self.action_shape)
        if np.any(action < low) or np.any(action > high):
            raise ValueError("Policy produced an out-of-bounds environment action")
        return action

    def metadata(self) -> dict[str, Any]:
        return {
            "policy_id": "strict_linear_policy_v1",
            "policy_class": self.__class__.__name__,
            "policy_family": "linear_bc",
            "checkpoint_path": self.path.as_posix(),
            "checkpoint_sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
            "checkpoint_format": "strict-linear-policy-v1",
            "observation_contract": {
                "feature_dim": self.feature_dim,
                "privileged_state_excluded": True,
            },
            "action_output_space": "environment",
            "action_shape": list(self.action_shape),
            "action_horizon": 1,
            "deterministic_eval": True,
        }


def create_policy() -> PolicyAdapter:
    return PolicyAdapter()
```

The example uses NyssaBench's deterministic numeric flattener for illustration.
Image-language policies should normally preserve structured camera and language
inputs instead of flattening them.

## Environment `module:attribute` Loader

Named policy adapters load external models through environment variables such
as `NYSSA_OPENVLA_POLICY` or `NYSSA_DIFFUSION_POLICY`. The value may be
`module:attribute` or `file.py:attribute`.

The current resolver behaves as follows:

- if the attribute is a class, NyssaBench instantiates it with no arguments;
- otherwise the attribute itself becomes the model;
- a function is therefore treated as the action callable, not invoked as a
  zero-argument factory.

Use a no-argument class as the factory when model construction is required:

```python
# my_project/policies.py
from __future__ import annotations

from typing import Any


class OpenVLAPolicyFactory:
    def __new__(cls):
        return load_model_and_processor()


def load_model_and_processor():
    # Return an object implementing predict_action, select_action, get_action,
    # act, or __call__, according to the selected NyssaBench named adapter.
    raise NotImplementedError


def direct_action_callable(observation: dict[str, Any]):
    # A module-level function is used directly and must accept one observation.
    raise NotImplementedError
```

```bash
NYSSA_OPENVLA_POLICY=my_project.policies:OpenVLAPolicyFactory \
uv run nyssa run --suite <suite> --engine maniskill --policy openvla ...
```

On PowerShell:

```powershell
$env:NYSSA_OPENVLA_POLICY = "my_project.policies:OpenVLAPolicyFactory"
uv run nyssa run --suite <suite> --engine maniskill --policy openvla ...
```

The generic loader does not pass a checkpoint path, device, task, or config to
the class constructor. Read explicit project environment variables or return a
custom direct policy file when richer construction is needed.

Named adapters differ in lifecycle forwarding. `BCPolicy` and `TaskBCPolicy`
forward model `reset` and `close`; RoboMimic invokes episode reset helpers.
LeRobot, diffusion, and OpenVLA adapter hooks currently call the model for
actions but do not forward model reset/close. Use a direct `Policy` adapter when
the external model requires those lifecycle guarantees.

## Observation Contracts

Every policy receives the wrapped observation:

```python
{
    "raw": simulator_observation,
    "action_space": {
        "type": "box",
        "shape": [7],
        "low": [...],
        "high": [...],
        "dtype": "float32",
    },
}
```

The live payload may contain NumPy arrays, tensors, nested dictionaries, batch
dimensions, state, images, or language fields depending on the engine and task.
Do not infer a production observation contract from one episode.

### Structured Policies

Generic external adapters pass the entire wrapped mapping to the model. A
structured adapter should explicitly:

- select policy-visible fields and exclude privileged simulator state;
- validate required keys, shapes, dtype, batch size, color channel order, and
  language instruction;
- move/cast tensors intentionally;
- record preprocessing and modality metadata;
- reject missing inputs instead of replacing them with silent zeros.

NyssaBench does not automatically inject `TaskSpec.description` or
`goal.instruction` into OpenVLA. A VLA adapter must route and record the exact
instruction itself.

### Flat Baselines

Repo-local simple BC and flat RoboMimic recursively collect numeric values from
`observation["raw"]`, sort mapping keys, flatten, then truncate or zero-pad to a
fixed feature dimension. Top-level simulator-state keys named `env_states`,
`states`, or `state` are removed from dictionary observations before these
baselines flatten them.

This representation is deterministic and useful for pipeline checks, but it
can discard fields beyond `feature_dim`, collapse structure, and include image
pixels without semantic preprocessing. A flat checkpoint is compatible only
with the exact feature order and feature dimension used during training.

RoboMimic export additionally requires at least 95% observation-payload
coverage and rejects effectively constant features when there are at least 32
steps. Action-only demonstrations are not valid policy training data.

## Action Contracts

### Generic External Models

`call_model()` tries adapter-specific method names and unwraps dictionaries under
`action`, `actions`, `pred_action`, or `pred_actions`. It detaches tensors and
moves them to CPU/NumPy where possible. It does not generally validate action
shape, finiteness, units, bounds, or control mode.

Production adapters should fail before `engine.step()` if the output does not
match the live action contract. Do not rely on a simulator error or implicit
clipping as validation.

### Environment-Space Actions

Policies may output actions directly in the live environment units and bounds.
Require exact element count and shape. Reject non-finite and out-of-range
values. Do not silently pad/truncate a production action.

Repo-local linear/KNN BC is deliberately more permissive: it pads or truncates
to the live action size and clips to bounds. That makes it a smoke baseline, not
proof of a strict production adapter.

### Normalized Actions

RoboMimic export maps finite bounded environment actions to `[-1, 1]` and stores
`nyssa-action-minmax-v1` with the original shape, low, and high values. The
policy adapter requires exactly the trained number of output values, clips the
normalized output to `[-1, 1]`, validates the live shape/bounds against the task
manifest when available, and denormalizes to environment units.

Do not treat an environment-space model as normalized or vice versa. Old task
exports before `nyssa-task-robomimic-export-v3` used raw actions and must be
re-exported and retrained.

## Action Chunks And Receding Execution

A policy that returns one action uses the default:

```text
policy_action_horizon = 1
policy_execution_horizon = 1
```

For an action-chunk model, set both CLI contracts:

```bash
uv run nyssa run \
  --suite <suite> \
  --engine maniskill \
  --policy path/to/chunk_policy.py \
  --policy-action-horizon 16 \
  --policy-execution-horizon 4 \
  --episodes 1 \
  --out runs/chunk_policy_smoke \
  --no-replay
```

The runner recognizes an action sequence when the returned array has at least
two dimensions or a list/tuple contains action-like elements. It executes:

```text
min(returned sequence length, policy action horizon, execution horizon)
```

actions from the chunk. The first action executes immediately; remaining
actions are cached and the policy is not called while the cache is non-empty.
After the committed prefix, the policy observes the new state and predicts a
new chunk. NyssaBench records chunk count, cached-action count/rate, declared
horizons, and `receding_horizon: true` when the policy action horizon exceeds
one.

If `policy_action_horizon` remains one, a 2-D output is passed as one action
rather than split. Configure the contract correctly. Verifier rejection or
recovery may clear/supersede cached policy actions; stressors apply to each
executed action, not to the chunk as one object.

Test call count, action order, cache clearing, early episode termination, and
shape validation at every chunk index.

## Task-Routed Checkpoints

Multi-task suites often have different action dimensions or controllers. Use
one checkpoint per canonical task key.

### Repo-Local BC

`task_bc_policy` reads JSON checkpoints from `NYSSA_TASK_BC_DIR` (default
`checkpoints/bc_by_task`) using:

```text
<task_key>.json
```

The built-in ManiSkill joint-control task IDs route to the corresponding base
keys (`maniskill_pick_cube_joint` to `maniskill_pick_cube`, and similarly for
push/stack). Other IDs remove a trailing `_joint`.

Missing checkpoints fail by default. `NYSSA_TASK_BC_MISSING=zero` executes a
bounded zero action for a missing task, but this is only a diagnostic fallback
and cannot support a learned-policy result.

### Task RoboMimic

`task_robomimic` reads `NYSSA_TASK_ROBOMIMIC_DIR` and discovers, in order:

1. a direct `<task_key>.pth` file;
2. checkpoint output referenced by `task_robomimic_manifest.json` and its
   generated task config;
3. recursive task-named `.pth` candidates under the export tree.

For multiple `model_epoch_*.pth` candidates, it chooses the greatest epoch,
then modification time and path as tie breakers. Avoid duplicate same-epoch
files because copying a tree can change which file is selected. Keep the
generated manifest beside the export. It records task config paths, action
transforms, observation quality, training episode counts, and training episode
seeds.

The task-routed policy fails if used before `reset(task=...)`, if no task
checkpoint is discoverable, if the manifest is malformed, or if the live action
contract differs from training.

Direct `.pth` files can load without a task manifest, but then the adapter has
no recorded training bounds or training seeds to compare. Such a run lacks
automatic controller compatibility and leakage evidence; keep or reconstruct
the manifest before treating it as a held-out result.

## Training And Evaluation Seed Isolation

Training and evaluation episodes must be disjoint. Compare stored episode seeds,
not only CLI run-seed labels. The actual episode seed formula is shown in the
lifecycle section.

When present, `task_robomimic_manifest.json` records training episode seeds. During
`task_robomimic.reset`, evaluation fails when the current episode seed appears
in that task's training set. The only override is:

```bash
NYSSA_ALLOW_TRAINING_SEED_EVAL=1
```

Use it only to diagnose training/inference plumbing. Label the run as
training-seed evaluation; it is not held-out or generalization evidence.

Repo-local JSON BC checkpoints do not currently carry an automatic training-seed
guard. Keep the source dataset manifest, record its seeds in policy metadata,
and choose held-out run seeds manually. Missing automatic enforcement does not
make overlap valid.

Training randomness and rollout randomness are separate. Record the framework
training seed, data-selection seed, checkpoint hash, and held-out evaluation
run seeds.

## Required Failure Behavior

Fail early and specifically for:

- unknown policy names or missing `create_policy`/`PolicyAdapter` entry points;
- missing environment model variables;
- missing checkpoint files or unsupported checkpoint formats;
- missing task-routed checkpoints or use before `reset(task=...)`;
- training/evaluation seed overlap;
- absent structured observation keys or degenerate flat observations;
- action count, shape, bounds, units, controller, or normalization mismatch;
- non-finite model outputs;
- unsupported action-chunk shape or horizon.

Do not catch these errors and substitute random, zero, or scripted actions in a
reported learned-policy run. Diagnostic fallbacks must be explicit in metadata
and result interpretation.

## Focused Adapter Tests

A production adapter PR should cover:

1. direct-file loading through both `create_policy()` and `PolicyAdapter`;
2. environment loading for a no-argument class and direct action callable;
3. reset order, exact task object, episode seed, recurrent/cache clearing, and
   close behavior;
4. metadata after close, including checkpoint identity and action contract;
5. structured observation key/shape/dtype validation;
6. deterministic flat feature order, truncation/padding, and privileged-state
   exclusion where flat features are intentional;
7. exact action shape, finite values, bounds, normalization, and controller
   mismatch failures;
8. action-chunk call count, committed prefix, cached actions, replanning, and
   verifier/recovery interruption;
9. task alias routing, manifest discovery, latest-epoch selection, missing-task
   behavior, and malformed manifests;
10. training-seed rejection and the explicitly labeled diagnostic override;
11. one fake-engine contract test and a real simulator smoke test on every
    supported observation/control mode.

Use tiny fake models/checkpoints in tests. Do not add downloaded weights or
training datasets to the repository.

## Pull Request Checklist

- [ ] Lifecycle methods clear task/episode state and seed every RNG.
- [ ] Metadata identifies the exact model, checkpoint, preprocessing, and action space.
- [ ] Observation visibility and structured/flat preprocessing are documented and tested.
- [ ] Action shape, units, bounds, normalization, and controller are strict.
- [ ] Chunk horizons and cache/replanning semantics are tested when applicable.
- [ ] Task routing and manifest discovery fail clearly for missing/incompatible tasks.
- [ ] Training and held-out evaluation seeds are demonstrably disjoint.
- [ ] Diagnostic overrides/fallbacks are excluded from result claims.
- [ ] Focused, full-suite, Ruff, and required simulator checks pass.
- [ ] No model weights, datasets, videos, or result archives are committed.
