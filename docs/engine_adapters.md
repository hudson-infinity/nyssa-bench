# Engine Adapters

Adapters isolate simulator APIs from NyssaBench task, policy, stressor, failure,
and artifact contracts. For robot integration, they own the executable mapping
from `TaskSpec.robot` to a simulator asset and controller.

See [Adding New Robots](adding_new_robots.md) for the complete compatibility,
state, provenance, and validation workflow.

## Shared Interface

Adapters implement `NyssaEngine`:

```python
class NyssaEngine:
    def load_task(self, task_spec): ...
    def reset(self, seed=None) -> tuple[dict, dict]: ...
    def step(self, action) -> tuple[dict, float, bool, bool, dict]: ...
    def render(self): ...
    def get_state(self) -> dict: ...
    def close(self): ...

    # Optional extension points with explicit unsupported/no-event defaults.
    def apply_stressor(self, stressor_id, parameters) -> dict: ...
    def drain_failure_events(self) -> list: ...
```

Reset and step follow Gymnasium shapes where possible. Observations returned to
policies should use `wrap_observation(env, observation)` so the result contains
`raw` simulator data and a live serializable action-space contract.

## Execution Lifecycle

`PolicyRunner.evaluate()` creates one engine instance and uses this order:

```text
construct engine
  for each task in suite order:
    load_task(task)
    for each episode index:
      policy/expert reset with namespaced episode seed
      stressor before_reset
      engine reset(seed)
      optional policy initial-state restore
      stressor after_reset
      repeat:
        policy/verifier/recovery action
        stressor action transform and before_step
        engine step(action)
        engine/stressor failure events and after_step
        observation stressor transform
      until terminated, truncated, or runner max steps
close engine in finally
```

`load_task` must fully replace task-specific environment state. `reset` must
fully replace episode state. Do not retain task assets, counters, randomization,
failure queues, renderer state, or controller targets from the previous task or
episode unless they are immutable and explicitly shared.

NyssaBench episode seeds use:

```text
episode_seed = run_seed * 1_000_000 + episode_index
```

The same episode seed is paired across tasks within a run. Pass it to every
simulator/task RNG involved in reset. Record and seed additional RNGs used by an
adapter; do not combine with wall-clock time or process-global randomness.

## Method Contracts

### `load_task(task_spec)`

This method is called once per suite task before any episode reset. It must:

- reject a task whose engine, environment/factory, robot, controller, sensors,
  or success contract is unsupported;
- resolve an explicit `success.engine_env_ids.<engine>` or supported
  `success.engine_factory.<engine>` mapping;
- create the environment and effective observation/action/rendering modes;
- set `max_steps` from the executable task horizon;
- inspect live robot, action-space, sensor, success, state, and rendering
  capabilities;
- capture immutable baselines needed to restore model/material parameters on
  episode reset;
- release/replace any environment previously owned by the adapter.

Environment construction errors should preserve the original exception as the
cause and add task/engine/mapping context. Never silently select a placeholder
task, fallback robot, or unrelated environment.

### `reset(seed)`

Return exactly:

```python
(wrapped_observation, info)
```

The method must:

- fail if no task is loaded;
- restore episode-mutated simulator/model baselines before reset;
- pass the supplied seed through the simulator's supported deterministic reset
  path;
- clear returns, elapsed steps, contacts, queued failure events, controllers,
  and adapter-side episode state;
- return a dictionary observation and a plain dictionary info payload;
- include a live action-space contract when the environment exposes one.

Calling `reset` twice with the same task/seed/config should produce equivalent
initial state and observation within the simulator's documented determinism.
Test this directly; accepting a seed argument without using it is invalid.

### `step(action)`

Return exactly five values:

```python
(wrapped_observation, reward, terminated, truncated, info)
```

Invariants:

- validate/coerce action according to the documented live action contract;
- execute one adapter transition (including documented action repeat only);
- return a finite scalar reward and Python booleans for termination flags;
- preserve environment `info` and add normalized success/failure/metric fields
  without overwriting stronger native evidence incorrectly;
- update adapter counters deterministically;
- wrap the next observation in the same outer contract as reset;
- emit simulator failure-event drafts under `info["failure_events"]` or the
  component queue when temporal evidence exists.

`terminated` means the underlying task/MDP reached a terminal state. `truncated`
means an external limit ended an otherwise non-terminal trajectory, commonly a
time limit. Do not mark every unsuccessful termination as truncation or every
time limit as task termination.

The runner stops on either flag. If the environment never sets one, the runner
still stops at its configured maximum steps; `FailureMapper` can classify that
as timeout. An adapter should preserve native `TimeLimit.truncated` evidence
where supplied.

### Success Extraction

`info["success"]` is the normalized per-transition signal consumed by the
runner. It must be false until the task's executable predicate is satisfied.

- ManiSkill checks configured `success_info_keys`, then `success`,
  `is_success`, and `success_once`.
- MuJoCo checks configured/default info keys, then supports final reward,
  episode return, or survival-step thresholds.
- RoboCasa checks `success` then `task_success`.
- Genesis factory environments must provide a meaningful success value; the
  adapter otherwise defaults false.

Test positive and negative cases around exact thresholds. A YAML predicate that
the adapter never computes is not success extraction. Do not infer success only
from `terminated` unless that is the documented task definition.

### `render()`

Return one frame compatible with replay normalization, normally an RGB/RGBA
NumPy array or a mapping/batch containing one. Rendering must:

- use a stable offscreen camera for headless runs;
- be callable immediately after reset and after every step;
- avoid advancing physics or mutating task/controller state;
- return consistent resolution/channel order/dtype/range;
- raise a useful setup error when rendering is requested but unavailable.

The runner attempts an initial frame and one frame per transition. Public
ManiSkill/MuJoCo runs request replay by default and fail if no MP4 can be
written. A `--no-replay` smoke run is not public replay evidence.

### `get_state()`

Return a serializable privileged snapshot for diagnostics, expert/recovery
planning, or future branch evaluation. At minimum document exactly what is and
is not included. A dictionary containing only time is not a full state snapshot.

State capture must not mutate the simulator and must not leak into policy input.
When the adapter claims restoration, provide `set_state(state)` even though it
is not yet an abstract base method. Restore robot/object/articulation/controller
and RNG state, then return a refreshed wrapped observation. Test round-trip and
next-transition equivalence.

### `close()`

Release simulator, renderer, native, subprocess, and file resources. `close`
is called in a `finally` block and should be safe after partial construction and
safe to call more than once. It must not delete user result/checkpoint/dataset
artifacts.

### `apply_stressor(stressor_id, parameters)`

The base implementation returns explicit `unsupported` evidence. An adapter
implementation must apply the requested backend mutation, verify the effective
value, and return an evidence mapping with `status: applied`, backend, affected
count, requested/resolved parameter, and useful before/after ranges.

Restore physical baselines before each episode. Do not return `applied` for a
declaration that was ignored, approximated without disclosure, or unsupported
on the selected CPU/GPU backend.

### `drain_failure_events()`

Return and clear queued `FailureEventDraft` objects or draft mappings emitted by
the engine outside the immediate step info. An event must carry temporal role,
category/subtype, confidence, evidence visibility/source, stressor context, and
recovery eligibility as applicable. Draining twice without new events should
return an empty list.

## Task Mapping And Randomization

Stable public engines require explicit task mappings. ManiSkill and MuJoCo use
`success.engine_env_ids`; Genesis and RoboCasa may use trusted factories under
their documented adapter contracts. Legacy `<engine>_env_id` keys may still be
read but do not satisfy the public mapping gate for new tasks.

Task mapping tests should assert the exact effective environment ID/factory,
robot, controller, observation mode, rendering mode, horizon, and success
predicate. Simulator version fallback must be recorded and shown semantically
equivalent before results are compared.

`randomization.seed` is the common executable baseline. Typed stressors execute
through the lifecycle around reset/step/action/observation. Legacy declarations
such as lighting or friction ranges are unsupported until mapped to typed,
backend-confirmed stressors. Reports must distinguish requested, applied,
skipped, and unsupported states.

See [Stressor Protocol](stressor_protocol.md) for composition, severity, state
restore, and robustness-sweep requirements.

## Headless Rendering And Replay

MuJoCo defaults to EGL on headless non-Windows hosts when `render_mode` is
`rgb_array` and no renderer/display override is present. Respect explicit
`MUJOCO_GL`/`PYOPENGL_PLATFORM` settings. Test EGL or OSMesa according to the
target deployment environment.

ManiSkill/SAPIEN requires a working Vulkan device/ICD for GPU rendering. An
installed Python package without accessible Vulkan/NVIDIA libraries is not a
render-capable backend. CPU simulation does not imply policy/replay cameras work.

For public replay evidence verify:

- initial and per-step frames are nonblank and correctly oriented;
- MP4 files exist for every episode and paths stay inside the run directory;
- replay manifests match episode identities and media paths;
- camera selection and frame rate are recorded;
- renderer errors are not swallowed into empty videos;
- assembled result packs are revalidated after copy/archive/pruning.

Use [Installation](installation.md) for system rendering packages and
[Validation Protocol](validation_protocol.md) for public evidence gates.

## Robot Mapping By Engine

| Engine | Executable robot selection | Controller/observation selection | Current state capability |
| --- | --- | --- | --- |
| ManiSkill | `success.robot_uids` or environment default | `success.control_mode`, `success.obs_mode`, render/backend/device fields | Capture and structured/flat restore |
| MuJoCo | Robot/model owned by `success.engine_env_ids.mujoco` | Environment ID; Nyssa passes render mode | Time-only capture, no adapter restore |
| RoboCasa | `task.robot` passed unchanged to robosuite, or factory-owned | Upstream/factory contract | Experimental capture-only |
| Genesis | Factory receives complete task spec | Factory contract | Experimental capture-only |

`configs/robots/*.yaml` is currently declarative. No engine adapter reads it
automatically. Adapter code and task `success` fields remain the executable
source.

## `load_task` Robot Responsibilities

Before an episode, an adapter should:

1. resolve a concrete task environment/factory;
2. map and validate the canonical robot ID;
3. construct or inspect the simulator asset;
4. select and verify observation/control modes;
5. set the executable maximum horizon;
6. inspect live action shape, bounds, dtype, and controller semantics;
7. verify required sensors, links, success signals, and state APIs;
8. capture immutable baselines needed by stressors/reset;
9. fail with task/robot/engine/mode context if any requirement is unsupported.

Do not silently choose an environment-default robot when the task requires a
specific embodiment. For ManiSkill, pass `robot_uids` explicitly for a new robot
mapping and inspect the live agent after creation.

## Observation Contract

The wrapped observation has this outer contract:

```python
{
    "raw": simulator_observation,
    "action_space": {
        "type": "box",
        "shape": [...],
        "low": [...],
        "high": [...],
        "dtype": "...",
    },
}
```

Adapters should preserve simulator structure rather than flattening it. Policies
own policy-specific preprocessing. The adapter/robot mapping must document:

- proprioceptive joint names, order, units, dtype, and batch shape;
- end-effector/TCP frame and orientation convention;
- gripper state convention;
- policy camera names, calibration, layout/range, depth units, and timing;
- which fields are policy-observable versus privileged.

`render()` is replay evidence and may use a different camera from the policy
observation. Test both independently.

## Action Contract

`action_space_spec()` serializes live Box or discrete spaces. For Box spaces,
shape, low, high, and dtype are required for strict learned-policy integration.

The current ManiSkill and MuJoCo adapters expand a scalar to the live Box shape
and clip that scalar-derived array. They otherwise pass array actions through.
RoboCasa converts lists/tuples to NumPy and can expand a scalar using
`env.action_dim`. These conveniences are not robot compatibility validation.

An adapter adding a robot/controller should reject:

- incorrect number of action values or array rank;
- non-finite values;
- mismatched controller representation or units;
- incompatible joint order;
- out-of-range environment-space actions unless documented clipping is part of
  the controller contract;
- normalized actions without an explicit denormalization manifest.

Learned policy action transforms should validate against the live action space,
not a hard-coded Panda or simulator default.

## Success And Failure Signals

Robot/task mappings must ensure success and failure signals refer to real loaded
objects, joints, links, contacts, and sensors. ManiSkill success comes from
configured/default environment info keys. MuJoCo supports info keys and
reward/return/survival thresholds. Factory adapters must emit a meaningful
`info["success"]`.

Engine-native temporal failure evidence can be returned in
`info["failure_events"]` or queued through `drain_failure_events()`. Keep
privileged simulator evidence separate from policy-visible observations.

## State Capture And Restore

`get_state()` does not imply the adapter can restore it. Robot integrations that
claim state-aligned replay or counterfactual recovery also need a tested
`set_state` path and refreshed observation.

ManiSkill currently unwraps common `raw`, `env_states`, `states`, or `state`
containers, sends dictionaries to `set_state_dict` where available, otherwise
uses `set_state`, and refreshes observations through `get_obs`/`_get_obs`.

MuJoCo currently returns only simulator time and has no `MuJoCoEngine.set_state`.
RoboCasa and Genesis delegate capture where available but have no adapter restore.
Mark these paths capture-only until full robot/object/controller/RNG round-trip
tests exist.

## Compatibility Errors

Raise before the first evaluated action and include:

- task ID;
- canonical robot ID;
- engine and simulator environment/asset;
- requested observation/control mode;
- observed versus expected action/sensor/state contract;
- supported alternatives.

Existing missing-mapping errors identify `success.engine_env_ids.<engine>` or a
required factory. New robot paths should add equally specific unsupported-robot,
controller, sensor, and state errors rather than relying on opaque upstream
exceptions.

## Robot Provenance

The adapter contributes selected engine, task mappings, live observations/action
spaces, package versions, and failure/stressor evidence to run artifacts. The
current result layout does not embed a resolved `configs/robots/*.yaml` manifest
or environment override map automatically.

Keep effective robot UID/asset, controller, joint order, camera set, action
contract, robot config hash, and overrides in task/policy/run metadata or
experiment notes until a versioned robot-provenance artifact exists. The task
source path plus `git_info.json` must resolve the canonical `task.robot` value.

## Robot-Facing Test Matrix

Every supported robot/controller/observation mapping needs:

- a fake-environment test for constructor kwargs and explicit incompatibility
  errors;
- live reset observation key/shape/dtype and action-space assertions;
- joint-name/order, bounds, TCP, gripper, camera, and success/failure tests;
- deterministic seeded reset tests;
- state capture/restore and next-transition equivalence tests where claimed;
- rendering/replay tests for each camera path;
- stressor tests against actual robot/simulator state where supported;
- an opt-in real simulator smoke using a compatible policy.

Run:

```bash
uv run pytest -q tests/test_core_flow.py
uv run pytest -q
uv run ruff check .
uv run python scripts/validate_configs.py
uv run python scripts/validate_backend.py maniskill --episodes 1
uv run python scripts/validate_backend.py mujoco --episodes 1
```

Run only backends affected by a mapping, but do not claim support for a backend
that was not executed on compatible infrastructure.

## Minimal Adapter Example

This minimal Gymnasium adapter is intentionally contract-only: it supports a
finite Box action space and native success info, but does not yet implement full
state capture/restore, rendering validation, stressors, temporal failure
evidence, or public-backend promotion.

```python
from __future__ import annotations

from typing import Any

import numpy as np

from nyssa_bench.core.task import TaskSpec
from nyssa_bench.engines.base import NyssaEngine
from nyssa_bench.engines.spaces import wrap_observation


class ExampleEngine(NyssaEngine):
    def __init__(self) -> None:
        self.env: Any | None = None
        self.task_spec: TaskSpec | None = None
        self.max_steps = 1000

    def load_task(self, task_spec: TaskSpec) -> None:
        if task_spec.engine != "example":
            raise RuntimeError(
                f"Task '{task_spec.task_id}' declares engine "
                f"'{task_spec.engine}', expected 'example'"
            )
        mapping = task_spec.success.get("engine_env_ids", {})
        env_id = mapping.get("example") if isinstance(mapping, dict) else None
        if not env_id:
            raise RuntimeError(
                f"Task '{task_spec.task_id}' is missing "
                "success.engine_env_ids.example"
            )

        self.close()
        try:
            import gymnasium as gym
        except ImportError as exc:
            raise RuntimeError(
                "Install the Example simulator extra before using this engine"
            ) from exc

        self.task_spec = task_spec
        self.max_steps = int(task_spec.success.get("max_steps", 1000))
        render_mode = task_spec.success.get("render_mode", "rgb_array")
        self.env = gym.make(
            str(env_id),
            render_mode=render_mode,
            max_episode_steps=self.max_steps,
        )
        self._validate_live_contract()

    def reset(
        self, seed: int | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        env = self._require_env()
        observation, info = env.reset(seed=seed)
        return wrap_observation(env, observation), dict(info or {})

    def step(
        self, action: Any
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        env = self._require_env()
        action_array = self._validate_action(action)
        observation, reward, terminated, truncated, info = env.step(action_array)
        reward_value = float(reward)
        if not np.isfinite(reward_value):
            raise RuntimeError("Example engine returned a non-finite reward")
        info = dict(info or {})
        info["success"] = bool(info.get("success", False))
        return (
            wrap_observation(env, observation),
            reward_value,
            bool(terminated),
            bool(truncated),
            info,
        )

    def render(self) -> Any:
        return self._require_env().render()

    def get_state(self) -> dict[str, Any]:
        # Capture-only placeholder. Do not claim restorable state from this.
        return {}

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None

    def _require_env(self) -> Any:
        if self.env is None:
            raise RuntimeError("No Example environment loaded; call load_task first")
        return self.env

    def _validate_live_contract(self) -> None:
        action_space = getattr(self._require_env(), "action_space", None)
        if (
            action_space is None
            or not getattr(action_space, "shape", None)
            or not hasattr(action_space, "low")
            or not hasattr(action_space, "high")
        ):
            raise RuntimeError("Example engine requires a finite Box action space")
        if not np.all(np.isfinite(action_space.low)) or not np.all(
            np.isfinite(action_space.high)
        ):
            raise RuntimeError("Example engine action bounds must be finite")

    def _validate_action(self, action: Any) -> np.ndarray:
        action_space = self._require_env().action_space
        value = np.asarray(action, dtype=action_space.dtype)
        if value.shape != action_space.shape:
            raise ValueError(
                f"Action shape {value.shape} does not match {action_space.shape}"
            )
        if not np.all(np.isfinite(value)):
            raise ValueError("Action contains non-finite values")
        if np.any(value < action_space.low) or np.any(value > action_space.high):
            raise ValueError("Action is outside the live environment bounds")
        return value
```

Add success extraction, metrics, failure events, stressors, state support, and
renderer diagnostics as explicit tested behavior rather than generic fallbacks.

## Registration

For a repository-integrated engine, import the class in
`nyssa_bench/core/registry.py`, add it to `ENGINE_REGISTRY`, and start it as
`experimental_contract_only` in `ENGINE_SUPPORT_TIER`. Add its dependency extra,
package version key, experiment/task configs, docs, and tests in the same PR.

An in-process external plugin can register without editing core:

```python
from nyssa_bench.plugins import register_plugin

from my_project.example_engine import ExampleEngine


class ExamplePlugin:
    name = "example_engine_plugin"

    def register(self, registry) -> None:
        registry.engines["example"] = ExampleEngine


register_plugin(ExamplePlugin())
```

The plugin module must be imported before `make_engine("example")` or runner
construction. NyssaBench does not currently auto-discover engine entry points
for the CLI, so a repo integration or explicit application bootstrap is needed
for plain `nyssa run --engine example`.

## Promotion To Supported Backend

An adapter remains experimental contract-only until all promotion evidence in
[Experimental Backends](experimental_backends.md) is complete. Promotion also
requires updating `ENGINE_SUPPORT_TIER`, public-claim engine/package-version
validation, installation extras/docs, and simulator-backed CI. Do not change
the support label based only on import success or a one-episode random smoke.
