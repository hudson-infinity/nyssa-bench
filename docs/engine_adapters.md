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
