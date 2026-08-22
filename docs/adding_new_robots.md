# Adding New Robots

Robot integration is a compatibility contract between a stable NyssaBench
identifier, task specs, simulator assets, controllers, live observations,
actions, state snapshots, policies, and result provenance. Matching a robot name
is not enough.

Use [Adding New Tasks](adding_new_tasks.md) for task-level fields and
[Engine Adapters](engine_adapters.md) for the robot-facing adapter boundary.

## Current Status

`nyssa_bench.specs.robot_spec.RobotSpec` currently contains only
`robot_id`, `description`, `end_effector`, and `supported_engines`.
NyssaBench does not yet load `configs/robots/*.yaml` into a runtime registry.
Those files are declarative compatibility manifests for contributors.

Executable robot selection still comes from task/adapter fields:

- ManiSkill receives `success.robot_uids` when present, otherwise the selected
  environment's default robot;
- RoboCasa passes `task.robot` directly to robosuite;
- MuJoCo embodiment is defined by `success.engine_env_ids.mujoco`;
- Genesis delegates robot/scene construction to `success.engine_factory.genesis`.

Do not claim that editing a robot YAML changes a rollout. The corresponding
task and engine adapter must consume and test the mapping.

## Annotated Panda Manifest

The repository's [Panda config](../configs/robots/panda.yaml) records the
current supported and experimental mappings. The important shape is:

```yaml
# Declarative only until a runtime robot registry consumes it.
robot_id: panda
description: Franka Panda manipulator used by the v0.1 manipulation suites.
end_effector: parallel_gripper

# Engines with a validated NyssaBench robot/task path.
supported_engines:
  - maniskill

# Mappings that exist only as experimental contracts.
experimental_engines:
  - robocasa

simulator_mappings:
  maniskill:
    status: supported
    asset_id: panda
    task_robot_field: panda
    runtime_robot_field: success.robot_uids
    default_robot_uid: panda
    validated_suites:
      - maniskill_smoke_v0
      - maniskill_manipulation_v0
      - maniskill_planner_bc_v0
    controllers:
      pd_ee_delta_pose:
        representation: end_effector_delta_pose_with_gripper
        joint_order: not_applicable_to_arm_command
        bounds_source: live_gymnasium_action_space
      pd_joint_pos:
        representation: absolute_joint_position_targets_with_gripper
        joint_order: simulator_action_space_order
        bounds_source: live_gymnasium_action_space
    state_restore:
      status: supported
      capture_api: env.get_state
      restore_api: env.set_state_dict_or_set_state

  robocasa:
    status: experimental_contract_only
    asset_id: Panda
    canonical_robot_id: panda
    required_runtime_value: Panda
    runtime_robot_field: task.robot_passed_without_translation
    mapping_status: adapter_translation_not_implemented

conventions:
  proprioception:
    joint_names: must_be_recorded_from_live_simulator
    joint_order: simulator_defined_not_inferred_from_vector_length
  end_effector:
    frame: simulator_tcp_frame_must_be_recorded
    position_units: meters
    quaternion_order: simulator_defined_must_be_recorded
  gripper:
    action_index: controller_defined_must_be_tested
    sign_and_range: live_action_space_and_controller_defined
  cameras:
    policy_cameras: selected_by_success.obs_mode
    replay_camera: selected_by_success.render_mode
    intrinsics_extrinsics: must_be_recorded_for_visual_policy_claims
  actions:
    normalization: environment_space_unless_policy_manifest_declares_transform
    shape_and_bounds: live_gymnasium_action_space
```

Values such as `must_be_recorded` are explicit unresolved requirements, not
wildcards that permit any convention.

## Stable Identifiers And Simulator Assets

Use lowercase snake case for `robot_id`. Keep it stable across tasks and reports.
Do not use a package class name, local asset path, or mutable nickname as the
canonical ID.

Each engine mapping should identify:

- the simulator package and tested version;
- asset/robot UID expected by that simulator;
- source and license of URDF/MJCF/mesh assets;
- base type and base frame;
- arm and gripper joint names/order;
- end-effector link/TCP frame;
- available cameras/sensors;
- supported controllers and task suites;
- whether the mapping is supported, experimental, or unsupported.

The canonical ID and simulator asset may differ. Translate explicitly in the
adapter rather than forcing task authors to use engine-specific names. For
example, a canonical `panda` may map to ManiSkill `panda` and upstream RoboCasa
`Panda`.

### Engine-Specific Selection

**ManiSkill:** task `robot: panda` is provenance; `success.robot_uids: panda`
is the executable override passed to `gym.make`. If `robot_uids` is omitted, the
environment default can silently select a robot. Set it for a new robot mapping
and assert the live robot UID after reset.

**MuJoCo:** the current adapter ignores `task.robot` when building the
environment. The Gymnasium environment ID owns the embodiment and model. Use a
robot-specific canonical ID such as `mujoco_reacher`, and test that the loaded
model matches it. The existing Panda manifest must not list MuJoCo support until
a Panda environment mapping is implemented and validated.

**RoboCasa:** the current experimental adapter passes `task.robot` unchanged to
`robosuite.make(robots=...)`. It does not consume the Panda config's canonical
to upstream-name mapping. Use a factory or extend the adapter before claiming
that lowercase `panda` maps to upstream `Panda`.

**Genesis:** a trusted `module:function` factory receives the full `TaskSpec`
and owns robot asset/controller construction. The factory must validate
`task.robot` and expose compatible reset/step/render/state behavior.

## Task And Robot Compatibility

A task using a robot should align all relevant fields:

```yaml
task_id: maniskill_pick_cube_joint
engine: maniskill
robot: panda
scene: maniskill_pick_cube
description: PickCube with absolute Panda joint-position control.
success:
  engine_env_ids:
    maniskill: PickCube-v1
  robot_uids: panda
  obs_mode: state_dict
  control_mode: pd_joint_pos
  max_steps: 100
action:
  mode: pd_joint_pos
  representation: absolute_joint_position_targets_with_gripper
```

Compatibility is the tuple:

```text
robot ID + engine asset + task + controller + observation mode + action space
```

An adapter should reject an incompatible tuple in `load_task` or immediately
after environment creation, before the first policy episode. At minimum verify:

- the task's canonical robot is mapped for the selected engine;
- the environment actually loaded the expected embodiment;
- the controller exists for that robot/task;
- live observation modalities match the task/policy contract;
- live action shape, ordering, dtype, and bounds match controller expectations;
- task success/failure signals refer to assets and links that exist;
- requested state capture/restore and cameras are available.

The current common adapters do not yet implement one centralized robot/task
compatibility gate. Contributors adding a robot must put checks in the affected
adapter and tests rather than relying on the declarative config.

### Expected Errors

Errors should name the task, canonical robot, engine, requested mode, and
supported alternatives. Prefer messages such as:

```text
Task 'pick_cube_xarm' requests robot 'xarm7' on engine 'maniskill', but no
validated robot mapping exists. Supported robots for this adapter: panda.
```

```text
Robot 'panda' controller 'pd_joint_pos' exposes action shape (9,), but the
policy/checkpoint declares (7,). Refusing to reshape the action silently.
```

Existing adapter errors also include:

- missing `success.engine_env_ids.<engine>`;
- missing RoboCasa mapping/factory;
- no environment loaded before reset/step/render;
- unavailable ManiSkill state restore;
- strict RoboMimic live/training action shape or bounds mismatch.

Do not catch compatibility errors and fall back to another robot, controller,
zero action, or default environment in a reported run.

## Controller And Action-Space Contracts

NyssaBench wraps a Gymnasium action space into each policy observation:

```python
{
    "action_space": {
        "type": "box",
        "shape": [9],
        "low": [...],
        "high": [...],
        "dtype": "float32",
    }
}
```

This live contract is authoritative. Robot configs and task `action` mappings
document expected semantics but do not override the environment.

For each controller, document and test:

- whether values are position, velocity, torque, delta pose, or normalized;
- units and reference frame;
- exact element count and tensor/array shape;
- arm joint order by name;
- gripper dimensions, index, sign, and open/closed convention;
- finite lower/upper bounds and whether they are physical or normalized;
- controller frequency, action repeat, latency, and action-chunk assumptions.

### Panda Examples

`pd_ee_delta_pose` is an end-effector delta-pose controller with a gripper
command. It does not expose arm-joint target ordering, but it still requires a
defined Cartesian frame, rotation representation, scaling, and gripper index.

`pd_joint_pos` uses absolute joint-position targets in the simulator action-space
order, including gripper dimensions according to the live controller. Do not
assume that every Panda vector is seven arm joints or that the gripper is always
the last scalar. Query and record the installed simulator/controller contract.

NyssaBench engine adapters expand and clip scalar actions for Box spaces, but do
not generally reshape or validate array actions. Production policies and robot
adapters must reject incompatible arrays before `env.step`.

RoboMimic uses `nyssa-action-minmax-v1` to normalize finite environment bounds
to `[-1, 1]`, then validates live shape/bounds before denormalization when its
task manifest is present. Keep this transform and robot/controller manifest
together with the checkpoint.

## Observation And Frame Conventions

Engine adapters return:

```python
{
    "raw": simulator_observation,
    "action_space": live_action_space_contract,
}
```

Robot integration must specify policy-visible versus privileged fields and avoid
assuming one simulator's names are universal.

### Proprioception

Record joint names and vector order for `qpos`, `qvel`, effort/torque, and
gripper state. State dictionaries may batch values and may include gripper
joints in the same vector. Vector length alone cannot recover joint identity.

Use radians/meters and radians/meters per second according to each joint type,
and record any normalization. Policies trained under a different order or unit
contract are incompatible even when shapes match.

### End Effector

Record TCP/end-effector link name, parent frame, world/base frame, position
units, orientation representation, and quaternion ordering. NyssaBench's
scripted heuristic searches aliases such as `tcp_pose`, `ee_pose`, and
`end_effector_pose`, but alias discovery is not a formal frame conversion.

### Gripper

Record finger joint names, width/position representation, action index, sign,
range, and whether values are absolute, delta, or normalized. Test fully open,
fully closed, and one intermediate command without an object before task use.

### Cameras

`success.obs_mode` selects policy observations; `success.render_mode` selects
replay rendering. They are different contracts. For every policy camera record:

- stable camera/sensor name and attachment frame;
- resolution, channel order, dtype, and numeric range;
- intrinsics and distortion model where applicable;
- extrinsics and coordinate convention;
- depth units, invalid value, clipping range, and alignment to RGB;
- batching and timestamp/synchronization behavior.

Do not infer policy-camera validity from an MP4 replay. A replay camera can look
correct while the policy receives missing, transposed, stale, or differently
normalized images.

## State Capture And Restore

Replay video capture only needs render frames. State-aligned demonstration
replay, counterfactual recovery, and deterministic branching need a much stronger
contract.

A restorable robot snapshot should include:

- robot base pose and all named joint positions/velocities;
- gripper state;
- controller integrators, targets, and queued actions;
- object/articulation state and simulator time;
- task stage and success/failure state;
- relevant simulator, task, policy, and stressor RNG state;
- sensor/camera state needed to reproduce the next observation.

After restore, the adapter must return a fresh observation with the same policy
contract. Test state round-trip and next-transition equivalence under a fixed
action and stochastic seed.

Current support is asymmetric:

| Engine | Capture | Restore | Current limitation |
| --- | --- | --- | --- |
| ManiSkill | `env.get_state()` | `set_state_dict` or `set_state` | Best current branch/replay path; structured and flat states require tests per backend/version. |
| MuJoCo | adapter reports simulator time only | none in `MuJoCoEngine` | Not sufficient for deterministic robot replay or counterfactual branching. |
| RoboCasa | delegates `get_state` when available | none in adapter | Capture-only experimental path. |
| Genesis | delegates `get_state` when available | none in adapter | Factory-defined capture-only experimental path. |

Do not label capture-only metadata as state restoration. Privileged state used
for restore must not be passed into the evaluated policy unless its declared
observation contract includes it.

## Run Provenance

Robot provenance is distributed across current artifacts:

- `run.yaml` records selected engine, task IDs, policy metadata, seed protocol,
  and stressor execution;
- `dataset_manifest.json` records each task source path plus success mapping,
  observation/action/goal contracts, experts, OOD splits, and randomization;
- `episodes.json` records live wrapped observations and action-space contracts;
- `package_versions.json`, `environment.json`, and `git_info.json` identify the
  software/host revision.

The top-level `task.robot` value and `configs/robots/<robot>.yaml` are not
currently embedded as a resolved robot manifest in each result pack. The task
source path and Git commit are therefore required to reconstruct the canonical
robot mapping. Environment-variable overrides such as
`NYSSA_MANISKILL_ROBOT_UIDS` are also not copied into one dedicated robot record.

Until a versioned robot manifest is embedded automatically, report the effective
canonical robot ID, simulator asset/UID, controller, observation mode, action
shape/bounds, camera set, robot-config hash, override variables, and simulator
version in experiment notes or policy/run metadata. Do not call a result fully
self-contained robot provenance when those values are absent.

## Integration Workflow

1. Add or update `configs/robots/<robot_id>.yaml` with canonical ID, license,
   engine mappings, support tier, controllers, conventions, tasks, and state
   capability.
2. Add the simulator asset/factory mapping in the affected engine adapter.
3. Add task specs with matching `robot`, explicit environment/factory mapping,
   observation/control modes, action contract, horizon, and success/failure
   semantics.
4. Implement explicit robot/task/controller compatibility checks and errors.
5. Wrap live observations/action space without exposing privileged state.
6. Implement and test capture/restore if the integration claims replay or
   counterfactual support.
7. Record mapping/provenance fields and package versions.
8. Add fake-environment contract tests and opt-in real simulator tests.
9. Run focused smoke tests for every supported task/controller/observation mode.

## Adapter Test Examples

Test declarative mapping and effective ManiSkill arguments:

```python
from nyssa_bench.core.task import TaskSpec
from nyssa_bench.engines.maniskill_adapter import _maniskill_env_kwargs


def test_panda_joint_task_mapping():
    task = TaskSpec.from_dict(
        {
            "task_id": "panda_joint_test",
            "engine": "maniskill",
            "robot": "panda",
            "scene": "pick_cube",
            "description": "Panda joint-control compatibility fixture.",
            "success": {
                "engine_env_ids": {"maniskill": "PickCube-v1"},
                "robot_uids": "panda",
                "obs_mode": "state_dict",
                "control_mode": "pd_joint_pos",
                "max_steps": 100,
                "success_info_keys": ["success"],
            },
        }
    )

    kwargs = _maniskill_env_kwargs(task)
    assert task.robot == "panda"
    assert kwargs["robot_uids"] == "panda"
    assert kwargs["obs_mode"] == "state_dict"
    assert kwargs["control_mode"] == "pd_joint_pos"
    assert kwargs["max_episode_steps"] == 100
```

Test live Box shape, ordering metadata, and bounds before policy evaluation:

```python
import numpy as np

from nyssa_bench.baselines.features import action_space_contract
from nyssa_bench.engines.spaces import wrap_observation


def test_live_panda_action_contract(fake_panda_env):
    observation = wrap_observation(fake_panda_env, {"agent": {"qpos": [0.0] * 9}})
    contract = action_space_contract(observation)
    action_size = int(np.prod(contract["shape"]))
    assert action_size == len(fake_panda_env.expected_action_order)
    assert len(contract["low"]) == action_size
    assert len(contract["high"]) == action_size
    assert np.all(np.asarray(contract["high"]) > np.asarray(contract["low"]))
    assert fake_panda_env.action_names == fake_panda_env.expected_action_order
    assert len(set(fake_panda_env.action_names)) == action_size
```

For state restore, test both structured and flat state APIs, compare restored
observations, then step the original/restored branches with the same action and
compare next state, reward, termination, success, and failure events.

Also test negative cases: unsupported robot ID, missing asset, unsupported
controller, observation-key mismatch, wrong action shape/bounds, camera absence,
and restore requested on a capture-only adapter.

## Validation Commands

Parse the declarative Panda manifest:

```bash
uv run python -c "from pathlib import Path; import yaml; data=yaml.safe_load(Path('configs/robots/panda.yaml').read_text()); assert data['robot_id']=='panda'"
```

Validate affected tasks/suites and repository configs:

```bash
uv run nyssa validate <task_id-or-path>
uv run nyssa validate <suite_id-or-path>
uv run python scripts/validate_configs.py
uv run pytest -q tests/test_core_flow.py
uv run pytest -q
uv run ruff check .
```

Run each supported backend/controller combination directly:

```bash
uv run nyssa run \
  --suite maniskill_planner_bc_v0 \
  --tasks maniskill_pick_cube_joint \
  --engine maniskill \
  --policy random \
  --episodes 1 \
  --seed 10000 \
  --out runs/panda_joint_robot_smoke \
  --no-replay

uv run python scripts/validate_backend.py maniskill --episodes 1
```

Repeat with a policy compatible with the controller, then with
`--capture-replay` for camera/rendering changes. A random one-episode run proves
pipeline compatibility only; it does not validate task success or policy
quality.

## Robot Integration Checklist

- [ ] Canonical robot ID and simulator asset names are distinct and mapped explicitly.
- [ ] Asset source, license, package version, base, joints, TCP, gripper, and cameras are documented.
- [ ] Supported and experimental engines/tasks/controllers are separated.
- [ ] Task robot, runtime robot selector, observation mode, and controller agree.
- [ ] Joint/action order, units, shape, dtype, bounds, and gripper convention are tested.
- [ ] Camera and proprioception fields are policy-visible only when declared.
- [ ] Unsupported robot/task/controller combinations fail before evaluation.
- [ ] Capture versus restore capability is stated and round-trip tested where claimed.
- [ ] Effective mapping and overrides are recoverable from run provenance/notes.
- [ ] Config, focused, full, Ruff, backend, and replay checks pass.
