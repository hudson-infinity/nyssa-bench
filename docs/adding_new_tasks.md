# Adding New Tasks

Task specifications are benchmark contracts. They determine which simulator is
executed, what counts as success, which failures can be diagnosed, which shifts
were actually applied, and whether two result packs are comparable. A YAML file
that merely loads is not necessarily a valid executable task.

This guide describes the current `TaskSpec` and `Suite` implementation. Follow
[CONTRIBUTING.md](../CONTRIBUTING.md) for the general branch, test, artifact, and
pull-request workflow.

## File Placement And Identity

Put built-in task files under a domain directory:

```text
nyssa_bench/tasks/<domain>/<task_id>.yaml
```

`TaskSpec.load()` also searches `configs/tasks/` for downstream or local task
files. Built-in contributions should use `nyssa_bench/tasks/` so they are
packaged with NyssaBench.

Use a globally unique, lowercase snake-case task ID. Keep the filename stem and
the `task_id` field identical. The current resolver searches by filename stem,
so duplicate stems in different domains are ambiguous even if the YAML IDs
differ.

For a benchmark-relevant semantic change, create a new task ID or explicit
version instead of silently changing the meaning of an existing task. Changes
to success, horizon, embodiment, control mode, assets, or evaluation splits can
invalidate comparisons with old result packs.

## Complete Annotated Task

This example describes a ManiSkill sensor-shift task. YAML comments explain
which fields are executable and which are declarative contracts.

```yaml
# Required. Globally unique and equal to this file's stem.
task_id: maniskill_pick_cube_sensor_shift_v0

# Required. The default engine expected for this task and the engine used for
# mismatch warnings. A run still selects its adapter with --engine.
engine: maniskill

# Required. Stable embodiment identifier used for provenance. RoboCasa consumes
# this value directly; ManiSkill uses success.robot_uids when an override is needed.
robot: panda

# Required. Stable scene or scenario identifier for provenance and analysis.
scene: maniskill_pick_cube

# Required. Human-readable task intent, not an automatically injected policy prompt.
description: Pick the cube and lift it while observing controlled sensor noise.

# Optional structured object inventory. Keep names stable because success, goal,
# split, and failure evidence may refer to them.
objects:
  - name: target_cube
    type: cube
    randomized_pose: true

# Optional declarative observation contract. The adapter still determines the
# live payload. success.obs_mode below is what ManiSkill receives at gym.make().
observation:
  mode: state_dict
  modalities: [proprioception, object_state]
  policy_observable: [agent, extra]
  privileged: [simulator_state]

# Optional declarative action contract. The live Gymnasium action space remains
# authoritative and is recorded with each wrapped observation.
action:
  type: box
  mode: pd_ee_delta_pose
  shape: [7]
  range: [-1.0, 1.0]

# Optional semantic goal and language contract. NyssaBench records this mapping,
# but a policy adapter must explicitly place the instruction in its input.
goal:
  type: object_lifted
  object: target_cube
  instruction: Pick up the cube.

# Optional expert/planner provenance. This does not register or enable a provider.
experts:
  recommended_provider: maniskill-scripted
  capabilities: [act, score_action, recover]

# Optional split lineage used by comparison and validity studies.
ood_splits:
  object_pose:
    train: nominal_workspace_v1
    test: held_out_pose_bins_v1

success:
  # Required for a real ManiSkill run. Keys are engine registry names; values
  # are real Gymnasium/SAPIEN environment IDs.
  engine_env_ids:
    maniskill: PickCube-v1

  # Environment info keys accepted as task success. At least one must be emitted
  # and tested; declaring a key does not make the environment produce it.
  success_info_keys: [success, is_success, success_once]

  # ManiSkill gym.make() settings. Keep these aligned with the declarative
  # observation/action contracts above.
  obs_mode: state_dict
  control_mode: pd_ee_delta_pose
  robot_uids: panda
  render_mode: rgb_array

  # Executable rollout horizon used by the engine and PolicyRunner.
  max_steps: 80

  # Declarative predicate for audit/provenance. The adapter still needs a tested
  # environment info signal that evaluates this predicate.
  object_lifted: target_cube

randomization:
  # Executable common randomization.
  seed: true

  # Typed stressors execute in list order. A task-level stressor is applied on
  # every run of this task, so reserve it for a task whose identity includes the shift.
  stressors:
    - format: nyssa-stressor-spec-v1
      stressor_id: observation_gaussian_noise
      severity: 0.25
      parameters:
        max_std: 0.1

# Optional metric requests. Use implemented metric IDs and test any new metric.
metrics:
  - success_rate
  - completion_time
  - collision_count
  - grasp_success_rate

# Optional but required for diagnosable failed public results. Include labels the
# environment or FailureMapper can actually support for this task.
failure_labels:
  - bad_grasp
  - object_slip
  - missed_target
  - timeout
  - unknown_failure
```

The example parses today, but it is not scientifically validated until the
mapping, live action/observation spaces, success signal, stressor effect, and
failure labels have simulator-backed tests.

## Field Reference

`TaskSpec.from_dict()` currently requires five non-empty top-level fields:

| Field | Required | Meaning |
| --- | --- | --- |
| `task_id` | Yes | Stable task identity. Keep it equal to the filename stem. |
| `engine` | Yes | Expected engine registry name and mismatch-warning contract. |
| `robot` | Yes | Stable embodiment identifier. Adapter consumption varies by engine. |
| `scene` | Yes | Stable scene/scenario identifier for provenance. |
| `description` | Yes | Human-readable intent; not automatically a language-policy input. |

All remaining top-level fields are optional to the parser:

| Field | Type | Current role |
| --- | --- | --- |
| `objects` | list of mappings | Object inventory and stable names used by predicates/evidence. |
| `success` | mapping | Engine mapping, environment options, horizon, and success semantics. |
| `randomization` | mapping | Seed declarations and executable stressor specs. |
| `observation` | mapping | Declared modalities, modes, and visibility/provenance contract. |
| `action` | mapping | Declared action representation, mode, shape, bounds, or horizon. |
| `goal` | mapping | Structured goal and optional language instruction contract. |
| `experts` | mapping | Recommended expert/planner identity and capability metadata. |
| `ood_splits` | mapping | Train/test asset, pose, scene, language, or dynamics split lineage. |
| `metrics` | list | Requested implemented metric identifiers. |
| `failure_labels` | list | Shared or task-specific labels that can be emitted or mapped. |

The loader preserves these mappings but does not define strict sub-schemas for
all of them. Treat their documented meaning as a compatibility contract and add
consumer tests. `nyssa validate` verifies parsing and references; it does not
prove the scientific meaning of arbitrary nested keys.

## Engine Mappings And Factories

### `engine_env_ids`

Real ManiSkill and MuJoCo tasks use an explicit mapping under `success`:

```yaml
success:
  engine_env_ids:
    maniskill: PickCube-v1
```

```yaml
success:
  engine_env_ids:
    mujoco: Reacher-v5
```

The mapping key must match the selected engine registry name. The environment
ID must exist in the installed simulator version. Do not use the legacy
`maniskill_env_id` or `mujoco_env_id` fields for new tasks: adapters may still
read them, but the public-claim validator requires `engine_env_ids` or a
supported factory mapping.

MuJoCo attempts compatible lower Gymnasium environment versions for known
Reacher, Pusher, and InvertedPendulum IDs. Record the package version and verify
which environment was actually created; a fallback is not evidence that two
versions have identical task semantics.

### `engine_factory`

Genesis and RoboCasa can import a task factory:

```yaml
success:
  engine_factory:
    genesis: my_project.tasks.pick_cube:create_env
```

The path must be `module:function`. The function receives the loaded `TaskSpec`
and returns an environment compatible with the selected adapter. The Genesis
factory environment may return Gymnasium-style four- or five-value steps. A
RoboCasa factory must match the robosuite-style API expected by its adapter.

ManiSkill and MuJoCo adapters currently consume `engine_env_ids`, not
`engine_factory`. Adding a factory entry for those engines may satisfy a generic
mapping check but will not make the adapter execute it. Extend and test the
adapter first.

Factory imports execute contributor code. Keep them in trusted source modules,
avoid machine-specific paths, and include unit plus real-backend tests.

## Robots, Modes, And Horizons

The top-level `robot` field identifies the embodiment in manifests and reports.
RoboCasa passes it to robosuite. ManiSkill normally receives its robot from the
environment default; set `success.robot_uids` when the task requires an explicit
robot override. MuJoCo embodiment is defined by the environment ID, so the
top-level value must describe that environment accurately.

For ManiSkill, these `success` fields are passed to `gym.make()`:

- `obs_mode`
- `control_mode`
- `robot_uids`
- `render_mode`
- `sim_backend`
- `render_device`
- `shader_dir`
- `max_steps` as `max_episode_steps`

`control_mode` must match the actions produced by every evaluated policy and
expert. For example, `pd_ee_delta_pose` and `pd_joint_pos` are not interchangeable.
Keep `success.control_mode` and `action.mode` consistent and assert the live
action shape and bounds in an integration test.

`max_steps` is the executable NyssaBench horizon. It configures the adapter and
runner, contributes to timeout failure mapping, and must be large enough for the
task without hiding policy stalls. Fields such as `max_time_seconds` and
`max_collisions` are currently declarative unless the environment/adapter turns
them into termination and success/failure info. Test that conversion rather than
assuming the YAML enforces it.

## Success Predicates

A task needs an executable, observed success signal. The public-claim validator
recognizes keys including `success_info_keys`, `success_metric`,
`reward_threshold`, `return_threshold`, `min_success_steps`, `object_lifted`,
`object_inside`, `object_on_top`, and `ee_at_target`. Recognition is not the same
as runtime implementation.

### ManiSkill

The adapter checks configured `success_info_keys`, then `success`, `is_success`,
and `success_once` in environment `info`. Use keys the selected environment
actually emits and test tensor/batched truth conversion. A declarative
`object_lifted` field does not independently compute success.

### MuJoCo

The adapter first checks configured/default info keys, then supports:

- `success_metric: final_reward_threshold` with `reward_threshold`;
- `success_metric: episode_return_threshold` with `return_threshold`;
- `success_metric: survival_steps` with `min_success_steps`.

Choose thresholds from the actual environment reward semantics, document the
rationale, and test values immediately below and above the boundary. Do not
tune a threshold against evaluation results without recording that process.

### Factory And Experimental Engines

Genesis and RoboCasa factory environments must emit a meaningful success value
in `info`; their adapters otherwise default to failure. A contract-only
experiment YAML is not an executable success predicate.

Every new task test should demonstrate at least one success and one failure. A
random-policy smoke run that never succeeds proves execution, not predicate
validity.

## Failure Taxonomy And Evidence

Prefer shared labels from `nyssa_bench.metrics.taxonomy.FAILURE_LABELS`:

```text
bad_grasp, object_slip, collision, missed_target, wrong_object,
occlusion_failure, planner_stuck, joint_limit_failure, timeout,
latency_failure, unstable_contact, unknown_failure,
out_of_distribution_layout
```

The environment can emit `failure_label` with `failure_label_source: env`.
Otherwise `FailureMapper` recognizes collision/safety events, wrong-object
events, drop/slip events, grasp failures, joint limits, stalls, latency,
out-of-distribution layouts, early unstable termination, and timeouts.

Only declare labels that can be produced for the task. The mapper converts an
unconfigured diagnostic into `unknown_failure`, and public failed episodes
cannot rely on `unknown_failure`. Add a task-specific label only when the shared
taxonomy is insufficient, document it, and test its environment or mapper
provenance.

New runs also create temporal failure ledgers. If the environment has localized
evidence, emit a versioned failure-event draft through the engine component API
rather than discarding it into one terminal label. See
[Failure Event Protocol](failure_event_protocol.md).

## Randomization And Stressors

`randomization.seed: true` is the common executable baseline for ManiSkill and
MuJoCo. Legacy declarations such as `lighting: true`, `camera_pose: true`, or
`friction_range: [...]` remain descriptive unless an adapter maps them to a
typed stressor. Reports list unsupported declarations; they cannot support a
robustness claim.

Executable stressors live under `randomization.stressors` and run in list order.
Each spec declares a stable ID, severity, parameters, and optional seed. The
current built-ins cover image brightness, observation noise, action noise,
action delay, and friction scaling. Friction scaling is supported by MuJoCo and
ManiSkill CPU PhysX, not ManiSkill GPU simulation.

A task-level stressor applies to every episode of every run containing that
task. Use it when the shift is part of the task identity. For matched severity
sweeps, keep the task clean and pass versioned `--stressor-config` files so the
same task, seeds, and policy are compared. Unsupported task-level stressors fail
by default; do not catch that error and report the shift as active.

See [Stressor Protocol](stressor_protocol.md) for severity, composition,
manifest, restoration, and robustness-AUC rules.

## Observation, Action, Goal, And Language Contracts

These optional mappings are preserved in task and dataset manifests and affect
comparison provenance. They are not universal automatic converters.

### Observation

Declare mode, modalities, expected structure, and which fields are policy
observable versus privileged. Engine adapters wrap live observations as:

```python
{"raw": simulator_observation, "action_space": live_action_space_contract}
```

For ManiSkill, `success.obs_mode` is the executable environment setting. Keep it
aligned with `observation.mode`. Test actual keys, tensor/array shapes, dtypes,
batch dimensions, and camera channel order.

### Action

Declare representation, control mode, shape, bounds, normalization, and action
horizon where relevant. The live simulator action space is authoritative and is
recorded in wrapped observations; task-level `action` fields do not override it.
Tests must show that policy output is accepted without silent reshaping or
clipping unless that behavior is the documented adapter contract.

### Goal And Language

Use `goal` for stable semantic targets such as object, receptacle, pose,
relation, stage sequence, or instruction. `description` is for humans and is not
automatically passed to a policy. NyssaBench currently has no dedicated
top-level language field or universal language injector. A language-conditioned
policy adapter must explicitly consume `goal.instruction` (or another documented
goal key), record the exact instruction, and test missing/paraphrased behavior.

Language perturbations and paraphrases are evaluation conditions. Version them
as task/split/stressor contracts instead of silently rewriting `description`.

### Experts And OOD Splits

`experts` records recommended provider identity and capabilities; it does not
register or enable a provider. The CLI/execution configuration still selects
the expert provider.

Use `ood_splits` to record train/evaluation lineage for assets, poses, scenes,
language, or dynamics. Do not label a run OOD when the task has no auditable
split definition or when training data overlaps the declared test split.

## Register A Suite

Suites are YAML files under `configs/suites/`; no Python registry edit is
required. Keep the filename stem and `suite_id` identical:

```yaml
suite_id: maniskill_sensor_shift_v0
description: ManiSkill manipulation tasks with controlled sensor shifts.
tasks:
  - maniskill_pick_cube_sensor_shift_v0
  - maniskill_push_cube_sensor_shift_v0
```

Task names are resolved by filename stem. Keep entries unique and order them
intentionally because `Suite.load()` preserves list order. A suite must contain
at least one task.

One `PolicyRunner` uses one selected engine for the full suite. Do not mix tasks
that lack equivalent mappings for that engine. Mixed declarations produce
warnings and cannot support a clean public comparison.

List and filter suites from the CLI:

```bash
uv run nyssa list-suites
uv run nyssa list-tasks
uv run nyssa run \
  --suite maniskill_sensor_shift_v0 \
  --tasks maniskill_pick_cube_sensor_shift_v0 \
  --engine maniskill \
  --policy random \
  --episodes 1 \
  --out runs/maniskill_sensor_shift_smoke \
  --no-replay
```

The equivalent API is:

```python
from nyssa_bench import Suite

suite = Suite.load("maniskill_sensor_shift_v0")
focused = suite.filter_tasks(["maniskill_pick_cube_sensor_shift_v0"])
```

Filtering retains the suite ID and raises if any requested task is absent.

## Required Tests

A new task contribution needs tests for more than YAML parsing:

1. **Loading and identity:** the task and suite resolve by ID; file stems, IDs,
   task order, and filtering are correct.
2. **Engine mapping:** the selected adapter resolves the intended environment or
   factory and passes the expected robot, observation, control, rendering, and
   horizon arguments.
3. **Success extraction:** positive and negative environment info/reward/return
   cases produce the intended boolean result, including the exact threshold.
4. **Failure diagnosis:** each declared label is environment-native or covered
   by a tested `FailureMapper`/failure-event path; unknown cases remain honest.
5. **Live contracts:** simulator integration confirms observation structure and
   action shape/bounds for each declared control mode.
6. **Stressor behavior:** every declared executable stressor reports `applied`
   and changes the intended policy/simulator quantity deterministically.

A minimal contract test can start with:

```python
from nyssa_bench import Suite
from nyssa_bench.core.task import TaskSpec
from nyssa_bench.metrics.failure_mapper import FailureMapper


def test_sensor_shift_task_contract():
    task = TaskSpec.load("maniskill_pick_cube_sensor_shift_v0")
    assert task.source_path is not None
    assert task.source_path.stem == task.task_id
    assert task.success["engine_env_ids"]["maniskill"] == "PickCube-v1"
    assert task.success["control_mode"] == "pd_ee_delta_pose"
    assert task.success["max_steps"] == 80

    suite = Suite.load("maniskill_sensor_shift_v0")
    assert [item.task_id for item in suite.tasks] == [
        "maniskill_pick_cube_sensor_shift_v0",
        "maniskill_push_cube_sensor_shift_v0",
    ]
    assert suite.filter_tasks([task.task_id]).tasks == (task,)

    failure = FailureMapper().classify(
        {"object_dropped": True},
        task_spec=task,
        step_count=12,
    )
    assert failure.label == "object_slip"
```

Add a backend-specific success test. For ManiSkill, verify every configured info
key against the adapter's extraction path. For MuJoCo, test values on both sides
of each configured threshold. Prefer an adapter-level fake environment test plus
an opt-in real simulator test.

## Validation And Smoke Runs

Validate the individual task, suite, and all repository configs:

```bash
uv run nyssa validate nyssa_bench/tasks/<domain>/<task_id>.yaml
uv run nyssa validate configs/suites/<suite_id>.yaml
uv run nyssa validate <task_id>
uv run nyssa validate <suite_id>
uv run python scripts/validate_configs.py
```

Run focused and full tests:

```bash
uv run pytest -q tests/test_core_flow.py
uv run pytest -q
uv run ruff check .
```

Then run the task on the real backend. A one-episode random run checks loading,
reset, action compatibility, stepping, termination, and artifact writing:

```bash
uv run nyssa run \
  --suite <suite_id> \
  --tasks <task_id> \
  --engine maniskill \
  --policy random \
  --episodes 1 \
  --out runs/<task_id>_smoke \
  --no-replay
```

Use a known success-capable policy or expert path to validate the positive
success predicate. If rendering, cameras, or public replay evidence changed,
repeat with `--capture-replay` and inspect the MP4 and replay manifest.

`scripts/validate_backend.py` exercises the built-in ManiSkill or MuJoCo
validation suite. It validates a new task only after that task is included in
the selected suite, so a direct filtered run remains required:

```bash
uv run python scripts/validate_backend.py maniskill --episodes 1
uv run python scripts/validate_backend.py mujoco --episodes 1
```

Contract loading, a random smoke run, and an experimental adapter check are not
public benchmark evidence. Public result packs still need the episode/seed,
replay, provenance, diagnosis, and clean-worktree gates in
[Validation Protocol](validation_protocol.md).

## Task Pull Request Checklist

- [ ] Task ID and filename are globally unique and equal.
- [ ] Suite ID and filename are equal; task entries are unique and intentional.
- [ ] Engine mapping/factory resolves on the supported simulator version.
- [ ] Robot, observation, control mode, action contract, and horizon match live behavior.
- [ ] Success has tested positive and negative cases.
- [ ] Every declared failure label has a tested evidence or mapper path.
- [ ] Randomization distinguishes executable stressors from unsupported declarations.
- [ ] Goal, language, expert, and OOD split provenance is explicit where applicable.
- [ ] Config validation, focused tests, full tests, and Ruff pass.
- [ ] Required real simulator and replay smoke tests are reported in the PR.
