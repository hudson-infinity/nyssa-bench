# Task Spec Reference

Task YAML files define the benchmark contract: identity, embodiment, simulator
mapping, policy interfaces, success, randomization, metrics, failure diagnosis,
and evaluation split lineage.

For the annotated end-to-end example, suite registration, required tests, and
smoke commands, use [Adding New Tasks](adding_new_tasks.md).

## Resolution

Built-in tasks live at:

```text
nyssa_bench/tasks/<domain>/<task_id>.yaml
```

`TaskSpec.load()` searches `nyssa_bench/tasks/` and then `configs/tasks/` by
filename stem. Task IDs and stems should be globally unique and equal. Duplicate
stems are ambiguous.

## Required Fields

The loader rejects a task if any of these fields are absent or empty:

| Field | Meaning |
| --- | --- |
| `task_id` | Stable task identity and expected filename stem. |
| `engine` | Expected NyssaBench engine registry name. |
| `robot` | Stable embodiment identifier. |
| `scene` | Stable scene/scenario identifier. |
| `description` | Human-readable task intent. |

The `engine` field does not select the runtime by itself; the runner's
`--engine` argument selects one adapter for the full suite. A mismatch is
reported and weakens claim validity.

## Optional Fields

| Field | Parsed type | Meaning |
| --- | --- | --- |
| `objects` | list | Named task objects and declarative object properties. |
| `success` | mapping | Environment mapping/options, horizon, and success predicates. |
| `randomization` | mapping | Seed declarations and typed executable stressors. |
| `observation` | mapping | Declared modality, mode, shape, and visibility contract. |
| `action` | mapping | Declared representation, control mode, shape, bounds, and horizon. |
| `goal` | mapping | Semantic target and optional language instruction. |
| `experts` | mapping | Recommended expert/planner identity and capabilities. |
| `ood_splits` | mapping | Auditable train/evaluation split lineage. |
| `metrics` | list | Implemented metrics requested for the task. |
| `failure_labels` | list | Environment-native or mapped diagnostic labels. |

The loader preserves nested keys inside these declared fields. Unknown
top-level fields are not retained by `TaskSpec.to_dict()`, so do not create a
new top-level contract without extending the dataclass, serialization,
documentation, and tests.

## Executable `success` Fields

### Mapping

ManiSkill and MuJoCo require explicit environment IDs:

```yaml
success:
  engine_env_ids:
    maniskill: PickCube-v1
```

Genesis requires `success.engine_factory.genesis` as `module:function`.
RoboCasa accepts `success.engine_factory.robocasa` or
`success.engine_env_ids.robocasa`. ManiSkill and MuJoCo do not currently execute
factories.

Legacy fields such as `maniskill_env_id` may be read by an adapter but do not
satisfy the public explicit-mapping gate. Do not use them for new tasks.

### Environment Options And Horizon

ManiSkill consumes these values from `success` when present:

```yaml
success:
  obs_mode: state_dict
  control_mode: pd_ee_delta_pose
  robot_uids: panda
  render_mode: rgb_array
  sim_backend: cpu
  render_device: cuda:0
  shader_dir: default
  max_steps: 80
```

Environment variables can override these values for a run. The result metadata
must preserve enough environment information to reproduce the effective setup.

MuJoCo consumes `render_mode` and `max_steps`; embodiment and control semantics
come from its environment ID.

### Success Extraction

ManiSkill reads configured `success_info_keys` followed by `success`,
`is_success`, and `success_once` from environment info.

MuJoCo supports environment info keys and these metric contracts:

```yaml
success:
  success_metric: final_reward_threshold
  reward_threshold: -0.2
```

```yaml
success:
  success_metric: episode_return_threshold
  return_threshold: 25.0
```

```yaml
success:
  success_metric: survival_steps
  min_success_steps: 500
```

Declarative predicates such as `object_lifted`, `object_inside`,
`object_on_top`, or `ee_at_target` are useful for audit and claim validation,
but current common adapters do not compute them from simulator state. The
environment or adapter must emit a tested success signal. A recognized YAML key
alone does not implement a predicate.

`max_steps` is executable and participates in timeout mapping. Other limits,
including `max_time_seconds` and `max_collisions`, require explicit adapter/task
logic before they affect termination or success.

## Policy Interface Contracts

`observation`, `action`, `goal`, `experts`, and `ood_splits` are recorded in
dataset provenance. They are declarative unless a policy, engine, expert, or
study consumes them explicitly.

- Engine adapters wrap observations under `raw` and add the live action-space
  contract when available.
- The live simulator action space is authoritative; task-level shape and bounds
  do not coerce actions automatically.
- `description` is not injected into a language-conditioned policy.
- Put a structured instruction under a documented `goal` key and make the
  policy adapter consume and record it explicitly.
- `experts` does not enable a provider; execution configuration selects one.
- `ood_splits` must identify actual non-overlapping train/evaluation lineage.

Any consumer-specific nested schema should be documented and covered by
round-trip and integration tests.

## Randomization

The common executable baseline is:

```yaml
randomization:
  seed: true
```

Typed stressors use the versioned specification:

```yaml
randomization:
  seed: true
  stressors:
    - format: nyssa-stressor-spec-v1
      stressor_id: observation_gaussian_noise
      severity: 0.25
      parameters:
        max_std: 0.1
```

Task-level stressors execute on every run and in list order. Legacy keys such as
`lighting: true`, `camera_pose: true`, or `friction_range: [...]` remain
declarations unless an adapter maps them to an executable stressor. Unsupported
declarations are reported and cannot support stress-test claims. See
[Stressor Protocol](stressor_protocol.md).

## Failure Labels

Use shared labels from the failure taxonomy where possible. A task may emit an
environment-native `failure_label`, rely on a tested `FailureMapper` diagnostic,
or emit temporal failure events with evidence. Do not list failure labels that
the task cannot produce, and do not hide unmapped cases behind a preferred
label. See [Failure Taxonomy](failure_taxonomy.md) and
[Failure Event Protocol](failure_event_protocol.md).

## Suites

Suites live under `configs/suites/<suite_id>.yaml`:

```yaml
suite_id: maniskill_smoke_v0
description: Smoke suite mapped to common ManiSkill manipulation environments.
tasks:
  - maniskill_pick_cube
  - maniskill_stack_cube
  - maniskill_push_cube
```

The task list must be non-empty. Entries resolve by task filename stem and retain
their declared order. Keep suite IDs equal to filename stems, avoid duplicate
tasks, and include only tasks executable under the engine selected for the run.

## Validation Boundary

These commands verify parsing and references:

```bash
uv run nyssa validate <task_id-or-path>
uv run nyssa validate <suite_id-or-path>
uv run python scripts/validate_configs.py
```

They do not prove that an environment ID exists, an engine factory returns a
compatible environment, a success predicate is correct, a stressor changes the
simulator, or a failure label is diagnosable. Those claims require focused unit
tests and a real simulator smoke run described in
[Adding New Tasks](adding_new_tasks.md).
