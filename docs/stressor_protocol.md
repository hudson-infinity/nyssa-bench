# Stressor Protocol

NyssaBench stressors are deterministic, executable interventions used to
measure policy behavior under controlled distribution shift. A declared
stressor is never treated as active until its application is confirmed by the
pipeline or simulator backend.

## Versioned Configuration

Pass one condition to `nyssa run`, `nyssa experiment`, or `nyssa ablate` with
`--stressor-config`:

```yaml
format: nyssa-stressor-config-v1
condition_id: action_delay_s05
unsupported_policy: error
stressors:
  - format: nyssa-stressor-spec-v1
    stressor_id: action_delay
    severity: 0.5
    parameters:
      max_delay_steps: 4
```

Severity is finite and normalized to `[0, 1]`. Each stressor maps severity to
concrete applied parameters and records both values. An optional non-negative
`seed` overrides the deterministic seed derived from the episode seed,
composition index, and stable stressor identifier.

Unknown fields, formats, stressors, invalid severities, and duplicate stressor
identifiers are rejected. Configuration order is execution order and appears
as `composition_order` in every stressor context.

## Lifecycle

The execution lifecycle is:

```text
validate support and composition
  -> derive deterministic stressor seeds
  -> before_reset
  -> simulator reset
  -> restore configured initial simulator state
  -> after_reset
  -> transform policy observation
  -> transform executed action
  -> before_step / simulator step / after_step
  -> capture restorable stressor state
```

Stressor state includes RNG state and stateful runtime data such as delayed
actions. Each application declares its normalized severity domain and episode
lifetime. `StressorPipeline.get_state()` and `set_state()` require the same
condition, episode seed, engine/task context, and composition order. This makes
stressor context available to replay and future state-fork recovery evaluation
instead of restoring simulator state alone.

## Built-In Stressors

| Identifier | Category | Application point | Severity mapping | Support |
| --- | --- | --- | --- | --- |
| `image_brightness` | visual | policy RGB observation | interpolation from `1.0` to `target_scale` | pixel observation modes on wrapped engines |
| `observation_gaussian_noise` | sensor | policy observation | `std = severity * max_std` | all wrapped engines with numeric `raw` observations |
| `action_gaussian_noise` | action | executed action | `std = severity * max_std` | all engines; clips to declared action bounds |
| `action_delay` | system | executed action | rounded steps up to `max_delay_steps` | all engines with numeric actions |
| `friction_scale` | dynamics | simulator after reset | interpolation from `1.0` to `target_scale` | MuJoCo; ManiSkill CPU PhysX |

ManiSkill GPU simulation cannot safely mutate PhysX materials after scene
creation. `friction_scale` therefore reports `unsupported` for that backend.
Visual, sensor, action, and system stressors remain available in compatible
ManiSkill GPU observation/control modes.

Use `nyssa list-stressors` to list the registered implementations.

## Support States

Every requested stressor ends an episode in one of these states:

| State | Meaning |
| --- | --- |
| `applied` | The transform or backend effect executed and exact parameters were recorded. |
| `skipped` | The request intentionally produced no shift, such as severity `0.0`. |
| `unsupported` | The task, mode, or backend could not execute the request. |

The transient `requested` state is also serialized during setup. The default
`unsupported_policy: error` stops the run. `unsupported_policy: record` permits
diagnostic execution but fails the `stressor_requests_resolved` claim check and
cannot support a public stress-test claim.

## Composition

Stressors run in declared order. For example, delay followed by action noise
adds noise to the delayed action, while noise followed by delay buffers the
noisy action. This distinction is intentional and tested. A stressor may
declare incompatible identifiers through `conflicts_with`; duplicate stable
identifiers are always incompatible.

## Artifacts

Each run writes:

- `stressor_manifest.json`: configured condition, aggregate support state, and
  exact per-episode requested/applied parameters and final state
- `episodes.json` and `episodes.jsonl`: episode and per-step stressor context,
  including policy action before stressors and executed action
- `replay_manifest.json`: stressor context associated with each replay
- `run.yaml`, `config.yaml`, and `dataset_manifest.json`: run-level stressor
  configuration, execution summary, and artifact hashes

`metrics.json` includes `stressor_execution`, while each episode reports
applied and unsupported stressor counts.

## Severity Sweeps

Run separate, matched result packs for each severity. The checked-in
`configs/stressors/action_delay_s*.yaml` files provide a minimal example:

```bash
uv run nyssa run \
  --suite mujoco_control_v0 \
  --engine mujoco \
  --policy random \
  --episodes 20 \
  --seed 0 \
  --stressor-config configs/stressors/action_delay_s0.yaml \
  --out benchmark_results/action_delay/s0 \
  --no-replay

uv run nyssa run \
  --suite mujoco_control_v0 \
  --engine mujoco \
  --policy random \
  --episodes 20 \
  --seed 0 \
  --stressor-config configs/stressors/action_delay_s05.yaml \
  --out benchmark_results/action_delay/s05 \
  --no-replay

uv run nyssa run \
  --suite mujoco_control_v0 \
  --engine mujoco \
  --policy random \
  --episodes 20 \
  --seed 0 \
  --stressor-config configs/stressors/action_delay_s1.yaml \
  --out benchmark_results/action_delay/s1 \
  --no-replay

uv run nyssa robustness-report \
  benchmark_results/action_delay/s0 \
  benchmark_results/action_delay/s05 \
  benchmark_results/action_delay/s1 \
  --out benchmark_results/action_delay/report
```

The aggregation requires identical suite, engine, policy, task set, episode
budget, and complete `(task_id, seed, episode_index)` coverage at every
severity. It emits:

- clean and per-severity shifted success with Wilson 95% intervals
- degradation from clean performance
- normalized trapezoidal robustness AUC over the observed severity span
- a paired episode-bootstrap 95% interval for AUC
- `robustness.json`, `robustness.csv`, and `robustness.html`

## Public Claim Rule

A public stress-test claim requires exact support confirmation for every
requested non-zero stressor, matched stressor distributions between compared
policies, complete paired episode coverage across severities, replay evidence,
declared parameter ranges, and uncertainty. A severity-zero condition is the
clean baseline, not evidence that the corresponding shift executed.
