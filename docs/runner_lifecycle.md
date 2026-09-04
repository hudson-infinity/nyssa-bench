# Runner lifecycle

NyssaBench separates execution into three internal responsibilities:

- `ExperimentRunner` expands immutable policy, seed, and ablation cells and
  executes each cell once.
- `EpisodeRunner` owns one episode's outer lifecycle, injected components,
  metric recorder finalization, and lifecycle hooks.
- `CounterfactualBranchRunner` restores branch state and executes matched
  continuation, recovery, and optional oracle branches.

`PolicyRunner` remains the public compatibility facade. Existing commands,
result paths, schemas, and seed derivation continue to use that class.

## Ordered phases

The stable internal lifecycle order is:

```text
task_load
  component_reset
    episode_start
      before_reset
        engine reset
        policy initial-state restore
        stressor after-reset and initial observation transform
      after_reset
      before_policy -> policy prediction -> after_policy
      before_verifier -> verifier decision -> after_verifier
      before_recovery -> recovery planning/state fork -> after_recovery
      before_step
        stressor before-step and action transform
        engine step
      after_engine_step
        engine failure evidence
        stressor after-step and observation transform
        failure detectors
        metric recorders
      after_step
      episode_finalize
  resource_cleanup
```

Cached policy or recovery actions skip policy prediction but still pass through
the same transition lifecycle. A counterfactual branch sets `branch_kind` on
its transition context and uses the same `TransitionLifecycle` as the live
episode. Branch execution does not run recursive verifier/recovery decisions.

## Component interfaces

`EpisodeComponents` receives the engine, policy, expert/verifier, stressor
factory, detector factory, optional branch factory, and metric recorders.
Structural protocols keep external adapters independent of NyssaBench concrete
classes.

A lifecycle hook implements:

```python
class Hook:
    component_id = "my_hook"

    def on_lifecycle(self, context, payload):
        ...
```

A metric recorder implements `reset(context)`,
`record_transition(context, transition)`, and `finalize(context, episode)`.
Final metrics are merged into the episode before run aggregation.

## Failure behavior

Component and hook exceptions are wrapped in `LifecycleExecutionError`. The
error identifies component, phase, task, step, and support status. Detector and
stressor-specific errors keep their existing diagnostics inside that cause.

Episode finalization runs after normal completion and exceptions. A failing
finalization hook does not replace the primary episode error; it is attached as
`lifecycle_finalize_error`. Run cleanup attempts engine, policy, and expert
closure even if an earlier close fails. Cleanup errors are raised when no
primary error exists, or attached to the primary error as
`lifecycle_cleanup_errors`.

Counterfactual branches restore the live snapshot in a `finally` block. A
branch execution error becomes branch evidence when state restoration remains
safe. A restore failure aborts the live episode because continuing would make
the result invalid.

## Experiment expansion

`policy_seed_cells` and `ablation_cells` define deterministic row-major matrix
order. `ExperimentRunner` rejects duplicate output directories before a second
cell can overwrite the first. Both CLI matrix commands use the same
`_matrix_policy_runner` construction path, including counterfactual options.

## Scope boundary

These runners orchestrate measurement. They do not build scenes, choose robot
assets, define controllers, or own simulator physics. Those responsibilities
remain in ManiSkill, MuJoCo, Isaac Lab, and other engine ecosystems consumed by
NyssaBench adapters.
