# Experimental Backends

RoboCasa and Genesis are tracked as experiments, not supported public
NyssaBench backends. Their registry entries exercise adapter boundaries and
experiment contracts; that status does not imply runnable task coverage,
reproducible rendering, or valid benchmark results.

## Evidence Levels

Keep these levels distinct:

1. **Import hook:** adapter class imports and is registered.
2. **Contract validation:** experiment YAML references loadable task specs and
   declares required mappings.
3. **Runnable smoke:** a concrete suite executes reset/step/artifact writing on
   the real simulator.
4. **Validated integration:** task mappings, success/failure semantics,
   deterministic seeds, state/rendering/stressors, policies, and tests are
   demonstrated on supported infrastructure.
5. **Supported public backend:** registry, claim gates, package provenance,
   simulator-backed CI, and a complete validated result pack are accepted.

Only level 5 is a supported public backend.

## Current Contract Validation

Experiment configs live in `configs/experiments/` and describe minimum future
integration requirements:

- task spec to scene asset mapping
- success predicate mapping
- randomization support
- replay or event export
- failure taxonomy mapping

Run contract validation with:

```bash
uv run python scripts/validate_backend.py robocasa
uv run python scripts/validate_backend.py genesis
```

These commands currently parse the experiment contract, load candidate tasks,
and ensure required mapping flags are enabled. They do not import/run the real
simulator, call an engine factory, capture replay, or validate policy success.

`--run-experimental` still fails until `scripts/validate_backend.py` has a real
suite entry for the backend. Do not use the contract-only command output as a
simulator test result.

## Concrete Mappings

The adapters can run real environments only after a task provides a mapping:

- RoboCasa: `success.engine_env_ids.robocasa` for a robosuite/RoboCasa
  environment, or `success.engine_factory.robocasa` as `module:function`.
- Genesis: `success.engine_factory.genesis` as `module:function`.

The factory receives a `TaskSpec` and must return an environment compatible with
the adapter's reset, step, render, state, and close expectations. Factory code
is trusted executable code and must be versioned, tested, and included in
provenance.

Until mappings, upstream assets, and dependencies exist, adapters must fail with
explicit setup guidance instead of selecting a placeholder environment.

## Promotion Checklist

### Packaging And Reproducibility

- [ ] A documented dependency extra installs a tested simulator version on the
  supported Python/OS/GPU matrix.
- [ ] Native assets, licenses, macros/download steps, rendering libraries, and
  driver requirements are documented.
- [ ] `package_versions.json` records the engine package and claim validation
  recognizes it.
- [ ] Seeds reach simulator, task, domain-randomization, and adapter RNGs.
- [ ] Same task/seed/config resets are equivalent within documented tolerance.

### Task, Robot, Observation, And Action Contracts

- [ ] At least one compact reference suite has explicit environment/factory
  mappings and no hidden fallback tasks.
- [ ] Canonical robot IDs map to real simulator assets/controllers with explicit
  compatibility errors.
- [ ] Reset/step observations preserve documented modalities and separate
  privileged state from policy-visible inputs.
- [ ] Live action spaces have validated shape, order, dtype, finite bounds,
  units, normalization, and controller semantics.
- [ ] Task filtering and paired episode identities work across the suite.

### Success, Failure, Stressors, And Recovery

- [ ] Every task has positive and negative success-predicate tests tied to real
  simulator signals.
- [ ] Expected failure labels/events have engine/task evidence and unknown
  failures are measured honestly.
- [ ] Supported typed stressors alter the intended backend state and return
  before/after application evidence; unsupported requests fail or downgrade.
- [ ] State capture content is documented. Restore and next-transition
  equivalence are tested before replay branching/recovery claims.
- [ ] Planner/verifier/recovery integrations use compatible state/action
  contracts and record interventions.

### Rendering And Artifacts

- [ ] Headless rendering works on documented infrastructure with stable,
  nonblank frames before/after steps.
- [ ] Every public episode has a valid MP4, replay manifest entry, safe path,
  and consistent denominator.
- [ ] Run/config/environment/package/Git/episode/stressor/failure artifacts are
  complete and revalidate after result-pack assembly.
- [ ] Effective engine factory, robot, controller, cameras, overrides, and asset
  versions are recoverable from provenance.

### Tests And Continuous Integration

- [ ] Shared fake-engine contract tests cover lifecycle, seeds, termination,
  truncation, success, actions, failures, stressors, state, and cleanup.
- [ ] Real simulator tests cover every supported task/controller/observation
  mapping.
- [ ] Scheduled or GPU-backed CI executes the simulator; lightweight import CI
  alone is insufficient.
- [ ] `validate_backend.py` runs a real backend suite rather than only an
  experiment contract.
- [ ] Installation, task, robot, policy, and engine guides are updated.

### Empirical Promotion Evidence

- [ ] At least 100 episodes per task per run and at least three independent run
  seeds are assembled with complete paired task/episode matrices.
- [ ] A non-placeholder policy and meaningful reference baseline produce
  interpretable success/failure distributions.
- [ ] Public run and replay validators pass from a clean identified Git commit.
- [ ] Results do not describe adapter hooks, factories, random runs, missing
  videos, or training-seed diagnostics as backend validity.
- [ ] Maintainers review and approve the evidence before support-tier changes.

## Promotion Changes

After evidence is accepted:

1. change the engine's `ENGINE_SUPPORT_TIER` entry;
2. add it to public-claim engine validation only if public evidence requirements
   are met;
3. map its package name in engine package-version validation;
4. add stable installation extras and backend validation suite;
5. add simulator-backed CI and release-smoke coverage;
6. publish the validated result pack and update result-tier documentation.

Promotion is reversible. If upstream packaging, assets, renderer, success
semantics, or tests stop satisfying the contract, downgrade the support tier and
invalidate affected claims until revalidated.

See [Engine Adapters](engine_adapters.md),
[Adding New Robots](adding_new_robots.md), and
[Validation Protocol](validation_protocol.md).
