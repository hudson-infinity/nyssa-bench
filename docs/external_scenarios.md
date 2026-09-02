# External scenario packages

NyssaBench accepts scenarios produced by external world-generation, scene,
rare-event, or adversarial systems through
`nyssa-external-scenario-package-v1`. The producer creates the package;
NyssaBench validates its identity and contracts, translates declared variation
into the Stressor Engine, executes the mapped task, and records the package in
the result pack.

NyssaBench does not implement scenario synthesis or optimize the producer.

## Package layout

A directory uses `scenario.yaml` as its manifest:

```text
scenario-package/
├── scenario.yaml
└── assets/
    └── scene_descriptor.json
```

The manifest records:

- stable scenario and generator identities with semantic versions and revisions;
- a canonical package SHA-256;
- NEP, task, stressor, and split contract versions;
- engine/runtime requirements and one explicit task/environment mapping;
- asset hashes, licenses, provenance, redistribution policy, and resolution
  location;
- deterministic reset identity, physical parameters, and signal visibility;
- stressor axes, severity ranges, defaults, parameters, and composition rules;
- train, validation, public-test, and hidden-test lineage with contamination
  status;
- success predicates, horizon, safety constraints, and solvability checks;
- optional rare-event or adversarial-search provenance.

Unknown fields are rejected. The package hash is calculated from canonical JSON
serialization of the complete manifest except `content_sha256`; asset content is
covered through each asset's declared hash.

## Validation modes

Execution validation requires every required asset to resolve within the package
and match its hash:

```bash
uv run nyssa validate-scenario path/to/scenario-package
```

Metadata validation can identify a protected package when licensed assets
cannot be redistributed:

```bash
uv run nyssa validate-scenario path/to/scenario-package --metadata-only
```

Protected assets still require a content hash, license, provenance URI, and
external locator. Metadata validation reports them as unresolved and sets
`execution_ready: false`; `run-scenario` will reject the package until they are
resolved.

The general validator also accepts scenario directories:

```bash
uv run nyssa validate path/to/scenario-package
```

## Stressor boundary

An external package declares available axes, but it cannot mutate the simulator
directly. Every axis must name a registered `nyssa-stressor-spec-v1` contract.
NyssaBench checks the registered severity domain, engine/task support, and
composition conflicts, then builds a normal `StressorConfig`.

Run the package with its default severities or explicit overrides:

```bash
uv run nyssa run-scenario path/to/scenario-package \
  --policy random \
  --episodes 20 \
  --severity action_delay=0.5 \
  --out benchmark_results/external_scenario_smoke \
  --no-replay
```

Overrides use `STRESSOR_ID=SEVERITY`, must be unique, and must remain inside the
producer-declared range. The resulting run writes `scenario_execution.json` and
includes its hash in `dataset_manifest.json`. Public artifacts preserve asset
identity, license, provenance, and resolution status without copying protected
locators.

The manifest's run seed and episode-seed protocol are part of scenario identity.
`run-scenario` rejects a different `--seed`; publish a new package version and
content hash for a new initial-state protocol.

Version 1 executes seeded resets against the mapped task's default physical
baseline. The manifest must declare `source: simulator_task_default` and
`mutation_policy: stressor_contracts_only`; masses, friction, damping, or other
physical variation must use registered stressor axes. Direct generator-specific
engine writes are rejected.

## Split and claim behavior

Split IDs and parents must resolve without cycles. Hidden-test splits must be
protected. Public and hidden evaluation splits cannot use `unknown`
contamination status, and a train/evaluation content-hash collision cannot be
declared clean.

Known overlap remains executable for diagnostic studies, but validation sets
`claim_ready: false`. It cannot pass the scenario claim check.

Scenario-backed public claims require both `scenario_package_valid` and
`scenario_execution_ready` and `scenario_claim_ready` in the run claim
validator. A metadata-only package can be cataloged, but it cannot support an
execution claim.

Comparison contracts include the full scenario identity. Runs from different
package versions or content hashes are incompatible even when they map to the
same task and stressor ID.

## Conformance fixture

The static package under
`conformance/scenario/v1/valid_seeded_mujoco/` demonstrates the producer and
consumer boundary without including a generator. External repositories can load
that fixture in their own tests, replace its producer-specific fields, recompute
the canonical package hash, and run `ScenarioPackageValidator` as a conformance
check. The `conformance/` tree is included in source distributions and wheels.
Use `scenario_conformance_fixture_path()` to locate the packaged fixture without
assuming a checkout layout.
