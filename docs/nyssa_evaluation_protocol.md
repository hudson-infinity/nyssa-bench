# Nyssa Evaluation Protocol 0.1

The Nyssa Evaluation Protocol (NEP) is the portable contract layer for a
NyssaBench evaluation. NEP has its own semantic version, currently `0.1.0`,
separate from the Python package version.

## Normative artifacts

The normative definitions are:

- Pydantic models in `nyssa_bench/nep/protocol.py`;
- generated JSON Schemas in `schemas/nep/0.1.0/`;
- valid and deliberately invalid fixtures in `conformance/nep/0.1.0/`;
- canonical JSON serialization and `content_sha256` validation;
- cross-contract checks in `NEPManifest`.

Documentation examples are guidance. If prose conflicts with a model or schema,
the model and committed schema control for NEP 0.1.

Regenerate schemas and fixtures after an intentional contract change:

```bash
uv run python scripts/generate_nep_artifacts.py
uv run pytest -q tests/test_nep.py
```

CI compares every committed schema and fixture with the runtime models.

## Six contracts

### Task

The Task Contract identifies the task version, engines, robot, scene, horizon,
observation modalities, action representation, success predicate, assets,
licenses, and split lineage. Evaluation splits cannot declare training overlap.

### Stressor

The Stressor Contract records condition identity, composition semantics,
stressor versions, categories, severity, seeds, application points, policy
observability, privilege, parameters, and backend application evidence.
Backend-confirmed status requires a content-addressed evidence artifact.

### Policy

The Policy Contract identifies the policy family, version, checkpoint and
preprocessing hashes, modalities, action bounds and dimension, prediction and
execution horizons, state semantics, deterministic seeding claim, and training
dataset/split lineage.

### Failure evidence

The Failure Evidence Contract references the temporal FailureEvent ledger and
detector contracts. It declares temporal precision, evidence visibility, and
whether causal links are hypotheses or supported by intervention evidence.

### Intervention

The Intervention Contract declares whether intervention is enabled, trigger
sources, intervention types, cost metrics, counterfactual branch evidence, and
required restoration fidelity. Disabled interventions cannot carry execution
evidence.

### Claim

The Claim Contract requests one tier and names every supporting artifact. The
manifest enforces tier-specific requirements:

| Tier | Additional requirement |
| --- | --- |
| `pipeline` | Complete contracts and RunValidity reference. No benchmark-performance claim. |
| `clean_simulation` | BenchmarkValidity evidence and no positive-severity stressor. |
| `ood_robustness` | Positive, backend-confirmed stressors plus BenchmarkValidity. |
| `recovery_effectiveness` | Enabled intervention, counterfactual branches, and qualified or exact restoration. |
| `cross_simulator` | At least two engines and BenchmarkValidity evidence. |
| `sim_real_predictive` | Real-evidence artifact and BenchmarkValidity evidence. |

Unknown artifact references, policy/task action mismatches, missing modalities,
training/evaluation split overlap, and policy horizons beyond the task horizon
invalidate the complete manifest.

## Canonical identity

`NEPManifest.create()` serializes the complete envelope as sorted finite JSON,
excluding only `content_sha256`, and hashes it with SHA-256. Readers recompute
the hash. Editing any nested contract without rebuilding the identity is an
error.

Every evidence reference resolves through the manifest's artifact table. An
artifact records its media type, SHA-256, URI, and whether it is required.

## Validation

```bash
uv run nyssa validate-nep path/to/nep-manifest.json \
  --out path/to/nep-validation.json
```

The command returns 0 for a valid envelope and 3 for invalid or incompatible
input. Its report includes field paths and actionable messages. Generate a copy
of the schemas with:

```bash
uv run nyssa write-nep-schemas --out exported-nep-schemas
```

## Compatibility and migration

NEP 0.x treats each minor version as a compatibility boundary. A 0.1 reader may
accept older 0.1 patch artifacts, but it must reject 0.2 or a newer 0.1 patch it
does not understand. After 1.0, equal major versions are compatible when the
artifact minor version is not newer than the reader.

Patch releases may clarify descriptions, add optional fields with defaults, or
add stricter diagnostics that do not reject previously valid data. Changing a
required field, meaning, enum, canonicalization rule, hash input, or claim gate
requires a new 0.x minor version and explicit migration.

`nyssa-nep-draft-v0` has one explicit migration into 0.1. The migration rejects
unknown draft fields, validates all six target contracts, recomputes canonical
identity, and emits `nyssa-nep-migration-v0.1`. Current manifests are never
silently rewritten.

## Evidence status

The committed MuJoCo and ManiSkill fixtures prove schema conformance only. They
are pipeline controls, not simulator result packs. The installed MuJoCo CI track
can now produce real execution evidence. A ManiSkill end-to-end NEP result still
requires the GPU workflow described in [Simulator-backed CI](simulator_ci.md).

NEP 0.1 is implemented in source but is not a published protocol release until
the PyPI/TestPyPI work in #59 and release bundle work in #61 execute.
