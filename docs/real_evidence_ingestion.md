# Real and reconstructed evidence ingestion

`nyssa-real-evidence-package-v1` is the boundary between NyssaBench, real-robot
programs, and external real-to-sim systems. NyssaBench validates and compares
supplied evidence. It does not reconstruct scenes, estimate physics, optimize a
digital twin, or manage customer data.

## Contract contents

A package records:

- robot, embodiment, controller, policy checkpoint, site, trial, and
  pseudonymous operator identity;
- clock domains, timestamp units, synchronization offsets, drift, and
  uncertainty;
- coordinate-frame topology and transforms;
- sensor modalities, artifact hashes, sample counts, units, and missing ranges;
- executed action representation, dimensions, units, bounds, timestamps, and
  latency calibration;
- task outcome, interventions, safety events, and missing-data markers;
- temporal `FailureEvent` records with `real_robot` provenance;
- camera, clock, geometry, dynamics, and latency calibration with uncertainty
  and fit quality;
- one or more reconstructed variants with tool identity, assumptions,
  parameter estimates, mismatches, outcomes, and
  `reconstructed_simulation` failure provenance;
- a one-to-many real/sim mapping with controlled axes and matching keys;
- privacy classification, consent basis, license, redaction, retention,
  redistribution, and artifact-access rules.

The package is content-addressed. Packaged artifacts are also hashed and must
resolve inside the package directory.

## Validation and ingestion

Validate all required artifacts:

```bash
uv run nyssa validate-real-evidence path/to/evidence-package
```

Protected artifacts can be cataloged without access:

```bash
uv run nyssa validate-real-evidence path/to/evidence-package --metadata-only
```

Metadata-only validation keeps `evidence_ready` and `claim_ready` false. Missing
calibration, uncertainty, fit quality, governance controls, or complete variant
mapping also downgrades claims instead of silently filling fields.

Write sanitized artifacts for analysis:

```bash
uv run nyssa import-real-evidence path/to/evidence-package \
  --out benchmark_results/real_evidence_import
```

The import contains:

- `real_evidence_manifest.json`, with operator IDs and artifact locations
  removed;
- `real_evidence_ledgers.json`, preserving real versus reconstructed event
  provenance as a separate artifact governed by the package access rules;
- `real_sim_pairs.json`, one row per controlled reconstruction variant;
- `real_evidence_report.html`, showing readiness, calibration uncertainty,
  fit quality, mismatches, and validation issues.

The sanitized manifest contains event summaries only. It omits operator IDs,
artifact locations, evidence payloads, and embedded ledgers.

Issue #38's paired sim-real study consumes `comparison_pairs()` and these
versioned artifacts rather than inventing another file layout.

## Conformance fixture

`conformance/real_evidence/v1/valid_reconstructed_family/` contains one
synthetic real failure and two reconstructed variants. It exercises clocks,
frames, units, actions, interventions, failure ledgers, calibration,
uncertainty, mismatch reporting, governance, and one-to-many mapping without
including a reconstruction method.

The fixture ships in wheels. Use
`real_evidence_conformance_fixture_path()` to locate it in external conformance
tests.
