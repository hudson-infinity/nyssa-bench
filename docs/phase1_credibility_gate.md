# Phase 1 credibility gate

The Phase 1 gate separates implemented measurement infrastructure from
validated benchmark evidence and predictive sim-real evidence. It applies only
to NyssaBench. Passing it does not complete the broader Hudson Labs roadmap.

The normative input is
[`claims/phase1_credibility.json`](../claims/phase1_credibility.json). Evidence
records and every referenced JSON artifact are pinned by SHA-256. The evaluator
rejects changed files, paths outside the declared repository root, unknown
fields, duplicate IDs, malformed native artifact formats, and documentation
files presented as result evidence.

## Gates

### Gate A: Measurement Core

Gate A is source verified. It requires the validated claim matrix, NEP
contracts, executable stressors, temporal failure events, counterfactual
recovery, BenchmarkValidity, metric-vector reporting, and machine-controlled
public wording. Every capability needs non-documentation source and test paths.

### Gate B: Reference Benchmark Evidence

Gate B requires Gate A plus:

- a compact reference manifest with a protected hidden-test split;
- at least one oracle control;
- valid result packs for two distinct learned policy families, identified by
  NEP policy contracts;
- a paired clean/shifted robustness sweep whose statistical-precision and
  paired-design audits pass;
- a content-valid BenchmarkValidity report; and
- installed-wheel MuJoCo and GPU ManiSkill CI evidence, including per-episode
  ManiSkill replay capture.

Run metrics must contain a valid `nyssa-metric-vector-v1` and passing
`public_claim_validation`. A string such as `status: validated` is not enough:
the evaluator parses each native artifact and recomputes BenchmarkValidity
derived fields and hashes.

### Gate C: Predictive Validity

Gate C requires Gate B, a prespecified claim-ready hardware package, and a
content-pinned `nyssa-sim-real-study-v1` whose complete report contains a
held-out incremental predictive analysis. The study must have at least three
training pairs and two held-out pairs and pass the statistical-precision and
sim-real BenchmarkValidity audits.

Either outcome is valid scientific evidence:

- a confidence interval wholly above zero passes
  `positive_incremental_result`; or
- any other adequately powered interval passes
  `well_powered_negative_result`.

The unused alternative is reported as `not_applicable`, not `passed`. This
keeps positive evidence, negative evidence, missing evidence, and invalid
evidence distinct.

## Status semantics

| Status | Meaning |
| --- | --- |
| `passed` | The required native artifacts were loaded, hash checked, and validated. |
| `failed` | Referenced evidence exists but is malformed, changed, contradictory, or invalid. |
| `missing` | No qualifying evidence was supplied, or an earlier gate has not passed. |
| `not_applicable` | A mutually exclusive, prespecified alternative was not the observed outcome. |

Gate dependencies are sequential. Gate B cannot pass while Gate A is missing or
failed, and Gate C cannot pass while Gate B is missing or failed.

## Evidence records

Each reference in the credibility spec identifies one
`nyssa-credibility-evidence-v1` record by ID, category, repository-relative
path, and SHA-256. The record then pins every JSON artifact it uses. A minimal
record is:

```json
{
  "format": "nyssa-credibility-evidence-v1",
  "evidence_id": "mujoco-ci-2026-09-05",
  "category": "simulator_ci",
  "status": "validated",
  "artifacts": [
    {
      "path": "simulator_smoke.json",
      "sha256": "<64 lowercase hexadecimal characters>",
      "media_type": "application/json"
    }
  ],
  "metadata": {"engine": "mujoco"}
}
```

The generated report repeats the immutable record and artifact references on
every check they satisfy. It also emits the complete gate definitions and maps
the applicable dependencies across issues #13 through #23.

## Commands

From the repository root, evaluate the committed Phase 1 spec:

```bash
uv run nyssa credibility-gate claims/phase1_credibility.json \
  --repo-root . \
  --out build/credibility
```

The command writes `phase1_credibility.json` and
`phase1_credibility.html`. Exit code `0` means all three gates pass. Exit code
`2` means the report is valid but Phase 1 evidence is incomplete. Invalid specs
or artifacts fail normally.

Repository and release checks use:

```bash
uv run python scripts/validate_credibility.py
```

That check succeeds while the honest committed state is Gate A passed with
Gates B and C missing. It fails if Gate A or the claim matrix regresses.

## Public wording

The report reads the current and milestone wording from the validated claim
matrix. Even a passing Gate C cannot authorize stronger wording by itself. The
matrix must also report `promotion_ready: true`, which requires a separate
reviewed evidence-promotion change. Until then, the report selects the current
failure-aware evaluation and audit framework wording.

Generated Worlds, real-to-sim reconstruction, interpretability-method
development, policy learning algorithms, and hosted product infrastructure are
explicitly outside this gate.
