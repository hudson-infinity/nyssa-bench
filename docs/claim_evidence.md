# Claim evidence and public positioning

NyssaBench keeps project positioning separate from research goals. The current
public description is:

> NyssaBench is an open-source failure-aware evaluation and audit framework for
> embodied AI policies, built toward foundational infrastructure for evaluating
> frontier robot systems.

That sentence describes the implemented framework. It does not claim validated
frontier-policy coverage, a completed reference benchmark, simulator-backed CI,
or predictive sim-real evidence.

## Evidence matrix

[`claims/claim_evidence.json`](../claims/claim_evidence.json) is the normative,
machine-readable inventory. Each claim records:

- capability status: `implemented`, `integration_only`, `experimental`, or
  `planned`;
- evidence tier and whether the wording is authorized as a public assertion;
- source, test, issue, and result-artifact requirements;
- limitations and the work required for promotion.

`source_verified` means the implementation and deterministic tests exist. It is
not equivalent to `result_validated`. A result claim needs immutable run and
BenchmarkValidity artifacts. `predictive_validated` additionally requires a
paired hardware study.

Run the same check used by CI and the release checklist:

```bash
uv run python scripts/validate_claim_evidence.py
```

The validator checks referenced source and test paths, required public wording,
forbidden stronger headlines, promotion dependencies, and README result links.
A headline result link is accepted only when the matrix lists the result pack
and its RunValidity and BenchmarkValidity artifacts exist.

## Capability labels

`implemented` means the repository contains the behavior and tests. It does not
say that every simulator, task, or policy supports it.

`integration_only` means an adapter or pipeline path exists, but it has not met
the evidence requirements for a validated track. Current learned-policy hooks
fall into this category until #16 is complete.

`experimental` means the contract may be useful for development, but the
backend or workflow is not eligible for a public benchmark claim.

`planned` means the repository may contain a roadmap, placeholder, or small
utility, but the named capability is not available as a validated workflow.

## Promotion gate

The stronger foundational/frontier headline is stored as milestone wording in
the matrix and is currently unauthorized. Promotion requires all of these
claims to reach result-backed evidence:

1. executable stressor measurement;
2. temporal failure evidence;
3. counterfactual recovery measurement;
4. benchmark-validity audits on the reference evidence;
5. validated learned-policy coverage;
6. the compact reference benchmark;
7. simulator-backed continuous integration;
8. sim-real predictive validity.

The measurement capabilities have source and test implementations, but that is
not enough for promotion. They still need applicable validated result packs.
The learned-policy, reference-benchmark, simulator-CI, and predictive claims
remain integration-only or planned. The
[Phase 1 credibility gate](phase1_credibility_gate.md) combines these entries
without treating source verification as result validation.

Changing the stronger milestone to an authorized claim requires a separate
reviewed change that attaches immutable evidence and makes the matrix validator
report `promotion_ready: true`.

## Headline results

The matrix currently lists no headline result packs. Command examples and smoke
directories in the README are instructions, not endorsements or benchmark
results. A future result link must identify its run-validity and
benchmark-validity artifacts in the matrix before it can appear as headline
evidence.
