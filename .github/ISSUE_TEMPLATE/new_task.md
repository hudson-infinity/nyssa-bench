---
name: New task
about: Propose a benchmark task
labels: area:task
---

Read the [task and suite authoring guide](https://github.com/hudson-infinity/nyssa-bench/blob/main/docs/adding_new_tasks.md)
before proposing an executable task.

## Task Goal

Describe the physical objective and why it adds evaluation coverage.

## Identity And Suite

- Proposed task ID and domain:
- Proposed suite ID:
- Existing task or result compatibility affected:

## Engine Mapping

- Engine:
- `engine_env_ids` entry or supported `engine_factory`:
- Simulator/package version:

## Robot

- Embodiment identifier:
- Observation mode and modalities:
- Control/action mode, shape, and bounds:
- Maximum steps:

## Success Predicate

List the exact environment info key, reward/return threshold, survival
criterion, or adapter logic. Include positive and negative test cases.

## Failure Diagnosis

List environment-native labels, `FailureMapper` mappings, and temporal evidence
available for each expected failure.

## Randomization

Separate executable typed stressors from unsupported/declarative shifts. State
severity, seed, backend support, and how application will be verified.

## Goal, Language, And Splits

Describe structured goal/instruction consumption, expert provenance, training
data lineage, and OOD splits where applicable.

## Metrics

List implemented metrics and any new metric contract required.

## Validation Plan

- [ ] Task and suite load by ID and path.
- [ ] Engine mapping and effective environment kwargs are tested.
- [ ] Success extraction has positive and negative tests.
- [ ] Failure labels/events have tested provenance.
- [ ] Live observation and action contracts are simulator-validated.
- [ ] Executable stressors produce backend-confirmed effects.
- [ ] Config validation, focused tests, full tests, and Ruff pass.
- [ ] Required real simulator and replay smoke commands are identified.
