---
name: New engine adapter
about: Propose or track a simulator adapter
labels: area:engine-adapter
---

Read the [engine adapter implementation guide](https://github.com/hudson-infinity/nyssa-bench/blob/main/docs/engine_adapters.md)
and [experimental backend promotion checklist](https://github.com/hudson-infinity/nyssa-bench/blob/main/docs/experimental_backends.md)
before proposing support.

## Engine

- Simulator/project and upstream URL:
- Proposed registry ID:
- Proposed initial support tier:
- Python/OS/GPU/native dependency matrix:

## Why This Engine

Describe evaluation coverage unavailable in current supported engines. Do not
use task count or import success alone as justification.

## Required Task Mapping

- Reference suite/tasks:
- Environment IDs or trusted factories:
- Robot/assets/controllers:
- Success predicates and failure evidence:
- Randomization/stressors:

## Lifecycle Contracts

- `load_task` mapping and compatibility checks:
- Seeded `reset` behavior:
- `step` termination/truncation/success semantics:
- Observation/action contracts and finite bounds:
- State capture/restore scope:
- Cleanup/idempotent close behavior:

## Rendering And Replay

- Headless renderer/backend:
- Policy versus replay cameras:
- Required system libraries/drivers:
- MP4/replay validation plan:

## Planner, Recovery, And Failure Events

Describe state/action interfaces, privileged evidence separation, temporal
failure emission, and intervention logging.

## Install Notes

- Dependency extra and package-version key:
- Asset/license/setup commands:
- Reproducible environment/container:

## Test And Promotion Strategy

- [ ] Fake-engine lifecycle and negative contract tests.
- [ ] Real task/robot/controller/observation integration tests.
- [ ] Deterministic paired seed/reset tests.
- [ ] Positive and negative success extraction tests.
- [ ] Failure and typed-stressor evidence tests.
- [ ] State round-trip/next-transition tests where restore is claimed.
- [ ] Headless render and nonblank MP4 replay tests.
- [ ] Simulator-backed scheduled/GPU CI plan.
- [ ] Real `validate_backend.py` suite plan.
- [ ] Complete public result-pack evidence plan.
- [ ] Registry, claim gate, package version, docs, and release updates identified.

## Current Evidence

Label each artifact as import, contract-only, runnable smoke, validated
integration, or supported-public evidence. Include commands, versions, Git
commit, seeds, and artifact links.
