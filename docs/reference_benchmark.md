# Reference benchmark and protected splits

NyssaBench ships a **candidate**, not a validated benchmark release, at
[`configs/reference/nyssa_reference_v0_1.json`](../configs/reference/nyssa_reference_v0_1.json).
The candidate fixes the intended task and statistical scope while keeping every
uncollected artifact visible as missing evidence.

## Candidate task set

The 12 tasks use registered ManiSkill 3.0.1 environments:

| Mechanism | Tasks |
| --- | --- |
| Grasp and place | PickCube, PlaceSphere |
| Non-prehensile manipulation | PushCube, PushT |
| Stacking | StackCube |
| Contact-rich insertion | PegInsertionSide, PlugCharger |
| Articulated manipulation | OpenCabinetDrawer, OpenCabinetDoor, TurnFaucet |
| Clutter and distractors | PickClutterYCB |
| Multi-stage tool use | PullCubeTool |

These mappings follow the upstream [ManiSkill task inventory](https://maniskill.readthedocs.io/en/latest/tasks/).
Each entry embeds a complete NEP Task Contract and pins its executable Nyssa
TaskSpec by SHA-256. The ManiSkill container smoke verifies that all environment
IDs are registered. Registration is not solvability evidence.

## Split contract

Every train, validation, public-test, and hidden-test split independently
commits five dimensions:

- assets;
- initial states;
- poses;
- task variants;
- demonstrations and training sources.

Each dimension has an item count, SHA-256 content commitment, and `pending` or
`committed` status. Public committed dimensions must point to a hash-matching
artifact. Hidden-test dimensions must never contain a path. Hidden splits also
require an independent evaluator, unpublished contents, protection enabled,
clean contamination status, unique hashes, and acyclic lineage.

The committed candidate intentionally uses `pending` commitments. Its hashes
are placeholders generated from labels and cannot be promoted. Real split
contents must be created privately, committed cryptographically, and audited
before changing the benchmark status to `release`.

## Evidence requirements

`nyssa audit-reference` validates native evidence rather than trusting status
text. A release requires all of the following:

- 12–20 hash-pinned TaskSpecs and NEP Task Contracts;
- verified per-task asset identity, license, and content provenance;
- execution-verified success semantics for every simulator task;
- clean commitments for every split dimension;
- one oracle solvability artifact per task with at least 100 episodes and at
  least 80% success;
- passing RunValidity and BenchmarkValidity inside every oracle artifact;
- passing shortcut, leakage, input-ablation, statistical-precision,
  paired-design, rank-stability, and hidden-test audits in each result;
- complete evidence from at least two distinct learned policy families;
- full task coverage by each learned-policy result;
- paired seeds, a worst-case target success interval width of 0.2, and 5,000
  paired bootstrap resamples.

The 100-episode design is a prespecified starting point based on worst-case
Wilson interval width. It must be recomputed from pilot variance before the
final experiment. A fixture passing this validator does not establish that the
real benchmark is powered.

## Commands

Regenerate and verify the candidate after changing a task:

```bash
uv run python scripts/generate_reference_candidate.py
uv run python scripts/generate_reference_candidate.py --check
```

Audit the current evidence state:

```bash
uv run nyssa audit-reference \
  configs/reference/nyssa_reference_v0_1.json \
  --repo-root . \
  --out build/reference-audit
```

Exit code `0` means a declared release has complete valid evidence. Exit code
`2` means the report was generated but release evidence is missing. Invalid or
tampered inputs fail. The current candidate reports 12 passing task contracts
and 57 missing asset, success-semantics, split, oracle, and learned-policy checks.

Run the executable suite only after installing ManiSkill and its assets:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
uv run nyssa run \
  --suite nyssa_reference_manipulation_v0_1 \
  --engine maniskill \
  --policy random \
  --episodes 1 \
  --out benchmark_results/reference_registration_smoke \
  --capture-replay
```

Random is only a pipeline control. It cannot satisfy oracle solvability or the
learned-policy requirement.

## Promotion boundary

Do not place hidden values, private asset lists, held-out poses, or evaluator
secrets in generated configs, run manifests, logs, or release bundles. Only
content hashes and counts may cross the hidden-test boundary.

The claim matrix keeps `compact_reference_benchmark` at `planned` until a
release-ready report and immutable result packs exist. The Phase 1 credibility
gate consumes the release report directly; a candidate report cannot satisfy
Gate B.
