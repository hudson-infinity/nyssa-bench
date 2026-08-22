# Contributing To NyssaBench

NyssaBench is an evaluation and failure-analysis layer for embodied AI. Keep
changes focused on measurement contracts, simulator and policy adapters,
reproducible artifacts, and evidence-backed reports. Environment generation,
robot training systems, and unrelated model implementations belong in separate
projects unless they are required to exercise an evaluation interface.

By participating, you agree to follow [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
Report vulnerabilities through [SECURITY.md](SECURITY.md), not a public issue.

## Development Baseline

NyssaBench declares Python 3.10 or newer. Use:

- Python 3.11 for core development, documentation, and the same baseline as CI;
- Python 3.10 for ManiSkill motion-planning and compiled `toppra`/`mplib`
  workflows;
- NumPy 1.26 for ManiSkill environments that load extensions built against the
  NumPy 1.x ABI.

Python 3.12 and newer may work for the core package, but a change is not
considered compatible with a simulator extra unless that simulator and its
compiled dependencies run on the selected Python version.

Install [uv](https://docs.astral.sh/uv/), create a GitHub fork, and clone the
fork. Keep the canonical repository as `upstream`:

```bash
git clone https://github.com/<your-user>/nyssa-bench.git
cd nyssa-bench
git remote add upstream https://github.com/hudson-infinity/nyssa-bench.git
git fetch upstream
uv sync --extra all --extra dev
uv run nyssa list-suites
uv run pytest -q
```

Install the repository hooks after the environment is ready:

```bash
uv run pre-commit install --install-hooks
```

The configuration installs both commit and push hooks. Commit hooks check file
structure, YAML/JSON/TOML syntax, accidental large files/private keys, Ruff,
Nyssa config changes, and required release files. The pre-push hook runs the
full pytest suite.

Organization members with direct write access may clone the canonical
repository instead. In either case, `origin` should be the remote where the
contributor branch will be pushed and `upstream` should track the canonical
`main` branch.

`uv sync` is exact by default. A later sync that omits an extra can remove
packages installed by an earlier command. Repeat the complete canonical command
after dependency changes. Do not run several exact sync commands expecting them
to accumulate.

For a smaller simulator-specific environment, choose one complete profile:

```bash
uv sync --extra dev --extra mujoco --extra video --extra reports
uv sync --extra dev --extra maniskill --extra video --extra reports
```

To add capabilities without removing extras already installed in a lean
environment, use one additive command:

```bash
uv sync --inexact --extra dataset --extra lerobot --extra robomimic --extra vla --extra diffusion
```

See [docs/installation.md](docs/installation.md) for rendering packages,
ManiSkill ABI constraints, and all optional extras.

## Repository Boundaries

| Path | Responsibility | Keep Out |
| --- | --- | --- |
| `nyssa_bench/core/` | Task, suite, episode, registry, and shared execution contracts | Simulator imports and model-specific logic |
| `nyssa_bench/engines/` | Translation between a simulator and `NyssaEngine` | Benchmark-wide scoring policy and hidden fallback tasks |
| `nyssa_bench/policies/` | Policy loading, observation/action adaptation, and policy metadata | Simulator ownership and benchmark result validation |
| `nyssa_bench/experts/` | Verifier, planner, expert, and recovery-provider interfaces | Claims that a heuristic is a learned or oracle policy |
| `nyssa_bench/stressors/` | Typed, deterministic distribution shifts and backend evidence | Unapplied randomization declarations |
| `nyssa_bench/failures/` | Temporal failure evidence, provenance, and ledger contracts | Detector algorithms that cannot expose supporting evidence |
| `nyssa_bench/datasets/` | Import/export, provenance, and training-data contracts | Undeclared checkpoints or opaque source data |
| `nyssa_bench/metrics/` | Episode and aggregate measurements and claim gates | Rendering and simulator control |
| `nyssa_bench/reports/` | Comparison, scorecard, result-pack, and HTML presentation | Recomputing simulator truth without recorded evidence |
| `nyssa_bench/replay/` | Replay media, timelines, and replay validation | Policy execution |
| `nyssa_bench/arena/` | Paired policy comparison on validated episode identities | Unpaired rankings |
| `configs/` | Versioned experiment, policy, and stressor configuration | Machine-specific paths and credentials |
| `nyssa_bench/tasks/`, `configs/suites/` | Task and suite YAML contracts | Generated trajectories and results |
| `tests/` | Unit, contract, migration, and opt-in simulator tests | Large datasets and model weights |
| `scripts/` | Validation, backend smoke, release, and setup entry points | Library APIs imported by normal package users |

Prefer an existing abstraction and registry over a parallel execution path.
New simulators should arrive as adapters, policies as policy adapters, and
result semantics as versioned contracts with migration tests.

## Choose A Scoped Change

Start from an issue or open one before a large change. State:

- the behavior or contract that is missing;
- which engines, policies, schemas, or artifacts are affected;
- how the change will be validated;
- whether a public result or compatibility claim changes.

Useful authoring guides include:

- [task and suite contracts](docs/task_spec.md)
- [engine adapters](docs/engine_adapters.md)
- [policy adapters](docs/policy_adapters.md)
- [stressor protocol](docs/stressor_protocol.md)
- [failure event protocol](docs/failure_event_protocol.md)
- [API stability](docs/api_stability.md)

Keep refactors separate from behavioral changes unless the refactor is required
to implement the behavior safely.

## Branch And Commit Workflow

1. Update the default branch and create a branch for one issue:

   ```bash
   git switch main
   git pull --ff-only upstream main
   git switch -c docs/contributor-workflow
   ```

2. Make the smallest coherent change. Do not rewrite unrelated files or remove
   a contributor's local artifacts.
3. Add tests at the contract boundary. A simulator adapter should have shared
   dummy-engine tests plus real simulator coverage where required.
4. Run the checks in the next section.
5. Review `git diff` and `git status` before staging. Stage named paths rather
   than the entire worktree when generated results are present.
6. Use short imperative commit subjects, for example:

   ```text
   Document contributor validation workflow
   Preserve legacy failure ledger fields
   Validate MuJoCo action bounds
   ```
7. Push the branch to the contributor remote:

   ```bash
   git push -u origin docs/contributor-workflow
   ```

   Open the pull request from that branch into
   `hudson-infinity/nyssa-bench:main`.

Do not combine multiple issues, bulk formatting, generated result archives, or
unrelated dependency updates in one commit.

## Required Local Checks

Run all commit-stage hooks manually before opening a pull request:

```bash
uv run pre-commit run --all-files
```

Run the pre-push stage explicitly when diagnosing hook behavior:

```bash
uv run pre-commit run --hook-stage pre-push --all-files
```

Hooks use the active environment, so invoke `pre-commit` through `uv run` unless
the project virtual environment is already activated. Update pinned third-party
hook revisions intentionally with `uv run pre-commit autoupdate`, review the
resulting diff, and rerun both stages.

Run these checks from the repository root before marking any pull request ready:

```bash
uv run pytest -q
uv run ruff check .
uv run python scripts/validate_configs.py
uv run python scripts/release_checklist.py
```

Focused checks are useful while editing, but documentation-only changes still
run the complete baseline before review because command, path, and API examples
are executable contributor contracts.

Pre-commit is a fast consistency gate. It does not replace required simulator
integration, replay inspection, held-out policy evaluation, or release smoke
tests.

When changing a task, suite, experiment, or stressor file, also validate the
specific target:

```bash
uv run nyssa validate path/to/config.yaml
uv run nyssa validate suite_id
uv run nyssa validate task_id
```

Run focused tests while iterating, but report the full-suite result in the pull
request. If a test is skipped, name it and explain which optional dependency or
host capability was unavailable.

Release, packaging, dependency, and artifact-layout changes require the clean
environment smoke test:

```bash
uv run python scripts/release_smoke.py
```

The script creates `.release-venv`, installs the development, video, dataset,
and report extras, runs the full tests, lists suites, validates all task/suite
configs, and runs the static release checklist. Maintainers should also follow
[docs/launch_v0.1.md](docs/launch_v0.1.md) when preparing a tag.

## Simulator Integration Tests

Simulator integration is optional for:

- prose-only documentation changes;
- core schema or report changes fully exercised with synthetic fixtures;
- refactors that do not alter an engine boundary, simulator dependency,
  rendering path, or simulator-derived evidence.

It is required when a change affects:

- an engine adapter or task-to-environment mapping;
- observation or action conversion, state save/restore, or success extraction;
- simulator-specific stressors, failure evidence, contacts, or dynamics;
- rendering, MP4 replay capture, camera observations, or GPU/CPU backend choice;
- simulator dependency constraints;
- a policy, verifier, or recovery path that reads simulator-specific state;
- any result presented as validated on that simulator.

Run the affected backend directly:

```bash
uv run python scripts/validate_backend.py mujoco --episodes 1
uv run python scripts/validate_backend.py maniskill --episodes 1
```

Add `--capture-replay` when rendering or replay evidence changes. ManiSkill
requires a compatible Linux Vulkan/NVIDIA or documented CPU setup. MuJoCo and
ManiSkill are not installed in lightweight CI, so passing CI does not replace a
required local or hosted simulator run. If the required backend cannot be run,
leave the PR in draft or arrange a run on compatible infrastructure and attach
the command, package versions, and result artifact.

RoboCasa and Genesis currently expose experimental contract validation by
default:

```bash
uv run python scripts/validate_backend.py robocasa
uv run python scripts/validate_backend.py genesis
```

Do not describe contract-only validation as a simulator rollout.

## Compatibility And Versioned Contracts

NyssaBench is pre-1.0, but public imports and serialized result packs already
have users. Follow [docs/api_stability.md](docs/api_stability.md):

- prefer additive fields and optional methods with no-op defaults;
- do not rename or reinterpret a serialized field in place;
- give a changed schema a new format/version identifier;
- keep readers for old result packs and add migration/round-trip fixtures;
- preserve top-level summary fields when introducing richer evidence;
- update public exports, type annotations, documentation, and tests together;
- document intentional breakage and provide a migration path before merge.

Changes to task/suite YAML, episode JSON, run manifests, comparison contracts,
failure/stressor schemas, plugin registration, CLI arguments, or environment
variables are compatibility changes even when Python call signatures do not
change.

Never weaken a validation gate to make an old or incomplete result pass. Migrate
the artifact or mark its claim tier honestly.

## Checkpoints, Datasets, Videos, And Results

Generated artifacts are not source code. Keep local outputs under the ignored
directories `runs/`, `reports/`, `checkpoints/`, or
`benchmark_results/<run>/`. Do not commit:

- model checkpoints or downloaded weights;
- imported or generated datasets;
- MP4 files, frame dumps, or replay directories;
- result-pack ZIP/TAR archives;
- machine-specific caches, absolute paths, credentials, or service tokens.

NyssaBench does not currently use Git LFS as a general artifact store. Put large
artifacts in durable external storage and include a stable URL, checksum,
license, source/provenance, generation command, package versions, and Git commit
in the PR or associated issue.

A small generated fixture may be committed only when it is required for an
automated test, deterministic, redistributable, stripped of private data, and
materially smaller than generating it during the test. Explain why a synthetic
fixture is insufficient.

Do not replace canonical published results with a convenient local rerun.
Result changes must include the exact command, seed protocol, validation status,
and replay evidence required by the benchmark tier.

## Pull Request Expectations

Open a draft while work or required simulator validation is incomplete. Mark it
ready when the diff is scoped, checks pass, and required artifacts are
available. A reviewable PR should:

- link the issue with `Closes #<number>` when it fully resolves it;
- explain what changed, why, and the user or research impact;
- identify public API, schema, CLI, dependency, and artifact-layout changes;
- list exact validation commands and pass/skip counts;
- distinguish synthetic, contract-only, simulator, and public-result evidence;
- include migration notes for compatibility changes;
- avoid unrelated files and generated binary artifacts.

Respond to review comments with a code or documentation update, or explain the
technical reason for keeping the current behavior. Resolve conversations only
after the corresponding commit is pushed.

## Contributor PR Checklist

- [ ] The change is scoped to one issue and follows repository boundaries.
- [ ] Tests cover the changed behavior and legacy compatibility where needed.
- [ ] `uv run pytest -q` passes.
- [ ] `uv run ruff check .` passes.
- [ ] Commit and pre-push pre-commit stages pass.
- [ ] Config and release checklist scripts pass.
- [ ] Required simulator integration ran, or the PR remains draft with a clear plan.
- [ ] Public APIs and versioned schemas remain compatible or include migration notes.
- [ ] Checkpoints, datasets, videos, and result archives are not committed.
- [ ] Documentation and examples match the implemented commands.
- [ ] The PR links its issue and reports exact validation results.
