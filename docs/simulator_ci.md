# Simulator-backed continuous integration

NyssaBench separates fast pull-request checks from real simulator evidence.

## Pull requests

Pull-request CI runs source tests, builds the package, inspects the archives,
and installs the wheel on Python 3.10 and 3.13. Its deterministic internal
engine checks lifecycle and artifact contracts. It does not claim that ManiSkill
or MuJoCo executed in those jobs.

## Scheduled MuJoCo

`.github/workflows/installed-simulators.yml` runs every Monday and can also be
dispatched manually. The job:

1. builds and validates the wheel;
2. installs that wheel with its declared `mujoco` extra;
3. runs `pip check` and records the complete installed package set;
4. loads InvertedPendulum through the real Gymnasium/MuJoCo backend;
5. checks three independently seeded reset, step, snapshot, and restore cycles;
6. verifies the proposed action shape and bounds;
7. executes two namespaced evaluation episodes with action noise;
8. checks stressor application and the result-pack contract;
9. uploads the result pack and `simulator_smoke.json`, including failures.

The smoke records NyssaBench, Python, simulator package, platform, and rendering
backend versions. MuJoCo uses OSMesa on the hosted Linux runner and does not
claim replay evidence in this lightweight scheduled job.

## GPU ManiSkill

The `maniskill-gpu` job uses `[self-hosted, linux, x64, gpu]` and runs only when
`workflow_dispatch` explicitly requests it. It installs the built wheel with the
`maniskill` extra, checks dependencies, imports the real package, performs the
same state/action/stressor/seed checks, and requires one MP4 per smoke episode.

No hosted CPU fallback is allowed. If Hudson does not have a compatible GPU
runner online, the job remains unexecuted and the claim matrix continues to mark
simulator-backed ManiSkill CI as planned.

## Failure diagnostics

The smoke module writes `simulator_smoke.json` before re-raising an execution
error. The artifact records the exception type, message, traceback, package
versions, and rendering environment. The workflow uploads the output directory
with `if: always()` and retains the normal result pack on success.

## Flakiness policy

Simulator jobs begin as non-required scheduled evidence. Do not make one a
merge gate until it has at least 20 consecutive completed runs on a pinned
runner class with at least 95% infrastructure-pass rate. Policy/task failures
must remain distinct from dependency, driver, renderer, timeout, and runner
failures. A changed simulator, driver, rendering backend, task, or runner class
starts a new baseline window.

The workflow does not use partial `uv sync` commands. It installs one built
wheel with one complete extra, runs `pip check`, and records `pip freeze`, which
makes accidental dependency removal or undeclared runtime imports visible.
