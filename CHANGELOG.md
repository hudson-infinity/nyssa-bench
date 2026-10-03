# Changelog

## [0.0.1](https://github.com/hudson-infinity/nyssa-bench/compare/v0.0.1...v0.0.1) (2026-10-03)


### Features

* **ci:** automate dependency PRs and version releases ([#91](https://github.com/hudson-infinity/nyssa-bench/issues/91)) ([f95a3f0](https://github.com/hudson-infinity/nyssa-bench/commit/f95a3f09498206b42c2161b6258ef4973ffe9997))


### Bug Fixes

* **ci:** verify individual Dependabot status records ([#97](https://github.com/hudson-infinity/nyssa-bench/issues/97)) ([89ea7f0](https://github.com/hudson-infinity/nyssa-bench/commit/89ea7f07c6534e390f4650edb9784df459c25727))
* **deps:** bump aiohttp from 3.14.1 to 3.14.3 ([#101](https://github.com/hudson-infinity/nyssa-bench/issues/101)) ([d51cbfe](https://github.com/hudson-infinity/nyssa-bench/commit/d51cbfe84801e882efa4a25adafbfe9d6442f8eb))
* **deps:** bump anyio from 4.14.1 to 4.14.2 ([#99](https://github.com/hudson-infinity/nyssa-bench/issues/99)) ([37e9b07](https://github.com/hudson-infinity/nyssa-bench/commit/37e9b072bac6bb5d9bab0e6900b8374146147245))
* **deps:** bump gitpython from 3.1.50 to 3.1.59 ([#98](https://github.com/hudson-infinity/nyssa-bench/issues/98)) ([88035ee](https://github.com/hudson-infinity/nyssa-bench/commit/88035ee628bd5dbae4962085768bdef1dd03649b))
* **deps:** bump gitpython from 3.1.59 to 3.1.62 ([#109](https://github.com/hudson-infinity/nyssa-bench/issues/109)) ([309ef34](https://github.com/hudson-infinity/nyssa-bench/commit/309ef34c62c0253ceed677fd9df61856519930f1))
* **deps:** bump gymnasium from 0.29.1 to 1.3.0 ([#107](https://github.com/hudson-infinity/nyssa-bench/issues/107)) ([a4b28e0](https://github.com/hudson-infinity/nyssa-bench/commit/a4b28e0bb039ed80e8efc3e32420b5578381c4f6))
* **deps:** bump numpy from 1.26.4 to 2.2.6 ([#93](https://github.com/hudson-infinity/nyssa-bench/issues/93)) ([904249d](https://github.com/hudson-infinity/nyssa-bench/commit/904249d0664a2f882bcbefe8553316ba4c0728d5))
* **deps:** bump plotly from 6.8.0 to 7.1.0 ([#96](https://github.com/hudson-infinity/nyssa-bench/issues/96)) ([17cafc2](https://github.com/hudson-infinity/nyssa-bench/commit/17cafc21e29ca23d9dcb27f7448b3b3fa8e179b3))
* **deps:** bump pyarrow from 24.0.0 to 25.0.1 ([#95](https://github.com/hudson-infinity/nyssa-bench/issues/95)) ([7918758](https://github.com/hudson-infinity/nyssa-bench/commit/79187589b7408800e6a030c9cda888da4ab03576))
* **deps:** bump setuptools from 82.0.1 to 83.0.0 ([#100](https://github.com/hudson-infinity/nyssa-bench/issues/100)) ([c39ce2f](https://github.com/hudson-infinity/nyssa-bench/commit/c39ce2f018242b49966f3c512c3286b2c884379a))
* **deps:** bump the python-patch-minor group across 1 directory with 5 updates ([#108](https://github.com/hudson-infinity/nyssa-bench/issues/108)) ([7e4e8b9](https://github.com/hudson-infinity/nyssa-bench/commit/7e4e8b90e0fef8ce1115f877917b20d935f0c249))
* **deps:** bump the python-patch-minor group with 11 updates ([#92](https://github.com/hudson-infinity/nyssa-bench/issues/92)) ([1158ed0](https://github.com/hudson-infinity/nyssa-bench/commit/1158ed0824e51ec50651b482e4dbb63985b561bd))
* **deps:** bump types-pyyaml from 6.0.12.20260518 to 6.0.12.20260906 ([#94](https://github.com/hudson-infinity/nyssa-bench/issues/94)) ([2c77700](https://github.com/hudson-infinity/nyssa-bench/commit/2c77700bd621876b0d133f09b41c7c2f2345443c))
* **deps:** bump urllib3 from 2.7.0 to 2.8.0 ([#111](https://github.com/hudson-infinity/nyssa-bench/issues/111)) ([c64d2c7](https://github.com/hudson-infinity/nyssa-bench/commit/c64d2c730ee2f065872d4cfeea3eac2657457beb))
* **deps:** bump virtualenv from 21.5.1 to 21.7.13 ([#110](https://github.com/hudson-infinity/nyssa-bench/issues/110)) ([3e0c8fb](https://github.com/hudson-infinity/nyssa-bench/commit/3e0c8fb25fe45eda6d3ed760cc1eca6f9c6cde40))
* **release:** start PyPI releases at 0.0.1 ([#104](https://github.com/hudson-infinity/nyssa-bench/issues/104)) ([0837733](https://github.com/hudson-infinity/nyssa-bench/commit/0837733e9e3769e8015c7db9fbc5bec5a7047df9))
* **release:** verify published package installations ([#103](https://github.com/hudson-infinity/nyssa-bench/issues/103)) ([12dea88](https://github.com/hudson-infinity/nyssa-bench/commit/12dea882d392ad55df76ead2992692ee3ccb9bbe))

## 0.0.1rc1

First release candidate for TestPyPI qualification. The first stable PyPI
version will be 0.0.1. This package release does not promote missing learned
policy, GPU simulator, or real-world benchmark evidence to validated status.

- Exclude local uv build caches from source distributions and reject cache
  contents during release validation, including inside example checkpoints.

- Set the first PyPI release to 0.0.1 and pin the initial release PR to that
  version; NEP retains its independent 0.1.0 protocol version.
- Add weekly dependency bots, automatic PR labels, conventional-title release
  PRs, and CI-gated squash merging with explicit downstream workflow dispatch.
- Require applicable container checks, automation tests, and workflow linting
  through a single aggregate CI gate; retain protected package publication.
- Fix episode-source omission/duplication and avoid traversing per-task copies
  beneath aggregate exports; decode BC checkpoints once per load.
- Preserve repeated episode identities in HDF5, reject non-finite RoboMimic
  features, and compute observation variance without cancellation.
- Propagate malformed checkpoint and prediction errors instead of silently
  substituting zero actions.
- Support dataclasses in file-loaded policies, experts, and monitors, Windows
  object references, zero-argument factories, and parameterless model resets.
- Keep file-policy experiment outputs inside the selected directory and reject
  equivalent output paths before executing any run.
- Repair the result-archive hook's ZIP filename filter.
- Align ManiSkill and learned-policy extras with the validated PyTorch 2.6.0,
  torchvision 0.21.0, CUDA 12.4 runtime instead of resolving an unbounded CUDA
  dependency on fresh installations.
- Avoid duplicate branch push CI, cancel superseded container runs, and cache
  simulator dependencies independently from the changing NyssaBench wheel.
- Pin and validate a homogeneous 20/20 retained MuJoCo simulator-CI baseline.
- Add a hardware calibration preregistration contract with factorial conditions,
  safety/governance, recovery arms, evidence completeness, and claim gating.
- Add a checkpoint-bound policy-track registry and native conformance,
  provenance, leakage, paired-run, replay, and failure-evidence audit.
- Add a strict 12-task reference benchmark candidate, protected five-dimension
  split commitments, power design, and native oracle/learned evidence audit.
- Publish wheel-only core, MuJoCo, and ManiSkill image workflows with immutable
  tags, installed-command smoke tests, SBOM/provenance, and release bundles.
- Establish the single-distribution PyPI contract, complete simulator replay
  extras, package metadata, version checks, and OIDC trusted-publishing workflow.
- Validate wheel and source-distribution contents, packaged resource identity,
  external working-directory execution, and installed simulator workflows.
- Add real installed-wheel MuJoCo state/stressor/result checks, an explicit GPU
  ManiSkill path, retained failure diagnostics, and simulator flakiness policy.
- Add Nyssa Evaluation Protocol 0.1 with six strict contracts, canonical hashes,
  claim-tier checks, schemas, migration rules, and conformance fixtures.
- Add external policy preflight, state-leak and action-chunk validation,
  machine/HTML conformance reports, and packaged state/RGB examples.
- Add versioned paired sim-real studies with rank, failure, shift, censoring,
  recovery, clustered uncertainty, and held-out predictive analyses.
- Add a content-addressed Phase 1 credibility gate spanning the measurement
  core, reference benchmark evidence, and predictive sim-real validity.
- Add pinned pre-commit and pre-push hooks for structural checks, Ruff, config
  validation, release files, and the full test suite.
- Bound numeric observation flattening to the requested feature size and avoid
  materializing full image tensors in Python.
- Vectorize paired robustness bootstraps, group episode summaries in one pass,
  and reuse stressor/episode serialization metadata in high-frequency paths.
- Normalize bounded RoboMimic actions to `[-1, 1]`, persist per-task action
  transforms, and validate live inference bounds.
- Record training episode seeds and reject task-policy evaluation leakage by
  default.
- Namespace simulator episode seeds by run seed so multi-seed experiments are
  disjoint while task comparisons remain paired.
- Make recovery-only ablations activate independently from verifier fallback
  intervention.
- Require real MP4 files, simulator versions, a commit, and a clean worktree for
  public run validation.
- Execute ManiSkill collection templates with the active interpreter and
  shell-safe placeholders.

## Initial development scaffold (unreleased)

- Initial benchmark harness scaffold.
- Dummy engine, ManiSkill/MuJoCo adapter boundaries, and experimental RoboCasa/Genesis boundaries.
- Task YAML specs, suite loading, policy adapters, metrics, reports, dataset export, comparison reports, and leaderboard export.
