# ADR-0001: Reuse adjacent evaluation patterns through Nyssa contracts

- Status: Accepted
- Date: 2026-09-05
- Decision owners: NyssaBench maintainers
- Scope: Evaluation architecture and external integration boundaries
- Related issues: #13, #14, #20, #25, #26, #27, #28, #34, #35, #36, #38, #39

## Context

Several open-source projects solve parts of the robot-policy evaluation problem.
NyssaBench should learn from those systems without turning into another
simulator, task platform, world generator, policy-training repository, or
failure-prediction method.

This record is based on repository code, not only papers or README claims. Each
source is pinned to the revision inspected on 2026-09-05. The decisions below
describe requirements for original NyssaBench interfaces; they do not copy
upstream implementation text.

## Decision

NyssaBench remains the simulator-independent measurement, failure-analysis,
recoverability, validity, and evidence layer. It will:

1. adopt per-step evaluator lifecycle requirements from damage and runtime
   monitoring systems;
2. consume controlled scenarios and stress conditions through versioned
   contracts with backend-confirmed application evidence;
3. keep environment construction, assets, robot definitions, and policy
   training in their upstream ecosystems;
4. compare simulation and real evidence through explicit paired manifests and
   rank/failure correspondence metrics;
5. integrate external monitors, policies, scenarios, and datasets through
   adapters or conformance fixtures instead of source copies or mandatory
   dependencies;
6. treat adaptive failure-boundary search as an evaluation technique, not the
   sole NyssaBench novelty claim.

The software boundary is:

```text
external simulator / task / policy / monitor / real evidence
                         |
                  versioned adapter
                         |
              Nyssa execution contracts
                         |
     stress -> temporal failure -> branch -> validity
                         |
               auditable result artifacts
```

## Inspected upstream snapshot

| Project | Repository and revision | License at revision | Inspected code paths | Nyssa role |
| --- | --- | --- | --- | --- |
| OopsieVerse | [`UT-Austin-RobIn/oopsieverse@a739f99`](https://github.com/UT-Austin-RobIn/oopsieverse/tree/a739f99e94f32c3229ae6932299e2e0e8c065480) | No repository-level license detected; GitHub license endpoint returned 404 | [`damagesim/core/evaluators/base.py`](https://github.com/UT-Austin-RobIn/oopsieverse/blob/a739f99e94f32c3229ae6932299e2e0e8c065480/damagesim/core/evaluators/base.py), [`damageable_mixin.py`](https://github.com/UT-Austin-RobIn/oopsieverse/blob/a739f99e94f32c3229ae6932299e2e0e8c065480/damagesim/core/damageable_mixin.py), [`damageable_env.py`](https://github.com/UT-Austin-RobIn/oopsieverse/blob/a739f99e94f32c3229ae6932299e2e0e8c065480/damagesim/core/damageable_env.py) | Per-step evaluator lifecycle and diagnostic evidence |
| RoboGate | Standalone `liveplex-cpu/robogate` returned 404; public code inspected at open [`IsaacLab-Arena#506@6bc4d50`](https://github.com/isaac-sim/IsaacLab-Arena/pull/506/commits/6bc4d501b64c60ec9815be65fed21417e1a7bcc0) | Open PR against Apache-2.0 Arena; unmerged contribution remains reference-only | [`scenarios.py`](https://github.com/isaac-sim/IsaacLab-Arena/blob/6bc4d501b64c60ec9815be65fed21417e1a7bcc0/isaaclab_arena/tasks/robogate_benchmark/scenarios.py), [`environments.py`](https://github.com/isaac-sim/IsaacLab-Arena/blob/6bc4d501b64c60ec9815be65fed21417e1a7bcc0/isaaclab_arena/tasks/robogate_benchmark/environments.py), [`run_benchmark.py`](https://github.com/isaac-sim/IsaacLab-Arena/blob/6bc4d501b64c60ec9815be65fed21417e1a7bcc0/isaaclab_arena/tasks/robogate_benchmark/scripts/run_benchmark.py), [`confidence_scorer.py`](https://github.com/isaac-sim/IsaacLab-Arena/blob/6bc4d501b64c60ec9815be65fed21417e1a7bcc0/isaaclab_arena/tasks/robogate_benchmark/confidence_scorer.py) | Scenario schema and failure-boundary research reference |
| SIMPLER | [`simpler-env/SimplerEnv@06accac`](https://github.com/simpler-env/SimplerEnv/tree/06accaca93535902d408da4855f21cece12bceb7) | MIT | [`evaluation/maniskill2_evaluator.py`](https://github.com/simpler-env/SimplerEnv/blob/06accaca93535902d408da4855f21cece12bceb7/simpler_env/evaluation/maniskill2_evaluator.py), [`utils/metrics.py`](https://github.com/simpler-env/SimplerEnv/blob/06accaca93535902d408da4855f21cece12bceb7/simpler_env/utils/metrics.py), [`main_inference.py`](https://github.com/simpler-env/SimplerEnv/blob/06accaca93535902d408da4855f21cece12bceb7/simpler_env/main_inference.py) | Paired sim-real study methodology |
| LIBERO-PRO | [`Zxy-MLlab/LIBERO-PRO@eafdb80`](https://github.com/Zxy-MLlab/LIBERO-PRO/tree/eafdb809426b13153aa1e4c42d6601844217dfec) | MIT | [`perturbation.py`](https://github.com/Zxy-MLlab/LIBERO-PRO/blob/eafdb809426b13153aa1e4c42d6601844217dfec/perturbation.py), [`evaluation_config.yaml`](https://github.com/Zxy-MLlab/LIBERO-PRO/blob/eafdb809426b13153aa1e4c42d6601844217dfec/evaluation_config.yaml), [`benchmark_scripts/`](https://github.com/Zxy-MLlab/LIBERO-PRO/tree/eafdb809426b13153aa1e4c42d6601844217dfec/benchmark_scripts) | Controlled and composed perturbation reference |
| FIPER | [`learnsyslab/fiper@13d79c5`](https://github.com/learnsyslab/fiper/tree/13d79c5c3069def843e454787ff128defc249838) (`utiasDSL/fiper` redirects here) | MIT | [`evaluation/evaluation_manager.py`](https://github.com/learnsyslab/fiper/blob/13d79c5c3069def843e454787ff128defc249838/evaluation/evaluation_manager.py), [`method_eval_classes/base_eval_class.py`](https://github.com/learnsyslab/fiper/blob/13d79c5c3069def843e454787ff128defc249838/evaluation/method_eval_classes/base_eval_class.py), [`results_manager.py`](https://github.com/learnsyslab/fiper/blob/13d79c5c3069def843e454787ff128defc249838/evaluation/results_manager.py), [`datasets/`](https://github.com/learnsyslab/fiper/tree/13d79c5c3069def843e454787ff128defc249838/datasets) | External runtime monitor and temporal metric reference |
| Sentinel | [`agiachris/sentinel@6dc89ca`](https://github.com/agiachris/sentinel/tree/6dc89ca4cb30f75ce834811c2d1b31f7ffadf7b5) | MIT | [`scripts/eval_detector/`](https://github.com/agiachris/sentinel/tree/6dc89ca4cb30f75ce834811c2d1b31f7ffadf7b5/scripts/eval_detector), [`scripts/data_generation/`](https://github.com/agiachris/sentinel/tree/6dc89ca4cb30f75ce834811c2d1b31f7ffadf7b5/scripts/data_generation), [`sentinel/bc/`](https://github.com/agiachris/sentinel/tree/6dc89ca4cb30f75ce834811c2d1b31f7ffadf7b5/sentinel/bc) | External consistency/progress monitor reference |
| IsaacLab-Arena | [`isaac-sim/IsaacLab-Arena@af3f24b`](https://github.com/isaac-sim/IsaacLab-Arena/tree/af3f24b054879a3886e80447772cd27e4bf208f1) | Apache-2.0 (`LICENSE.md`; GitHub reports `NOASSERTION`) | [`evaluation/experiment_runner.py`](https://github.com/isaac-sim/IsaacLab-Arena/blob/af3f24b054879a3886e80447772cd27e4bf208f1/isaaclab_arena/evaluation/experiment_runner.py), [`evaluation/policy_runner.py`](https://github.com/isaac-sim/IsaacLab-Arena/blob/af3f24b054879a3886e80447772cd27e4bf208f1/isaaclab_arena/evaluation/policy_runner.py), [`environment_spec/`](https://github.com/isaac-sim/IsaacLab-Arena/tree/af3f24b054879a3886e80447772cd27e4bf208f1/isaaclab_arena/environment_spec), [`scene/`](https://github.com/isaac-sim/IsaacLab-Arena/tree/af3f24b054879a3886e80447772cd27e4bf208f1/isaaclab_arena/scene), [`embodiments/`](https://github.com/isaac-sim/IsaacLab-Arena/tree/af3f24b054879a3886e80447772cd27e4bf208f1/isaaclab_arena/embodiments), [`metrics/`](https://github.com/isaac-sim/IsaacLab-Arena/tree/af3f24b054879a3886e80447772cd27e4bf208f1/isaaclab_arena/metrics) | Orchestration separation and environment composition reference |
| RoboVerse | [`RoboVerseOrg/RoboVerse@6fbafcf`](https://github.com/RoboVerseOrg/RoboVerse/tree/6fbafcff73d77d5c916b50c92614102a43244602) | Apache-2.0; third-party notices and component licenses also apply | [`packages/metasim/metasim/scenario/`](https://github.com/RoboVerseOrg/RoboVerse/tree/6fbafcff73d77d5c916b50c92614102a43244602/packages/metasim/metasim/scenario), [`sim/`](https://github.com/RoboVerseOrg/RoboVerse/tree/6fbafcff73d77d5c916b50c92614102a43244602/packages/metasim/metasim/sim), [`task/`](https://github.com/RoboVerseOrg/RoboVerse/tree/6fbafcff73d77d5c916b50c92614102a43244602/packages/metasim/metasim/task), [`randomization/`](https://github.com/RoboVerseOrg/RoboVerse/tree/6fbafcff73d77d5c916b50c92614102a43244602/packages/metasim/metasim/randomization), [`THIRD_PARTY_NOTICES.md`](https://github.com/RoboVerseOrg/RoboVerse/blob/6fbafcff73d77d5c916b50c92614102a43244602/THIRD_PARTY_NOTICES.md) | External unified simulation/task/data platform |
| RoboTwin | [`RoboTwin-Platform/RoboTwin@96c1fea`](https://github.com/RoboTwin-Platform/RoboTwin/tree/96c1feab536306b50c26af200044fcdf126e8904) | MIT; submodule and asset terms must be checked separately | [`envs/`](https://github.com/RoboTwin-Platform/RoboTwin/tree/96c1feab536306b50c26af200044fcdf126e8904/envs), [`env_cfg/`](https://github.com/RoboTwin-Platform/RoboTwin/tree/96c1feab536306b50c26af200044fcdf126e8904/env_cfg), [`collect_data.sh`](https://github.com/RoboTwin-Platform/RoboTwin/blob/96c1feab536306b50c26af200044fcdf126e8904/collect_data.sh), [`assets/`](https://github.com/RoboTwin-Platform/RoboTwin/tree/96c1feab536306b50c26af200044fcdf126e8904/assets), [`XPolicyLab`](https://github.com/RoboTwin-Platform/RoboTwin/tree/96c1feab536306b50c26af200044fcdf126e8904/XPolicyLab) | External task, asset, data-generation, and policy ecosystem |
| RoboCasa | [`robocasa/robocasa@4f8a298`](https://github.com/robocasa/robocasa/tree/4f8a2980def75a55dff96b990745b83540425f09) | MIT for primary code; `LICENSE` identifies partial MuJoCo code under Apache-2.0 | [`robocasa/environments/`](https://github.com/robocasa/robocasa/tree/4f8a2980def75a55dff96b990745b83540425f09/robocasa/environments), [`models/`](https://github.com/robocasa/robocasa/tree/4f8a2980def75a55dff96b990745b83540425f09/robocasa/models), [`demos/`](https://github.com/robocasa/robocasa/tree/4f8a2980def75a55dff96b990745b83540425f09/robocasa/demos), [`wrappers/`](https://github.com/robocasa/robocasa/tree/4f8a2980def75a55dff96b990745b83540425f09/robocasa/wrappers), [`LICENSE`](https://github.com/robocasa/robocasa/blob/4f8a2980def75a55dff96b990745b83540425f09/LICENSE) | External household task, scene, asset, and demonstration ecosystem |

## Project decisions

### OopsieVerse: evaluator lifecycle, not a health model

The `DamageEvaluator` base exposes one signal-specific evaluator contract, while
`DamageableMixin` owns evaluator registration, reset, per-step execution, and
evaluator-specific diagnostic records. Backend-specific code supplies simulator
state.

NyssaBench adopts the lifecycle requirement:

```text
construct detectors -> validate support -> reset -> observe each transition
                    -> emit evidence -> finalize -> retain diagnostics
```

This maps to `nyssa_bench/failures/detectors/` from #34 and the temporal
`FailureEvent`/ledger contracts from #14. Nyssa does not adopt per-object health,
damage subtraction, simulator-specific multiple inheritance, or damage as the
universal failure representation. Damage evaluators can integrate later by
emitting external or simulator-state FailureEvents.

Because the inspected OopsieVerse revision has no detectable repository-level
license, no implementation may be copied or vendored. The lifecycle above is an
independently expressed interface requirement.

### RoboGate: scenario identity and boundary search, with execution proof

The public PR defines scenario category/variant/parameter records and a
benchmark report. Code inspection also exposes why Nyssa requires applied-state
evidence: the PR's `RoboGateValidationTask.get_events_cfg()` returns `None`, the
Arena loop does not apply `scenario.params`, and the loop samples random actions.
Those facts do not invalidate the research question, but they mean the PR cannot
serve as executable evidence that every declared scenario was realized.

NyssaBench adopts scenario identity, parameter domains, deterministic seeds, and
held-out boundary confirmation as requirements. This maps to the Stressor
Contract from #13 and `nyssa_bench/stress_search/` from #36. Every condition must
record backend-confirmed parameters; declaration alone is insufficient.

Nyssa does not adopt a fixed weighted deployment-confidence scalar. Boundary
search remains one sampler family and is explicitly excluded as the sole novelty
claim. The scientific contribution is the connected protocol from controlled
stress through temporal failure and counterfactual recoverability.

### SIMPLER: paired predictive validity

SIMPLER's evaluator varies task setup and visual matching while retaining policy
rollouts. `simpler_env/utils/metrics.py` pairs simulator and real policy
performance and implements Pearson correlation and mean maximum rank violation.

NyssaBench adopts explicit real/sim pairing, policy/task rank correspondence,
rank-violation metrics, and condition aggregation. This maps to the real-evidence
contract from #27, the hardware calibration track in #20, and `SimRealStudy` in
#38.

Nyssa extends the analysis target beyond terminal success to failure-event
distributions, temporal signatures, perturbation response, censoring, and
recovery effects. It will not copy SIMPLER's hard-coded performance tables or
make SIMPLER a core dependency. A converter should emit Nyssa real/sim manifests.

### LIBERO-PRO: controlled perturbation categories and composition

LIBERO-PRO's `perturbation.py` implements object, spatial, language, task, and
environment changes and a combined perturbator with explicit ordering and
mutual-exclusion rules.

NyssaBench adopts named perturbation categories, explicit composition order,
incompatibility checks, deterministic seeds, and separate generated conditions.
These requirements map to #13 and `nyssa_bench/stressors/`.

Nyssa does not adopt direct BDDL regex rewriting, process-global RNG mutation,
hard-coded environment replacement, or a LIBERO-only perturbation interface.
LIBERO-PRO conditions should enter through an adapter or conformance fixture
that produces a Stressor/Scenario manifest and reports what the backend applied.

### FIPER and Sentinel: external methods, common measurement

FIPER separates method evaluators, evaluation management, result management,
and rollout datasets. Sentinel provides data generation and detector-evaluation
scripts for consistency/progress and OOD monitors across simulated and real
settings. Both are method repositories, not generic benchmark contracts.

NyssaBench adopts the measurement requirements: timestamped risk predictions,
lead time, calibration/classification behavior, runtime cost, evidence identity,
and identical-episode comparison. These map to the external monitor contract in
#28 and FailureEvent evidence boundaries in #34/#14.

Nyssa does not train FIPER/Sentinel methods, own activation or representation
research, or import their task stacks. A method integrates as an external
`FailureMonitor` module with a checkpoint hash and declared observable or
privileged inputs. Its prediction quality remains separate from the recovery
policy's counterfactual effect.

### IsaacLab-Arena: orchestration separation, not environment ownership

Arena has distinct environment specifications, scenes, embodiments, tasks,
policies, metrics, experiment runs, experiment results, and policy runners.
That separation supports Nyssa's runner refactor in #35.

NyssaBench adopts distinct experiment, episode, transition, metric, and artifact
lifecycles. It does not import Arena's scene graph, asset registry, embodiment,
controller, environment-generation, or task ownership. Arena environments may
be exposed through an engine adapter; Arena policies may use policy adapters;
generated Arena scenarios may use #26's content-addressed scenario package.

### RoboVerse, RoboTwin, and RoboCasa: consume ecosystems

These repositories already own broad combinations of simulator backends, robot
and object assets, tasks, environment configuration, data generation,
demonstrations, and policy tooling. Their code structure is direct evidence that
replicating those responsibilities would turn NyssaBench into another robotics
platform.

NyssaBench therefore consumes their stable outputs:

- task and success contracts through engine/task adapters;
- scenarios and randomization through #26 and the Stressor Contract;
- demonstrations and rollout datasets through import/export adapters;
- policy checkpoints through the Policy Contract;
- upstream asset, dataset, and submodule licenses as explicit provenance.

Nyssa will not mirror asset trees, fork task suites into its source tree, own
their data-generation scripts, or absorb their learning stacks. A small
conformance fixture is preferred when it can verify the boundary without making
the upstream project a mandatory dependency.

## Integration policy

| Situation | Preferred mechanism | Direct dependency allowed? |
| --- | --- | --- |
| External simulator or task suite | Optional engine/task adapter and conformance fixture | Optional extra only when runtime import is required |
| Generated scenario or perturbation pack | #26 scenario package plus Stressor manifest | No |
| Real or reconstructed trial | #27 real-evidence package | No |
| External failure predictor | #28 `FailureMonitor` module | No; method environment remains external |
| External learned policy | Policy adapter and checkpoint contract | Optional policy extra, never base install |
| External demonstration data | Import/export converter with source license and hashes | No |
| Upstream metric definition | Original Nyssa implementation with cited semantics and reference tests | No source copy by default |

Any proposal to vendor or copy upstream code requires a separate license review,
attribution plan, compatibility rationale, and ADR update. Repository-level
licenses do not automatically cover downloaded assets, datasets, model weights,
submodules, or third-party simulator components.

## Consequences

Positive consequences:

- NyssaBench can compare systems from different ecosystems without replacing
  those ecosystems.
- Upstream execution claims are checked through applied evidence rather than
  trusted from configuration names.
- Failure monitors and learning methods remain independent research artifacts.
- Sim-real work reuses established rank methodology while extending it to
  failure profiles and recoverability.
- License and provenance boundaries remain visible at every adapter.

Costs and constraints:

- Adapters must translate upstream identities into Nyssa contracts and may lag
  upstream API changes.
- Conformance fixtures are required to prevent semantic drift.
- Some integrations remain optional because GPU, simulator, dataset, or license
  requirements cannot be moved into the core package.
- Revision pins in this ADR are historical evidence, not automatic endorsements
  of later upstream changes.

## Rejected alternatives

**Fork one adjacent framework.** Rejected because no inspected project combines
Nyssa's intended stress, temporal failure, counterfactual recovery, validity,
and portable evidence contracts, and a fork would inherit unrelated platform
scope.

**Build a universal simulator/task platform.** Rejected because Arena,
RoboVerse, RoboTwin, RoboCasa, ManiSkill, and other ecosystems already own that
layer.

**Make failure-boundary search the headline identity.** Rejected because the
RoboGate research question already occupies that space and because search alone
does not standardize temporal evidence or causal recoverability.

**Embed failure-prediction methods.** Rejected because FIPER and Sentinel show
that method development has separate models, datasets, and research assumptions.
Nyssa evaluates those methods through #28.

**Copy perturbation code into each engine.** Rejected because it loses common
identity, composition, severity, support, and applied-state semantics.

## Review triggers

Revisit this ADR when:

- an upstream project publishes a stable protocol that subsumes a Nyssa
  contract;
- a direct dependency is proposed instead of an adapter;
- source or asset code is proposed for vendoring;
- Nyssa begins owning scene generation, robot definitions, or policy training;
- #38's sim-real study requires a pairing primitive not covered here;
- an upstream license or canonical repository changes.
