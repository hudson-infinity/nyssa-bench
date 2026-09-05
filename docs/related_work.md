# Related Work

The code-level ownership and integration decisions behind this landscape are
recorded in
[ADR-0001: Reuse adjacent evaluation patterns through Nyssa contracts](architecture_decisions/0001-adjacent-evaluation-frameworks.md).

NyssaBench is not intended to replace robot simulators, robot datasets, or task
zoos. Its role is to sit above them as a failure-aware evaluation, validation,
reporting, and comparison layer for embodied AI policies.

## P0 Evaluation Systems

RoboArena treats robot-policy evaluation itself as a frontier problem. It uses
distributed real-world evaluation and double-blind pairwise comparisons across
evaluators and institutions. NyssaBench should take the pairwise and blinded
comparison model, but apply it first to simulation and failure debugging before
real-world deployment. Source: https://arxiv.org/abs/2506.18123

SIMPLER evaluates real-world manipulation policies in simulation and reports
paired sim-and-real evidence. Its lesson for NyssaBench is that simulator
results become stronger when they are tested for correlation with real robot
behavior, not merely reported as standalone numbers. Source:
https://arxiv.org/abs/2405.05941

PolaRiS reconstructs real scenes from short video scans into interactive
simulation environments for policy evaluation. It points toward a future
NyssaBench real-to-sim path where real robot logs or scene captures become
repeatable evaluation cases. Source: https://arxiv.org/abs/2512.16881

RobotArena Infinity converts robot videos into simulated counterparts and uses
automated and human preference scoring. It supports NyssaBench's long-term
direction toward real-to-sim, perturbation-based, preference-compatible
evaluation. Source: https://arxiv.org/abs/2510.23571

## P0 Robustness and Benchmark Audits

What Are We Actually Benchmarking in Robot Manipulation? identifies shortcut
solvability, weak statistics, creeping overfitting, and data-source dependence
as ways benchmark scores can stop measuring real capability. NyssaBench should
turn these concerns into validation checks, audit notes, and scorecard caveats.
Source: https://arxiv.org/abs/2606.04233

LIBERO-PRO shows that high standard benchmark performance can collapse under
reasonable perturbations. This supports NyssaBench's emphasis on stressors,
generalization checks, and anti-overclaim validation. Source:
https://arxiv.org/abs/2510.03827

LIBERO-Plus studies VLA robustness under perturbations to layout, camera,
initial robot state, language, lighting, textures, and sensor noise. It
motivates NyssaBench stressor dimensions beyond seed-only randomization. Source:
https://arxiv.org/abs/2510.13626

Eva-VLA frames physical robustness evaluation as worst-case search over object
transformations, illumination changes, and adversarial patches. It motivates a
future NyssaBench stress-search mode rather than only uniform random sampling.
Source: https://arxiv.org/abs/2509.18953

ROBOGATE studies adaptive failure discovery through large simulation sweeps and
boundary-focused sampling. This is closely aligned with NyssaBench's planned
failure-region discovery workflows. Source: https://arxiv.org/abs/2603.22126

The SO-101 failure and recovery benchmark emphasizes failure taxonomy and
recovery-aware metrics on low-cost real robots. This supports NyssaBench's
decision to report failure labels, failure sources, and recovery metrics instead
of only binary success. Source: https://arxiv.org/abs/2606.08881

## P1 Simulator and Manipulation Benchmarks

ManiSkill, RoboCasa, RoboCasa365, RoboTwin, RoboVerse, RLBench, Meta-World,
robosuite, CALVIN, LIBERO, and VIMA are best treated as backends, datasets, or
task ecosystems. NyssaBench should not compete with them as a simulator. It
should provide:

- result validation
- replay-first evidence requirements
- failure taxonomy
- stress-condition reporting
- policy comparison
- dataset export from evaluated episodes
- result tiers and publication caveats

Representative sources:

- ManiSkill2: https://arxiv.org/abs/2302.04659
- RoboCasa: https://arxiv.org/abs/2406.02523
- RoboCasa365: https://arxiv.org/abs/2603.04356
- RoboTwin 2.0: https://arxiv.org/abs/2506.18088
- RoboVerse: https://arxiv.org/abs/2504.18904
- RLBench: https://arxiv.org/abs/1909.12271
- Meta-World: https://arxiv.org/abs/1910.10897
- robosuite: https://arxiv.org/abs/2009.12293
- CALVIN: https://arxiv.org/abs/2112.03227
- LIBERO: https://arxiv.org/abs/2306.03310
- VIMA: https://arxiv.org/abs/2210.03094

## P1 Embodied AI Environments

BEHAVIOR-1K and OmniGibson, AI2-THOR, ProcTHOR, Habitat, and iGibson matter if
NyssaBench expands beyond tabletop manipulation into household, mobile, or
long-horizon embodied AI. They are not v0.1 priorities.

Representative sources:

- BEHAVIOR-1K: https://arxiv.org/abs/2403.09227
- AI2-THOR: https://arxiv.org/abs/1712.05474
- ProcTHOR: https://arxiv.org/abs/2206.06994
- Habitat: https://arxiv.org/abs/1904.01201
- iGibson: https://arxiv.org/abs/2012.02924

## P1 Datasets and Policy Stacks

Open X-Embodiment, DROID, BridgeData V2, RH20T, AgiBot World, LeRobot,
OpenVLA, Octo, SmolVLA, pi0, RDT, GR00T N1, Diffusion Policy, and MimicGen
define the policy/data landscape NyssaBench should evaluate against later.

As of the July 2026 open-source landscape, the highest-leverage integrations are
not new hand-written learners. They are compatibility paths to existing stacks:

- ManiSkill official baselines for PPO, SAC, TD-MPC2, behavior cloning, diffusion
  policy, and VLA policies, with GPU-parallel simulation and public result logs.
  NyssaBench should treat these as strong upstream baselines and add failure,
  stressor, replay, and public-claim validation around them.
- RoboMimic for low-dimensional and visual imitation learning baselines such as
  BC, BC-RNN, BC-Transformer, BC-GMM, and Diffusion Policy. NyssaBench should
  keep exporting clean HDF5 datasets and loading trained checkpoints rather than
  duplicating RoboMimic training internals.
- LeRobot for modern ACT, diffusion, VQ-BeT, multitask DiT, VLA, reward-model,
  and real-robot dataset workflows. NyssaBench should support LeRobot-format
  export/import and evaluate LeRobot policies through a stable adapter.
- LIBERO, Meta-World, RLBench, RoboCasa365, and Open X-Embodiment as task and
  dataset ecosystems that already define scale, diversity, transfer splits, or
  real-robot data mixtures. NyssaBench should align its result tiers, suite
  metadata, and adapter contracts with these ecosystems.

Immediate baseline order should stay simple:

1. random policy as a weak sanity check
2. scripted or planner-backed policy as a task sanity check
3. behavior cloning, robomimic, or diffusion policy as the first learned result
4. VLA/generalist policies only after the harness has credible video-backed
   result packs

Representative sources:

- Open X-Embodiment: https://arxiv.org/abs/2310.08864
- DROID: https://arxiv.org/abs/2403.12945
- BridgeData V2: https://arxiv.org/abs/2308.12952
- RH20T: https://arxiv.org/abs/2307.00595
- AgiBot World Colosseo: https://arxiv.org/abs/2503.06669
- LeRobot: https://arxiv.org/abs/2602.22818
- OpenVLA: https://arxiv.org/abs/2406.09246
- Octo: https://arxiv.org/abs/2405.12213
- SmolVLA: https://arxiv.org/abs/2506.01844
- pi0: https://arxiv.org/abs/2410.24164
- RDT-1B: https://arxiv.org/abs/2410.07864
- GR00T N1: https://arxiv.org/abs/2503.14734
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- MimicGen: https://arxiv.org/abs/2310.17596

## P1/P2 Real-to-Sim and World-Model Evaluation

GSWorld, Splatting Physical Scenes, SoMA, GE-Sim, RoboWorld, and NVIDIA Cosmos
represent the future frontier where policies are evaluated across photorealistic
real-to-sim reconstructions and neural/world-model simulators. NyssaBench should
not compete with these systems directly. It should compare which evaluation
signals best predict real-world failures.

Representative sources:

- GSWorld: https://arxiv.org/abs/2510.20813
- Splatting Physical Scenes: https://arxiv.org/abs/2506.04120
- SoMA: https://arxiv.org/abs/2602.02402
- GE-Sim 2.0: https://arxiv.org/abs/2605.27491
- RoboWorld: https://arxiv.org/abs/2607.01060
- Cosmos: https://arxiv.org/abs/2501.03575

## P2 Low-Cost and Real-World Collection

REPLAB, UMI, and UMI-FT matter for accessible real-world evaluation and future
low-cost robot support. They motivate future import/export paths for real logs,
force/contact-aware metrics, and SO-101 or LeRobot hardware workflows.

Representative sources:

- REPLAB: https://arxiv.org/abs/1905.07447
- UMI: https://arxiv.org/abs/2402.10329
- UMI-FT: https://arxiv.org/abs/2601.09988

## Positioning

NyssaBench's lane is:

> Open-source infrastructure for failure-aware, stress-tested,
> pairwise-compatible evaluation of embodied AI policies.

RoboArena ranks policies in the real world. SIMPLER, PolaRiS, and RobotArena
Infinity validate sim and real-to-sim evaluation. ManiSkill, RoboCasa,
RoboTwin, and RoboVerse provide simulation and task ecosystems. NyssaBench
should be the reliability measurement layer across those systems.
