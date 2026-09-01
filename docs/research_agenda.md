# Research Agenda

NyssaBench's north star:

> NyssaBench is foundational because it standardizes embodied AI evaluation. It
> is frontier because it evaluates the hardest current robot-policy failures:
> robustness, recoverability, VLA reliability, and predictive sim-real validity.

NyssaBench provides the measurement system for this agenda. World generation,
real-to-sim reconstruction, interpretability-method development, policy
learning, and hosted products remain separate projects that can provide inputs
or consume evidence through versioned interfaces.

## First Paper Framing

Working title:

> NyssaBench: Failure-Aware and Audit-Safe Evaluation for Frontier Embodied AI
> Policies

Subtitle:

> A foundational benchmark framework for stress-testing, comparing, and
> debugging robot policies under real-world variation.

## Core Contributions

1. Simulator-agnostic `TaskSpec` and adapter interfaces for embodied policy
   evaluation.
2. Failure taxonomy and mapper for manipulation and embodied policy failures.
3. Stressor protocol for physical, visual, language, and control perturbations.
4. Result-tier system that prevents overclaiming benchmark results.
5. Pairwise Arena Mode for blinded policy comparison under identical
   conditions.
6. Replay-first reports with failure evidence.
7. Failure-episode export for retraining workflows.
8. Initial video-backed ManiSkill result pack with random, scripted, and learned
   baselines.

## Build Order

1. Keep the foundational protocol docs and validation APIs stable.
2. Make the ManiSkill result pack video-backed.
3. Improve the scripted/planner baseline until it solves the tasks reliably.
4. Train or evaluate a learned baseline that clearly beats random.
5. Add pairwise arena reporting on top of validated run artifacts.
6. Add stress search and ingest externally reconstructed experiments only after
   the reference result is credible.

## What Not To Build First

- new simulator engines
- cloud dashboards
- leaderboard websites
- OpenVLA-first demos
- humanoid-first benchmarks
- real-to-sim reconstruction
- world-model simulators

Those are future layers. The immediate bottleneck is trustworthy video-backed
evaluation.
