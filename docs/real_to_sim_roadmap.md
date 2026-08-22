# Real-to-Sim Roadmap

Real-to-sim is a future NyssaBench direction, not a v0.1 requirement. The goal
is to turn real robot scenes, logs, or videos into repeatable evaluation cases
that can be stress-tested before policies are deployed again.

## Motivation

SIMPLER shows that simulated evaluation becomes more credible when paired with
real-world evidence. PolaRiS and RobotArena Infinity show that real scene/video
data can become scalable evaluation environments. GSWorld and related
Gaussian-splatting work point toward photorealistic closed-loop simulation.

NyssaBench should use these systems as sources of evaluation environments and
evidence, not as features to reimplement immediately.

## Proposed Stages

### Stage 0: Video-Backed Simulator Results

Current priority. Public results require MP4 replay evidence, explicit task
mappings, confidence intervals, failure labels, and reproducibility metadata.

### Stage 1: Real Log Import

Import real robot logs or datasets and align them with NyssaBench episode
schema:

- observations
- actions
- rewards or success labels
- failure labels
- recovery markers
- metadata
- video evidence

### Stage 2: Real-to-Sim Case Registry

Store reconstructed scenes as evaluation cases:

- source log or scan id
- reconstruction method
- simulator backend
- known limitations
- supported perturbations
- validation status

### Stage 3: Paired Sim/Real Scorecards

Report policy performance in both settings:

- sim success rate
- real success rate
- failure-mode agreement
- rank correlation
- stressor sensitivity
- examples where sim and real disagree

### Stage 4: World-Model Evaluation

Use video or world-model simulators as an additional evaluator and compare their
failure predictions against physics simulation and real-world outcomes.

## Non-Goals for v0.1

- building a neural renderer
- building a world model
- reconstructing scenes from raw video
- claiming sim-to-real validity without paired real-world evidence
