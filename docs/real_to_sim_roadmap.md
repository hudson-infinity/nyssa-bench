# Real evidence and reconstructed experiment integration

Real-to-sim reconstruction is a separate research program, not a NyssaBench
subsystem. NyssaBench's responsibility begins when a reconstruction system or
hardware program supplies a versioned scene, log, mapping, or evidence manifest.
The goal here is to make those external outputs usable as repeatable evaluation
cases without implementing reconstruction in this repository.

## Motivation

SIMPLER shows that simulated evaluation becomes more credible when paired with
real-world evidence. PolaRiS and RobotArena Infinity show that real scene/video
data can become scalable evaluation environments. GSWorld and related
Gaussian-splatting work point toward photorealistic closed-loop simulation.

NyssaBench should use these systems as producers of evaluation environments and
evidence, not as features to reimplement.

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

### Stage 2: Reconstructed-case ingestion

Validate externally reconstructed scenes as evaluation cases:

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
