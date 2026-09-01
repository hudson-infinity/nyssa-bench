# Frontier Questions

NyssaBench is frontier when it is used to study the hardest open problems in
embodied AI evaluation.

## Research Questions

NyssaBench should enable research on:

- Do simulation failures predict real robot failures?
- Can supplied real robot videos or logs become reusable evaluation evidence?
- Can VLA policies survive physical, visual, language, and control
  perturbations?
- Can failure modes be predicted before action execution?
- Can pairwise evaluation reveal policy differences hidden by success rate?
- Can externally generated worlds expose failures that fixed benchmarks miss?
- Can failure episodes identify useful downstream training data?
- Can world-model rollouts become useful robot-policy evaluators?
- Which evaluation signals best predict deployment failures?

## Frontier Scope

The near-term frontier is not building a new simulator. It is building the
measurement layer that compares:

- physics simulation
- real-to-sim simulation
- world-model rollouts
- real robot evaluation

NyssaBench should track how these signals agree, where they disagree, and which
failures transfer to the real world.

It does this by validating and evaluating inputs from physics simulators,
generated-world systems, reconstruction systems, world models, and hardware
programs. It does not implement those producers. Likewise, NyssaBench can
measure an interpretability monitor or export failure data without owning the
monitoring method or the learning algorithm that consumes the export.
