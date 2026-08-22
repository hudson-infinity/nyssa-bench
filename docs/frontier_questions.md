# Frontier Questions

NyssaBench is frontier when it is used to study the hardest open problems in
embodied AI evaluation.

## Research Questions

NyssaBench should enable research on:

- Do simulation failures predict real robot failures?
- Can real robot videos or logs become reusable benchmark tasks?
- Can VLA policies survive physical, visual, language, and control
  perturbations?
- Can failure modes be predicted before action execution?
- Can pairwise evaluation reveal policy differences hidden by success rate?
- Can generated worlds expose failures that fixed benchmarks miss?
- Can failure episodes become targeted retraining data?
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
