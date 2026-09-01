# Foundational Contributions

NyssaBench is foundational when it defines reusable standards and infrastructure
that other embodied AI projects can build on, even when they use different
simulators, policies, robots, or datasets.

## Core Standards

NyssaBench contributes:

- `TaskSpec`: a portable way to describe embodied evaluation tasks.
- `EngineAdapter`: a boundary between evaluation logic and simulators.
- `PolicyAdapter`: a boundary between evaluation logic and policy
  implementations.
- Failure taxonomy: shared names for robot-policy failure modes.
- Failure mapper: conservative mapping from simulator events to failure labels.
- Stressor protocol: how physical, visual, language, and control perturbations
  are declared, applied, and reported.
- Result tiers: how benchmark claims are scoped and prevented from overclaiming.
- Benchmark audit checklist: what a result must answer before publication.
- Pairwise protocol: how two policies are compared under identical conditions.
- Replay-first report format: how evidence is inspected.
- Dataset export format: how evaluated and failed episodes are handed to
  downstream analysis or learning systems.

## Why This Is Foundational

These are not just repo features. They are evaluation conventions:

- `TaskSpec` defines what was tested.
- Failure taxonomy defines what went wrong.
- Stressor protocol defines what variation was applied.
- Result tiers define what can be claimed.
- Pairwise protocol defines fair policy comparison.
- Replay reports define evidence.

The goal is for another lab to use NyssaBench as its audit/evaluation layer even
if it uses a different simulator or policy stack.

The repository owns these evaluation conventions, not the upstream systems it
evaluates or the downstream methods that consume its artifacts. See
[Project scope](project_scope.md) for the interface boundary.
