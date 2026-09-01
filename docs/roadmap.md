# Roadmap

NyssaBench is the failure-aware evaluation and audit layer for embodied AI
policies. The roadmap covers measurement infrastructure and evidence. It does
not absorb generated-world systems, real-to-sim reconstruction, interpretability
methods, policy-learning research, or hosted product infrastructure. See
[Project scope](project_scope.md) for the proposal gate.

## Measurement core

- Version and publish the Nyssa Evaluation Protocol contracts.
- Complete executable stressor coverage with backend application evidence.
- Complete temporal failure detectors and evidence provenance.
- Add matched counterfactual recovery evaluation.
- Replace scalar-first reporting with a versioned metric vector.
- Keep run validity and benchmark validity as separate executable results.

## Reference benchmark evidence

- Define a compact task set with protected split lineage.
- Validate task solvability with planner or oracle controls.
- Evaluate materially different external policy families through the same
  policy contract.
- Run paired clean and shifted experiments with justified sample sizes.
- Add scheduled MuJoCo and capable-runner ManiSkill integration jobs.
- Publish immutable result packs that pass the applicable claim gates.

Reference checkpoints and baseline scripts exist to test the protocol. General
policy training and method development remain in policy projects.

## Predictive validity

- Ingest versioned real-world evidence supplied by hardware programs.
- Pair simulation and hardware conditions without hiding unavoidable mismatch.
- Compare policy rankings, temporal failures, shift response, and recovery
  effects with uncertainty.
- Test whether failure profiles add predictive value beyond clean task success.
- Keep sim-real metrics unavailable until hardware evidence passes its contract.

NyssaBench ingests reconstructed experiments but does not reconstruct scenes.
Real-to-sim systems remain external producers.

## Stable interfaces

- External scenario and generated-world ingestion contract.
- Real-world and reconstructed-experiment evidence contract.
- External failure-monitor adapter and calibration contract.
- Failure and recovery evidence export for downstream learning systems.
- Conformance fixtures that avoid unnecessary direct dependencies.

## Explicit non-goals

- building simulators, world generators, renderers, or asset platforms;
- implementing real-scene reconstruction;
- developing new VLA, RL, interpretability, or continual-learning methods;
- operating robots or deployment fleets;
- building hosted evaluation, customer, billing, or deployment products.

Those programs can use NyssaBench artifacts and adapters without moving their
implementation into this repository.
