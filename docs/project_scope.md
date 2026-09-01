# Project scope

NyssaBench is the failure-aware evaluation and audit layer for embodied AI
policies. It is Hudson Labs' first foundational project, but it is not the
umbrella repository for every Hudson research program or product.

NyssaBench answers one question:

> How do we know an embodied policy is actually reliable?

The repository owns measurement contracts, controlled execution, failure
evidence, recovery evaluation, validity checks, and reproducible result
artifacts. It can consume scenarios, policies, monitors, and real-world evidence
from other projects, and it can export evaluated failure data to them. Producing
those upstream systems or training downstream methods remains outside this
repository.

## In scope

NyssaBench owns:

- versioned task, engine, policy, stressor, failure-evidence, intervention, and
  claim contracts;
- simulator adapters needed to execute those contracts without owning simulator
  scenes, robots, controllers, or assets;
- controlled stressors and paired experimental protocols;
- temporal failure detection, evidence provenance, and failure-aware reports;
- verifier and intervention measurement, including counterfactual recovery
  evaluation where state restoration is defensible;
- run validity, benchmark validity, statistical checks, and claim gates;
- reproducibility metadata, result packs, replay evidence, and metric vectors;
- adapters for external policies and failure monitors;
- ingestion of externally generated scenarios or real evidence through stable
  contracts;
- exports for downstream analysis, data selection, or learning systems;
- compact reference policies and tasks when they are necessary to test the
  evaluation protocol.

Reference implementations are controls, not a transfer of project ownership.
For example, a small BC baseline can verify the policy contract, but NyssaBench
does not become a general imitation-learning framework.

## Out of scope

Separate projects should own:

- procedural, adversarial, or learned world generation;
- asset generation, scene authoring platforms, and simulator development;
- reconstruction of real scenes and estimation of geometry or physics from
  video, scans, or telemetry;
- new interpretability, activation-probing, representation-steering, or policy
  confidence methods;
- general policy training, continual learning, RL algorithms, and data-selection
  research;
- robot fleet operation and deployment control;
- hosted evaluation services, customer data planes, deployment-readiness
  products, billing, and organization management;
- Hudson Labs' organization-wide research strategy.

These systems may integrate with NyssaBench, but their implementation should not
land here merely because NyssaBench evaluates their outputs.

## Interface boundaries

| External producer or consumer | NyssaBench responsibility | External responsibility |
| --- | --- | --- |
| Generated-world system | Validate and ingest a scenario contract; execute and audit the scenario | Generate scenes, assets, distributions, and adversarial cases |
| Real-to-sim system | Ingest provenance, mappings, uncertainty, and evidence references | Reconstruct the scene and estimate physical properties |
| Policy project | Validate observations, actions, checkpoints, and training-data declarations | Design and train the policy |
| Interpretability or failure-monitor project | Run the monitor through a declared adapter and measure calibration, latency, and intervention effects | Develop probes, internal representations, or steering methods |
| Failure-driven learning system | Export versioned failure, intervention, and branch evidence | Select data, optimize the policy, and manage training |
| Real-robot program | Pair hardware trials with simulation records and audit claims | Operate hardware, enforce safety procedures, and collect trials |
| Hosted product | Produce portable result and claim artifacts | Schedule private jobs, isolate customer data, and provide product workflows |

## Triage examples

| Proposal | Decision | Reason |
| --- | --- | --- |
| Add a typed camera-occlusion stressor | NyssaBench | It changes a controlled evaluation condition |
| Build a procedural kitchen generator | Separate project | It produces worlds rather than measures policies |
| Add an external scenario manifest and conformance test | NyssaBench | It is an ingestion and validation boundary |
| Reconstruct a kitchen from customer video | Separate project | It is real-to-sim estimation |
| Add an adapter for a VLA checkpoint | NyssaBench | It lets the protocol evaluate an external policy |
| Develop a new VLA architecture or training recipe | Separate project | It is policy research |
| Add temporal failure-monitor calibration metrics | NyssaBench | It measures monitor behavior |
| Train a latent probe on VLA activations | Separate project | It develops an interpretability method |
| Export recovery branches as a dataset | NyssaBench | It is a versioned evidence handoff |
| Implement continual RL over those branches | Separate project | It is a learning algorithm |
| Generate a static local HTML report | NyssaBench | It is a portable evaluation artifact |
| Build a multi-tenant hosted dashboard | Separate product repository | It is service and customer infrastructure |

## Proposal gate

Every feature proposal must answer:

1. Which NyssaBench contract or measurement responsibility does this change?
2. Is the feature evaluating an external artifact, or implementing the external
   producer or consumer itself?
3. Can the integration be expressed as an adapter, manifest, conformance
   fixture, or export instead of a direct dependency?
4. Which evidence tier or claim becomes possible after the change?
5. Which responsibilities explicitly remain outside NyssaBench?

Maintainers should redirect a proposal when its primary output is a world,
reconstruction, interpretability method, trained policy, learning algorithm, or
hosted service. Cross-project work belongs here only at the stable measurement
interface.
