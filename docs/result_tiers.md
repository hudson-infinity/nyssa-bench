# Result Tiers

NyssaBench separates "a run completed" from "a result should be cited." The
tier should be written into reports and scorecards so readers know what kind of
claim the artifacts support.

The table defines protocol tiers, not evidence that NyssaBench has already
produced a result at every tier. The current public wording and promotion gate
live in the [claim evidence matrix](claim_evidence.md).

| Tier | Name | Meaning | Public claim |
| --- | --- | --- | --- |
| 0 | Scaffold | Dummy, unit, placeholder, or adapter-contract run. | No |
| 1 | Real simulator | Supported simulator backend, explicit task mappings, success predicates, enough episodes, metadata, diagnosed failures, and MP4 replay videos. | Yes, if validation passes |
| 2 | Simulator plus stressors | Tier 1 plus supported and reported stress conditions such as seed, object pose, camera pose, friction, lighting, or distractors. | Yes |
| 3 | Pairwise arena | Tier 2 plus paired A/B policy comparisons on the same tasks, seeds, and stressors, preferably blinded. | Yes |
| 4 | Sim-real validated | Tier 3 plus paired real-robot evidence showing simulator results correlate with real outcomes. | Stronger public claim |
| 5 | Distributed real-world | Multi-site or multi-evaluator real-world evaluation with blinded pairwise or preference aggregation. | Strongest real-world claim |
| 6 | Real-to-sim or world model | Real scene/log/video reconstruction or world-model evaluation with measured real-world correlation. | Research claim, requires caveats |

## Current Code Mapping

The current `RunClaimValidator` gates individual runs. A run is public only when
all required checks pass. Under the current gate, MP4 replay video evidence is
required for every episode.

Result-pack tiering is broader than a single run. For example, a pack with
three policies and three seeds can be Tier 1 only if every included run passes
the run-level validator and the pack includes enough seed coverage.

## Anti-Overclaim Rule

If a result is missing videos, has weak scripted baselines, uses unsupported
stressors, has too few episodes, or relies on placeholder policy adapters, it
must stay below Tier 1 even if the code path executed successfully.
