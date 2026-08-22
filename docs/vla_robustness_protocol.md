# VLA Robustness Protocol

Vision-language-action policies can look strong under fixed benchmark settings
and fail under modest physical or language perturbations. NyssaBench should make
those failures visible.

## Required Stressor Dimensions

The protocol should track these dimensions separately:

- object pose
- object identity
- camera viewpoint
- robot initial state
- lighting
- background texture
- sensor noise
- distractor objects
- instruction paraphrase
- instruction corruption
- adversarial visual patches

Unsupported stressors must be reported as unsupported. They must not appear as
active perturbations in scorecards.

## Metrics

For each policy, report:

- base success rate
- success by stressor
- worst-stressor success rate
- drop from base to worst stressor
- primary failure mode per stressor
- language sensitivity when instruction perturbations are present
- replay links for representative failures

## Worst-Case Search

Random perturbation is not enough for robust evaluation. Later NyssaBench
versions should support boundary or worst-case search:

```txt
nyssa stress-search
nyssa failure-boundary
nyssa risk-map
```

The first implementation should remain simple: fixed perturbation grids and
clear reporting. Optimization-based adversarial search can come later.

## Publication Rule

A VLA robustness claim is public only if:

- every episode has MP4 replay evidence
- stressor ranges are declared
- unsupported stressors are listed
- policies are compared under identical stressor distributions
- confidence intervals are reported
- task prompts and prompt perturbations are archived
