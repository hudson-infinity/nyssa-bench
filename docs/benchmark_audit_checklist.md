# Benchmark Audit Checklist

Use this checklist before publishing a NyssaBench result pack.

## Statistical Evidence

- At least 100 episodes per task.
- At least 3 seeds per policy.
- Confidence intervals are reported.
- Per-task and per-seed breakdowns are included.
- Differences between policies are not described as meaningful when confidence
  intervals overlap heavily without further testing.

## Replay Evidence

- Every episode has an MP4 replay.
- Every declared replay path resolves inside its run directory and points to a
  distinct existing episode MP4.
- The assembled pack's expected episode count agrees across `run.yaml`,
  `metrics.json`, and `episodes.json`.
- `replay_manifest.json` agrees with the episode replay and failure-clip paths.
- Failure clips, duplicate-content media, and unreferenced media are reported
  separately and do not increase episode replay coverage.
- Pack-level replay validation passes after copying, pruning, or archiving the
  result directory; cached run validation is not treated as sufficient.
- Failure episodes are easy to find.
- Reports include top failure cases.
- The result is not described as video-backed if videos are absent.

## Task Validity

- Every task has an explicit simulator environment mapping.
- Success predicates are mapped from environment info or explicit task metrics.
- Unsupported stressors are listed as unsupported, not silently claimed.
- Randomization ranges are identical when comparing policies.

## Policy Validity

- Random is treated as a sanity check only.
- Scripted baselines are not called oracles unless they solve the task reliably.
- Learned baselines identify the checkpoint, training data, and policy adapter.
- VLA or diffusion adapters are not listed as evaluated baselines unless a real
  model was loaded and run.

## Failure Analysis

- Failed episodes have mapper or environment labels.
- `unknown_failure` is investigated before publication.
- Primary failure mode is reported per policy and per task.
- Recovery metrics are reported when the task/policy supports recovery.

## Overclaim Guardrails

- Do not turn the metric vector into an undocumented weighted score or label
  simulator-only measurements as sim-real evidence.
- Do not call simulator-only results real-world validation.
- Do not publish Tier 0 or video-less runs as public benchmark results.
- Do not compare results produced under different task mappings, stressors, or
  success predicates without calling that out.
