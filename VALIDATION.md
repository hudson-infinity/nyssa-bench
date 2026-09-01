# NyssaBench Validation

A NyssaBench run can be described as a public benchmark result only when its
`public_claim_validation.status` is `validated`.

## Required Checks

- Real simulator backend: the run uses a supported real simulator adapter.
- Explicit task mapping: every task declares `success.engine_env_ids.<engine>` or an engine factory.
- Known success predicate: every task maps simulator success information or a task-specific success metric.
- Episode count: at least 100 episodes per task.
- Seed coverage: public result packs should include at least 3 seeds per policy.
- Evidence: every public run includes episode artifacts and MP4 replay videos.
- Failure labels: failed episodes use environment or mapper labels, not silent placeholder defaults.
- Stressors: unsupported stressors are reported and not claimed as active perturbations.
- Reproducibility metadata: package versions, environment metadata, run config, and git metadata are present.

## Current Gate

The code-level gate is `nyssa_bench.metrics.run_claims.RunClaimValidator`.
It validates individual runs. Result-pack seed coverage is documented in the
generated `RESULTS.md` and `manifest.json`.

## What Not To Claim

- Metric-vector fields must be interpreted separately. Simulator-only runs do
  not provide sim-real evidence, and old scalar scores are audit-only legacy
  fields after migration.
- A run without replay videos is a smoke or debugging run. It must not be
  described as a public NyssaBench benchmark result.
- Adapter hooks for OpenVLA, diffusion, LeRobot, and RoboMimic are not public
  baseline results unless a concrete model/checkpoint and validated run artifacts
  are included.
- Genesis and RoboCasa are experimental contract adapters until concrete task
  mappings and upstream assets are configured.
