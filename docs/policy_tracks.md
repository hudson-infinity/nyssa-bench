# Validated policy tracks

NyssaBench policy adapters are integration surfaces. A validated policy track
is a stronger object that binds a model and its training provenance to
conformance and benchmark evidence.

The candidate registry is
[`configs/policy_tracks/nyssa_policy_tracks_v0_1.json`](../configs/policy_tracks/nyssa_policy_tracks_v0_1.json).
It contains an oracle control, random sanity control, RoboMimic BC, diffusion
action-chunking, and OpenVLA track. Every track remains `integration_only` until
its real artifacts pass `nyssa audit-policy-tracks`. Oracle, random, RoboMimic,
and diffusion are required for the first release; OpenVLA is represented but
does not block that release until a compatible RGB/language task contract and
checkpoint are available.

## Required evidence

Every validated track must provide:

- a NEP policy contract with checkpoint and preprocessing hashes;
- an actual checkpoint and preprocessing artifact matching those hashes;
- task, asset, split, demonstration, run-seed, and episode-seed training
  provenance;
- action normalization, representation, prediction horizon, execution horizon,
  reset semantics, and training compute;
- one conformant report for every task in the common subset;
- one clean and one shifted run fingerprint for every evaluation run seed;
- passing RunValidity and BenchmarkValidity in every underlying result pack;
- complete paired episode keys, temporal failure ledgers, uncertainty metrics,
  and content-pinned MP4 replay evidence.

The evaluator reopens each run fingerprint and rehashes its underlying result
pack. A copied status field or adapter import cannot satisfy the track.

## Common subset

The initial comparison subset is:

```text
maniskill_pick_cube
maniskill_push_cube
maniskill_stack_cube
```

All tracks use the same task order, held-out split identity, evaluation run
seeds, and shifted condition. Learned training sources must not contain the
evaluation split, evaluation assets, or any evaluated episode seed.

This subset is itself downstream of the reference benchmark. The track registry
cannot become a release while the reference benchmark remains a candidate.

## Track setup

### Planner oracle

Use ManiSkill's Linux motion-planning solutions and record their exact source
commit, planning dependencies, generated action representation, and any
task-specific fallback. The repository-local `scripted_oracle` is not accepted
as the planner upper bound unless it independently meets the oracle threshold.

### RoboMimic BC

Export state-aligned demonstrations with the existing task exporter, train one
checkpoint per task, and retain `task_robomimic_manifest.json`:

```bash
uv run nyssa export-task-robomimic \
  benchmark_results/reference_training_demos \
  --out-dir datasets/reference_robomimic \
  --config-dir configs/generated/reference_robomimic \
  --feature-dim 512 \
  --epochs 50 \
  --batch-size 64
```

The final policy contract must identify a frozen multi-task release or a
content-addressed collection of per-task checkpoints. Direct checkpoint folders
without reconstructed training provenance remain diagnostics.

### Diffusion action chunking

Provide a real `NYSSA_DIFFUSION_POLICY=module:factory` implementation that
returns 16-action chunks and commits four actions before replanning. The factory
must implement reset semantics compatible with the NEP contract and expose the
same normalized end-effector action representation used during training.

```bash
NYSSA_DIFFUSION_POLICY=my_policy.factory:create \
uv run nyssa conform-policy \
  --policy my_policy/adapter.py \
  --policy-contract checkpoints/diffusion/policy_contract.json \
  --suite nyssa_reference_manipulation_v0_1 \
  --task maniskill_pick_cube \
  --engine maniskill \
  --episodes 1 \
  --out conformance/diffusion/pick_cube
```

Repeat conformance for all three tasks.

### OpenVLA

The VLA track must declare RGB camera calibration, language preprocessing,
action decoding, training datasets, and robot/action-space alignment. The
existing `openvla` registry entry is only a callable hook and is not a validated
track.

## Evaluation

Run each policy under the same clean and shifted design, then fingerprint every
result directory:

```bash
uv run nyssa regression-fingerprint \
  benchmark_results/<track>/clean/seed_10000 \
  --out policy_track_evidence/<track>/clean_seed_10000.json

uv run nyssa regression-fingerprint \
  benchmark_results/<track>/shifted/seed_10000 \
  --out policy_track_evidence/<track>/shifted_seed_10000.json
```

After filling the registry with immutable references:

```bash
uv run nyssa audit-policy-tracks \
  configs/policy_tracks/nyssa_policy_tracks_v0_1.json \
  --repo-root . \
  --out build/policy-track-audit
```

Exit code `0` means the registry is declared `release` and every required
artifact passes. Exit code `2` means evidence is missing. Invalid or changed
artifacts fail.

Random remains a sanity control and cannot become a validated headline track.
Policy conformance remains `integration_only` by design; it is necessary but
not sufficient for track validation.
