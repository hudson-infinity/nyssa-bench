# Recovery and ablation workflows

Use these recipes after completing the [README quickstart](../README.md#quickstart).
Commands run from the repository root and use Bash syntax for environment
variables. Small runs below exercise the pipeline; public results need the
replay, sampling, and audit evidence described in the
[validation protocol](validation_protocol.md).

## Compare verifier and recovery variants

`ablate` runs a policy across a seed and intervention matrix:

```bash
uv run nyssa ablate \
  --suite mujoco_control_v0 \
  --tasks mujoco_pusher \
  --engine mujoco \
  --policy random \
  --seeds 0 \
  --episodes 5 \
  --variants base verifier recovery verifier_recovery \
  --expert-provider mujoco-heuristic \
  --out benchmark_results/recovery_smoke \
  --no-replay
```

| Variant | Assistance |
| --- | --- |
| `base` | Policy alone |
| `verifier` | Action verification |
| `recovery` | Recovery assistance |
| `verifier_recovery` | Verification and recovery |

Each variant has its own run directory, such as
`benchmark_results/recovery_smoke/verifier_recovery/seed_0`. The result pack
records run and replay validation in its manifest and scorecard.

Available expert providers include:

| Provider | Use |
| --- | --- |
| `none` | No expert assistance |
| `bounds-verifier` | Reject actions outside the live action bounds |
| `maniskill-scripted` | Built-in ManiSkill manipulation heuristic; `scripted-oracle` is an alias |
| `mujoco-heuristic` | MuJoCo rollout verifier and recovery provider; `mujoco-random-shooting` is an alias |
| `policy:<name>` | Use a registered policy as the expert action source |

For a ManiSkill ablation, install its runtime and configure a working render
device first, then run:

```bash
uv run nyssa ablate \
  --suite maniskill_smoke_v0 \
  --engine maniskill \
  --policy random \
  --seeds 0 \
  --episodes 5 \
  --variants base verifier recovery verifier_recovery \
  --expert-provider maniskill-scripted \
  --out benchmark_results/maniskill_ablation_smoke \
  --capture-replay
```

The same `--expert-provider`, `--enable-verifier`, and `--enable-recovery`
controls are available on `run` and `experiment`. Use `--tasks` to focus a study
on the tasks for which the expert has been evaluated.

## Tune the MuJoCo verifier

The built-in provider scores candidate rollouts. More candidates spend more
simulator work searching for an action; a larger fixed rollout margin rejects
fewer policy actions. These settings configure a heuristic and need evaluation
on the tasks where it will be used.

```bash
NYSSA_MUJOCO_ROLLOUT_HORIZON=3 \
NYSSA_MUJOCO_ROLLOUT_MARGIN=0.25 \
NYSSA_MUJOCO_CANDIDATES=32 \
NYSSA_MUJOCO_PUSHER_SHAPING=5.0 \
NYSSA_MUJOCO_ADAPTIVE_MARGIN=auto \
NYSSA_MUJOCO_MARGIN_FRACTION=0.25 \
NYSSA_MUJOCO_MARGIN_TOP_K=2 \
NYSSA_MUJOCO_MARGIN_TOP_FRACTION=0.10 \
NYSSA_MUJOCO_RECOVERY_TASKS=mujoco_pusher \
uv run nyssa ablate \
  --suite mujoco_control_v0 \
  --tasks mujoco_pusher \
  --engine mujoco \
  --policy random \
  --seeds 0 \
  --episodes 5 \
  --variants base verifier_recovery \
  --expert-provider mujoco-heuristic \
  --out benchmark_results/pusher_calibration \
  --no-replay
```

| Variable | Effect |
| --- | --- |
| `NYSSA_MUJOCO_ROLLOUT_HORIZON` | Number of rollout steps used for action evaluation |
| `NYSSA_MUJOCO_ROLLOUT_MARGIN` | Fixed tolerance before rejecting a policy action |
| `NYSSA_MUJOCO_CANDIDATES` | Number of candidate action sequences to evaluate |
| `NYSSA_MUJOCO_PUSHER_SHAPING` | Pusher terminal shaping from object-goal and arm-object distances |
| `NYSSA_MUJOCO_PUSHER_ACTION_SCALES` | Guided action scales, for example `0.5,1.0,1.5,2.0` |
| `NYSSA_MUJOCO_PUSHER_FINISH_SCALES` | Low-control push-and-settle candidates near the success threshold |
| `NYSSA_MUJOCO_PUSHER_PLANNING_HORIZON` | Pusher planning horizon; defaults to `15` |
| `NYSSA_MUJOCO_PUSHER_RECOVERY_EXECUTION_HORIZON` | Cap on committed recovery actions before replanning |
| `NYSSA_MUJOCO_RECOVERY_TASKS` | Tasks using macro recovery; defaults to `mujoco_pusher`, and accepts `all` or a comma-separated list |
| `NYSSA_MUJOCO_ADAPTIVE_MARGIN` | `auto` enables a Pusher margin based on near-best candidate returns |
| `NYSSA_MUJOCO_MARGIN_FRACTION` | Fraction of the local return spread used for the adaptive margin |
| `NYSSA_MUJOCO_MARGIN_TOP_K` | Number of top candidates defining that spread; defaults to `2` |
| `NYSSA_MUJOCO_MARGIN_TOP_FRACTION` | Fraction of top candidates used when the top-k setting is disabled |

Pusher recovery considers local probes, approach actions, pushes toward the
goal, and mixed approach-then-push sequences. Sequential mixed plans commit
multiple actions; single-mode approach or push plans replan at each step by
default. Candidate scoring also rewards crossing the task's configured
`reward_threshold`.

The default macro-recovery task restriction leaves the rollout verifier active
on Reacher and InvertedPendulum without applying Pusher plans to them. Expand
`NYSSA_MUJOCO_RECOVERY_TASKS` only after evaluating the provider on those tasks.

## Inspect recovery evidence

Recovery runs write the usual episode, metric, and provenance artifacts plus:

- `recovery_dataset/manifest.json`, with supervised-target and context counts;
- `recovery_dataset/episodes.jsonl`, with executed actions and target provenance;
- `failure_ledger.json`, with temporal failures and recovery eligibility;
- `failure_gallery.html`, with failed episodes and available replay links.

Recovery dataset v2 keeps unsuccessful attempts as `negative_context` records
with `target_valid: false`. BC training accepts eligible targets from `expert`
or `recovery` sources. Rejected policy actions remain recorded as executed
actions; they do not become corrective training targets.

`recovery_success_rate` counts successes within the configured attribution
window over attempts with an applied plan. The default window is five
transitions or the plan length, whichever is longer. Eventual episode success
outside that window does not count. `recovery_episode_success_rate` separately
reports success among episodes with applied recovery. Set
`--recovery-attribution-horizon` when the default does not fit the study.

For recovery effects, enable matched state-fork evaluation with
`--counterfactual-repeats`, `--counterfactual-horizon`, and
`--counterfactual-max-branch-points`. The
[counterfactual recovery guide](counterfactual_recovery.md) explains the full
command, restoration grades, coverage, and uncertainty. Bounded recovery
outcomes alone establish temporal attribution, not a causal effect.

## Train from valid recovery targets

Given a collection run with valid expert or recovery targets, train one
checkpoint per task:

```bash
uv run nyssa train-recovery-bc \
  benchmark_results/recovery_smoke \
  --routing task \
  --out-dir checkpoints/recovery_bc_by_task \
  --merged-out benchmark_results/recovery_training/episodes.json
```

The command accepts a run directory or an ablation root. Task routing keeps
different action dimensions separate. `--by-task` is also supported.
If a smoke run produced no valid targets, collect suitable episodes before
training; negative-context records are not a substitute.

Evaluate the Pusher checkpoint on a held-out run seed:

```bash
NYSSA_TASK_BC_DIR=checkpoints/recovery_bc_by_task \
uv run nyssa ablate \
  --suite mujoco_control_v0 \
  --tasks mujoco_pusher \
  --engine mujoco \
  --policy task_bc_policy \
  --seeds 10000 \
  --episodes 5 \
  --variants base verifier recovery verifier_recovery \
  --expert-provider mujoco-heuristic \
  --recovery-attribution-horizon 5 \
  --out benchmark_results/recovery_bc_eval \
  --no-replay
```

Keep training and evaluation episode seeds disjoint. Under
`nyssa-episode-seed-v2`, run seeds identify separate episode-seed namespaces.
Repo-local BC checkpoints do not enforce seed exclusion automatically.
`NYSSA_TASK_BC_MISSING=zero` is available for diagnosing missing task checkpoints,
but such a fallback run cannot support a learned-policy claim.

To compare against the collection policy, evaluate that policy separately on
the same held-out task/seed settings, then use `nyssa compare` on those run
directories. The [policy comparison guide](policy_comparison.md) explains
contract compatibility and pairing requirements.

## Action chunks and larger studies

For a policy returning eight actions per call, `--policy-action-horizon 8`
declares the chunk length and `--policy-execution-horizon 4` commits four
actions before the next call. See [policy adapters](policy_adapters.md) for
observation and action contracts.

After smoke tests, choose episode counts and seeds from the study's sampling
and power requirements, configure rendering, and use `--capture-replay`.
Increasing the sample size or enabling video alone does not establish a public
claim. The result pack must pass its run and benchmark evidence gates.

For ManiSkill demonstration collection, state-aligned replay, and external
RoboMimic training, continue with [learned baselines](learned_baselines.md).
