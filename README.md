# NyssaBench

**NyssaBench is open-source foundational infrastructure for failure-aware evaluation of frontier embodied AI policies.**

Run reproducible robot-policy benchmarks, stress-test failures, audit benchmark
claims, compare policies, and export simulation data for training.

NyssaBench is not a simulator. It is the measurement layer for frontier embodied
AI: a foundational benchmark and audit framework for evaluating how robot
policies fail under real-world variation. It sits on top of robotics simulators
such as ManiSkill and MuJoCo, with experimental adapter boundaries for RoboCasa
and Genesis. Public scorecards must pass the run claim validator; experimental
adapters and placeholder task mappings are not public benchmark claims.

```python
from nyssa_bench import PolicyRunner, Suite

suite = Suite.load("mujoco_control_v0")

runner = PolicyRunner(
    policy="random",
    engine="mujoco",
    episodes=10,
    seed=42,
    out="runs/random_mujoco",
)

report = runner.evaluate(suite)
report.save("runs/random_mujoco/report.html")
```

## Install

Use Python 3.10 for ManiSkill/Linux runs. Some ManiSkill planning dependencies
publish wheels for CPython 3.10 but not newer Python ABIs, so Python 3.12 can
fail during dependency resolution.
The ManiSkill install path pins NumPy below 2.0 because ManiSkill
motion-planning dependencies such as `toppra` include compiled extensions that
can fail with a NumPy 2 ABI mismatch.

```bash
uv python install 3.10
uv venv --python 3.10 .venv
source .venv/bin/activate
python --version
```

Install the canonical stable benchmark environment:

```bash
uv sync --extra all --extra dev
```

`uv sync` is exact by default. A later sync that omits an extra can remove
packages installed by an earlier sync, so repeat the canonical command after
pulling dependency changes. Do not run separate exact sync commands as additive
installation steps.

For a dedicated lightweight environment, select one complete workflow in a
single command:

```bash
uv sync --extra dev --extra mujoco --extra dataset --extra video --extra reports
uv sync --extra dev --extra maniskill --extra dataset --extra video --extra reports
```

Use the first command for MuJoCo-only work or the second for ManiSkill-only
work. To add policy stacks to an existing lean environment without removing its
installed packages, use one inexact sync:

```bash
uv sync --inexact --extra lerobot --extra robomimic --extra vla --extra diffusion
```

If an older environment already installed NumPy 2, reinstall the planning stack
after pulling the latest dependency pin:

```bash
uv pip install "numpy==1.26.4"
uv pip install --force-reinstall --no-build-isolation --no-cache-dir "toppra==0.6.3"
```

Plain `python -m pip install -e ".[all,dev]"` still works if you are not using
`uv`.

Simulator video capture also requires system rendering libraries. On
Ubuntu/Debian GPU machines, install and verify them before running public
benchmark commands:

```bash
bash scripts/setup_rendering_linux.sh
vulkaninfo --summary
nvidia-smi
```

If `vulkaninfo --summary` cannot see a Vulkan device, ManiSkill can still run
some CPU-side simulation paths but replay videos will not be produced. Public
NyssaBench benchmark claims require MP4 replay evidence for every episode.
If `vulkaninfo --summary` reports only `llvmpipe`, the machine is using CPU
Vulkan. Install the NVIDIA Vulkan ICD/GL packages that match the host driver,
for example `nvidia-utils-535` and `libnvidia-gl-535` on driver branch 535.
MuJoCo `rgb_array` replay on headless Linux defaults to EGL when `DISPLAY` is
missing, which avoids GLFW/X11 crashes in Colab-style sessions. If your machine
requires a specific backend, set `MUJOCO_GL=egl` or `MUJOCO_GL=osmesa` before
launching `nyssa`.

On macOS, MuJoCo smoke runs usually need the Python extras plus native GLFW:

```bash
brew install glfw
```

ManiSkill video-backed result packs are expected to run on Linux machines with
a working NVIDIA/Vulkan stack.
On managed notebooks such as Lightning AI, ManiSkill may fail with
`Failed to find a supported physical device "cuda:0"` if the session has no
compatible render device. Select a GPU/Vulkan-capable runtime, or set
`NYSSA_MANISKILL_RENDER_DEVICE` and `NYSSA_MANISKILL_SIM_BACKEND` before
launching `nyssa` to match the device exposed by the host. For non-public
debugging on CPU-only sessions, set `NYSSA_MANISKILL_RENDER_MODE=none` and use
`--no-replay`; public claims still require replay videos.

Do not use `pip install -e ".[full]"` for normal benchmark runs. The `full`
extra intentionally pulls heavy experimental stacks, including Genesis,
RoboMimic, LeRobot, VLA, and diffusion dependencies, and native packages in
those stacks can fail on otherwise valid MuJoCo or ManiSkill machines. Install
only the extras for the workflow you are running.

Docker images are provided under `docker/` for core, ManiSkill, and MuJoCo environments.

Experimental adapters for RoboCasa and Genesis are included in the repo. Their integration contracts live in `configs/experiments/`; contract validation works without heavyweight assets, and full simulator runs require concrete task-to-scene mappings plus upstream setup. Install RoboCasa from upstream in a separate environment when working on that adapter; its current dependency stack is not compatible with the ManiSkill motion-planning NumPy 1.26 setup.

## First Run

The MuJoCo commands require the MuJoCo install command above. They create two
real run directories before report, export, compare, leaderboard, and scorecard
commands are called.

```bash
uv run nyssa list-suites

uv run nyssa run \
  --suite mujoco_control_v0 \
  --engine mujoco \
  --policy random \
  --episodes 10 \
  --seed 0 \
  --out runs/random_mujoco_seed0

uv run nyssa run \
  --suite mujoco_control_v0 \
  --engine mujoco \
  --policy random \
  --episodes 10 \
  --seed 1 \
  --out runs/random_mujoco_seed1

uv run nyssa report runs/random_mujoco_seed0
uv run nyssa export --run runs/random_mujoco_seed0 --format lerobot
uv run nyssa export --run runs/random_mujoco_seed0 --format jsonl
uv run nyssa compare runs/random_mujoco_seed0 runs/random_mujoco_seed1 --out reports/compare.html
uv run nyssa leaderboard runs/random_mujoco_seed0 runs/random_mujoco_seed1 --out reports/leaderboard.json
uv run nyssa scorecard runs/random_mujoco_seed0 runs/random_mujoco_seed1 --out benchmark_results/baselines_v0.json
```

## Recovery And Ablation Runs

Use `ablate` to run base, verifier, recovery, and verifier+recovery variants
with one command. Start small before running public-scale episodes.

MuJoCo smoke ablation:

```bash
uv run nyssa ablate \
  --suite mujoco_control_v0 \
  --engine mujoco \
  --policy random \
  --seeds 0 \
  --episodes 5 \
  --variants base verifier recovery verifier_recovery \
  --expert-provider mujoco-heuristic \
  --out benchmark_results/mujoco_ablation_smoke \
  --no-replay
```

ManiSkill smoke ablation:

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

Useful built-in expert providers:

- `none`: no expert, verifier, or recovery assistance.
- `bounds-verifier`: rejects actions outside the live action space.
- `maniskill-scripted`: built-in ManiSkill scripted manipulation heuristic.
- `scripted-oracle`: alias for the built-in ManiSkill scripted expert.
- `mujoco-heuristic`: calibrated short-horizon MuJoCo rollout verifier and recovery provider.
- `mujoco-random-shooting`: current alias for the MuJoCo heuristic scaffold.
- `policy:<name>`: use any registered Nyssa policy as the expert action source.

MuJoCo verifier calibration can be tuned without code changes:

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
uv run nyssa ablate ...
```

Higher `NYSSA_MUJOCO_ROLLOUT_MARGIN` rejects fewer learned-policy actions.
Lower values make the verifier more intervention-heavy. Higher
`NYSSA_MUJOCO_CANDIDATES` spends more simulator rollouts searching for a better
recovery action, which is most useful on higher-dimensional tasks such as
`mujoco_pusher`. `NYSSA_MUJOCO_PUSHER_SHAPING` adds Pusher-specific terminal
rollout shaping from object-goal and arm-object distances when those MuJoCo
body positions are available. Pusher also uses body-geometry guided recovery
macro-plans: sparse local arm-control probes, approach behind the object, push
toward the goal, and mixed approach-then-push sequences. Recovery executes the
selected short-horizon plan instead of discarding every action after the first,
but Pusher only commits sequential mixed plans by default so single-mode push or
approach plans can replan every step.
`NYSSA_MUJOCO_PUSHER_ACTION_SCALES` controls the guided action scales considered
by the test-time planner, for example `0.5,1.0,1.5,2.0`.
`NYSSA_MUJOCO_PUSHER_FINISH_SCALES` adds low-control push-and-settle candidates
for near-threshold Pusher states where reducing control penalty can decide
success.
`NYSSA_MUJOCO_PUSHER_PLANNING_HORIZON` lets Pusher score candidates over a
longer horizon than the execution horizon; the default is `15`. 
`NYSSA_MUJOCO_PUSHER_RECOVERY_EXECUTION_HORIZON` caps how many committed
recovery actions run before replanning.
`NYSSA_MUJOCO_RECOVERY_TASKS` controls where MuJoCo macro recovery is active;
the default is `mujoco_pusher`, so full-suite `verifier_recovery` uses the
rollout verifier on Reacher and InvertedPendulum without applying Pusher-tuned
macro actions there. Set it to `all` or a comma-separated task list when a
recovery planner has been validated for those tasks.
MuJoCo rollout scoring also gives a large bonus to candidates that cross a
task's configured `reward_threshold`, so near-success states prefer actions
that satisfy the benchmark predicate rather than only improving shaped progress.
`NYSSA_MUJOCO_ADAPTIVE_MARGIN=auto` switches Pusher to a margin derived from
the near-best candidate return spread, which avoids fixed margins that are too
large for small Pusher score gaps. `NYSSA_MUJOCO_MARGIN_TOP_FRACTION` controls
how much of the top candidate cluster defines that local spread, unless
`NYSSA_MUJOCO_MARGIN_TOP_K` is set. The default top-k setting is `2`, which uses
only the best two rollout returns for local margin scale.

The same hooks are available on `run` and `experiment`:

```bash
uv run nyssa run \
  --suite mujoco_control_v0 \
  --engine mujoco \
  --policy random \
  --episodes 10 \
  --seed 0 \
  --expert-provider mujoco-heuristic \
  --enable-verifier \
  --enable-recovery \
  --out runs/mujoco_recovery_smoke \
  --no-replay
```

Focus on a single task while debugging a weak task-specific result:

```bash
NYSSA_TASK_BC_DIR=checkpoints/recovery_bc_by_task \
NYSSA_TASK_BC_MISSING=zero \
NYSSA_MUJOCO_ROLLOUT_HORIZON=5 \
NYSSA_MUJOCO_CANDIDATES=64 \
NYSSA_MUJOCO_PUSHER_SHAPING=10.0 \
NYSSA_MUJOCO_ADAPTIVE_MARGIN=auto \
NYSSA_MUJOCO_MARGIN_FRACTION=0.25 \
NYSSA_MUJOCO_MARGIN_TOP_K=2 \
NYSSA_MUJOCO_MARGIN_TOP_FRACTION=0.10 \
NYSSA_MUJOCO_RECOVERY_TASKS=mujoco_pusher \
uv run nyssa ablate \
  --suite mujoco_control_v0 \
  --tasks mujoco_pusher \
  --engine mujoco \
  --policy task_bc_policy \
  --seeds 0 \
  --episodes 20 \
  --variants base verifier recovery verifier_recovery \
  --expert-provider mujoco-heuristic \
  --out benchmark_results/mujoco_pusher_calibrated_debug \
  --no-replay
```

Action-sequence policies can report and execute action chunks:

```bash
uv run nyssa run \
  --suite mujoco_control_v0 \
  --engine mujoco \
  --policy path/to/action_chunk_policy.py \
  --episodes 10 \
  --seed 0 \
  --policy-action-horizon 8 \
  --policy-execution-horizon 4 \
  --out runs/mujoco_action_chunk_smoke \
  --no-replay
```

Each recovery-aware run writes:

- `dataset_manifest.json`: provenance, task contracts, artifact hashes.
- `recovery_dataset/manifest.json`: counts of supervised recovery targets and negative/context records.
- `recovery_dataset/episodes.jsonl`: recovery context with explicit executed-action and target provenance.
- `failure_gallery.html`: representative failed episodes and replay links.
- `metrics.json`: success, intervention, recovery, verifier, action-chunk, and compute metrics.

Train the next task-routed BC checkpoints directly from one run directory or an
entire ablation result root. This is the safe default for multi-task suites such
as MuJoCo control, where tasks can have different action dimensions:

Recovery dataset v2 records unsuccessful recovery attempts as
`record_type: negative_context` with `target_valid: false`. Only actions whose
`target_source` is `expert` or `recovery` are eligible for BC training; rejected
policy actions remain auditable as `executed_action` and are never used as
supervised targets. Legacy datasets are accepted only when their step metadata
proves the action source.

Recovery outcomes use `nyssa-recovery-outcomes-v1`. An attempt is counted when
recovery is requested, but `recovery_success_rate` uses only attempts with a
non-empty applied plan as its denominator. Success is attributed only when the
task success predicate becomes true before a newer attempt and within the
configured step window. The window starts at the first recovery action and is
the larger of the configured horizon (five transitions by default) or the full
recovery-plan length. `recovery_episode_success_rate` separately uses episodes
with at least one applied recovery as its denominator. Configure the window for
`run`, `experiment`, or `ablate` with `--recovery-attribution-horizon`.

```bash
uv run nyssa train-recovery-bc \
  benchmark_results/mujoco_ablation_smoke \
  --routing task \
  --out-dir checkpoints/recovery_bc_by_task \
  --merged-out benchmark_results/mujoco_recovery_training/episodes.json

NYSSA_TASK_BC_DIR=checkpoints/recovery_bc_by_task \
NYSSA_TASK_BC_MISSING=zero \
NYSSA_MUJOCO_RECOVERY_TASKS=mujoco_pusher \
uv run nyssa ablate \
  --suite mujoco_control_v0 \
  --engine mujoco \
  --policy task_bc_policy \
  --seeds 0 \
  --episodes 5 \
  --variants base verifier recovery verifier_recovery \
  --expert-provider mujoco-heuristic \
  --recovery-attribution-horizon 5 \
  --out benchmark_results/mujoco_recovery_bc_ablation_smoke \
  --no-replay
```

Closed-loop MuJoCo recovery smoke:

```bash
NYSSA_MUJOCO_RECOVERY_TASKS=mujoco_pusher \
uv run nyssa ablate \
  --suite mujoco_control_v0 \
  --engine mujoco \
  --policy random \
  --seeds 0 \
  --episodes 20 \
  --variants base verifier recovery verifier_recovery \
  --expert-provider mujoco-heuristic \
  --out benchmark_results/mujoco_recovery_collect_v0 \
  --no-replay

uv run nyssa train-recovery-bc \
  benchmark_results/mujoco_recovery_collect_v0 \
  --routing task \
  --out-dir checkpoints/recovery_bc_by_task \
  --merged-out benchmark_results/mujoco_recovery_training_v0/episodes.json

NYSSA_TASK_BC_DIR=checkpoints/recovery_bc_by_task \
NYSSA_TASK_BC_MISSING=zero \
uv run nyssa ablate \
  --suite mujoco_control_v0 \
  --engine mujoco \
  --policy task_bc_policy \
  --seeds 0 \
  --episodes 20 \
  --variants base verifier recovery verifier_recovery \
  --expert-provider mujoco-heuristic \
  --out benchmark_results/mujoco_recovery_bc_eval_v0 \
  --no-replay

uv run nyssa compare \
  benchmark_results/mujoco_recovery_collect_v0/verifier_recovery/seed_0 \
  benchmark_results/mujoco_recovery_bc_eval_v0/verifier_recovery/seed_0 \
  --out reports/mujoco_recovery_bc_compare.html
```

For task-routed policies, emit one checkpoint per task under the existing
`task_bc_policy` directory layout:

```bash
uv run nyssa train-recovery-bc \
  benchmark_results/mujoco_ablation_smoke \
  --by-task \
  --out-dir checkpoints/bc_by_task

NYSSA_TASK_BC_DIR=checkpoints/bc_by_task \
NYSSA_TASK_BC_MISSING=zero \
uv run nyssa run \
  --suite mujoco_control_v0 \
  --engine mujoco \
  --policy task_bc_policy \
  --episodes 10 \
  --seed 0 \
  --out runs/mujoco_task_recovery_bc_smoke \
  --no-replay
```

After smoke runs pass, scale to public-claim settings:

```bash
NYSSA_MUJOCO_RECOVERY_TASKS=mujoco_pusher \
uv run nyssa ablate \
  --suite mujoco_control_v0 \
  --engine mujoco \
  --policy random \
  --seeds 0 1 2 \
  --episodes 100 \
  --variants base verifier recovery verifier_recovery \
  --expert-provider mujoco-heuristic \
  --out benchmark_results/mujoco_ablation_v0 \
  --capture-replay
```

Run the focused ManiSkill baseline matrix after installing the ManiSkill extras,
collecting scripted demos, and training the repo-local BC checkpoint:

```bash
uv run nyssa experiment \
  --suite maniskill_manipulation_v0 \
  --engine maniskill \
  --policies scripted_oracle \
  --seeds 0 1 2 \
  --episodes 100 \
  --out benchmark_results/maniskill_manipulation_v0_demos

uv run nyssa train-bc \
  benchmark_results/maniskill_manipulation_v0_demos/scripted_oracle/seed_0/episodes.json \
  benchmark_results/maniskill_manipulation_v0_demos/scripted_oracle/seed_1/episodes.json \
  benchmark_results/maniskill_manipulation_v0_demos/scripted_oracle/seed_2/episodes.json \
  --out checkpoints/bc_policy.json

uv run nyssa export \
  --run benchmark_results/maniskill_manipulation_v0_demos/scripted_oracle/seed_0 \
  --format robomimic \
  --out benchmark_results/maniskill_manipulation_v0_demos/scripted_oracle/seed_0/robomimic.hdf5

NYSSA_BC_CHECKPOINT=checkpoints/bc_policy.json \
uv run nyssa experiment \
  --suite maniskill_manipulation_v0 \
  --engine maniskill \
  --policies random scripted_oracle bc_policy \
  --seeds 0 1 2 \
  --episodes 100 \
  --out benchmark_results/maniskill_manipulation_v0
```

The run folder contains:

```txt
runs/random_mujoco_seed0/
|-- config.yaml
|-- run.yaml
|-- environment.json
|-- package_versions.json
|-- git_info.json
|-- metrics.json
|-- metrics.csv
|-- episodes.json
|-- episodes.jsonl
|-- dataset_manifest.json
|-- recovery_dataset/
|-- replay_manifest.json
|-- replay.html
|-- failure_gallery.html
|-- videos/
|-- failures/
|-- plots/
|-- lerobot/
`-- report.html
```

Import official ManiSkill motion-planning demonstrations before training
planner-backed learned baselines:

```bash
uv run nyssa collect-maniskill-demos \
  --env-ids PickCube-v1 PushCube-v1 StackCube-v1 \
  --num-traj 100 \
  --raw-dir demos/maniskill_motionplanning_raw \
  --out benchmark_results/maniskill_manipulation_v0_planner_demos

uv run nyssa import-maniskill-demos \
  --input demos/maniskill_motionplanning \
  --out benchmark_results/maniskill_manipulation_v0_planner_demos
```

`collect-maniskill-demos` runs ManiSkill's Panda motion-planning example and
then imports the generated HDF5 files into Nyssa. If your installed ManiSkill
version uses a different generator command, override it with
`--command-template` or `NYSSA_MANISKILL_DEMO_COMMAND`. The template can use
`{python}`, `{env_id}`, `{task_id}`, `{num_traj}`, `{raw_dir}`, and
`{raw_task_dir}` placeholders.

Evaluate behavior-cloned policies from those planner demos with the
planner-aligned suite:

```bash
uv run nyssa train-task-bc \
  benchmark_results/maniskill_manipulation_v0_planner_demos \
  --out-dir checkpoints/maniskill_planner_task_bc \
  --model sequence-knn \
  --feature-dim 512 \
  --action-horizon 16

NYSSA_TASK_BC_DIR=checkpoints/maniskill_planner_task_bc \
NYSSA_TASK_BC_MISSING=zero \
uv run nyssa ablate \
  --suite maniskill_planner_bc_v0 \
  --engine maniskill \
  --policy task_bc_policy \
  --seeds 0 \
  --episodes 20 \
  --variants base \
  --expert-provider maniskill-scripted \
  --policy-action-horizon 16 \
  --policy-execution-horizon 4 \
  --out benchmark_results/maniskill_task_bc_smoke \
  --capture-replay
```

`train-task-bc` accepts extracted result/import directories, direct
`episodes.json` files, or zipped result packs. It recursively discovers nested
`episodes.json` files and skips generated `recovery_dataset` folders.

Use state-aligned demonstration replay as the validated ManiSkill teacher upper
bound. This is an oracle/reference result, not a learned policy:

```bash
NYSSA_DEMO_REPLAY_DIR=benchmark_results/maniskill_manipulation_v0_planner_demos \
NYSSA_DEMO_REPLAY_FEATURE_DIM=512 \
uv run nyssa run \
  --suite maniskill_planner_bc_v0 \
  --engine maniskill \
  --policy demo_replay_policy \
  --episodes 10 \
  --seed 0 \
  --out runs/maniskill_demo_replay_smoke \
  --capture-replay
```

For the repo-local BC baseline, prefer `task_bc_policy` with one checkpoint per
task. See `docs/learned_baselines.md` for the exact training
commands. For stronger learned baselines, use the RoboMimic export and
`robomimic` or `task_robomimic` policy adapters documented there.

For task-routed RoboMimic training, export one dataset and config per task from
state-aligned rollout episodes containing live policy observations. Action-only
motion-planning imports are not sufficient. The exporter rejects sources with
less than 95% observation payload coverage or degenerate features and records
per-task quality statistics in `task_robomimic_manifest.json`. It also maps
every bounded simulator action to RoboMimic's required `[-1, 1]` training range
and records the original per-task bounds. At evaluation, `task_robomimic`
checks those bounds against the live environment and maps predictions back to
the simulator action space. The manifest records training episode seeds, and
the task policy refuses to evaluate those seeds unless
`NYSSA_ALLOW_TRAINING_SEED_EVAL=1` is explicitly set for a nonpublishable
diagnostic run.

RoboMimic datasets exported before manifest format
`nyssa-task-robomimic-export-v3` contain raw actions and must be re-exported and
retrained. Increasing epochs on those older checkpoints cannot repair the
action-space mismatch.

Evaluate checkpoints from result-pack training sources on simulator seeds that
do not occur in those sources. Reusing training rollout seeds is acceptable for
a pipeline smoke test but invalidates a public generalization claim.

Run seed values are namespaces, not the first element of an overlapping range.
Under `nyssa-episode-seed-v2`, run seed `1` produces simulator episode seeds
starting at `1000000`, while every task within that run receives the same seed
sequence for paired analysis. Result packs generated before this protocol must
be rerun before making an independent multi-seed claim.

Export one source directory or result ZIP:

```bash
uv run nyssa export-task-robomimic \
  benchmark_results/maniskill_manipulation_v0_planner_demos \
  --out-dir datasets/maniskill_robomimic_by_task \
  --config-dir configs/generated/maniskill_robomimic_by_task \
  --feature-dim 512 \
  --epochs 50 \
  --batch-size 64
```

Train the generated configs, then evaluate from the export directory directly.
`task_robomimic` recursively discovers the latest per-task
`model_epoch_*.pth` files and reads each checkpoint's observation feature
dimension, so manual checkpoint copying and
`NYSSA_ROBOMIMIC_FEATURE_DIM` are normally unnecessary:

```bash
uv run nyssa train-robomimic --config configs/generated/maniskill_robomimic_by_task/maniskill_pick_cube_bc.json
uv run nyssa train-robomimic --config configs/generated/maniskill_robomimic_by_task/maniskill_push_cube_bc.json
uv run nyssa train-robomimic --config configs/generated/maniskill_robomimic_by_task/maniskill_stack_cube_bc.json

NYSSA_TASK_ROBOMIMIC_DIR=datasets/maniskill_robomimic_by_task \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
uv run nyssa ablate \
  --suite maniskill_planner_bc_v0 \
  --engine maniskill \
  --policy task_robomimic \
  --seeds 10000 \
  --episodes 20 \
  --variants base \
  --expert-provider maniskill-scripted \
  --out benchmark_results/maniskill_task_robomimic_smoke \
  --capture-replay
```

Validate optional simulator backends:

```bash
uv run python scripts/validate_backend.py maniskill
uv run python scripts/validate_backend.py mujoco
uv run python scripts/validate_backend.py robocasa
uv run python scripts/validate_backend.py genesis
```

`robocasa` and `genesis` validate their experiment contracts by default. They only run full simulator checks after concrete task-to-scene mappings are added.

Run the release check from a clean virtual environment:

```bash
uv run python scripts/release_smoke.py
```

## What Is Included In v0.1

- Core benchmark API: load suite, load policy, run episodes, collect metrics, save replay videos, export datasets, generate reports.
- Engine adapters: ManiSkill, MuJoCo, and experimental RoboCasa/Genesis adapters with explicit validation contracts.
- Task YAML DSL for tabletop, warehouse, articulated-object, and stress-test suites.
- Policy adapters: random, repo-local scripted heuristic, repo-local linear BC, robomimic checkpoint loading, LeRobot checkpoint loading, plus hook-only OpenVLA and diffusion adapters. External policy adapters accept `NYSSA_SCRIPTED_ORACLE_POLICY`, `NYSSA_BC_POLICY`, `NYSSA_LEROBOT_POLICY`, `NYSSA_OPENVLA_POLICY`, `NYSSA_ROBOMIMIC_POLICY`, or `NYSSA_DIFFUSION_POLICY` as `module:factory` entry points.
- Baseline experiment command for policy/seed matrices and result-pack generation.
- Recovery/ablation command for base, verifier, recovery, and verifier+recovery variants.
- Expert-provider interface with built-in `bounds-verifier`, `maniskill-scripted`, `mujoco-heuristic`, and `policy:<name>` providers.
- Action-sequence metadata and execution hooks for chunked policies.
- Failure taxonomy, mapper-based failure labels, and aggregate metrics.
- HTML reports, JSON metrics, recovery datasets, failure galleries, public-claim validation, and unsupported-stressor reporting.
- Policy comparison reports, prototype reliability scores, and leaderboard export.
- Static leaderboard shell, protocol draft, scorecard structure, Docker files, and plugin API.
- CLI, docs, examples, and tests.

## Positioning

NyssaBench helps robotics teams evaluate embodied AI policies under realistic variation before deploying them to real robots. It focuses on policy-agnostic evaluation, failure analysis, stress testing, replay-first reports, and dataset export rather than owning a physics engine.
