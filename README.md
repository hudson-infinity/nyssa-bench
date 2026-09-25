# NyssaBench

[![CI](https://github.com/hudson-infinity/nyssa-bench/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/hudson-infinity/nyssa-bench/actions/workflows/ci.yml)
[Documentation](docs/index.md) · [Quickstart](#quickstart) · [Contributing](CONTRIBUTING.md) · [Apache 2.0](LICENSE)

NyssaBench is an open-source evaluation and audit framework for robot policies.

Evaluate a policy in MuJoCo or ManiSkill with controlled disturbances. Each run
records actions, failure events, recovery attempts, and replay evidence alongside
its metrics, so you can inspect where a policy failed and assess changes to it.

## What you can evaluate

- Run registered policies or your own Python adapter on shared task definitions
  with reproducible episode seeds.
- Apply visual, sensor, action, system, and dynamics stressors. Results record
  which changes the backend actually applied.
- Inspect temporal failure events, including collision, contact loss, and
  stalled progress, with their supporting evidence.
- Compare verifier and recovery interventions. Matched continuation/recovery
  branches measure recovery effects when the components support state restoration.
- Compare checkpoints, audit result validity, and export rollout or recovery
  data for downstream training.

Success, robustness, failure, recovery, safety, compute, and sim-real measurements
remain separate in the [metric vector](docs/metrics.md).

## Quickstart

Use Python 3.11 and [uv](https://docs.astral.sh/uv/getting-started/installation/)
for this MuJoCo example. Start from a source checkout and install the simulator
and report dependencies in one environment:

```bash
git clone https://github.com/hudson-infinity/nyssa-bench.git
cd nyssa-bench
uv sync --locked --python 3.11 --extra mujoco --extra reports
```

Run three episodes of the inverted-pendulum task:

```bash
uv run nyssa run \
  --suite mujoco_control_v0 \
  --tasks mujoco_inverted_pendulum \
  --engine mujoco \
  --policy random \
  --episodes 3 \
  --seed 0 \
  --out runs/quickstart \
  --no-replay
```

Open `runs/quickstart/report.html` in a browser. The random policy is a control
for checking the evaluation pipeline; use your own policy for performance studies.
Remove `--tasks mujoco_inverted_pendulum` to run the whole MuJoCo control suite.

`--no-replay` lets you check the pipeline before setting up rendering. To record
videos, configure the host graphics libraries described in the
[installation guide](docs/installation.md#rendering-system-packages), then omit
that flag and choose a new output directory. Public result claims require replay
evidence and the relevant validity checks.

Discover the available configurations and commands:

```bash
uv run nyssa list-suites
uv run nyssa list-tasks
uv run nyssa list-policies
uv run nyssa list-stressors
uv run nyssa --help
```

### Python API

The same evaluation is available from Python:

```python
from nyssa_bench import PolicyRunner, Suite

suite = Suite.load("mujoco_control_v0").filter_tasks(["mujoco_inverted_pendulum"])
runner = PolicyRunner(
    policy="random",
    engine="mujoco",
    episodes=3,
    seed=0,
    out="runs/python_quickstart",
    capture_replay=False,
)
report = runner.evaluate(suite)
print(report.summary["success_rate"])
```

## Inspect and reuse a run

Each run keeps its measurements and the evidence used to produce them together.
Start with these files:

| Artifact | Contents |
| --- | --- |
| `report.html` | Metrics, failure summaries, and validation status |
| `metrics.json` | Aggregate metrics, uncertainty, and the metric vector |
| `episodes.json` / `episodes.jsonl` | Episode identities and recorded transitions |
| `failure_ledger.json` | Temporal failure events and supporting evidence |
| `stressor_manifest.json` | Requested conditions, application status, and backend evidence |
| `dataset_manifest.json` | Provenance, task contracts, and artifact hashes |
| `run.yaml` | Run configuration and evaluation settings |

Replay-enabled runs also include videos and replay pages. Recovery and
counterfactual studies add their own records. See the
[report guide](docs/reports.md) and [recovery workflows](docs/recovery_workflows.md).

Regenerate the report or export the quickstart episodes:

```bash
uv run nyssa report runs/quickstart
uv run nyssa export --run runs/quickstart --format jsonl
```

The [dataset export guide](docs/dataset_export.md) covers HDF5, RoboMimic,
Parquet, and the lightweight LeRobot format. For evidence handoff to a learning
system, use the [learning evidence export](docs/learning_evidence_export.md).

## Add your policy

Start with a runnable state-policy example and its contract:

```bash
uv run nyssa write-policy-example --kind state --out runs/policy_example
uv run nyssa conform-policy \
  --policy runs/policy_example/state_policy.py \
  --policy-contract runs/policy_example/state_policy_contract.json \
  --suite mujoco_control_v0 \
  --task mujoco_inverted_pendulum \
  --engine mujoco \
  --episodes 1 \
  --out runs/policy_example/conformance
```

A policy file exposes `create_policy()` or `PolicyAdapter` and returns an
object with `act(observation)`. The conformance check exercises metadata,
reset behavior, observations, actions, and a small rollout before a larger study.

Follow the [external policy quickstart](docs/external_policy_quickstart.md) to
adapt the example to your checkpoint. The [adapter reference](docs/policy_adapters.md)
covers the RoboMimic, LeRobot, diffusion, and VLA hooks. Their checkpoint and
runtime dependencies are installed separately.

## Run a study

Apply a built-in action-delay condition to the quickstart task:

```bash
uv run nyssa run \
  --suite mujoco_control_v0 \
  --tasks mujoco_inverted_pendulum \
  --engine mujoco \
  --policy random \
  --episodes 3 \
  --seed 0 \
  --stressor-config configs/stressors/action_delay_s05.yaml \
  --out runs/action_delay \
  --no-replay
```

Use the same task definitions and episode seeds across conditions. This small
run checks the stressor pipeline; a robustness study also needs a matched clean
condition, severity coverage, and uncertainty estimates.

| Study | Guide |
| --- | --- |
| Clean and shifted performance across stressor severities | [Stressor protocol](docs/stressor_protocol.md) |
| Verifier/recovery ablations and recovery-data collection | [Recovery workflows](docs/recovery_workflows.md) |
| Recovery versus continuation from the same state | [Counterfactual recovery](docs/counterfactual_recovery.md) |
| Policy comparisons with compatible evaluation contracts | [Policy comparison](docs/policy_comparison.md) |
| Prespecified checkpoint pass/fail decisions | [Policy regression gates](docs/policy_regression_gates.md) |
| Searching for failure boundaries | [Stress search](docs/stress_search.md) |
| Training and evaluating the included BC baselines | [Learned baselines](docs/learned_baselines.md) |

## Simulator support

| Backend | Current scope |
| --- | --- |
| MuJoCo | Implemented adapter and Gymnasium control suite; installed-wheel smoke tests run in CI |
| ManiSkill | Implemented manipulation adapter; use Python 3.10 and the documented Linux/Vulkan setup for planning and replay |
| RoboCasa / Genesis | Experimental integration contracts; executable tasks require concrete scene mappings and upstream setup |

The project pins ManiSkill and PyTorch versions. GPU execution and replay need
a capable host. Container metadata checks do not establish GPU simulator coverage.

Choose a complete dependency profile in the
[installation guide](docs/installation.md#canonical-environments).
Use `uv sync --inexact --extra <name>` when adding extras to an existing environment;
an exact sync can remove extras omitted from the command. The
[Docker guide](docs/docker.md) covers the repository's container definitions.

## Current evidence and limits

NyssaBench is under active development. The measurement implementation has
source and test evidence, while the reference benchmark, validated learned-policy
tracks, and predictive hardware studies still need their required result evidence.
The [claim evidence matrix](docs/claim_evidence.md) currently approves no headline
benchmark result pack.

[Run validation](docs/validation_protocol.md) checks completeness and recorded
evidence. [Benchmark audits](docs/benchmark_validity.md) check the evaluation
design, including leakage and shortcuts. The
[Phase 1 credibility gate](docs/phase1_credibility_gate.md) records what evidence
is present and what is missing. A smoke test or passing adapter check does not
establish robot performance or real-world predictive validity.

NyssaBench owns evaluation and evidence contracts. World generation, scene
reconstruction, general policy training, and hosted products belong in external
projects that connect through those contracts. See the
[project scope](docs/project_scope.md) and
[Nyssa Evaluation Protocol](docs/nyssa_evaluation_protocol.md).

## Contribute

Bug reports, task contracts, simulator adapters, and policy integrations are
welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) for the validation requirements
and [API stability](docs/api_stability.md) before changing public contracts.

To add development tools to the quickstart environment and run the baseline checks:

```bash
uv sync --locked --inexact --extra dev
uv run pytest -q
uv run pre-commit run --all-files
```

Pull requests use conventional titles and squash merges after required CI passes.
The [automation guide](docs/github_automation.md) explains version PRs,
dependency updates, and the `automerge` and `hold` labels.

Report vulnerabilities through [SECURITY.md](SECURITY.md). This project follows
the [Code of Conduct](CODE_OF_CONDUCT.md).

Built by Hudson Labs. Licensed under [Apache 2.0](LICENSE).
