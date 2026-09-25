# Getting started

The first run uses MuJoCo's inverted-pendulum task and a random policy. It
checks that evaluation and report generation work before you configure a
larger study or a rendering device.

## Install from source

Use Python 3.11 and [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
git clone https://github.com/hudson-infinity/nyssa-bench.git
cd nyssa-bench
uv sync --locked --python 3.11 --extra mujoco --extra reports
```

## Run three episodes

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

Open `runs/quickstart/report.html`. `metrics.json` contains the aggregate
measurements, while `episodes.json` records each episode's actions and outcomes.
The report also shows validation status. A small run without replay is a
pipeline check and cannot support a public performance claim.

Regenerate the report or export its episodes as JSONL:

```bash
uv run nyssa report runs/quickstart
uv run nyssa export --run runs/quickstart --format jsonl
```

## Continue with your experiment

- [Installation](installation.md) covers simulator profiles and replay rendering.
- [External policy quickstart](external_policy_quickstart.md) shows how to load
  your policy and run conformance checks.
- [Stressor protocol](stressor_protocol.md) describes clean/shifted comparisons.
- [Recovery workflows](recovery_workflows.md) covers ablations and recovery data.
- [Benchmark validity](benchmark_validity.md) explains evidence requirements for
  stronger result claims.
