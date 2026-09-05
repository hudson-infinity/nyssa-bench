# v0.1 Launch Checklist

NyssaBench v0.1 is ready to launch when these checks pass from a clean checkout:

```bash
uv run python scripts/release_smoke.py
uv run python scripts/validate_backend.py maniskill
uv run python scripts/validate_backend.py mujoco
uv run python scripts/validate_backend.py robocasa
uv run python scripts/validate_backend.py genesis
```

Generated release artifacts:

- `runs/maniskill_validation/report.html`
- `runs/mujoco_validation/report.html`
- `benchmark_results/baselines_v0.json`

Run artifacts and scorecards are generated during release validation. Do not
treat stale local result files as source artifacts.

Tag and publish:

```bash
git tag -s v0.1.0rc1 -m "NyssaBench 0.1.0rc1"
git push origin v0.1.0rc1
```

Qualify the TestPyPI candidate before creating the stable signed tag. Follow
[Python package releases](releasing.md); do not bypass the protected `pypi`
environment.

Announcement draft:

```text
We are open-sourcing NyssaBench v0.1, an evaluation and failure-analysis framework for embodied AI policies under real-world variation.

It lets robotics teams run reproducible manipulation benchmarks, stress-test policies, generate replay-first HTML reports, classify failure modes, compare policy versions, and export trajectories for training.

The v0.1 release includes ManiSkill and MuJoCo adapter paths, task YAML specs, policy adapter entry points, LeRobot-style export, reports, and release checks.

Repo: https://github.com/hudson-infinity/nyssa-bench
```
