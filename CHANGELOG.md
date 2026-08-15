# Changelog

## Unreleased

- Normalize bounded RoboMimic actions to `[-1, 1]`, persist per-task action
  transforms, and validate live inference bounds.
- Record training episode seeds and reject task-policy evaluation leakage by
  default.
- Namespace simulator episode seeds by run seed so multi-seed experiments are
  disjoint while task comparisons remain paired.
- Make recovery-only ablations activate independently from verifier fallback
  intervention.
- Require real MP4 files, simulator versions, a commit, and a clean worktree for
  public run validation.
- Execute ManiSkill collection templates with the active interpreter and
  shell-safe placeholders.

## 0.1.0

- Initial benchmark harness scaffold.
- Dummy engine, ManiSkill/MuJoCo adapter boundaries, and experimental RoboCasa/Genesis boundaries.
- Task YAML specs, suite loading, policy adapters, metrics, reports, dataset export, comparison reports, and leaderboard export.
