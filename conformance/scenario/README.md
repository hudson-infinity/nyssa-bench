# Scenario conformance fixtures

These fixtures test the external scenario producer contract. They contain no
world-generation algorithm.

`v1/valid_seeded_mujoco/` is a complete, redistributable package that maps to a
checked-in MuJoCo task and a registered action-delay stressor. Validate it with:

```bash
uv run nyssa validate-scenario conformance/scenario/v1/valid_seeded_mujoco
```

External producer repositories may use the fixture and
`nyssa_bench.scenarios.ScenarioPackageValidator` in their own conformance tests.
When any identity-bearing field changes, recompute `content_sha256` with
`ScenarioPackage.compute_content_sha256()`.
