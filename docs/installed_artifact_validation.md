# Installed artifact validation

NyssaBench validates built distributions separately from source-checkout tests.
This catches package data, metadata, optional dependency, and path bugs that an
editable install can hide.

## Distribution inspection

Build and inspect the wheel and source distribution:

```bash
uv build
uvx twine check --strict dist/*
uv run python scripts/validate_distributions.py dist/* \
  --out distribution-validation.json
```

The validator requires one wheel and one `.tar.gz` source distribution with the
same version. It checks:

- distribution name, Python requirement, version, and every declared extra;
- the package, license, README, task files, suite files, and conformance
  fixtures;
- byte-for-byte identity between source configs/fixtures and wheel resources;
- duplicate or path-traversing archive members;
- accidental result packs, checkpoints, virtual environments, nested ZIPs, and
  files larger than 5 MiB.

The JSON report records the artifact names, sizes, SHA-256 digests, versions,
and member counts.

## External working-directory smoke

After installing only the wheel, run:

```bash
python -m nyssa_bench.packaging_smoke --out /tmp/nyssa-installed-smoke
```

The smoke module loads packaged suite, task, stressor, and conformance resources
through `importlib.resources`. It then executes one deterministic internal
episode and writes metrics, episodes, a dataset manifest, and an HTML report.
The result is intentionally `prototype` and `public_claim: false`; this test
validates installation and reporting, not robot-policy performance.

`installed_artifact_smoke.json` records the installed package path, working
directory, resource paths, discovered resource counts, result status, and
artifact hashes. The package path and resource paths should point into the
clean environment, not the source checkout.

## Continuous integration

Pull-request CI has three independent layers:

1. source tests on Python 3.11;
2. one artifact build plus strict archive inspection;
3. installed-wheel smoke jobs on Python 3.10 and 3.13 without a checkout.

The release workflow consumes the same built artifact and repeats the installed
smoke before either package index job can run. A publish job cannot proceed if
artifact inspection or either Python smoke fails.

The scheduled simulator workflow builds the wheel first, installs its MuJoCo
extra, and runs a result-pack smoke on Ubuntu. A manually selected
`maniskill-gpu` job targets a self-hosted Linux GPU runner and verifies the
ManiSkill import plus replay-producing execution. If no matching GPU runner is
registered, that optional job cannot execute; it is not silently replaced with
a CPU or mocked claim.

See [Simulator-backed continuous integration](simulator_ci.md) for the executed
checks, diagnostic contract, and flakiness promotion rule.

## Resource layout

Source checkouts keep configuration in `configs/` and fixtures in
`conformance/`. Wheels install both under `nyssa_bench/_resources/`.
`nyssa_bench.package_resources` selects packaged data first and uses the source
tree only as a contributor fallback. Runtime loaders do not search the current
working directory.
