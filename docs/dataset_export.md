# Dataset Export

v0.1 writes JSON rollouts and a lightweight LeRobot-compatible manifest without requiring LeRobot at runtime.

Optional HDF5, robomimic HDF5, and Parquet exporters are available through the `dataset` extra:

```bash
uv sync --inexact --extra dataset
```

`--inexact` adds the exporter dependencies without removing simulator or policy
extras already installed in the active environment. A fresh full benchmark
environment can use `uv sync --extra all --extra dev` instead.

Export a run for robomimic BC training:

```bash
uv run nyssa export --run runs/scripted_oracle --format robomimic --out runs/scripted_oracle/robomimic.hdf5
```

The RoboMimic exporter requires a finite Box action contract on every step,
requires consistent action shape and bounds within one HDF5 file, rejects
out-of-bounds demonstrations, and affinely normalizes actions to `[-1, 1]`.
The original bounds are stored in the HDF5 `data` group's
`nyssa_action_transform` attribute. Use `export-task-robomimic` when tasks have
different action contracts.

JSON and JSONL episode exports preserve the complete
`nyssa-failure-ledger-v1` payload. Legacy failed episodes with only a flat
failure label receive a terminal-only migration event when loaded for export.
Specialized learned-policy exports continue to use the flat label until the
downstream failure-evidence export contract is versioned separately.
