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
