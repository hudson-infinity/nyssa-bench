# Installation

## Choose an installation path

Most users should install the released package:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install nyssa-bench
nyssa --help
nyssa --version
```

This installs the Python API and CLI together. The base distribution excludes
simulators, GPU/model stacks, plotting libraries, video codecs, and dataset
libraries.

Install one workflow extra when the core package is not enough:

```bash
python -m pip install "nyssa-bench[mujoco]"
python -m pip install "nyssa-bench[maniskill]"
python -m pip install "nyssa-bench[dataset]"
python -m pip install "nyssa-bench[lerobot]"
python -m pip install "nyssa-bench[robomimic]"
python -m pip install "nyssa-bench[vla]"
python -m pip install "nyssa-bench[diffusion]"
```

The MuJoCo and ManiSkill extras include video encoding because their supported
run path captures replay evidence by default. The ManiSkill profile pins
ManiSkill 3.0.1 and PyTorch 2.6.0, matching the tested CUDA 12.4 container; VLA
and diffusion use the corresponding torchvision 0.21.0 release. Host graphics
drivers and Vulkan, EGL, or X11 libraries remain system dependencies and cannot
be supplied by a Python wheel. ManiSkill supports CPU simulation under WSL but
does not support GPU simulation or rendering there, so replay-backed evaluation
requires native Linux with a Vulkan-capable device.

Clone the repository and use the `uv sync` commands below only for contributor
development or unreleased source testing.

## Python Versions

`pyproject.toml` declares Python 3.10 or newer. The contributor validation
matrix is narrower:

| Workflow | Python | Notes |
| --- | --- | --- |
| Core package, tests, docs, and lightweight CI | 3.11 | Canonical contributor and GitHub Actions baseline |
| ManiSkill motion planning | 3.10 | Required for reliable `toppra`/`mplib` wheel and ABI compatibility |
| MuJoCo development | 3.10 or 3.11 | Install the `mujoco` extra and run the backend smoke test |
| Python 3.12+ | Conditional | Core may work, but simulator and learned-policy extras require separate validation |

A contribution that changes an optional runtime is supported only on Python
versions where that runtime and its compiled dependencies were exercised. State
the interpreter version in the pull request when simulator behavior changes.

Install `uv` using its official instructions, then confirm the interpreter
selected for the project:

```bash
uv --version
uv run python --version
```

## Canonical Environments

Use `uv` to install the canonical stable development and benchmark environment:

```bash
uv sync --extra all --extra dev
uv run nyssa list-suites
uv run pytest -q
uv run ruff check .
```

The `all` extra includes every supported stable runtime stack and excludes the
experimental Genesis dependency. `uv sync` performs an exact sync by default,
so a later invocation that omits extras can remove packages installed by an
earlier invocation. Repeat `uv sync --extra all --extra dev` after dependency
updates instead of treating multiple exact sync commands as additive.

For external contributions, this is the default setup. The GitHub Actions job
installs only `.[dev]`, so CI exercises core contracts but does not install or
validate MuJoCo, ManiSkill, learned-policy, dataset, or plotting extras.

For a dedicated lightweight environment, select one complete workflow:

```bash
uv sync --extra dev --extra maniskill --extra video --extra reports
uv sync --extra dev --extra mujoco --extra video --extra reports
```

Use `mujoco` for the lightest real backend path and `maniskill` for manipulation
tasks. To add packages to an existing lean environment without removing
previously installed extras, use `--inexact` and include all additions in one
command:

```bash
uv sync --inexact --extra dataset --extra lerobot --extra robomimic --extra vla --extra diffusion
```

Use `--inexact` only when intentionally extending an existing environment.
Before reproducing CI or release behavior, return to a declared exact profile.

## Contributor Verification

After installation, verify the repository and static configuration contracts:

```bash
uv run nyssa list-suites
uv run pytest -q
uv run ruff check .
uv run python scripts/validate_configs.py
uv run python scripts/release_checklist.py
```

Changes to a simulator boundary also require the corresponding command:

```bash
uv run python scripts/validate_backend.py mujoco --episodes 1
uv run python scripts/validate_backend.py maniskill --episodes 1
```

See [CONTRIBUTING.md](../CONTRIBUTING.md) for the complete validation matrix,
artifact rules, and pull-request checklist.

## ManiSkill Motion-Planning ABI

Use Python 3.10 and NumPy 1.26 for ManiSkill motion-planning demos. The
planner stack imports compiled packages such as `toppra`; NumPy 2 can trigger
`numpy.core.multiarray failed to import` when those extensions were built
against the NumPy 1.x ABI.

If an existing venv has NumPy 2 installed, repair it before generating
motion-planning demonstrations:

```bash
uv pip install "numpy==1.26.4"
uv pip install --force-reinstall --no-build-isolation --no-cache-dir "toppra==0.6.3"
python - <<'PY'
import numpy, toppra, mplib
print("numpy", numpy.__version__)
print("toppra", toppra.__version__)
print("mplib import ok")
PY
```

## Rendering System Packages

Python packages are not enough for video-backed robotics benchmarks. Public
NyssaBench benchmark claims require MP4 replay evidence, so install simulator
rendering libraries before running result packs.

On Ubuntu/Debian GPU machines:

```bash
bash scripts/setup_rendering_linux.sh
vulkaninfo --summary
nvidia-smi
```

The helper installs common GL/Vulkan/X11 runtime libraries:

```txt
libvulkan1 vulkan-tools mesa-vulkan-drivers libglvnd0 libgl1 libegl1
libglfw3 libx11-6 libxext6 libxrender1 libxrandr2 libxinerama1
libxcursor1 libxi6
```

On NVIDIA GPU machines, Vulkan also needs the NVIDIA ICD/GL packages matching
the installed driver branch. For example, on driver branch 535:

```bash
sudo apt-get install -y nvidia-utils-535 libnvidia-gl-535
```

If `vulkaninfo --summary` cannot see a Vulkan device, fix the host NVIDIA
driver/ICD setup before running ManiSkill result packs. Video-less runs are
allowed only for local smoke tests with `--no-replay`; they are not public
benchmark results.

If `vulkaninfo --summary` lists only `llvmpipe`, Vulkan is using CPU rendering.
That is not sufficient for ManiSkill video-backed public result packs; rerun on
a machine or container where the NVIDIA Vulkan ICD is exposed.

On macOS, MuJoCo smoke runs usually need native GLFW:

```bash
brew install glfw
```

ManiSkill video-backed result packs are expected to run on Linux with a working
NVIDIA/Vulkan stack.

If you are not using `uv`, the equivalent pip command is:

```bash
python -m venv .venv
python -m pip install -e ".[all,dev]"
```

## Extras

| Extra | Purpose |
| --- | --- |
| `cli` | Empty compatibility alias; the argparse CLI is included in the base package. |
| `dataset` | HDF5 and Parquet export. |
| `reports` | Template and plotting dependencies. |
| `video` | MP4/frame export dependencies. |
| `maniskill` | ManiSkill adapter runtime dependencies. |
| `mujoco` | MuJoCo adapter runtime dependencies. |
| `lerobot` | LeRobot policy and dataset integration dependencies. |
| `robomimic` | robomimic baseline dependencies. |
| `robocasa` | Experimental adapter contract only; install RoboCasa from upstream in a separate environment. |
| `vla` | Shared PyTorch/Transformers dependencies for VLA adapters such as OpenVLA. |
| `diffusion` | Diffusion policy baseline dependencies. |
| `experimental` | Experimental Genesis dependency. RoboCasa may still require a source install depending on upstream packaging. |
| `all` | Everything except experimental backends. |
| `full` | Declared heavy extras except RoboCasa, which currently needs a separate upstream environment. |

OpenVLA and some robotics diffusion-policy codebases are commonly installed from their upstream GitHub repositories rather than as stable PyPI packages. NyssaBench declares their common runtime stack in `vla` and `diffusion`, while the model code/checkpoints should be installed according to the upstream project instructions.

RoboCasa requires additional setup beyond Python package installation. Follow the upstream RoboCasa docs to set up macros and download kitchen assets after installing the source packages:

```bash
python -m robocasa.scripts.setup_macros
python -m robocasa.scripts.download_kitchen_assets
```
