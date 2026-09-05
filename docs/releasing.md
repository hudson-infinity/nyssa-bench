# Releasing the Python package

NyssaBench publishes one Python distribution named `nyssa-bench`. The wheel
contains both the importable `nyssa_bench` package and the `nyssa` console
command. Simulator and policy stacks remain optional extras.

## Repository setup

The release workflow uses PyPI trusted publishing and does not accept a stored
API token. Repository administrators must configure two protected GitHub
environments:

| Environment | Publisher | Use |
| --- | --- | --- |
| `testpypi` | TestPyPI trusted publisher for `hudson-infinity/nyssa-bench` and `.github/workflows/release.yml` | Tags containing `rc` |
| `pypi` | PyPI trusted publisher for the same repository and workflow | Stable tags |

Require reviewer approval on the `pypi` environment. TestPyPI can use the same
protection while the first release is being qualified. PyPI supports a pending
trusted publisher for a project that does not exist yet; configure that before
pushing the first tag.

## Version contract

`nyssa_bench/version.py` is the only package-version source. Hatch reads it
through `[tool.hatch.version]`, and the CLI exposes it through `nyssa --version`.
The release workflow rejects a tag unless it exactly equals `v<version>`.

Validate locally before creating a tag:

```bash
uv run python scripts/validate_release_version.py --tag v0.1.0rc1
uv build
uvx twine check --strict dist/*
```

Update `nyssa_bench/version.py` and `CHANGELOG.md` in a reviewed pull request.
Do not change version metadata on the tag itself.

## Release candidate

Create a PEP 440 release-candidate version such as `0.1.0rc1`, merge it, and tag
the exact main commit:

```bash
git tag -s v0.1.0rc1 -m "NyssaBench 0.1.0rc1"
git push origin v0.1.0rc1
```

The workflow builds a wheel and source distribution, runs strict metadata
checks, attests build provenance, and installs the wheel outside the checkout on
Python 3.10 and 3.13. Only then does the `testpypi` job request an OIDC token and
publish to TestPyPI.

Install the candidate from TestPyPI in a clean environment. PyPI remains the
fallback index for dependencies:

```bash
python -m venv .test-release
source .test-release/bin/activate
python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  nyssa-bench==0.1.0rc1
nyssa --help
nyssa --version
```

Record the TestPyPI workflow run and installed smoke result in the stable
release pull request.

## Stable release

After the candidate passes installation review, set the stable version, update
the changelog, merge that release pull request, and create the matching signed
tag:

```bash
git tag -s v0.1.0 -m "NyssaBench 0.1.0"
git push origin v0.1.0
```

Stable tags run the same build and installed-wheel jobs, then wait at the
protected `pypi` environment. Approve that job only after checking the tag,
attestation, distribution metadata, and recorded TestPyPI candidate.

Published files are immutable. Fix a bad release with a new version; never
replace an existing wheel or source distribution.

## Distribution profiles

- Base: Python API, argparse CLI, contracts, bundled configurations, NumPy,
  Pydantic, and PyYAML.
- `mujoco`: Gymnasium, MuJoCo, and replay encoding.
- `maniskill`: Python 3.10-oriented ManiSkill, Gymnasium, NumPy 1.x, and replay
  encoding. System CUDA/Vulkan/driver requirements still apply.
- `reports`, `video`, and `dataset`: optional artifact tooling.
- `robomimic`, `lerobot`, `vla`, and `diffusion`: external policy workflows.
- `all`: supported stable extras without experimental Genesis.
- `full`: stable extras plus the experimental Genesis dependency.

RoboCasa remains an empty compatibility extra because its supported source and
asset installation is managed upstream. It must not be described as a complete
PyPI workflow until its dependency contract stabilizes.
