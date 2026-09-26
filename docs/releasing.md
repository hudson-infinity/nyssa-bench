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

Keep reviewer approval and tag deployment restrictions on both environments.
Register the publishers before pushing the first release tag:

1. Sign in to [TestPyPI publishing settings](https://test.pypi.org/manage/account/publishing/)
   and [PyPI publishing settings](https://pypi.org/manage/account/publishing/).
   These are separate services; configure each account separately.
2. Add a pending GitHub publisher on each service using these exact values:

   | Field | TestPyPI | PyPI |
   | --- | --- | --- |
   | PyPI project name | `nyssa-bench` | `nyssa-bench` |
   | Owner | `hudson-infinity` | `hudson-infinity` |
   | Repository name | `nyssa-bench` | `nyssa-bench` |
   | Workflow name | `release.yml` | `release.yml` |
   | Environment name | `testpypi` | `pypi` |

3. Confirm that both publishers appear in their settings pages and that the
   matching GitHub environments still require approval. No API token or new
   repository secret is needed.

The workflow field takes the filename `release.yml`, without `.github/workflows/`.
For an existing project that you own, use its **Manage → Publishing** page instead
of adding a pending publisher. A pending publisher creates the project on its
first successful upload; it does not reserve the project name. See the
[PyPI setup guide](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
and [existing-project guide](https://docs.pypi.org/trusted-publishers/adding-a-publisher/).

## Version contract

The first stable PyPI package version is **0.0.1**. Its TestPyPI candidate is
`0.0.1rc1`. NEP's independent protocol version remains `0.1.0`; do not rename
its schemas or conformance fixtures when changing the package version.

Normal version bumps are proposed by Release Please after successful main CI.
Conventional PR titles determine the bump, and the version PR updates the
changelog and package version together. See [GitHub automation](github_automation.md)
for the release PR, explicit CI dispatch, and squash-merge flow. The existing
protected publishing environments still govern PyPI and TestPyPI publication.

`nyssa_bench/version.py` is the only package-version source. Hatch reads it
through `[tool.hatch.version]`, and the CLI exposes it through `nyssa --version`.
The release workflow rejects a tag unless it exactly equals `v<version>`.

Validate locally before creating a tag:

```bash
uv run python scripts/validate_claim_evidence.py
uv run python scripts/validate_credibility.py
uv run python scripts/validate_release_version.py --tag v0.0.1
uv build
uvx twine check --strict dist/*
uv run python scripts/validate_distributions.py dist/*
```

The credibility validator requires the committed Measurement Core to remain
valid and prints the status of all three gates. A package release does not
require unavailable reference or hardware evidence to be relabeled as passed;
the full `nyssa credibility-gate` command exits with `2` until Phase 1 is
scientifically complete.

Update `nyssa_bench/version.py`, `.release-please-manifest.json`, and `CHANGELOG.md`
in a reviewed pull request. The manifest version must match the package version.
Do not change version
metadata on the tag itself. The commands below use the required initial version;
when preparing the candidate, validate `--tag v0.0.1rc1` instead.

The temporary `release-as: 0.0.1` setting in `release-please-config.json`
prevents automatic bumps from skipping the first release. Keep it through
candidate qualification and the stable release. Once 0.0.1 is published and
verified, remove the override in a separate PR to resume conventional bumps.

## Release candidate

Create the PEP 440 release-candidate version `0.0.1rc1`, merge it, and tag
the exact main commit:

```bash
git tag -s v0.0.1rc1 -m "NyssaBench 0.0.1rc1"
git push origin v0.0.1rc1
```

The workflow builds a wheel and source distribution, runs strict metadata
and content checks, attests build provenance, and installs the wheel outside the
checkout on Python 3.10 and 3.13. The installed jobs also load bundled resources
and generate a smoke result pack and HTML report. Only then does the `testpypi`
job request an OIDC token and publish to TestPyPI.

After publication, `Verify TestPyPI installation` runs on Python 3.10 and 3.13.
It fetches the release metadata from TestPyPI, requires the wheel and source
archive filenames, sizes, and SHA-256 digests to match the build job, and checks
the downloaded bytes again. Missing uploads and temporary registry errors are
retried; wrong, yanked, or corrupt artifacts fail verification.

Each job installs the verified wheel into a fresh virtual environment, fetching
dependencies only from PyPI. It runs `pip check`, verifies the API, distribution,
and CLI versions, loads bundled suites and stressors, and generates a packaging
smoke result and HTML report outside the checkout.

To repeat this check locally, download the same run's `python-distributions`
artifact into `dist/` and run from the matching tag checkout:

```bash
python scripts/verify_published_release.py \
  --index testpypi \
  --version 0.0.1rc1 \
  --dist-dir dist \
  --out /tmp/nyssa-testpypi-verification
```

Use an empty output directory; an existing report is never overwritten. The
script needs Python 3.10 or later with `venv` support and network access. Use the
original build artifacts, since rebuilding a source archive can change its hash.

Download both `published-testpypi-python-*` workflow artifacts. Each contains
`published_release_verification.json`, the verified distributions,
`installation.log`, and successful smoke outputs. Failure reports and partial
downloads are retained too. Artifacts expire after 30 days, so retain the evidence
and record the workflow link, both Python results, and distribution hashes in
the stable release pull request. Packaging smoke does not establish scientific
benchmark or GPU validation.

## Stable release

After the candidate passes installation review, set the stable version, update
the changelog, merge that release pull request, and create the matching signed
tag:

```bash
git tag -s v0.0.1 -m "NyssaBench 0.0.1"
git push origin v0.0.1
```

Stable tags run the same build and installed-wheel jobs, then wait at the
protected `pypi` environment. Approve that job only after checking the tag,
attestation, distribution metadata, and recorded TestPyPI candidate.

After publication, `Verify PyPI installation` repeats the registry and clean
installation checks against production PyPI on both Python versions. Retain the
`published-pypi-python-*` artifacts with the release evidence. A successful upload
alone is insufficient: both verification jobs must pass before declaring the
release complete. For a temporary download or installation failure, rerun only
the failed verification jobs; they do not publish anything. To retry locally,
use the command above with `--index pypi` and the stable version.

Published files are immutable. Fix a bad release with a new version; never
replace an existing wheel or source distribution.

## Containers and GitHub release artifacts

The same tag builds and smoke-tests the core, MuJoCo, and ManiSkill Dockerfiles
from the exact wheel produced by the package job. It publishes version and
version-plus-commit tags to GHCR with SBOM and provenance attestations. Core and
MuJoCo publish `linux/amd64` and `linux/arm64`; ManiSkill publishes
`linux/amd64` under the CUDA/Vulkan assumptions in [Docker](docker.md).

The release-bundle job gathers the three immutable image digests, validates
their version and commit identity, and creates the compatibility manifest,
Phase 1 credibility report, deterministic ZIP, release notes, and checksums.
PyPI publication waits for the container and bundle jobs. A failed container or
bundle therefore blocks the package publication path rather than leaving an
apparently complete release with missing runtime artifacts.

The generated release notes use the claim matrix's current wording and
capability counts. The bundle includes only content-addressed headline evidence
that passes the applicable Phase 1 credibility checks.

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
