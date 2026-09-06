# Docker and release bundles

PyPI is the default installation path for the core CLI and Python API. Use a
release container when simulator system libraries, rendering backends, or an
exact runtime image are part of the experiment. Use a source checkout for
contributor development and editable installs.

## Published images

Tagged releases publish three images to GitHub Container Registry:

| Profile | Image | Platforms | Runtime contract |
| --- | --- | --- | --- |
| Core | `ghcr.io/hudson-infinity/nyssa-bench-core` | `linux/amd64`, `linux/arm64` | Python 3.11, core wheel |
| MuJoCo | `ghcr.io/hudson-infinity/nyssa-bench-mujoco` | `linux/amd64`, `linux/arm64` | Python 3.11, MuJoCo 3.12.0, Gymnasium 1.3.0, OSMesa, EGL, FFmpeg |
| ManiSkill | `ghcr.io/hudson-infinity/nyssa-bench-maniskill` | `linux/amd64` | Python 3.10, ManiSkill 3.0.1, PyTorch 2.5.1 CUDA 12.4, Vulkan 1.3, FFmpeg |

The ManiSkill image requires the NVIDIA Container Toolkit, a host driver
compatible with CUDA 12.4, and a Vulkan-capable physical device exposed to the
container. Arm64 ManiSkill is not published because this CUDA/Vulkan dependency
combination is not supported by the release contract. The PR smoke validates
the installed CLI and metadata without a GPU; simulator claims still require
the capable-runner workflow in [simulator-backed CI](simulator_ci.md).

Each image records OCI labels and environment metadata for the NyssaBench
version, full commit SHA, Python version, platform, build date, and simulator
profile. The release workflow publishes BuildKit provenance and an SBOM
attestation alongside each image.

## Immutable identity

A `v0.1.0` release publishes at least these tags:

```text
0.1.0
0.1.0-<12-character-commit>
```

Stable releases may also update `latest`. Release candidates do not. Scientific
results must record the OCI digest from `compatibility-manifest.json`:

```text
ghcr.io/hudson-infinity/nyssa-bench-mujoco@sha256:<digest>
```

Do not use `latest` as a result identity. Semantic-version and commit tags make
the release discoverable; the digest identifies the exact manifest that ran.

## Run a release image

Core package smoke:

```bash
docker run --rm \
  ghcr.io/hudson-infinity/nyssa-bench-core@sha256:<digest> \
  nyssa list-suites
```

MuJoCo evaluation with a writable host result directory:

```bash
docker run --rm \
  -v "$PWD/benchmark_results:/workspace/benchmark_results" \
  ghcr.io/hudson-infinity/nyssa-bench-mujoco@sha256:<digest> \
  nyssa run \
    --suite mujoco_control_v0 \
    --engine mujoco \
    --policy random \
    --episodes 1 \
    --out benchmark_results/mujoco_container_smoke
```

ManiSkill requires GPU and Vulkan device access configured for the host. The
exact flags vary by NVIDIA Container Toolkit and driver installation. Verify
device availability before a benchmark run:

```bash
docker run --rm --gpus all \
  ghcr.io/hudson-infinity/nyssa-bench-maniskill@sha256:<digest> \
  python3 -c "import torch; print(torch.cuda.is_available())"
```

A successful import is not simulator evidence. Run the installed simulator
smoke with replay capture before using the image for a claim.

## Build locally

Build the wheel first. Dockerfiles never install the source tree editable.
Simulator and base dependencies are installed below the changing wheel layer,
so BuildKit can reuse the large ManiSkill/CUDA and MuJoCo dependency layers when
only NyssaBench source changes. The final wheel installation uses `--no-deps`
and `pip check`, preventing that optimization from silently omitting required
runtime packages.

```bash
uv build
VERSION=$(uv run python -c "from nyssa_bench import __version__; print(__version__)")
COMMIT=$(git rev-parse HEAD)
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')

docker build -f docker/Dockerfile \
  --build-arg NYSSA_VERSION="$VERSION" \
  --build-arg NYSSA_COMMIT="$COMMIT" \
  --build-arg NYSSA_BUILD_DATE="$BUILD_DATE" \
  -t nyssa-bench:core .
```

Use `docker/Dockerfile.mujoco` or `docker/Dockerfile.maniskill` for the other
profiles. Local tags are development conveniences and are not release evidence.

## GitHub release bundle

Every tagged workflow attaches:

- the wheel and source distribution;
- `nyssa-bench-<version>-release.zip`;
- `compatibility-manifest.json`;
- `phase1-credibility.json`;
- `release-bundle-manifest.json`;
- `SHA256SUMS`.

The deterministic ZIP includes NEP schemas and conformance fixtures, suite and
stressor configs, migration and compatibility docs, Dockerfiles, and the
machine-readable claim state. It does not copy large result packs. The
compatibility manifest lists only claim-approved headline result packs and
content-hashes their RunValidity and BenchmarkValidity artifacts. An empty list
means the release makes no benchmark-result claim.

Verify downloaded files from the release directory:

```bash
sha256sum --check SHA256SUMS
```

GitHub Actions authenticates to GHCR with the repository `GITHUB_TOKEN` and
publishes BuildKit SBOM and provenance attestations. This follows the official
[GitHub container publishing](https://docs.github.com/en/actions/tutorials/publish-packages/publish-docker-images)
and [Docker build attestation](https://docs.docker.com/build/ci/github-actions/attestations/)
workflows.

Kubernetes, private worker scheduling, and hosted evaluation orchestration are
product infrastructure and remain outside NyssaBench.
