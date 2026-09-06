from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest
from pydantic import ValidationError

from nyssa_bench import container_smoke
from nyssa_bench.release_bundle import (
    PROFILES,
    ContainerReleaseRecord,
    build_release_bundle,
    write_container_record,
)
from nyssa_bench.version import __version__


ROOT = Path(__file__).resolve().parents[1]
COMMIT = "a" * 40
BUILD_DATE = "2026-09-05T20:00:00Z"


def test_release_bundle_is_deterministic_and_content_addressed(
    tmp_path: Path,
) -> None:
    records = _container_records(tmp_path)
    wheel = tmp_path / f"nyssa_bench-{__version__}-py3-none-any.whl"
    sdist = tmp_path / f"nyssa_bench-{__version__}.tar.gz"
    wheel.write_bytes(b"wheel")
    sdist.write_bytes(b"sdist")
    out = tmp_path / "release"

    paths = build_release_bundle(
        repo_root=ROOT,
        out_dir=out,
        version=__version__,
        commit_sha=COMMIT,
        build_date=BUILD_DATE,
        container_records=records,
        distributions=[wheel, sdist],
    )
    first_digest = _sha(paths["bundle"])
    rebuilt = build_release_bundle(
        repo_root=ROOT,
        out_dir=out,
        version=__version__,
        commit_sha=COMMIT,
        build_date=BUILD_DATE,
        container_records=records,
        distributions=[wheel, sdist],
    )

    assert _sha(rebuilt["bundle"]) == first_digest
    compatibility = json.loads(paths["compatibility"].read_text(encoding="utf-8"))
    assert compatibility["format"] == "nyssa-release-compatibility-v1"
    assert compatibility["credibility"]["highest_completed_gate"] == "measurement_core"
    assert compatibility["headline_result_packs"] == []
    assert {item["profile"] for item in compatibility["containers"]} == set(PROFILES)
    assert all(
        item["immutable_identity"].startswith("ghcr.io/")
        for item in compatibility["containers"]
    )
    checksums = paths["checksums"].read_text(encoding="ascii")
    assert wheel.name in checksums
    assert paths["bundle"].name in checksums
    assert paths["notes"].name in checksums
    with ZipFile(paths["bundle"]) as archive:
        names = set(archive.namelist())
        assert "compatibility-manifest.json" in names
        assert "phase1-credibility.json" in names
        assert "schemas/nep/0.1.0/nep-manifest.schema.json" in names
        assert "conformance/nep/0.1.0/valid/mujoco-pipeline.json" in names
        assert "docker/Dockerfile.mujoco" in names
        assert not any(name.startswith("benchmark_results/") for name in names)


def test_release_bundle_rejects_cross_release_container_record(
    tmp_path: Path,
) -> None:
    records = _container_records(tmp_path)
    payload = json.loads(Path(records[0]).read_text(encoding="utf-8"))
    payload["commit_sha"] = "b" * 40
    payload["tags"] = [
        f"{payload['image']}:{__version__}",
        f"{payload['image']}:{__version__}-{'b' * 12}",
    ]
    Path(records[0]).write_text(json.dumps(payload), encoding="utf-8")
    wheel = tmp_path / "package.whl"
    sdist = tmp_path / "package.tar.gz"
    wheel.write_bytes(b"wheel")
    sdist.write_bytes(b"sdist")

    with pytest.raises(ValueError, match="identities differ"):
        build_release_bundle(
            repo_root=ROOT,
            out_dir=tmp_path / "release",
            version=__version__,
            commit_sha=COMMIT,
            build_date=BUILD_DATE,
            container_records=records,
            distributions=[wheel, sdist],
        )


def test_container_record_requires_immutable_tags_and_profile_platforms() -> None:
    image = "ghcr.io/hudson-infinity/nyssa-bench-core"
    with pytest.raises(ValidationError, match="immutable version and commit tags"):
        ContainerReleaseRecord(
            profile="core",
            image=image,
            digest=f"sha256:{'1' * 64}",
            tags=(f"{image}:latest",),
            platforms=("linux/amd64", "linux/arm64"),
            version=__version__,
            commit_sha=COMMIT,
            python_version="3.11",
            build_date=BUILD_DATE,
            compatibility=PROFILES["core"],
        )


def test_container_smoke_validates_metadata_and_runs_outside_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _container_environment(monkeypatch, "core")
    monkeypatch.setattr(container_smoke.shutil, "which", lambda _: "/usr/bin/nyssa")
    monkeypatch.setattr(
        container_smoke.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="ok\n"),
    )
    monkeypatch.setattr(
        container_smoke,
        "run_packaging_smoke",
        lambda _: {"format": "nyssa-installed-artifact-smoke-v1"},
    )

    report = container_smoke.run_container_smoke("core", tmp_path / "out")

    assert report["status"] == "passed"
    assert report["container"]["commit"] == COMMIT
    assert report["runtime"]["kind"] == "installed_artifact"
    assert report["cli_checks"]["version"] == ["ok"]


def test_maniskill_metadata_smoke_stays_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _container_environment(monkeypatch, "maniskill")
    monkeypatch.setenv("NYSSA_CONTAINER_CUDA", "12.4")
    monkeypatch.setenv("NYSSA_CONTAINER_VULKAN", "1.3")
    monkeypatch.setattr(container_smoke.shutil, "which", lambda _: "/usr/bin/nyssa")
    monkeypatch.setattr(
        container_smoke.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="ok\n"),
    )
    monkeypatch.setattr(
        container_smoke,
        "_maniskill_registry_evidence",
        lambda: {"status": "registered_not_executed", "environment_count": 12},
    )

    report = container_smoke.run_container_smoke(
        "maniskill", tmp_path / "out", metadata_only=True
    )

    assert report["status"] == "partial"
    assert report["runtime"]["status"] == "not_run"
    assert report["runtime"]["registry"]["environment_count"] == 12


def test_container_smoke_rejects_unlabelled_or_source_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="profile metadata"):
        container_smoke.run_container_smoke("core", tmp_path / "out")


@pytest.mark.parametrize(
    "dockerfile",
    ["docker/Dockerfile", "docker/Dockerfile.mujoco", "docker/Dockerfile.maniskill"],
)
def test_release_dockerfiles_install_wheels_and_record_oci_identity(
    dockerfile: str,
) -> None:
    text = (ROOT / dockerfile).read_text(encoding="utf-8")

    assert "COPY dist/*.whl" in text
    assert "--no-deps" in text
    assert text.index("pip install --no-cache-dir") < text.index("COPY dist/*.whl")
    assert text.index("pip install --no-cache-dir") < text.index("ARG NYSSA_VERSION")
    assert text.index("ARG NYSSA_BUILD_DATE") < text.index("COPY dist/*.whl")
    assert "pip install -e" not in text
    assert "COPY . ." not in text
    assert "org.opencontainers.image.version" in text
    assert "org.opencontainers.image.revision" in text
    assert "org.opencontainers.image.created" in text
    assert "USER nyssa" in text


def test_heavy_simulator_dependencies_are_cached_below_the_wheel_layer() -> None:
    mujoco = (ROOT / "docker/Dockerfile.mujoco").read_text(encoding="utf-8")
    maniskill = (ROOT / "docker/Dockerfile.maniskill").read_text(encoding="utf-8")

    assert mujoco.index('"mujoco==${MUJOCO_VERSION}"') < mujoco.index("COPY dist/*.whl")
    assert maniskill.index('"torch==${TORCH_VERSION}"') < maniskill.index(
        "COPY dist/*.whl"
    )
    assert maniskill.index('"mani-skill==${MANISKILL_VERSION}"') < maniskill.index(
        "COPY dist/*.whl"
    )


def test_release_workflow_publishes_attested_images_and_bundle() -> None:
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )

    assert "ghcr.io/hudson-infinity/nyssa-bench-core" in workflow
    assert "ghcr.io/hudson-infinity/nyssa-bench-mujoco" in workflow
    assert "ghcr.io/hudson-infinity/nyssa-bench-maniskill" in workflow
    assert "linux/amd64,linux/arm64" in workflow
    assert "provenance: mode=max" in workflow
    assert "sbom: true" in workflow
    assert "nyssa_bench.release_bundle bundle" in workflow
    assert "nyssa_bench.container_smoke" in workflow


def _container_records(root: Path) -> list[Path]:
    records = []
    for index, profile in enumerate(PROFILES, start=1):
        image = f"ghcr.io/hudson-infinity/nyssa-bench-{profile}"
        records.append(
            write_container_record(
                profile=profile,  # type: ignore[arg-type]
                image=image,
                digest=f"sha256:{str(index) * 64}",
                tags=[
                    f"{image}:{__version__}",
                    f"{image}:{__version__}-{COMMIT[:12]}",
                ],
                version=__version__,
                commit_sha=COMMIT,
                build_date=BUILD_DATE,
                out_path=root / f"{profile}.json",
            )
        )
    return records


def _container_environment(monkeypatch: pytest.MonkeyPatch, profile: str) -> None:
    python_version = str(PROFILES[profile]["python"])
    monkeypatch.setenv("NYSSA_CONTAINER_PROFILE", profile)
    monkeypatch.setenv("NYSSA_CONTAINER_VERSION", __version__)
    monkeypatch.setenv("NYSSA_CONTAINER_COMMIT", COMMIT)
    monkeypatch.setenv("NYSSA_CONTAINER_BUILD_DATE", BUILD_DATE)
    monkeypatch.setenv(
        "NYSSA_CONTAINER_PLATFORM",
        "linux/amd64",
    )
    monkeypatch.setenv("NYSSA_CONTAINER_PYTHON", python_version)
    monkeypatch.setattr(
        container_smoke.platform,
        "python_version",
        lambda: f"{python_version}.1",
    )
    versions = {"nyssa-bench": __version__}
    if profile == "core":
        monkeypatch.setenv("NYSSA_CONTAINER_SIMULATOR", "none")
    elif profile == "mujoco":
        monkeypatch.setenv("NYSSA_CONTAINER_MUJOCO", "3.12.0")
        monkeypatch.setenv("NYSSA_CONTAINER_GYMNASIUM", "1.3.0")
        versions.update({"mujoco": "3.12.0", "gymnasium": "1.3.0"})
    else:
        monkeypatch.setenv("NYSSA_CONTAINER_MANISKILL", "3.0.1")
        monkeypatch.setenv("NYSSA_CONTAINER_TORCH", "2.6.0")
        versions.update({"mani-skill": "3.0.1", "torch": "2.6.0+cu124"})
    monkeypatch.setattr(container_smoke, "package_versions", lambda: versions)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
