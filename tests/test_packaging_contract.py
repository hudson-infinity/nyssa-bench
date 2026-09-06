from __future__ import annotations

from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 CI
    import tomli as tomllib

from nyssa_bench import __version__
from nyssa_bench.cli import main
from scripts.validate_release_version import validate_release_version


ROOT = Path(__file__).resolve().parents[1]


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_distribution_identity_and_version_are_single_sourced() -> None:
    pyproject = _pyproject()
    project = pyproject["project"]

    assert project["name"] == "nyssa-bench"
    assert project["dynamic"] == ["version"]
    assert "version" not in project
    assert pyproject["tool"]["hatch"]["version"]["path"] == ("nyssa_bench/version.py")
    assert validate_release_version(f"v{__version__}") == __version__
    with pytest.raises(ValueError, match="does not match package version"):
        validate_release_version("v9.9.9")


def test_package_metadata_uses_hudson_identity_and_public_urls() -> None:
    project = _pyproject()["project"]

    assert project["authors"] == [{"name": "Hudson Labs"}]
    assert project["maintainers"] == [{"name": "Hudson Labs"}]
    assert project["license"] == "Apache-2.0"
    assert project["license-files"] == ["LICENSE"]
    assert project["urls"] == {
        "Homepage": "https://github.com/hudson-infinity/nyssa-bench",
        "Documentation": "https://github.com/hudson-infinity/nyssa-bench/tree/main/docs",
        "Repository": "https://github.com/hudson-infinity/nyssa-bench",
        "Issues": "https://github.com/hudson-infinity/nyssa-bench/issues",
        "Changelog": "https://github.com/hudson-infinity/nyssa-bench/blob/main/CHANGELOG.md",
    }


def test_base_install_is_lightweight_and_cli_is_not_optional() -> None:
    project = _pyproject()["project"]
    dependencies = "\n".join(project["dependencies"]).lower()
    excluded = {
        "gymnasium",
        "imageio",
        "mani-skill",
        "matplotlib",
        "mujoco",
        "pandas",
        "pyarrow",
        "rich",
        "robomimic",
        "torch",
        "transformers",
        "typer",
    }

    assert not any(package in dependencies for package in excluded)
    assert project["scripts"] == {"nyssa": "nyssa_bench.cli:main"}
    assert project["optional-dependencies"]["cli"] == []


@pytest.mark.parametrize("extra", ["mujoco", "maniskill"])
def test_simulator_profiles_include_default_replay_runtime(extra: str) -> None:
    requirements = "\n".join(
        _pyproject()["project"]["optional-dependencies"][extra]
    ).lower()

    assert "gymnasium" in requirements
    assert "imageio" in requirements
    assert "imageio-ffmpeg" in requirements


def test_cli_reports_package_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])

    assert exc_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"nyssa {__version__}"


def test_release_workflow_uses_oidc_and_protected_indices() -> None:
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )

    assert "id-token: write" in workflow
    assert "environment:\n      name: testpypi" in workflow
    assert "environment:\n      name: pypi" in workflow
    assert "repository-url: https://test.pypi.org/legacy/" in workflow
    assert "password: ${{ secrets.GITHUB_TOKEN }}" in workflow
    assert "password: ${{ secrets.PYPI_API_TOKEN }}" not in workflow
    assert "PYPI_API_TOKEN" not in workflow
    assert 'python-version: ["3.10", "3.13"]' in workflow
    assert "working-directory: ${{ runner.temp }}" in workflow


def test_pr_ci_avoids_duplicate_push_runs_and_cancels_stale_containers() -> None:
    ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    containers = (ROOT / ".github/workflows/container-ci.yml").read_text(
        encoding="utf-8"
    )

    assert "push:\n    branches: [main]" in ci
    assert "pull_request:" in ci
    assert "cancel-in-progress: true" in containers
    assert "nyssa_bench/release_bundle.py" not in containers
    assert "tests/test_release_bundle.py" not in containers


def test_optional_dependency_errors_use_release_install_commands() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "nyssa_bench").rglob("*.py")
    )

    assert "uv sync --extra" not in source
    assert "pip install -e" not in source
    for extra in (
        "dataset",
        "diffusion",
        "experimental",
        "lerobot",
        "maniskill",
        "mujoco",
        "robomimic",
        "vla",
    ):
        assert f"nyssa-bench[{extra}]" in source or extra in {"diffusion", "vla"}
