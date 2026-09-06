from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def test_dependency_extras_are_declared():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    extras = pyproject["project"]["optional-dependencies"]

    expected = {
        "all",
        "cli",
        "dataset",
        "dev",
        "diffusion",
        "experimental",
        "lerobot",
        "maniskill",
        "mujoco",
        "reports",
        "robomimic",
        "robocasa",
        "video",
        "vla",
        "full",
    }
    assert expected.issubset(extras)


def test_all_extra_covers_every_stable_runtime_extra():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    extras = pyproject["project"]["optional-dependencies"]
    stable_runtime_extras = {
        "cli",
        "dataset",
        "diffusion",
        "lerobot",
        "maniskill",
        "mujoco",
        "reports",
        "robomimic",
        "robocasa",
        "video",
        "vla",
    }
    required = {
        requirement
        for extra in stable_runtime_extras
        for requirement in extras[extra]
    }

    assert required.issubset(extras["all"])


def test_cli_compatibility_extra_does_not_duplicate_base_runtime():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["optional-dependencies"]["cli"] == []


def test_gpu_extras_share_the_validated_pytorch_runtime():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    extras = pyproject["project"]["optional-dependencies"]

    assert "mani-skill==3.0.1" in extras["maniskill"]
    assert "torch==2.6.0" in extras["maniskill"]
    for extra in ("vla", "diffusion", "all", "full"):
        assert "torch==2.6.0" in extras[extra]
        assert "torchvision==0.21.0" in extras[extra]

    dockerfile = Path("docker/Dockerfile.maniskill").read_text(encoding="utf-8")
    assert "ARG TORCH_VERSION=2.6.0" in dockerfile
    assert "ARG MANISKILL_VERSION=3.0.1" in dockerfile
