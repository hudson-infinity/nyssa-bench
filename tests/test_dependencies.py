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
