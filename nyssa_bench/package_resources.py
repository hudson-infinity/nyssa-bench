from __future__ import annotations

from importlib.resources import files
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PACKAGE_ROOT.parent


def resource_root(name: str) -> Path:
    if not name or "/" in name or "\\" in name or name in {".", ".."}:
        raise ValueError("package resource name must be one path segment")
    packaged = files("nyssa_bench").joinpath("_resources", name)
    packaged_path = Path(str(packaged))
    if packaged_path.exists():
        return packaged_path
    source_path = SOURCE_ROOT / name
    if source_path.exists():
        return source_path
    raise FileNotFoundError(f"NyssaBench resource bundle is missing: {name}")


def config_root(section: str) -> Path:
    root = resource_root("configs") / section
    if not root.is_dir():
        raise FileNotFoundError(f"NyssaBench config section is missing: {section}")
    return root


def policy_example_root() -> Path:
    packaged = files("nyssa_bench").joinpath("_resources", "policy_examples")
    packaged_path = Path(str(packaged))
    if packaged_path.is_dir():
        return packaged_path
    source_path = SOURCE_ROOT / "examples" / "policies"
    if source_path.is_dir():
        return source_path
    raise FileNotFoundError("NyssaBench policy examples are missing")
