from __future__ import annotations

import argparse
import re
import runpy
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 release path
    import tomli as tomllib  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parents[1]
__version__ = str(
    runpy.run_path(str(ROOT / "nyssa_bench" / "version.py"))["__version__"]
)
VERSION_PATTERN = re.compile(
    r"(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)"
    r"(?:rc(?:0|[1-9]\d*))?"
)


def validate_release_version(tag: str | None = None) -> str:
    if not VERSION_PATTERN.fullmatch(__version__):
        raise ValueError(f"unsupported release version: {__version__}")
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject.get("project", {})
    hatch = pyproject.get("tool", {}).get("hatch", {})
    if project.get("name") != "nyssa-bench":
        raise ValueError("release distribution name must be nyssa-bench")
    if project.get("dynamic") != ["version"]:
        raise ValueError("project version must be single-sourced through Hatch")
    if hatch.get("version", {}).get("path") != "nyssa_bench/version.py":
        raise ValueError("Hatch version path must be nyssa_bench/version.py")
    if tag is not None and tag != f"v{__version__}":
        raise ValueError(
            f"release tag {tag!r} does not match package version v{__version__}"
        )
    return __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate NyssaBench package and Git tag versions."
    )
    parser.add_argument("--tag")
    args = parser.parse_args(argv)
    try:
        version = validate_release_version(args.tag)
    except ValueError as exc:
        print(f"release version validation failed: {exc}")
        return 1
    print(f"release version: {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
