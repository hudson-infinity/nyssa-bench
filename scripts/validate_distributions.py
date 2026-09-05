from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_EXTRAS = {
    "all",
    "cli",
    "dataset",
    "dev",
    "diffusion",
    "experimental",
    "full",
    "lerobot",
    "maniskill",
    "mujoco",
    "reports",
    "robocasa",
    "robomimic",
    "video",
    "vla",
}
REQUIRED_WHEEL_SUFFIXES = {
    "nyssa_bench/__init__.py",
    "nyssa_bench/packaging_smoke.py",
    "nyssa_bench/_resources/configs/suites/tabletop_manipulation_v0.yaml",
    "nyssa_bench/tasks/tabletop/pick_cube.yaml",
    "nyssa_bench/_resources/conformance/scenario/README.md",
}
REQUIRED_SDIST_SUFFIXES = {
    "LICENSE",
    "README.md",
    "pyproject.toml",
    "configs/suites/tabletop_manipulation_v0.yaml",
    "conformance/scenario/README.md",
    "nyssa_bench/__init__.py",
}
FORBIDDEN_PARTS = {
    ".git",
    ".venv",
    "benchmark_results",
    "checkpoints",
    "dist",
    "runs",
}
MAX_MEMBER_BYTES = 5 * 1024 * 1024


def validate_distributions(paths: list[str | Path]) -> dict[str, Any]:
    artifacts = [Path(path).resolve() for path in paths]
    wheels = [path for path in artifacts if path.suffix == ".whl"]
    sdists = [path for path in artifacts if path.name.endswith(".tar.gz")]
    if len(wheels) != 1 or len(sdists) != 1 or len(artifacts) != 2:
        raise ValueError("distribution validation requires one wheel and one .tar.gz sdist")
    reports = [_validate_wheel(wheels[0]), _validate_sdist(sdists[0])]
    versions = {report["version"] for report in reports}
    if len(versions) != 1:
        raise ValueError("wheel and source distribution versions do not match")
    return {
        "format": "nyssa-distribution-validation-v1",
        "distribution": "nyssa-bench",
        "version": next(iter(versions)),
        "artifacts": reports,
    }


def _validate_wheel(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"wheel not found: {path}")
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        _validate_members(names, {info.filename: info.file_size for info in infos})
        _require_suffixes(names, REQUIRED_WHEEL_SUFFIXES, "wheel")
        metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
        if len(metadata_names) != 1:
            raise ValueError("wheel must contain exactly one METADATA file")
        metadata = BytesParser().parsebytes(archive.read(metadata_names[0]))
        if metadata.get("Name") != "nyssa-bench":
            raise ValueError("wheel metadata distribution name is incorrect")
        version = str(metadata.get("Version", ""))
        if not version:
            raise ValueError("wheel metadata version is missing")
        if metadata.get("Requires-Python") != ">=3.10":
            raise ValueError("wheel metadata Requires-Python is incorrect")
        extras = set(metadata.get_all("Provides-Extra", []))
        if extras != REQUIRED_EXTRAS:
            raise ValueError(
                f"wheel extras differ from the release contract: {sorted(extras)}"
            )
        _compare_resource_bytes(archive)
    return _artifact_report(path, "wheel", version, names)


def _validate_sdist(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"source distribution not found: {path}")
    with tarfile.open(path, "r:gz") as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        names = [member.name for member in members]
        _validate_members(names, {member.name: member.size for member in members})
        _require_suffixes(names, REQUIRED_SDIST_SUFFIXES, "source distribution")
        pkg_info = [member for member in members if member.name.endswith("/PKG-INFO")]
        if len(pkg_info) != 1:
            raise ValueError("source distribution must contain one PKG-INFO file")
        handle = archive.extractfile(pkg_info[0])
        if handle is None:
            raise ValueError("cannot read source distribution PKG-INFO")
        metadata = BytesParser().parsebytes(handle.read())
        if metadata.get("Name") != "nyssa-bench":
            raise ValueError("source distribution name is incorrect")
        version = str(metadata.get("Version", ""))
        if not version:
            raise ValueError("source distribution version is missing")
    return _artifact_report(path, "sdist", version, names)


def _validate_members(names: list[str], sizes: dict[str, int]) -> None:
    if len(names) != len(set(names)):
        raise ValueError("distribution archive contains duplicate members")
    for name in names:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"distribution member escapes the archive: {name}")
        if FORBIDDEN_PARTS.intersection(path.parts):
            raise ValueError(f"distribution contains forbidden content: {name}")
        if name.lower().endswith(".zip"):
            raise ValueError(f"distribution contains a result archive: {name}")
        if sizes[name] > MAX_MEMBER_BYTES:
            raise ValueError(f"distribution member exceeds 5 MiB: {name}")


def _require_suffixes(names: list[str], required: set[str], label: str) -> None:
    missing = [
        suffix
        for suffix in sorted(required)
        if not any(name == suffix or name.endswith(f"/{suffix}") for name in names)
    ]
    if missing:
        raise ValueError(f"{label} is missing required content: {', '.join(missing)}")


def _compare_resource_bytes(archive: zipfile.ZipFile) -> None:
    members = set(archive.namelist())
    for source_root in (ROOT / "configs", ROOT / "conformance"):
        for source in source_root.rglob("*"):
            if not source.is_file():
                continue
            relative = source.relative_to(ROOT).as_posix()
            packaged = f"nyssa_bench/_resources/{relative}"
            if packaged not in members:
                raise ValueError(f"wheel is missing bundled resource: {relative}")
            if archive.read(packaged) != source.read_bytes():
                raise ValueError(f"wheel resource changed during packaging: {relative}")


def _artifact_report(
    path: Path, kind: str, version: str, names: list[str]
) -> dict[str, Any]:
    return {
        "kind": kind,
        "filename": path.name,
        "version": version,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "member_count": len(names),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate NyssaBench wheel and source distribution contents."
    )
    parser.add_argument("artifacts", nargs="+")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    try:
        report = validate_distributions(args.artifacts)
    except ValueError as exc:
        print(f"distribution validation failed: {exc}")
        return 1
    text = json.dumps(report, indent=2) + "\n"
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
