from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any
from uuid import uuid4

from nyssa_bench.version import __version__


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_run_id(suite_id: str, policy: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = uuid4().hex[:8]
    return f"{suite_id}_{policy}_{stamp}_{suffix}"


def environment_metadata() -> dict[str, Any]:
    return {
        "nyssa_bench_version": __version__,
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def package_versions() -> dict[str, str]:
    names = [
        "nyssa-bench",
        "numpy",
        "pydantic",
        "pyyaml",
        "gymnasium",
        "mujoco",
        "mani-skill",
        "sapien",
        "torch",
        "torchvision",
        "robomimic",
        "lerobot",
        "diffusers",
        "transformers",
        "h5py",
        "imageio",
    ]
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def git_info(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root)
    return {
        "commit": _git(root, ["rev-parse", "HEAD"]),
        "branch": _git(root, ["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(_git(root, ["status", "--short"])),
    }


def write_json(path: str | Path, data: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def _git(root: Path, args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()
