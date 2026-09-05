from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from nyssa_bench.packaging_smoke import run_packaging_smoke
from nyssa_bench.simulator_smoke import run_simulator_smoke
from nyssa_bench.utils.reproducibility import package_versions
from nyssa_bench.version import __version__


CONTAINER_SMOKE_FORMAT = "nyssa-container-smoke-v1"
ContainerProfile = Literal["core", "mujoco", "maniskill"]
SUPPORTED_PLATFORMS = {
    "core": {"linux/amd64", "linux/arm64"},
    "mujoco": {"linux/amd64", "linux/arm64"},
    "maniskill": {"linux/amd64"},
}


def run_container_smoke(
    profile: ContainerProfile,
    out_dir: str | Path,
    *,
    metadata_only: bool = False,
) -> dict[str, Any]:
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    metadata = _container_metadata(profile)
    versions = package_versions()
    _validate_installed_versions(profile, metadata, versions)
    _validate_outside_checkout()
    cli = shutil.which("nyssa")
    if not cli:
        raise RuntimeError("installed nyssa console command is unavailable")
    cli_checks = {}
    for name, arguments in {
        "version": ["--version"],
        "suites": ["list-suites"],
        "stressors": ["list-stressors"],
    }.items():
        result = subprocess.run(
            [cli, *arguments],
            cwd=out,
            check=True,
            capture_output=True,
            text=True,
        )
        cli_checks[name] = result.stdout.strip().splitlines()

    if profile == "core":
        runtime = {
            "status": "passed",
            "kind": "installed_artifact",
            "evidence": run_packaging_smoke(out / "result_pack"),
        }
    elif metadata_only:
        runtime = {
            "status": "not_run",
            "kind": "simulator",
            "reason": "metadata-only smoke; capable GPU execution is required",
            "registry": _maniskill_registry_evidence(),
        }
    else:
        runtime = {
            "status": "passed",
            "kind": "simulator",
            "evidence": run_simulator_smoke(
                profile,
                out / "result_pack",
                capture_replay=profile == "maniskill",
            ),
        }

    payload = {
        "format": CONTAINER_SMOKE_FORMAT,
        "status": "passed" if runtime["status"] == "passed" else "partial",
        "profile": profile,
        "container": metadata,
        "runtime": runtime,
        "cli_checks": cli_checks,
        "python_version": platform.python_version(),
        "package_versions": versions,
        "working_directory": Path.cwd().resolve().as_posix(),
        "package_path": Path(__file__).resolve().as_posix(),
    }
    path = out / "container_smoke.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def _container_metadata(profile: ContainerProfile) -> dict[str, str]:
    values = {
        "profile": os.getenv("NYSSA_CONTAINER_PROFILE", ""),
        "version": os.getenv("NYSSA_CONTAINER_VERSION", ""),
        "commit": os.getenv("NYSSA_CONTAINER_COMMIT", ""),
        "build_date": os.getenv("NYSSA_CONTAINER_BUILD_DATE", ""),
        "platform": os.getenv("NYSSA_CONTAINER_PLATFORM", ""),
        "python": os.getenv("NYSSA_CONTAINER_PYTHON", ""),
    }
    if values["profile"] != profile:
        raise RuntimeError(
            "container profile metadata does not match the smoke profile"
        )
    if values["version"] != __version__:
        raise RuntimeError("container and installed package versions differ")
    if not re.fullmatch(r"[0-9a-f]{40}", values["commit"]):
        raise RuntimeError("container commit must be a full lowercase Git SHA")
    try:
        built_at = datetime.fromisoformat(values["build_date"].replace("Z", "+00:00"))
    except ValueError as exc:
        raise RuntimeError("container build date must be RFC 3339") from exc
    if built_at.tzinfo is None:
        raise RuntimeError("container build date must include a timezone")
    if values["platform"] not in SUPPORTED_PLATFORMS[profile]:
        raise RuntimeError(
            f"unsupported {profile} container platform: {values['platform']}"
        )
    if not platform.python_version().startswith(f"{values['python']}."):
        raise RuntimeError("container Python metadata differs from the runtime")
    if profile == "core":
        values["simulator"] = os.getenv("NYSSA_CONTAINER_SIMULATOR", "")
        if values["simulator"] != "none":
            raise RuntimeError("core container must declare simulator metadata as none")
    elif profile == "mujoco":
        values["mujoco"] = os.getenv("NYSSA_CONTAINER_MUJOCO", "")
        values["gymnasium"] = os.getenv("NYSSA_CONTAINER_GYMNASIUM", "")
    if profile == "maniskill":
        values["cuda"] = os.getenv("NYSSA_CONTAINER_CUDA", "")
        values["vulkan"] = os.getenv("NYSSA_CONTAINER_VULKAN", "")
        values["mani-skill"] = os.getenv("NYSSA_CONTAINER_MANISKILL", "")
        values["torch"] = os.getenv("NYSSA_CONTAINER_TORCH", "")
        if not values["cuda"] or not values["vulkan"]:
            raise RuntimeError("ManiSkill container lacks CUDA or Vulkan metadata")
    return values


def _validate_installed_versions(
    profile: ContainerProfile,
    metadata: dict[str, str],
    installed: dict[str, str],
) -> None:
    expected = {
        "core": {},
        "mujoco": {
            "mujoco": metadata.get("mujoco", ""),
            "gymnasium": metadata.get("gymnasium", ""),
        },
        "maniskill": {
            "mani-skill": metadata.get("mani-skill", ""),
            "torch": metadata.get("torch", ""),
        },
    }[profile]
    failures = {
        name: {"expected": version, "installed": installed.get(name)}
        for name, version in expected.items()
        if not version or not installed.get(name, "").startswith(version)
    }
    if failures:
        raise RuntimeError(f"container package versions differ: {failures}")


def _maniskill_registry_evidence() -> dict[str, Any]:
    import gymnasium as gym
    import mani_skill  # noqa: F401

    from nyssa_bench.reference_benchmark.candidate import CANDIDATE_TASKS

    expected = sorted({task.env_id for task in CANDIDATE_TASKS})
    missing = [env_id for env_id in expected if env_id not in gym.registry]
    if missing:
        raise RuntimeError(
            "ManiSkill reference environments are not registered: "
            + ", ".join(missing)
        )
    return {
        "status": "registered_not_executed",
        "environment_ids": expected,
        "environment_count": len(expected),
    }


def _validate_outside_checkout() -> None:
    working = Path.cwd().resolve()
    package = Path(__file__).resolve()
    if (working / "pyproject.toml").is_file():
        raise RuntimeError("container smoke must run outside a source checkout")
    try:
        package.relative_to(working)
    except ValueError:
        return
    raise RuntimeError("NyssaBench is imported from the smoke working directory")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate an installed NyssaBench container runtime."
    )
    parser.add_argument("--profile", choices=sorted(SUPPORTED_PLATFORMS), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="validate package and runtime metadata without starting the simulator",
    )
    args = parser.parse_args(argv)
    if args.metadata_only and args.profile != "maniskill":
        parser.error("--metadata-only is only valid for the GPU ManiSkill profile")
    payload = run_container_smoke(
        args.profile,
        args.out,
        metadata_only=args.metadata_only,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "passed" or args.metadata_only else 1


if __name__ == "__main__":
    raise SystemExit(main())
