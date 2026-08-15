from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from nyssa_bench.datasets.import_maniskill import import_maniskill_demos


DEFAULT_ENV_IDS = ["PickCube-v1", "PushCube-v1", "StackCube-v1"]
DEFAULT_COMMAND_TEMPLATE = (
    "{python} -m mani_skill.examples.motionplanning.panda.run "
    "-e {env_id} --num-traj {num_traj} --record-dir {raw_task_dir}"
)


def collect_maniskill_demos(
    *,
    out: str | Path,
    raw_dir: str | Path,
    env_ids: list[str] | None = None,
    num_traj: int = 100,
    command_template: str | None = None,
    continue_on_error: bool = False,
) -> dict[str, Path]:
    """Collect ManiSkill motion-planning demos, then import them into Nyssa.

    ManiSkill's demo generation entry points can move between releases. The
    default template targets the ManiSkill3 Panda motion-planning example, and
    callers can override it without changing Nyssa code.
    """

    out = Path(out)
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    env_ids = env_ids or DEFAULT_ENV_IDS
    command_template = command_template or os.getenv("NYSSA_MANISKILL_DEMO_COMMAND", DEFAULT_COMMAND_TEMPLATE)

    runs = []
    failures = []
    for env_id in env_ids:
        raw_task_dir = raw_dir / _safe_env_id(env_id)
        raw_task_dir.mkdir(parents=True, exist_ok=True)
        command = _format_command(
            command_template,
            env_id=env_id,
            num_traj=num_traj,
            raw_dir=raw_dir,
            raw_task_dir=raw_task_dir,
        )
        result = subprocess.run(command, shell=True, text=True, capture_output=True, check=False)
        run_record = {
            "env_id": env_id,
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout[-4000:],
            "stderr": result.stderr[-4000:],
            "raw_task_dir": raw_task_dir.as_posix(),
        }
        runs.append(run_record)
        if result.returncode != 0:
            failures.append(run_record)
            if not continue_on_error:
                _write_collect_manifest(raw_dir, out, runs, failures)
                raise RuntimeError(
                    f"ManiSkill demo collection failed for {env_id} with exit code {result.returncode}. "
                    f"Command: {command}\nSTDERR:\n{result.stderr[-2000:]}"
                )

    imported = import_maniskill_demos(raw_dir, out)
    manifest_path = _write_collect_manifest(raw_dir, out, runs, failures)
    return {"collect_manifest": manifest_path, **imported}


def _format_command(
    template: str,
    *,
    env_id: str,
    num_traj: int,
    raw_dir: Path,
    raw_task_dir: Path,
) -> str:
    return template.format(
        python=_shell_argument(sys.executable),
        env_id=_shell_argument(env_id),
        task_id=_shell_argument(env_id.replace("-v1", "").replace("-", "_").lower()),
        num_traj=int(num_traj),
        raw_dir=_shell_argument(raw_dir),
        raw_task_dir=_shell_argument(raw_task_dir),
    )


def _shell_argument(value: str | Path) -> str:
    text = str(value)
    if os.name == "nt":
        return subprocess.list2cmdline([text])
    return shlex.quote(text)


def _write_collect_manifest(raw_dir: Path, out: Path, runs: list[dict[str, Any]], failures: list[dict[str, Any]]) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "nyssa-maniskill-demo-collect-v1",
        "raw_dir": raw_dir.as_posix(),
        "out": out.as_posix(),
        "runs": runs,
        "failure_count": len(failures),
        "failures": failures,
    }
    manifest_path = out / "collect_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def _safe_env_id(env_id: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in env_id)
