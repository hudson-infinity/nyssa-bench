from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def load_lerobot_policy(path: str | Path | None = None) -> Any:
    policy_path = Path(path or os.getenv("NYSSA_LEROBOT_POLICY_PATH", "checkpoints/lerobot_policy"))
    if not policy_path.exists():
        raise RuntimeError(
            f"LeRobot policy path not found: {policy_path}. Set NYSSA_LEROBOT_POLICY=module:factory "
            "or NYSSA_LEROBOT_POLICY_PATH to a trained LeRobot policy directory."
        )
    try:
        from lerobot.common.policies.factory import make_policy
    except ImportError as exc:
        raise RuntimeError(
            'LeRobot integration requires: python -m pip install "nyssa-bench[lerobot]"'
        ) from exc
    return make_policy(policy_path=str(policy_path))


def create_lerobot_policy() -> Any:
    return load_lerobot_policy()
