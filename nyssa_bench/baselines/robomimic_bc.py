from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def load_robomimic_checkpoint(path: str | Path | None = None) -> Any:
    checkpoint = Path(path or os.getenv("NYSSA_ROBOMIMIC_CHECKPOINT", "checkpoints/robomimic_policy.pth"))
    if not checkpoint.exists():
        raise RuntimeError(
            f"RoboMimic checkpoint not found: {checkpoint}. Train one with `nyssa train-robomimic` "
            "or set NYSSA_ROBOMIMIC_POLICY=module:factory."
        )
    try:
        from robomimic.utils.file_utils import policy_from_checkpoint
    except ImportError as exc:
        raise RuntimeError(
            'RoboMimic integration requires: python -m pip install "nyssa-bench[robomimic]"'
        ) from exc
    policy, checkpoint_data = policy_from_checkpoint(ckpt_path=str(checkpoint))
    feature_dim = _flat_feature_dim_from_checkpoint(checkpoint_data)
    if feature_dim is not None:
        setattr(policy, "_nyssa_flat_feature_dim", feature_dim)
    return policy


def create_robomimic_policy() -> Any:
    return load_robomimic_checkpoint()


def train_robomimic(config: str | Path, *, name: str | None = None, debug: bool = False) -> None:
    try:
        import robomimic  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            'RoboMimic training requires: python -m pip install "nyssa-bench[robomimic]"'
        ) from exc

    config_path = Path(config)
    command = [sys.executable, "-m", "robomimic.scripts.train", "--config", str(config_path)]
    if name:
        command.extend(["--name", name])
    if debug:
        command.append("--debug")
    before = _checkpoint_set(_output_dir_from_config(config_path))
    subprocess.run(command, check=True)
    after = _checkpoint_set(_output_dir_from_config(config_path))
    if not after.difference(before):
        raise RuntimeError(
            "RoboMimic training finished without producing a new model_epoch_*.pth checkpoint. "
            "Check the RoboMimic log for the underlying training error."
        )


def write_robomimic_bc_config(
    *,
    data: str | Path,
    out: str | Path,
    output_dir: str | Path = "checkpoints/robomimic",
    name: str = "nyssa_robomimic_bc_flat",
    epochs: int = 50,
    batch_size: int = 64,
    seed: int = 1,
    learning_rate: float = 1e-4,
) -> Path:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "algo_name": "bc",
        "experiment": {
            "name": name,
            "validate": False,
            "logging": {
                "terminal_output_to_txt": True,
                "log_tb": True,
            },
            "save": {
                "enabled": True,
                "every_n_seconds": 600,
                "every_n_epochs": 1,
            },
            "rollout": {
                "enabled": False,
            },
            "epoch_every_n_steps": 1000,
            "validation_epoch_every_n_steps": 100,
        },
        "train": {
            "data": str(Path(data).resolve()),
            "output_dir": str(Path(output_dir).resolve()),
            "num_epochs": int(epochs),
            "batch_size": int(batch_size),
            "num_data_workers": 0,
            "hdf5_cache_mode": "low_dim",
            "hdf5_normalize_obs": False,
            "seed": int(seed),
        },
        "observation": {
            "modalities": {
                "obs": {
                    "low_dim": ["flat"],
                    "rgb": [],
                    "depth": [],
                    "scan": [],
                }
            }
        },
        "algo": {
            "optim_params": {
                "policy": {
                    "learning_rate": {
                        "initial": float(learning_rate),
                    }
                }
            },
            "actor_layer_dims": [1024, 1024],
            "gmm": {
                "enabled": False,
            },
            "gaussian": {
                "enabled": False,
            },
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _output_dir_from_config(config: Path) -> Path:
    try:
        payload = json.loads(config.read_text(encoding="utf-8"))
        return Path(payload.get("train", {}).get("output_dir", "checkpoints/robomimic"))
    except (OSError, json.JSONDecodeError):
        return Path("checkpoints/robomimic")


def _checkpoint_set(output_dir: Path) -> set[Path]:
    if not output_dir.exists():
        return set()
    return set(output_dir.rglob("model_epoch_*.pth"))


def _flat_feature_dim_from_checkpoint(checkpoint_data: Any) -> int | None:
    if not isinstance(checkpoint_data, dict):
        return None
    shape_metadata = checkpoint_data.get("shape_metadata")
    if not isinstance(shape_metadata, dict):
        return None
    all_shapes = shape_metadata.get("all_shapes")
    if not isinstance(all_shapes, dict):
        return None
    flat_shape = all_shapes.get("flat")
    if flat_shape is None:
        return None
    try:
        dimensions = tuple(int(value) for value in flat_shape)
    except (TypeError, ValueError):
        return None
    if not dimensions or any(value <= 0 for value in dimensions):
        return None
    return math.prod(dimensions)
