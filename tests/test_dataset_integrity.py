from __future__ import annotations

import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from nyssa_bench.baselines.simple_bc import (
    TaskRoutedLinearBCPolicy,
    load_episode_sources,
)
from nyssa_bench.core.episode import EpisodeResult, StepRecord
from nyssa_bench.datasets.export_hdf5 import export_hdf5
from nyssa_bench.datasets.export_robomimic import validate_robomimic_observations


@pytest.mark.parametrize("archive", [False, True])
def test_episode_sources_select_aggregates_per_subtree(tmp_path: Path, archive: bool):
    # A bundle may mix aggregated runs, per-task runs, and custom task names.
    members = {
        "run_a/episodes.json": [{"task_id": "aggregate"}],
        "run_a/custom_task/episodes.json": [{"task_id": "duplicate"}],
        "run_b/maniskill_pick_cube/episodes.json": [{"task_id": "independent"}],
        "run_c/custom_task/episodes.json": [{"task_id": "custom"}],
        "run_c/recovery_dataset/episodes.json": [{"task_id": "recovery"}],
    }
    if archive:
        source = tmp_path / "bundle.zip"
        with zipfile.ZipFile(source, "w") as handle:
            for name, payload in members.items():
                handle.writestr(name, json.dumps(payload))
    else:
        source = tmp_path / "bundle"
        for name, payload in members.items():
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload), encoding="utf-8")

    assert [item["task_id"] for item in load_episode_sources([source])] == [
        "aggregate", "independent", "custom",
    ]


def test_episode_zip_ignores_similarly_named_files(tmp_path: Path):
    source = tmp_path / "bundle.zip"
    with zipfile.ZipFile(source, "w") as handle:
        handle.writestr("episodes.json", '[{"task_id": "valid"}]')
        handle.writestr("not_episodes.json", "invalid JSON")

    assert load_episode_sources([source]) == [{"task_id": "valid"}]


@pytest.mark.parametrize("missing_task", ["error", "zero"])
def test_task_bc_rejects_corrupt_checkpoint(tmp_path: Path, missing_task: str):
    (tmp_path / "task.json").write_text('{"format": "nyssa-linear-bc-v1"}')
    policy = TaskRoutedLinearBCPolicy(tmp_path, missing_task=missing_task)
    policy.reset(task=SimpleNamespace(task_id="task"))

    with pytest.raises(KeyError, match="weights"):
        policy.predict_action({"raw": [0.0]})


def test_task_bc_propagates_prediction_error(tmp_path: Path):
    class BrokenModel:
        def predict_action(self, observation):
            raise KeyError("required observation")

    policy = TaskRoutedLinearBCPolicy(tmp_path, missing_task="zero")
    policy._models["task"] = BrokenModel()
    policy.reset(task=SimpleNamespace(task_id="task"))

    with pytest.raises(KeyError, match="required observation"):
        policy.predict_action({"raw": [0.0]})


def _episode(task: str, index: int, seed: int, values: list[float]) -> EpisodeResult:
    return EpisodeResult(
        task_id=task, episode_index=index, seed=seed, success=True,
        failure_label=None, metrics={},
        steps=[
            StepRecord({"raw": [value]}, [0.0], value, False, False, {})
            for value in values
        ],
    )


def test_hdf5_preserves_repeated_episode_indices_and_identity(tmp_path: Path):
    h5py = pytest.importorskip("h5py")
    episodes = [
        _episode("task_a", 0, 7, [1.0]),
        _episode("task_b", 0, 7, [2.0]),
        _episode("task_a", 0, 8, [3.0]),
        _episode("task_a", 1, 7, [4.0]),
    ]
    path = export_hdf5(episodes, tmp_path / "episodes.hdf5")

    with h5py.File(path) as handle:
        assert len(handle) == len(episodes)
        actual = {
            (group.attrs["task_id"], group.attrs["episode_index"], group.attrs["seed"]):
            group["rewards"][:].tolist()
            for group in handle.values()
        }
    assert actual == {
        (episode.task_id, episode.episode_index, episode.seed):
        [step.reward for step in episode.steps]
        for episode in episodes
    }


def test_hdf5_keeps_unique_legacy_group_names(tmp_path: Path):
    h5py = pytest.importorskip("h5py")
    path = export_hdf5([_episode("task", 12, 7, [1.0])], tmp_path / "out.hdf5")
    with h5py.File(path) as handle:
        assert list(handle) == ["episode_0012"]


def test_observation_variance_is_stable_for_large_offsets():
    values = (1e10 + np.arange(64) * 0.01).tolist()
    quality = validate_robomimic_observations(
        [_episode("task", 0, 0, values)], feature_dim=1,
    )
    assert quality["feature_variance_max"] == pytest.approx(np.var(values), rel=1e-3)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_robomimic_rejects_nonfinite_observations(value: float):
    with pytest.raises(ValueError, match="non-finite"):
        validate_robomimic_observations(
            [_episode("task", 0, 0, [value])], feature_dim=1,
        )
