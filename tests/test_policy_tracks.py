from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from nyssa_bench.core.episode import EpisodeResult
from nyssa_bench.metrics.vector import build_metric_vector
from nyssa_bench.cli import main
from nyssa_bench.policy_tracks import evaluator as track_evaluator
from nyssa_bench.policy_tracks import (
    PolicyTrack,
    PolicyTrackRegistry,
    evaluate_policy_tracks,
)
from nyssa_bench.policy_tracks.candidate import build_policy_track_candidate
from nyssa_bench.policies.diffusion_policy_adapter import DiffusionPolicyAdapter
from nyssa_bench.policies.lerobot_adapter import LeRobotPolicy
from nyssa_bench.policies.openvla_adapter import OpenVLAPolicy
from nyssa_bench.reference_benchmark import ArtifactReference
from nyssa_bench.regression.evidence import RunEvidence


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "configs" / "policy_tracks" / "nyssa_policy_tracks_v0_1.json"


def test_committed_policy_candidate_is_deterministic_and_honest() -> None:
    expected = build_policy_track_candidate(ROOT).model_dump(mode="json")
    committed = json.loads(CANDIDATE.read_text(encoding="utf-8"))

    assert committed == expected
    report = evaluate_policy_tracks(
        PolicyTrackRegistry.model_validate(committed), root=ROOT
    )
    assert report["status"] == "evidence_missing"
    assert report["release_ready"] is False
    assert report["validated_oracle"] is False
    assert report["validated_learned_policy_families"] == []
    assert report["status_counts"] == {
        "passed": 5,
        "failed": 0,
        "missing": 19,
        "not_applicable": 7,
    }
    assert all(track["validated"] is False for track in report["tracks"])


def test_validated_track_requires_every_native_artifact() -> None:
    track = build_policy_track_candidate(ROOT).tracks[1]
    payload = track.model_dump(mode="json")
    payload["status"] = "validated"

    with pytest.raises(ValidationError, match="validated policy track is missing"):
        PolicyTrack.model_validate(payload)


def test_sanity_control_cannot_be_promoted() -> None:
    track = build_policy_track_candidate(ROOT).tracks[-1]
    payload = track.model_dump(mode="json")
    payload["status"] = "validated"

    with pytest.raises(ValidationError, match="sanity controls"):
        PolicyTrack.model_validate(payload)


def test_all_tracks_must_share_the_same_evaluation_design() -> None:
    registry = build_policy_track_candidate(ROOT)
    tracks = list(registry.tracks)
    tracks[1] = tracks[1].model_copy(update={"evaluation_seeds": (42,)})

    with pytest.raises(ValidationError, match="same evaluation design"):
        PolicyTrackRegistry.model_validate(
            {**registry.model_dump(mode="json"), "tracks": tracks}
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("split_ids", ["nyssa-reference-v0.1-hidden_test"], "evaluation split"),
        ("asset_ids", ["reference-hidden-assets-v0.1"], "assets overlap"),
    ],
)
def test_training_leakage_is_rejected(
    tmp_path: Path, field: str, value: list[str], message: str
) -> None:
    registry = build_policy_track_candidate(ROOT)
    track = registry.tracks[1]
    source = track.contract.training_data[0]
    training = {
        "format": "nyssa-policy-training-provenance-v1",
        "policy_id": track.contract.policy_id,
        "checkpoint_sha256": track.contract.checkpoint_sha256,
        "preprocessing_sha256": track.contract.preprocessing_sha256,
        "training_data": [
            {
                "dataset_id": source.dataset_id,
                "dataset_version": source.dataset_version,
                "sha256": source.sha256,
                "split_ids": list(source.split_ids),
                "episode_seeds": [0, 1],
                "asset_ids": ["train-assets"],
                "task_ids": list(track.evaluation_task_ids),
                "demonstration_ids": ["demo-0"],
            }
        ],
        "action_transform": {
            "representation": track.contract.action_representation,
            "lower_bounds": list(track.contract.action_lower_bounds),
            "upper_bounds": list(track.contract.action_upper_bounds),
        },
        "temporal_context": {
            "prediction_horizon": track.contract.prediction_horizon,
            "execution_horizon": track.contract.execution_horizon,
        },
        "compute": {
            "hardware": "unit-gpu",
            "accelerator_count": 1,
            "training_hours": 1.0,
            "peak_memory_gb": 8.0,
            "precision": "fp32",
        },
    }
    training["training_data"][0][field] = value
    path = tmp_path / "training.json"
    path.write_text(json.dumps(training), encoding="utf-8")
    reference = ArtifactReference(path="training.json", sha256=_sha(path))
    updated = track.model_copy(update={"training_provenance": reference})
    tracks = tuple(
        updated if item.track_id == track.track_id else item for item in registry.tracks
    )
    local_registry = registry.model_copy(update={"tracks": tracks})

    report = evaluate_policy_tracks(local_registry, root=tmp_path)
    check = next(
        item
        for item in report["checks"]
        if item["check_id"] == "track:robomimic_bc:training_provenance"
    )

    assert check["status"] == "failed"
    assert message in check["reason"]


def test_changed_setup_document_fails_instead_of_becoming_missing(
    tmp_path: Path,
) -> None:
    registry = build_policy_track_candidate(ROOT)
    copied = tmp_path / "setup.md"
    copied.write_text("changed", encoding="utf-8")
    track = registry.tracks[0].model_copy(
        update={
            "setup_document": ArtifactReference(
                path="setup.md",
                sha256=registry.tracks[0].setup_document.sha256,
            )
        }
    )
    local = registry.model_copy(update={"tracks": (track, *registry.tracks[1:])})

    report = evaluate_policy_tracks(local, root=tmp_path)
    check = next(
        item
        for item in report["checks"]
        if item["check_id"] == "track:planner_oracle:setup"
    )

    assert check["status"] == "failed"
    assert "SHA-256 mismatch" in check["reason"]


def test_cli_writes_policy_track_reports(tmp_path: Path) -> None:
    out = tmp_path / "report"

    exit_code = main(
        [
            "audit-policy-tracks",
            str(CANDIDATE),
            "--repo-root",
            str(ROOT),
            "--out",
            str(out),
        ]
    )

    assert exit_code == 2
    assert (out / "policy_tracks.json").is_file()
    assert "evidence_missing" in (out / "policy_tracks.html").read_text(
        encoding="utf-8"
    )


@pytest.mark.parametrize(
    ("adapter_type", "adapter_id"),
    [
        (DiffusionPolicyAdapter, "diffusion"),
        (LeRobotPolicy, "lerobot"),
        (OpenVLAPolicy, "openvla"),
    ],
)
def test_external_adapters_delegate_lifecycle_and_metadata(
    adapter_type: type, adapter_id: str
) -> None:
    class Model:
        def __init__(self) -> None:
            self.reset_args = None
            self.closed = False

        def reset(self, *, task=None, seed=None) -> None:
            self.reset_args = (task, seed)

        def close(self) -> None:
            self.closed = True

        def metadata(self) -> dict[str, str]:
            return {"checkpoint_id": "unit"}

        def act(self, observation):
            return [0.0]

    model = Model()
    adapter = adapter_type(model=model)

    adapter.reset(task="pick", seed=7)
    metadata = adapter.metadata()
    adapter.close()

    assert model.reset_args == ("pick", 7)
    assert model.closed is True
    assert metadata["adapter"] == adapter_id
    assert metadata["checkpoint_id"] == "unit"


def test_native_run_pair_requires_complete_keys_replays_ledgers_and_uncertainty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    track = build_policy_track_candidate(ROOT).tracks[1]
    clean = _run_evidence(tmp_path / "clean", shifted=False)
    shifted = _run_evidence(tmp_path / "shifted", shifted=True)
    monkeypatch.setattr(track_evaluator, "run_validity_available", lambda _: True)
    monkeypatch.setattr(track_evaluator, "benchmark_validity_available", lambda _: True)

    track_evaluator._validate_run_pair(
        clean,
        shifted,
        track,
        training_episode_seeds=set(),
        minimum_episodes_per_task=2,
    )

    shifted.episodes[0].episode_index = 99
    with pytest.raises(ValueError, match="not completely paired"):
        track_evaluator._validate_run_pair(
            clean,
            shifted,
            track,
            training_episode_seeds=set(),
            minimum_episodes_per_task=2,
        )


def _run_evidence(root: Path, *, shifted: bool) -> RunEvidence:
    root.mkdir()
    episodes = []
    hashes = {}
    for task_id in (
        "maniskill_pick_cube",
        "maniskill_push_cube",
        "maniskill_stack_cube",
    ):
        for index in range(2):
            replay = Path("videos") / f"{task_id}-{index}.mp4"
            path = root / replay
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b"mp4-evidence")
            hashes[replay.as_posix()] = _sha(path)
            episodes.append(
                EpisodeResult(
                    task_id=task_id,
                    episode_index=index,
                    seed=100_000 + index,
                    success=index == 0,
                    failure_label=None if index == 0 else "missed_target",
                    metrics={},
                    replay_path=replay.as_posix(),
                    stressor_context=(
                        {
                            "applications": [
                                {
                                    "status": "applied",
                                    "requested": {"severity": 0.5},
                                }
                            ]
                        }
                        if shifted
                        else {}
                    ),
                    failure_ledger=object(),  # type: ignore[arg-type]
                )
            )
    summary = {
        "episodes": len(episodes),
        "success_count": sum(episode.success for episode in episodes),
    }
    summary["metric_vector"] = build_metric_vector(summary, episodes)
    return RunEvidence(
        root=root,
        metadata={
            "seed": 10000,
            "stressor_config": (
                {"condition_id": "reference-action-delay-s05"} if shifted else None
            ),
        },
        manifest={},
        summary=summary,
        episodes=tuple(episodes),
        artifacts_sha256=hashes,
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
