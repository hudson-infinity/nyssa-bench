from __future__ import annotations

import hashlib
import json
from pathlib import Path

from nyssa_bench.nep import PolicyContract, TrainingDataContract
from nyssa_bench.policy_tracks.protocol import (
    ComputeContract,
    PolicyTrack,
    PolicyTrackRegistry,
)
from nyssa_bench.reference_benchmark import ArtifactReference
from nyssa_bench.reference_benchmark import load_reference_benchmark


TASK_SUBSET = (
    "maniskill_pick_cube",
    "maniskill_push_cube",
    "maniskill_stack_cube",
)
EVALUATION_SEEDS = (10000, 10001, 10002)
EVALUATION_ASSETS = ("reference-hidden-assets-v0.1",)
EVALUATION_SPLIT_ID = "nyssa-reference-v0.1-hidden_test"
STRESSOR_CONDITION_ID = "reference-action-delay-s05"


def build_policy_track_candidate(repo_root: str | Path) -> PolicyTrackRegistry:
    root = Path(repo_root).resolve()
    setup = root / "docs" / "policy_tracks.md"
    reference = root / "configs" / "reference" / "nyssa_reference_v0_1.json"
    reference_spec = load_reference_benchmark(reference)
    evaluation_split = next(
        split
        for split in reference_spec.splits
        if split.split_id == EVALUATION_SPLIT_ID
    )
    setup_ref = _reference(root, setup)
    shared = {
        "status": "integration_only",
        "setup_document": setup_ref,
        "evaluation_task_ids": TASK_SUBSET,
        "evaluation_split_id": EVALUATION_SPLIT_ID,
        "evaluation_split_sha256": _model_sha256(
            evaluation_split.model_dump(mode="json")
        ),
        "evaluation_seeds": EVALUATION_SEEDS,
        "evaluation_asset_ids": EVALUATION_ASSETS,
        "stressor_condition_id": STRESSOR_CONDITION_ID,
    }
    tracks = (
        PolicyTrack(
            track_id="planner_oracle",
            role="oracle_control",
            required_for_release=True,
            adapter_id="scripted_oracle",
            contract=_contract(
                "planner_oracle",
                "planner_oracle",
                prediction_horizon=1,
                execution_horizon=1,
            ),
            evaluation_compute=_evaluation_compute("fp32"),
            **shared,  # type: ignore[arg-type]
        ),
        PolicyTrack(
            track_id="robomimic_bc",
            role="learned",
            required_for_release=True,
            adapter_id="task_robomimic",
            contract=_contract("robomimic_bc", "robomimic_bc", training=True),
            evaluation_compute=_evaluation_compute("fp32"),
            **shared,  # type: ignore[arg-type]
        ),
        PolicyTrack(
            track_id="diffusion_action_chunk",
            role="learned",
            required_for_release=True,
            adapter_id="diffusion",
            contract=_contract(
                "diffusion_action_chunk",
                "diffusion_policy",
                prediction_horizon=16,
                execution_horizon=4,
                training=True,
            ),
            evaluation_compute=_evaluation_compute("fp16"),
            **shared,  # type: ignore[arg-type]
        ),
        PolicyTrack(
            track_id="openvla",
            role="vla",
            required_for_release=False,
            adapter_id="openvla",
            contract=_contract(
                "openvla",
                "vision_language_action",
                modalities=("rgb", "language"),
                training=True,
            ),
            evaluation_compute=_evaluation_compute("bf16"),
            **shared,  # type: ignore[arg-type]
        ),
        PolicyTrack(
            track_id="random_sanity",
            role="sanity_control",
            required_for_release=True,
            adapter_id="random",
            contract=_contract("random_sanity", "random"),
            evaluation_compute=_evaluation_compute("fp32", accelerator_count=0),
            **shared,  # type: ignore[arg-type]
        ),
    )
    return PolicyTrackRegistry(
        registry_id="nyssa_policy_tracks_v0_1",
        registry_version="0.1.0",
        status="candidate",
        reference_benchmark=_reference(root, reference),
        benchmark_task_subset=TASK_SUBSET,
        minimum_episodes_per_task=100,
        required_learned_policy_families=2,
        tracks=tracks,
        metadata={
            "claim_boundary": (
                "All tracks are integration-only until real checkpoints, provenance, "
                "conformance, and paired result fingerprints are attached."
            )
        },
    )


def _contract(
    policy_id: str,
    family: str,
    *,
    modalities: tuple[str, ...] = ("state_dict",),
    prediction_horizon: int = 1,
    execution_horizon: int = 1,
    training: bool = False,
) -> PolicyContract:
    training_data = (
        (
            TrainingDataContract(
                dataset_id=f"pending-{policy_id}-training",
                dataset_version="0.1.0",
                sha256=_digest(f"pending-training:{policy_id}"),
                split_ids=("nyssa-reference-v0.1-train",),
                license_id="pending-training-data-license-audit",
            ),
        )
        if training
        else ()
    )
    return PolicyContract(
        policy_id=policy_id,
        policy_version="0.1.0",
        policy_family=family,
        checkpoint_id=f"pending-{policy_id}-checkpoint",
        checkpoint_sha256=_digest(f"pending-checkpoint:{policy_id}"),
        preprocessing_sha256=_digest(f"pending-preprocessing:{policy_id}"),
        observation_modalities=modalities,
        action_representation="pd_ee_delta_pose",
        action_dimension=7,
        action_lower_bounds=(-1.0,) * 7,
        action_upper_bounds=(1.0,) * 7,
        prediction_horizon=prediction_horizon,
        execution_horizon=execution_horizon,
        state_semantics="resettable",
        deterministic_seeding=True,
        training_data=training_data,
    )


def _evaluation_compute(
    precision: str, *, accelerator_count: int = 1
) -> ComputeContract:
    return ComputeContract(
        hardware="pending-capable-runner",
        accelerator_count=accelerator_count,
        training_hours=0.0,
        peak_memory_gb=None,
        precision=precision,
    )


def _reference(root: Path, path: Path) -> ArtifactReference:
    return ArtifactReference(
        path=path.relative_to(root).as_posix(),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _model_sha256(value: dict[str, object]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
