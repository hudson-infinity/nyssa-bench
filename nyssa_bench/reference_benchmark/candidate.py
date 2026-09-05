from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from nyssa_bench.nep import AssetContract, SplitLineageContract, TaskContract
from nyssa_bench.reference_benchmark.protocol import (
    ArtifactReference,
    BenchmarkSplit,
    ExperimentalDesign,
    Mechanism,
    ReferenceBenchmarkSpec,
    ReferenceTask,
    SplitDimensionCommitment,
)


@dataclass(frozen=True)
class CandidateTask:
    task_id: str
    env_id: str
    mechanism: Mechanism
    horizon: int
    robot_id: str = "panda"


CANDIDATE_TASKS = (
    CandidateTask("maniskill_pick_cube", "PickCube-v1", "grasp_place", 80),
    CandidateTask("maniskill_place_sphere", "PlaceSphere-v1", "grasp_place", 80),
    CandidateTask("maniskill_stack_cube", "StackCube-v1", "stacking", 120),
    CandidateTask("maniskill_push_cube", "PushCube-v1", "nonprehensile", 80),
    CandidateTask("maniskill_push_t", "PushT-v1", "nonprehensile", 150),
    CandidateTask(
        "maniskill_peg_insertion_side", "PegInsertionSide-v1", "contact_insertion", 200
    ),
    CandidateTask("maniskill_plug_charger", "PlugCharger-v1", "contact_insertion", 250),
    CandidateTask(
        "maniskill_open_cabinet_drawer", "OpenCabinetDrawer-v1", "articulated", 250
    ),
    CandidateTask(
        "maniskill_open_cabinet_door", "OpenCabinetDoor-v1", "articulated", 250
    ),
    CandidateTask(
        "maniskill_turn_faucet",
        "TurnFaucet-v1",
        "articulated",
        250,
        "panda_wristcam",
    ),
    CandidateTask(
        "maniskill_pick_clutter_ycb",
        "PickClutterYCB-v1",
        "clutter_distractors",
        150,
    ),
    CandidateTask("maniskill_pull_cube_tool", "PullCubeTool-v1", "multi_stage", 150),
)


def build_reference_candidate(repo_root: str | Path) -> ReferenceBenchmarkSpec:
    root = Path(repo_root).resolve()
    tasks = tuple(_reference_task(root, definition) for definition in CANDIDATE_TASKS)
    partitions = ("train", "validation", "public_test", "hidden_test")
    splits = tuple(
        BenchmarkSplit(
            split_id=f"nyssa-reference-v0.1-{partition}",
            partition=partition,  # type: ignore[arg-type]
            parent_split_ids=(f"nyssa-reference-v0.1-{partitions[index - 1]}",)
            if index
            else (),
            producer_id="hudson-reference-data-owner",
            evaluator_id=(
                "independent-hidden-evaluator"
                if partition == "hidden_test"
                else "hudson-reference-data-owner"
            ),
            protected=partition == "hidden_test",
            contents_published=False,
            contamination_status="unknown",
            dimensions=tuple(
                SplitDimensionCommitment(
                    dimension=dimension,  # type: ignore[arg-type]
                    content_sha256=_digest(f"pending:{partition}:{dimension}:v0.1"),
                    item_count=1,
                    status="pending",
                )
                for dimension in (
                    "assets",
                    "initial_states",
                    "poses",
                    "task_variants",
                    "demonstrations",
                )
            ),
        )
        for index, partition in enumerate(partitions)
    )
    return ReferenceBenchmarkSpec(
        benchmark_id="nyssa_reference_manipulation_v0_1",
        benchmark_version="0.1.0",
        status="candidate",
        tasks=tasks,
        splits=splits,
        experimental_design=ExperimentalDesign(
            paired_seeds=True,
            minimum_episodes_per_condition=100,
            target_success_ci95_width=0.2,
            bootstrap_samples=5000,
            minimum_oracle_success_rate=0.8,
            required_learned_policy_families=2,
            required_controls=("oracle", "zero_action", "random"),
            primary_metrics=(
                "clean_success_rate",
                "shifted_success_rate",
                "failure_event_distribution",
                "mean_time_to_failure_steps",
                "counterfactual_recovery_gain",
            ),
            rationale=(
                "One hundred paired episodes per condition supports Wilson intervals "
                "no wider than 0.2 in the worst case; paired bootstrap intervals use "
                "five thousand resamples. Final power must be recomputed from pilot data."
            ),
        ),
        metadata={
            "upstream": "ManiSkill 3.0.1",
            "upstream_task_documentation": (
                "https://maniskill.readthedocs.io/en/latest/tasks/"
            ),
            "candidate_boundary": (
                "Environment mappings are source-backed candidates. Asset commitments, "
                "split contents, oracle evidence, and learned-policy evidence are pending."
            ),
        },
    )


def _reference_task(root: Path, definition: CandidateTask) -> ReferenceTask:
    task_path = _task_path(root, definition.task_id)
    return ReferenceTask(
        contract=TaskContract(
            task_id=definition.task_id,
            task_version="0.1.0",
            engine_ids=("maniskill",),
            robot_id=definition.robot_id,
            scene_id=definition.task_id,
            horizon_steps=definition.horizon,
            observation_modalities=("state_dict",),
            action_representation="pd_ee_delta_pose",
            success_predicate={
                "engine_env_id": definition.env_id,
                "info_keys": ["success", "is_success", "success_once"],
            },
            assets=(
                AssetContract(
                    asset_id=f"maniskill-environment:{definition.env_id}",
                    asset_version="3.0.1",
                    sha256=_digest(f"pending-asset-audit:{definition.env_id}"),
                    license_id="pending-upstream-asset-license-audit",
                    split="hidden_test",
                ),
            ),
            split_lineage=SplitLineageContract(
                split_id="nyssa-reference-v0.1-hidden_test",
                partition="hidden_test",
                lineage_sha256=_digest(
                    f"pending-task-lineage:{definition.task_id}:v0.1"
                ),
            ),
        ),
        task_spec=ArtifactReference(
            path=task_path.relative_to(root).as_posix(),
            sha256=_sha256_file(task_path),
        ),
        mechanisms=(definition.mechanism,),
        supported_stressors=(
            "action_delay",
            "action_gaussian_noise",
            "observation_gaussian_noise",
        ),
        failure_capabilities=(
            "info.success",
            "contact",
            "grasp",
            "progress",
            "action_bounds",
        ),
        asset_provenance_status="pending",
        success_predicate_status="pending",
    )


def _task_path(root: Path, task_id: str) -> Path:
    matches = list((root / "nyssa_bench" / "tasks").rglob(f"{task_id}.yaml"))
    if len(matches) != 1:
        raise ValueError(f"expected one TaskSpec for {task_id}, found {len(matches)}")
    return matches[0]


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
