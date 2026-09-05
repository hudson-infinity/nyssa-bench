from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from nyssa_bench.cli import main
from nyssa_bench.reference_benchmark.candidate import (
    CANDIDATE_TASKS,
    build_reference_candidate,
)
from nyssa_bench.nep import (
    AssetContract,
    PolicyContract,
    SplitLineageContract,
    TaskContract,
)
from nyssa_bench.reference_benchmark import (
    ArtifactReference,
    BenchmarkSplit,
    ExperimentalDesign,
    ReferenceBenchmarkSpec,
    ReferenceTask,
    SplitDimensionCommitment,
    evaluate_reference_benchmark,
)
from nyssa_bench.validity import AuditResult, BenchmarkValidityReport


MECHANISMS = (
    "grasp_place",
    "nonprehensile",
    "stacking",
    "contact_insertion",
    "articulated",
    "clutter_distractors",
    "multi_stage",
)
DIMENSIONS = (
    "assets",
    "initial_states",
    "poses",
    "task_variants",
    "demonstrations",
)
ROOT = Path(__file__).resolve().parents[1]


def test_committed_candidate_is_generated_from_runtime_task_specs() -> None:
    expected = build_reference_candidate(ROOT).model_dump(mode="json")
    committed = json.loads(
        (ROOT / "configs/reference/nyssa_reference_v0_1.json").read_text(
            encoding="utf-8"
        )
    )
    suite = yaml.safe_load(
        (ROOT / "configs/suites/nyssa_reference_manipulation_v0_1.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert committed == expected
    assert suite["tasks"] == [task.task_id for task in CANDIDATE_TASKS]
    report = evaluate_reference_benchmark(
        ReferenceBenchmarkSpec.model_validate(committed), root=ROOT
    )
    assert report["status"] == "evidence_missing"
    assert report["status_counts"] == {
        "passed": 12,
        "failed": 0,
        "missing": 57,
        "not_applicable": 0,
    }


def test_candidate_reports_missing_evidence_without_claiming_release(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path, status="candidate", committed=False, evidence=False)

    report = evaluate_reference_benchmark(spec, root=tmp_path)

    assert report["status"] == "evidence_missing"
    assert report["release_ready"] is False
    assert report["task_count"] == 12
    assert set(report["mechanism_coverage"]) == set(MECHANISMS)
    assert report["status_counts"]["missing"] == 57
    assert report["status_counts"]["failed"] == 0


def test_release_requires_and_validates_all_evidence(tmp_path: Path) -> None:
    spec = _spec(tmp_path, status="release", committed=True, evidence=True)

    report = evaluate_reference_benchmark(spec, root=tmp_path)

    assert report["status"] == "release_ready"
    assert report["release_ready"] is True
    assert report["status_counts"]["missing"] == 0
    assert report["status_counts"]["failed"] == 0
    assert report["status_counts"]["passed"] == 69


def test_tampered_task_and_evidence_are_failed_not_missing(tmp_path: Path) -> None:
    spec = _spec(tmp_path, status="release", committed=True, evidence=True)
    task_path = tmp_path / spec.tasks[0].task_spec.path
    task_path.write_text("task_id: changed\n", encoding="utf-8")

    report = evaluate_reference_benchmark(spec, root=tmp_path)

    check = next(
        item for item in report["checks"] if item["check_id"] == "task:task-00:contract"
    )
    assert check["status"] == "failed"
    assert "SHA-256 mismatch" in check["reason"]
    assert report["status"] == "failed"


def test_hidden_content_cannot_be_exposed() -> None:
    artifact = ArtifactReference(path="hidden.json", sha256="a" * 64)
    dimensions = tuple(
        SplitDimensionCommitment(
            dimension=dimension,
            content_sha256="a" * 64,
            item_count=1,
            status="committed",
            public_artifact=artifact,
        )
        for dimension in DIMENSIONS
    )

    with pytest.raises(ValidationError, match="cannot expose artifact paths"):
        BenchmarkSplit(
            split_id="hidden",
            partition="hidden_test",
            producer_id="owner",
            evaluator_id="evaluator",
            protected=True,
            contents_published=False,
            contamination_status="clean",
            dimensions=dimensions,
        )


def test_split_hash_collisions_and_cycles_are_rejected(tmp_path: Path) -> None:
    spec = _spec(tmp_path, status="candidate", committed=False, evidence=False)
    payload = spec.model_dump(mode="json")
    payload["splits"][1]["parent_split_ids"] = [payload["splits"][2]["split_id"]]
    payload["splits"][2]["parent_split_ids"] = [payload["splits"][1]["split_id"]]

    with pytest.raises(ValidationError, match="cycle"):
        ReferenceBenchmarkSpec.model_validate(payload)

    payload = spec.model_dump(mode="json")
    payload["splits"][1]["dimensions"][0]["content_sha256"] = payload["splits"][0][
        "dimensions"
    ][0]["content_sha256"]
    with pytest.raises(ValidationError, match="content collision"):
        ReferenceBenchmarkSpec.model_validate(payload)


def test_cli_writes_machine_and_human_reports(tmp_path: Path) -> None:
    spec = _spec(tmp_path, status="candidate", committed=False, evidence=False)
    path = tmp_path / "reference.json"
    path.write_text(
        json.dumps(spec.model_dump(mode="json"), indent=2), encoding="utf-8"
    )
    out = tmp_path / "report"

    exit_code = main(
        [
            "audit-reference",
            str(path),
            "--repo-root",
            str(tmp_path),
            "--out",
            str(out),
        ]
    )

    assert exit_code == 2
    assert (out / "reference_benchmark.json").is_file()
    assert "evidence_missing" in (out / "reference_benchmark.html").read_text(
        encoding="utf-8"
    )


def _spec(
    root: Path,
    *,
    status: str,
    committed: bool,
    evidence: bool,
) -> ReferenceBenchmarkSpec:
    tasks = tuple(_task(root, index, evidence=evidence) for index in range(12))
    splits = tuple(
        _split(root, partition, committed=committed)
        for partition in ("train", "validation", "public_test", "hidden_test")
    )
    learned = (
        tuple(
            _learned_evidence(root, family)
            for family in ("diffusion", "transformer_bc")
        )
        if evidence
        else ()
    )
    return ReferenceBenchmarkSpec(
        benchmark_id="unit_reference",
        benchmark_version="1.0.0",
        status=status,  # type: ignore[arg-type]
        tasks=tasks,
        splits=splits,
        experimental_design=ExperimentalDesign(
            paired_seeds=True,
            minimum_episodes_per_condition=50,
            target_success_ci95_width=0.2,
            bootstrap_samples=1000,
            minimum_oracle_success_rate=0.8,
            required_learned_policy_families=2,
            required_controls=("oracle", "zero_action"),
            primary_metrics=("clean_success_rate", "failure_event_distribution"),
            rationale="Fifty paired episodes bound the prespecified success interval.",
        ),
        learned_policy_evidence=learned,
    )


def _task(root: Path, index: int, *, evidence: bool) -> ReferenceTask:
    task_id = f"task-{index:02d}"
    path = root / "tasks" / f"{task_id}.yaml"
    path.parent.mkdir(exist_ok=True)
    task_yaml = {
        "task_id": task_id,
        "engine": "maniskill",
        "robot": "panda",
        "scene": task_id,
        "description": f"Reference task {index}",
        "success": {
            "engine_env_ids": {"maniskill": "PickCube-v1"},
            "success_info_keys": ["success"],
            "obs_mode": "state_dict",
            "control_mode": "pd_ee_delta_pose",
            "max_steps": 100,
        },
        "randomization": {"seed": True},
        "failure_labels": ["missed_target", "timeout"],
    }
    path.write_text(yaml.safe_dump(task_yaml, sort_keys=False), encoding="utf-8")
    contract = TaskContract(
        task_id=task_id,
        task_version="1.0.0",
        engine_ids=("maniskill",),
        robot_id="panda",
        scene_id=task_id,
        horizon_steps=100,
        observation_modalities=("state_dict",),
        action_representation="pd_ee_delta_pose",
        success_predicate={
            "info_key": "success",
            "engine_env_id": "PickCube-v1",
        },
        assets=(
            AssetContract(
                asset_id=f"asset-{index}",
                asset_version="1.0.0",
                sha256=_digest(f"asset-{index}"),
                license_id="Apache-2.0",
                split="hidden_test",
            ),
        ),
        split_lineage=SplitLineageContract(
            split_id="hidden_test",
            partition="hidden_test",
            lineage_sha256=_digest(f"lineage-{index}"),
        ),
    )
    solvability = _solvability_evidence(root, task_id, contract) if evidence else None
    return ReferenceTask(
        contract=contract,
        task_spec=ArtifactReference(
            path=path.relative_to(root).as_posix(), sha256=_sha(path)
        ),
        mechanisms=(MECHANISMS[index] if index < len(MECHANISMS) else "grasp_place",),
        supported_stressors=("action_delay", "observation_gaussian_noise"),
        failure_capabilities=("info.success", "progress", "action_bounds"),
        asset_provenance_status="verified" if evidence else "pending",
        success_predicate_status="verified" if evidence else "pending",
        solvability_evidence=solvability,
    )


def _split(root: Path, partition: str, *, committed: bool) -> BenchmarkSplit:
    dimensions = []
    for dimension in DIMENSIONS:
        artifact = None
        content_hash = _digest(f"{partition}:{dimension}:pending")
        if committed:
            content = root / "splits" / partition / f"{dimension}.json"
            content.parent.mkdir(parents=True, exist_ok=True)
            content.write_text(
                json.dumps({"partition": partition, "dimension": dimension}),
                encoding="utf-8",
            )
            content_hash = _sha(content)
            if partition != "hidden_test":
                artifact = ArtifactReference(
                    path=content.relative_to(root).as_posix(), sha256=content_hash
                )
        dimensions.append(
            SplitDimensionCommitment(
                dimension=dimension,  # type: ignore[arg-type]
                content_sha256=content_hash,
                item_count=10,
                status="committed" if committed else "pending",
                public_artifact=artifact,
            )
        )
    order = ["train", "validation", "public_test", "hidden_test"]
    index = order.index(partition)
    return BenchmarkSplit(
        split_id=partition,
        partition=partition,  # type: ignore[arg-type]
        parent_split_ids=(order[index - 1],) if index else (),
        producer_id="owner",
        evaluator_id="independent-evaluator" if partition == "hidden_test" else "owner",
        protected=partition == "hidden_test",
        contents_published=False if partition == "hidden_test" else True,
        contamination_status="clean" if committed else "unknown",
        dimensions=tuple(dimensions),
    )


def _solvability_evidence(
    root: Path, task_id: str, contract: TaskContract
) -> ArtifactReference:
    path = root / "evidence" / f"{task_id}.json"
    path.parent.mkdir(exist_ok=True)
    payload = {
        "format": "nyssa-reference-solvability-v1",
        "task_id": task_id,
        "task_contract_sha256": _contract_hash(contract),
        "oracle_policy_id": "planner-oracle",
        "episodes": 100,
        "success_rate": 0.95,
        **_valid_result(),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return ArtifactReference(path=path.relative_to(root).as_posix(), sha256=_sha(path))


def _learned_evidence(root: Path, family: str) -> ArtifactReference:
    path = root / "evidence" / f"policy-{family}.json"
    policy = PolicyContract(
        policy_id=f"policy-{family}",
        policy_version="1.0.0",
        policy_family=family,
        checkpoint_id="checkpoint",
        checkpoint_sha256=_digest(f"checkpoint-{family}"),
        preprocessing_sha256=_digest(f"preprocess-{family}"),
        observation_modalities=("state_dict",),
        action_representation="pd_ee_delta_pose",
        action_dimension=7,
        action_lower_bounds=(-1.0,) * 7,
        action_upper_bounds=(1.0,) * 7,
        prediction_horizon=8,
        execution_horizon=4,
        state_semantics="resettable",
        deterministic_seeding=True,
    )
    payload = {
        "format": "nyssa-reference-learned-policy-v1",
        "policy_contract": policy.model_dump(mode="json"),
        "policy_id": policy.policy_id,
        "checkpoint_sha256": policy.checkpoint_sha256,
        "task_ids": [f"task-{index:02d}" for index in range(12)],
        **_valid_result(),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return ArtifactReference(path=path.relative_to(root).as_posix(), sha256=_sha(path))


def _valid_result() -> dict[str, Any]:
    audit_ids = (
        "shortcut_solvability",
        "train_evaluation_leakage",
        "language_observation_ablations",
        "statistical_precision",
        "paired_design",
        "rank_stability",
        "hidden_test_integrity",
    )
    audits = tuple(
        AuditResult(
            audit_id=audit_id,
            category="unit",
            status="passed",
            severity="blocking",
            inputs={"prespecified": True},
            evidence={"valid": True},
            remediation="No remediation required.",
            claim_impact="block",
            summary="Reference fixture passed.",
        )
        for audit_id in audit_ids
    )
    validity = BenchmarkValidityReport(
        benchmark_id="unit_reference",
        benchmark_version="1.0.0",
        claim_tier="public_simulation",
        spec_sha256="a" * 64,
        audits=audits,
        metadata={"required_audits": list(audit_ids)},
    ).to_dict()
    return {
        "run_validity": {
            "status": "validated",
            "public_claim": True,
            "failures": [],
        },
        "benchmark_validity": validity,
    }


def _contract_hash(contract: TaskContract) -> str:
    encoded = json.dumps(
        contract.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
