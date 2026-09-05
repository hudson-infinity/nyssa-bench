from __future__ import annotations

import hashlib
import json
from pathlib import Path

from nyssa_bench.nep.protocol import (
    ArtifactContract,
    AssetContract,
    ClaimContract,
    FailureEvidenceContract,
    InterventionContract,
    NEPManifest,
    PolicyContract,
    SplitLineageContract,
    StressorContract,
    StressorEntryContract,
    TaskContract,
)


REFERENCE_TASKS = {
    "mujoco": "mujoco_inverted_pendulum",
    "maniskill": "maniskill_pick_cube",
}
REFERENCE_ACTION_DIMENSIONS = {"mujoco": 1, "maniskill": 7}


def reference_pipeline_manifest(engine: str) -> NEPManifest:
    if engine not in REFERENCE_TASKS:
        raise ValueError(f"unsupported NEP reference engine: {engine}")
    task_id = REFERENCE_TASKS[engine]
    artifacts = tuple(
        ArtifactContract(
            artifact_id=artifact_id,
            media_type="application/json",
            sha256=_digest(f"{engine}:{artifact_id}"),
            uri=f"nyssa-reference://{engine}/{artifact_id}.json",
        )
        for artifact_id in ("failure-ledger", "detector-contracts", "run-validity")
    )
    return NEPManifest.create(
        evaluation_id=f"nep-reference-{engine}",
        task=TaskContract(
            task_id=task_id,
            task_version="1.0.0",
            engine_ids=(engine,),
            robot_id="reference_robot",
            scene_id="reference_scene",
            horizon_steps=200,
            observation_modalities=("state",),
            action_representation="environment_action",
            success_predicate={"info_key": "success"},
            assets=(
                AssetContract(
                    asset_id=f"{task_id}-definition",
                    asset_version="1.0.0",
                    sha256=_digest(f"{engine}:{task_id}:asset"),
                    license_id="upstream-simulator-license",
                    split="public_test",
                ),
            ),
            split_lineage=SplitLineageContract(
                split_id=f"{task_id}-public-test",
                partition="public_test",
                lineage_sha256=_digest(f"{engine}:{task_id}:split"),
            ),
        ),
        stressor=StressorContract(
            condition_id="clean",
            composition_semantics="ordered",
        ),
        policy=PolicyContract(
            policy_id="random_pipeline_control",
            policy_version="1.0.0",
            policy_family="integration_control",
            checkpoint_id="builtin-random-policy-v1",
            checkpoint_sha256=_digest("builtin-random-policy-v1"),
            preprocessing_sha256=_digest("identity-preprocessing-v1"),
            observation_modalities=("state",),
            action_representation="environment_action",
            action_dimension=REFERENCE_ACTION_DIMENSIONS[engine],
            action_lower_bounds=(-1.0,) * REFERENCE_ACTION_DIMENSIONS[engine],
            action_upper_bounds=(1.0,) * REFERENCE_ACTION_DIMENSIONS[engine],
            prediction_horizon=1,
            execution_horizon=1,
            state_semantics="stateless",
            deterministic_seeding=True,
        ),
        failure_evidence=FailureEvidenceContract(
            ledger_artifact_id="failure-ledger",
            detector_contract_artifact_id="detector-contracts",
            temporal_precision=("exact_step", "terminal_only"),
            evidence_visibility=("policy_observable", "privileged"),
            causal_semantics="hypothesis_only",
        ),
        intervention=InterventionContract(enabled=False),
        claim=ClaimContract(
            requested_tier="pipeline",
            evidence_artifact_ids=("failure-ledger", "run-validity"),
            run_validity_artifact_id="run-validity",
        ),
        artifacts=artifacts,
    )


def result_pack_pipeline_manifest(engine: str, run_dir: str | Path) -> NEPManifest:
    base = reference_pipeline_manifest(engine)
    run_dir = Path(run_dir).resolve()
    artifact_files = {
        "failure-ledger": "failure_ledger.json",
        "detector-contracts": "failure_detector_manifest.json",
        "run-validity": "metrics.json",
        "stressor-backend": "stressor_manifest.json",
    }
    artifacts = tuple(
        ArtifactContract(
            artifact_id=artifact_id,
            media_type="application/json",
            sha256=hashlib.sha256((run_dir / filename).read_bytes()).hexdigest(),
            uri=f"nyssa-result-pack://{filename}",
        )
        for artifact_id, filename in artifact_files.items()
    )
    stressor_payload = json.loads(
        (run_dir / "stressor_manifest.json").read_text(encoding="utf-8")
    )
    configured = stressor_payload.get("configured") or {}
    configured_stressors = configured.get("stressors") or []
    episode_records = stressor_payload.get("episodes") or []
    applications = (
        episode_records[0].get("stressor_context", {}).get("applications", [])
        if episode_records
        else []
    )
    application_by_id = {
        item.get("stressor_id"): item for item in applications if isinstance(item, dict)
    }
    stressors = []
    for item in configured_stressors:
        application = application_by_id.get(item.get("stressor_id"), {})
        stressors.append(
            StressorEntryContract(
                stressor_id=str(item["stressor_id"]),
                stressor_version="1.0.0",
                category=str(application.get("category", "system")),  # type: ignore[arg-type]
                severity=float(item["severity"]),
                seed=int(item["seed"]),
                application_points=tuple(application.get("application_points", [])),
                parameters=dict(item.get("parameters", {})),
                observable_by_policy=bool(
                    application.get("observable_by_policy", False)
                ),
                privileged=bool(application.get("privileged", False)),
                backend_confirmed=application.get("status") == "applied",
                backend_evidence_artifact_id="stressor-backend"
                if application.get("status") == "applied"
                else None,
            )
        )
    return NEPManifest.create(
        evaluation_id=f"{base.evaluation_id}-executed",
        task=base.task,
        stressor=StressorContract(
            condition_id=str(configured.get("condition_id", "clean")),
            composition_semantics="ordered",
            stressors=tuple(stressors),
        ),
        policy=base.policy,
        failure_evidence=base.failure_evidence,
        intervention=base.intervention,
        claim=base.claim,
        artifacts=artifacts,
    )


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()
