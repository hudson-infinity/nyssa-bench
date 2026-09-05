from __future__ import annotations

from typing import Any, Mapping

from nyssa_bench.nep.protocol import (
    ArtifactContract,
    ClaimContract,
    FailureEvidenceContract,
    InterventionContract,
    NEPManifest,
    PolicyContract,
    StressorContract,
    TaskContract,
)


NEP_DRAFT_FORMAT = "nyssa-nep-draft-v0"


def migrate_nep_data(data: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if data.get("format") != NEP_DRAFT_FORMAT:
        return dict(data), None
    allowed = {
        "format",
        "evaluation_id",
        "task",
        "stressor",
        "policy",
        "failure_evidence",
        "intervention",
        "claim",
        "artifacts",
    }
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError("draft NEP contains unknown fields: " + ", ".join(unknown))
    manifest = NEPManifest.create(
        evaluation_id=data.get("evaluation_id"),
        task=TaskContract.model_validate(data.get("task")),
        stressor=StressorContract.model_validate(data.get("stressor")),
        policy=PolicyContract.model_validate(data.get("policy")),
        failure_evidence=FailureEvidenceContract.model_validate(
            data.get("failure_evidence")
        ),
        intervention=InterventionContract.model_validate(data.get("intervention")),
        claim=ClaimContract.model_validate(data.get("claim")),
        artifacts=tuple(
            ArtifactContract.model_validate(item)
            for item in _artifact_list(data.get("artifacts"))
        ),
    )
    return (
        manifest.model_dump(mode="json"),
        {
            "format": "nyssa-nep-migration-v0.1",
            "source_format": NEP_DRAFT_FORMAT,
            "target_format": manifest.format,
            "target_nep_version": manifest.nep_version,
            "semantics": "explicit_breaking_draft_to_0.1_migration",
            "content_sha256": manifest.content_sha256,
        },
    )


def _artifact_list(value: Any) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError("draft NEP artifacts must be a list")
    return value
