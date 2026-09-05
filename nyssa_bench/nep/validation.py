from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from pydantic import ValidationError

from nyssa_bench.nep.compatibility import check_nep_compatibility
from nyssa_bench.nep.protocol import NEPManifest


@dataclass(frozen=True)
class NEPValidationIssue:
    code: str
    path: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "path": self.path, "message": self.message}


@dataclass(frozen=True)
class NEPValidationReport:
    valid: bool
    claim_ready: bool
    evaluation_id: str | None
    nep_version: str | None
    content_sha256: str | None
    requested_claim_tier: str | None
    issues: tuple[NEPValidationIssue, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "nyssa-nep-validation-report-v0.1",
            "valid": self.valid,
            "claim_ready": self.claim_ready,
            "evaluation_id": self.evaluation_id,
            "nep_version": self.nep_version,
            "content_sha256": self.content_sha256,
            "requested_claim_tier": self.requested_claim_tier,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def validate_nep_manifest(
    data: Mapping[str, Any],
) -> tuple[NEPValidationReport, NEPManifest | None]:
    version = data.get("nep_version")
    issues = []
    if isinstance(version, str):
        compatibility = check_nep_compatibility(version)
        if not compatibility.compatible:
            issues.append(
                NEPValidationIssue(
                    "incompatible_nep_version",
                    "nep_version",
                    compatibility.reason,
                )
            )
    if issues:
        return _report(data, issues), None
    try:
        manifest = NEPManifest.model_validate(data)
    except ValidationError as exc:
        for error in exc.errors(include_url=False):
            path = ".".join(str(item) for item in error["loc"]) or "$"
            issues.append(
                NEPValidationIssue(
                    "contract_validation_failed",
                    path,
                    str(error["msg"]),
                )
            )
        return _report(data, issues), None
    return (
        NEPValidationReport(
            valid=True,
            claim_ready=True,
            evaluation_id=manifest.evaluation_id,
            nep_version=manifest.nep_version,
            content_sha256=manifest.content_sha256,
            requested_claim_tier=manifest.claim.requested_tier,
            issues=(),
        ),
        manifest,
    )


def _report(
    data: Mapping[str, Any], issues: list[NEPValidationIssue]
) -> NEPValidationReport:
    claim = data.get("claim")
    return NEPValidationReport(
        valid=False,
        claim_ready=False,
        evaluation_id=str(data["evaluation_id"])
        if data.get("evaluation_id") is not None
        else None,
        nep_version=str(data["nep_version"])
        if data.get("nep_version") is not None
        else None,
        content_sha256=str(data["content_sha256"])
        if data.get("content_sha256") is not None
        else None,
        requested_claim_tier=str(claim.get("requested_tier"))
        if isinstance(claim, Mapping) and claim.get("requested_tier") is not None
        else None,
        issues=tuple(issues),
    )
