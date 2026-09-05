from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from nyssa_bench.validity import BenchmarkValidityReport


CLAIM_EVIDENCE_FORMAT = "nyssa-claim-evidence-matrix-v1"
CLAIM_EVIDENCE_REPORT_FORMAT = "nyssa-claim-evidence-report-v1"
CAPABILITY_STATUSES = {
    "implemented",
    "integration_only",
    "experimental",
    "planned",
}
EVIDENCE_TIERS = {
    "source_verified",
    "integration_only",
    "experimental",
    "planned",
    "result_validated",
    "predictive_validated",
}


def load_claim_evidence(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid claim evidence matrix: {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError("claim evidence matrix must contain a mapping")
    return dict(value)


def validate_claim_evidence(
    matrix: Mapping[str, Any], *, repo_root: str | Path
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    _validate_top_level(matrix)
    claims = _mapping_list(matrix.get("claims"), "claims")
    claim_ids = [str(claim.get("claim_id", "")) for claim in claims]
    if any(not claim_id for claim_id in claim_ids) or len(claim_ids) != len(
        set(claim_ids)
    ):
        raise ValueError("claim IDs must be non-empty and unique")
    by_id = {str(claim["claim_id"]): claim for claim in claims}
    for claim in claims:
        _validate_claim(claim, root)

    current_claim_id = str(matrix.get("current_public_claim_id", ""))
    if current_claim_id not in by_id:
        raise ValueError("current_public_claim_id does not reference a claim")
    current = by_id[current_claim_id]
    if current.get("status") != "implemented" or not current.get(
        "authorized_public_assertion"
    ):
        raise ValueError("current public positioning must be implemented and authorized")

    promotion = _mapping(matrix.get("promotion_gate"), "promotion gate")
    _reject_unknown(
        promotion,
        {"claim_id", "milestone_wording", "required_claim_ids"},
        "promotion gate",
    )
    promotion_claim_id = str(promotion.get("claim_id", ""))
    if promotion_claim_id not in by_id:
        raise ValueError("promotion gate does not reference a claim")
    milestone = by_id[promotion_claim_id]
    if milestone.get("authorized_public_assertion") is not False:
        raise ValueError("promotion wording must remain unauthorized before the gate passes")
    if promotion.get("milestone_wording") != milestone.get("wording"):
        raise ValueError("promotion milestone wording does not match its claim")
    required_ids = _string_list(
        promotion.get("required_claim_ids"), "promotion required claim IDs"
    )
    unknown_required = sorted(set(required_ids) - set(by_id))
    if unknown_required:
        raise ValueError(
            "promotion gate references unknown claims: " + ", ".join(unknown_required)
        )
    if promotion_claim_id in required_ids or len(required_ids) != len(set(required_ids)):
        raise ValueError("promotion gate requirements must be unique and non-recursive")

    _validate_public_surfaces(matrix, root, by_id)
    missing_claims = [
        claim_id
        for claim_id in required_ids
        if not _claim_satisfies_promotion(by_id[claim_id])
    ]
    headline_results = _validate_headline_results(matrix, root)
    return {
        "format": CLAIM_EVIDENCE_REPORT_FORMAT,
        "current_public_claim_id": current_claim_id,
        "current_wording": current["wording"],
        "promotion_claim_id": promotion_claim_id,
        "promotion_ready": not missing_claims,
        "missing_promotion_claims": missing_claims,
        "claim_status_counts": _counts(
            str(claim["status"]) for claim in claims
        ),
        "headline_result_count": len(headline_results),
        "validated_headline_result_count": sum(
            result.get("validation_status") == "validated"
            for result in headline_results
        ),
    }


def _validate_top_level(matrix: Mapping[str, Any]) -> None:
    allowed = {
        "format",
        "schema_version",
        "current_public_claim_id",
        "public_surfaces",
        "required_assertions",
        "forbidden_assertions",
        "headline_result_packs",
        "claims",
        "promotion_gate",
    }
    _reject_unknown(matrix, allowed, "claim evidence matrix")
    if matrix.get("format") != CLAIM_EVIDENCE_FORMAT:
        raise ValueError(f"unsupported claim evidence format: {matrix.get('format')}")
    if matrix.get("schema_version") != 1:
        raise ValueError("unsupported claim evidence schema version")


def _validate_claim(claim: Mapping[str, Any], root: Path) -> None:
    allowed = {
        "claim_id",
        "status",
        "evidence_tier",
        "wording",
        "authorized_public_assertion",
        "issue_ids",
        "source_paths",
        "test_paths",
        "artifact_requirements",
        "promotion_requirements",
        "limitations",
    }
    _reject_unknown(claim, allowed, f"claim {claim.get('claim_id')}")
    status = claim.get("status")
    evidence_tier = claim.get("evidence_tier")
    if status not in CAPABILITY_STATUSES:
        raise ValueError(f"unsupported claim status: {status}")
    if evidence_tier not in EVIDENCE_TIERS:
        raise ValueError(f"unsupported claim evidence tier: {evidence_tier}")
    compatible_tiers = {
        "implemented": {
            "source_verified",
            "result_validated",
            "predictive_validated",
        },
        "integration_only": {"integration_only"},
        "experimental": {"experimental"},
        "planned": {"planned"},
    }
    if evidence_tier not in compatible_tiers[str(status)]:
        raise ValueError(
            f"claim status {status} is incompatible with evidence tier {evidence_tier}"
        )
    wording = claim.get("wording")
    if not isinstance(wording, str) or not wording.strip():
        raise ValueError("claim wording must be non-empty")
    if not isinstance(claim.get("authorized_public_assertion"), bool):
        raise ValueError("authorized_public_assertion must be a boolean")
    issue_ids = claim.get("issue_ids")
    if not isinstance(issue_ids, list) or not all(
        isinstance(value, int) and not isinstance(value, bool) and value > 0
        for value in issue_ids
    ):
        raise ValueError("claim issue_ids must contain positive integers")
    if len(issue_ids) != len(set(issue_ids)):
        raise ValueError("claim issue_ids must be unique")
    source_paths = _string_list(claim.get("source_paths"), "claim source paths")
    test_paths = _string_list(claim.get("test_paths"), "claim test paths")
    _string_list(
        claim.get("artifact_requirements"), "claim artifact requirements"
    )
    promotion = _string_list(
        claim.get("promotion_requirements"), "claim promotion requirements"
    )
    _string_list(claim.get("limitations"), "claim limitations")
    for path in (*source_paths, *test_paths):
        candidate = _repo_path(root, path)
        if not candidate.exists():
            raise ValueError(
                f"claim '{claim.get('claim_id')}' references missing path: {path}"
            )
    if status == "implemented" and (not source_paths or not test_paths):
        raise ValueError("implemented claims require source and test evidence")
    if status != "implemented" and not promotion:
        raise ValueError("non-implemented claims require promotion requirements")
    if status != "implemented" and claim.get("authorized_public_assertion"):
        raise ValueError("non-implemented claims cannot authorize public assertions")


def _validate_public_surfaces(
    matrix: Mapping[str, Any], root: Path, claims: Mapping[str, Mapping[str, Any]]
) -> None:
    surfaces = _string_list(matrix.get("public_surfaces"), "public surfaces")
    required = _mapping_list(
        matrix.get("required_assertions"), "required assertions"
    )
    forbidden = _string_list(
        matrix.get("forbidden_assertions"), "forbidden assertions"
    )
    files = []
    for surface in surfaces:
        path = _repo_path(root, surface)
        if not path.is_file():
            raise ValueError(f"public claim surface is not a file: {surface}")
        files.append(path)
    for item in required:
        _reject_unknown(item, {"path", "claim_id", "text"}, "required assertion")
        claim_id = str(item.get("claim_id", ""))
        if claim_id not in claims or not claims[claim_id].get(
            "authorized_public_assertion"
        ):
            raise ValueError("required assertion references an unauthorized claim")
        path = _repo_path(root, str(item.get("path", "")))
        text = item.get("text")
        if not isinstance(text, str) or text not in path.read_text(encoding="utf-8"):
            raise ValueError(f"required public assertion is missing from {path}")
        if text != claims[claim_id].get("wording"):
            raise ValueError("required public assertion differs from claim wording")
    for phrase in forbidden:
        locations = [
            path.relative_to(root).as_posix()
            for path in files
            if phrase.lower() in path.read_text(encoding="utf-8").lower()
        ]
        if locations:
            raise ValueError(
                f"unsupported strong claim appears in: {', '.join(locations)}"
            )


def _validate_headline_results(
    matrix: Mapping[str, Any], root: Path
) -> list[Mapping[str, Any]]:
    results = _mapping_list(
        matrix.get("headline_result_packs"), "headline result packs"
    )
    allowed = {
        "result_id",
        "path",
        "validation_status",
        "run_validity_artifact",
        "benchmark_validity_artifact",
    }
    known_paths = set()
    for result in results:
        _reject_unknown(result, allowed, "headline result pack")
        if result.get("validation_status") != "validated":
            raise ValueError("headline result packs must be validated")
        resolved = {}
        for key in ("path", "run_validity_artifact", "benchmark_validity_artifact"):
            value = result.get(key)
            if not isinstance(value, str) or not _repo_path(root, value).exists():
                raise ValueError(f"headline result pack has missing {key}")
            resolved[key] = _repo_path(root, value)
        if not resolved["path"].is_dir():
            raise ValueError("headline result pack path must be a directory")
        run_payload = _load_json_mapping(
            resolved["run_validity_artifact"], "headline run validity"
        )
        run_validation = run_payload.get("public_claim_validation", run_payload)
        if not isinstance(run_validation, Mapping) or not bool(
            run_validation.get("status") == "validated"
            and run_validation.get("public_claim") is True
            and run_validation.get("failures") == []
        ):
            raise ValueError("headline result pack did not pass RunValidity")
        benchmark_payload = _load_json_mapping(
            resolved["benchmark_validity_artifact"],
            "headline benchmark validity",
        )
        try:
            benchmark = BenchmarkValidityReport.from_dict(benchmark_payload)
        except ValueError as exc:
            raise ValueError(
                "headline result pack has invalid BenchmarkValidity evidence"
            ) from exc
        if not benchmark.claim_ready:
            raise ValueError("headline result pack did not pass BenchmarkValidity")
        known_paths.add(str(result["path"]))
    markdown_link = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    readme = (root / "README.md").read_text(encoding="utf-8")
    linked_results = {
        target
        for target in markdown_link.findall(readme)
        if "benchmark_results/" in target or target.endswith("RESULTS.md")
    }
    undeclared = sorted(linked_results - known_paths)
    if undeclared:
        raise ValueError(
            "README links undeclared headline result packs: " + ", ".join(undeclared)
        )
    return results


def _claim_satisfies_promotion(claim: Mapping[str, Any]) -> bool:
    return bool(
        claim.get("status") == "implemented"
        and claim.get("evidence_tier")
        in {"result_validated", "predictive_validated"}
    )


def _repo_path(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"claim evidence path escapes the repository: {value}") from exc
    return path


def _load_json_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label} artifact: {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} artifact must contain a mapping")
    return dict(value)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _mapping_list(value: Any, label: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or not all(
        isinstance(item, Mapping) for item in value
    ):
        raise ValueError(f"{label} must be a list of mappings")
    return value


def _string_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(f"{label} must be a list of non-empty strings")
    return list(value)


def _reject_unknown(data: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"unknown {label} fields: {', '.join(unknown)}")


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))
