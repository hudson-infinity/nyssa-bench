from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from nyssa_bench.claims import load_claim_evidence, validate_claim_evidence


REPO_ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = REPO_ROOT / "claims" / "claim_evidence.json"


def test_repository_claim_matrix_is_valid_and_promotion_is_blocked() -> None:
    report = validate_claim_evidence(
        load_claim_evidence(MATRIX_PATH), repo_root=REPO_ROOT
    )

    assert report["current_public_claim_id"] == "current_public_positioning"
    assert report["promotion_ready"] is False
    assert "validated_learned_policy_coverage" in report[
        "missing_promotion_claims"
    ]
    assert "sim_real_predictive_validity" in report["missing_promotion_claims"]
    assert report["headline_result_count"] == 0


def test_claim_matrix_rejects_missing_evidence_paths() -> None:
    matrix = copy.deepcopy(load_claim_evidence(MATRIX_PATH))
    matrix["claims"][0]["source_paths"] = ["missing/source.py"]

    with pytest.raises(ValueError, match="references missing path"):
        validate_claim_evidence(matrix, repo_root=REPO_ROOT)


def test_claim_matrix_rejects_unsupported_headline_on_public_surface(
    tmp_path: Path,
) -> None:
    matrix = _minimal_matrix()
    (tmp_path / "README.md").write_text(
        "Current framework wording.\n"
        "NyssaBench is open-source foundational infrastructure\n",
        encoding="utf-8",
    )
    (tmp_path / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "test_source.py").write_text("def test_value(): pass\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported strong claim"):
        validate_claim_evidence(matrix, repo_root=tmp_path)


def test_claim_matrix_rejects_unvalidated_headline_result(tmp_path: Path) -> None:
    matrix = _minimal_matrix()
    matrix["headline_result_packs"] = [
        {
            "result_id": "smoke",
            "path": "results/smoke",
            "validation_status": "prototype",
            "run_validity_artifact": "results/smoke/metrics.json",
            "benchmark_validity_artifact": "results/smoke/benchmark_validity.json",
        }
    ]
    (tmp_path / "README.md").write_text(
        "Current framework wording.\n", encoding="utf-8"
    )
    (tmp_path / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "test_source.py").write_text("def test_value(): pass\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must be validated"):
        validate_claim_evidence(matrix, repo_root=tmp_path)


def test_claim_matrix_parses_headline_run_validity(tmp_path: Path) -> None:
    matrix = _minimal_matrix()
    matrix["headline_result_packs"] = [
        {
            "result_id": "failed-run",
            "path": "results/failed-run",
            "validation_status": "validated",
            "run_validity_artifact": "results/failed-run/metrics.json",
            "benchmark_validity_artifact": "results/failed-run/benchmark_validity.json",
        }
    ]
    (tmp_path / "README.md").write_text(
        "Current framework wording.\n", encoding="utf-8"
    )
    (tmp_path / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "test_source.py").write_text(
        "def test_value(): pass\n", encoding="utf-8"
    )
    result = tmp_path / "results" / "failed-run"
    result.mkdir(parents=True)
    (result / "metrics.json").write_text(
        json.dumps(
            {
                "public_claim_validation": {
                    "status": "not_public",
                    "public_claim": False,
                    "failures": ["missing_replay"],
                }
            }
        ),
        encoding="utf-8",
    )
    (result / "benchmark_validity.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="did not pass RunValidity"):
        validate_claim_evidence(matrix, repo_root=tmp_path)


def test_claim_matrix_loader_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "claims.json"
    path.write_text(json.dumps([]), encoding="utf-8")

    with pytest.raises(ValueError, match="must contain a mapping"):
        load_claim_evidence(path)


def _minimal_matrix() -> dict:
    return {
        "format": "nyssa-claim-evidence-matrix-v1",
        "schema_version": 1,
        "current_public_claim_id": "current",
        "public_surfaces": ["README.md"],
        "required_assertions": [
            {
                "path": "README.md",
                "claim_id": "current",
                "text": "Current framework wording.",
            }
        ],
        "forbidden_assertions": [
            "NyssaBench is open-source foundational infrastructure"
        ],
        "headline_result_packs": [],
        "claims": [
            {
                "claim_id": "current",
                "status": "implemented",
                "evidence_tier": "source_verified",
                "wording": "Current framework wording.",
                "authorized_public_assertion": True,
                "issue_ids": [23],
                "source_paths": ["source.py"],
                "test_paths": ["test_source.py"],
                "artifact_requirements": ["metrics.json"],
                "promotion_requirements": [],
                "limitations": ["Source evidence only."],
            },
            {
                "claim_id": "milestone",
                "status": "planned",
                "evidence_tier": "planned",
                "wording": "Milestone wording.",
                "authorized_public_assertion": False,
                "issue_ids": [30],
                "source_paths": [],
                "test_paths": [],
                "artifact_requirements": ["credibility.json"],
                "promotion_requirements": ["Complete the evidence gate."],
                "limitations": ["Not current."],
            },
        ],
        "promotion_gate": {
            "claim_id": "milestone",
            "milestone_wording": "Milestone wording.",
            "required_claim_ids": ["current"],
        },
    }
