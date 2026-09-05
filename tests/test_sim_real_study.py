from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from nyssa_bench.cli import main
from nyssa_bench.core.episode import EpisodeResult, StepRecord
from nyssa_bench.failures import FailureEventDraft, FailureEventLedger
from nyssa_bench.metrics.success import aggregate_episodes
from nyssa_bench.metrics.vector import build_metric_vector
from nyssa_bench.real_evidence import (
    RealEvidencePackage,
    real_evidence_conformance_fixture_path,
)
from nyssa_bench.regression import file_sha256
from nyssa_bench.simreal import (
    RealReference,
    SimRealPair,
    SimRealStudySpec,
    SimulationReference,
    evaluate_sim_real_study,
    failure_distribution_similarity,
    kendall_tau_b,
    load_sim_real_report,
    mean_maximum_rank_violation,
    pearson_correlation,
    spearman_correlation,
)
from nyssa_bench.simreal.study import _predictive_scores


def test_rank_metrics_cover_perfect_inverse_and_ties() -> None:
    ascending = [0.0, 1.0, 2.0]
    descending = [2.0, 1.0, 0.0]

    assert pearson_correlation(ascending, ascending) == pytest.approx(1.0)
    assert spearman_correlation(ascending, ascending) == pytest.approx(1.0)
    assert kendall_tau_b(ascending, ascending) == pytest.approx(1.0)
    assert pearson_correlation(ascending, descending) == pytest.approx(-1.0)
    assert spearman_correlation(ascending, descending) == pytest.approx(-1.0)
    assert kendall_tau_b(ascending, descending) == pytest.approx(-1.0)
    assert mean_maximum_rank_violation(ascending, ascending) == 0.0
    assert mean_maximum_rank_violation(ascending, descending) > 0.0
    assert kendall_tau_b([1.0, 1.0, 2.0], [1.0, 2.0, 2.0]) == pytest.approx(0.5)


def test_failure_distribution_similarity_has_reference_extremes() -> None:
    assert failure_distribution_similarity({"slip": 4}, {"slip": 2}) == 1.0
    assert failure_distribution_similarity({"slip": 4}, {"collision": 2}) == 0.0
    assert failure_distribution_similarity({}, {"collision": 2}) is None


def test_incremental_predictive_analysis_can_report_negative_result() -> None:
    train = [
        _predictive_row(False, False, severity=0.0, failure=False),
        _predictive_row(False, False, severity=0.0, failure=False),
        _predictive_row(True, True, severity=1.0, failure=True),
        _predictive_row(True, True, severity=1.0, failure=True),
    ]
    test = [
        _predictive_row(False, False, severity=1.0, failure=True),
        _predictive_row(True, True, severity=0.0, failure=False),
    ]

    scores = _predictive_scores(train, test)

    assert scores is not None
    baseline, enhanced = scores
    assert baseline < enhanced


def test_protocol_rejects_many_to_one_pairing(tmp_path: Path) -> None:
    package = RealEvidencePackage.load(real_evidence_conformance_fixture_path())
    simulation = _make_simulation_reference(tmp_path / "sim", package)
    pair = _pair("pair-1", simulation, package)

    with pytest.raises(ValueError, match="many-to-one simulation"):
        SimRealStudySpec(
            study_id="duplicate",
            study_version="1.0.0",
            prespecified_at="2026-09-05T00:00:00Z",
            pairs=(pair, pair.model_copy(update={"pair_id": "pair-2"})),
            primary_metrics=("failure_distribution",),
            bootstrap_samples=200,
            bootstrap_seed=0,
            cluster_fields=("trial_id",),
            holdout_shift_ids=(),
        )

    with pytest.raises(ValueError, match="recovery metric requires"):
        SimRealStudySpec(
            study_id="unsupported-recovery",
            study_version="1.0.0",
            prespecified_at="2026-09-05T00:00:00Z",
            pairs=(pair,),
            primary_metrics=("recovery_effect",),
            bootstrap_samples=200,
            bootstrap_seed=0,
            cluster_fields=("trial_id",),
            holdout_shift_ids=(),
            recovery_assumption="disabled",
        )


def test_study_loads_validated_real_and_pinned_simulation_evidence(
    tmp_path: Path,
) -> None:
    package_path = real_evidence_conformance_fixture_path()
    package = RealEvidencePackage.load(package_path)
    simulation = _make_simulation_reference(tmp_path / "sim", package)
    pair = _pair("pair-1", simulation, package)
    spec = SimRealStudySpec(
        study_id="single-pair-conformance",
        study_version="1.0.0",
        prespecified_at="2026-09-05T00:00:00Z",
        pairs=(pair,),
        primary_metrics=("failure_distribution",),
        bootstrap_samples=200,
        bootstrap_seed=3,
        cluster_fields=("trial_id",),
        holdout_shift_ids=(),
    )

    report = evaluate_sim_real_study(spec, spec_root=tmp_path)

    assert report["status"] == "complete"
    assert report["pair_count"] == 1
    assert report["metrics"]["failure_distribution"]["status"] == "available"
    assert report["metrics"]["policy_rank"]["status"] == "unavailable"
    assert report["metrics"]["recovery_effect"]["status"] == "unavailable"
    assert report["pairs"][0]["task_id"] == "mujoco_inverted_pendulum"

    spec_path = tmp_path / "study.json"
    spec_path.write_text(json.dumps(spec.model_dump(mode="json")), encoding="utf-8")
    assert main(["sim-real-study", str(spec_path), "--out", str(tmp_path / "out")]) == 0
    saved = json.loads(
        (tmp_path / "out" / "sim_real_study.json").read_text(encoding="utf-8")
    )
    assert saved["status"] == "complete"
    assert len(saved["report_sha256"]) == 64
    assert load_sim_real_report(tmp_path / "out" / "sim_real_study.json") == saved
    assert (tmp_path / "out" / "sim_real_study.html").is_file()


def _pair(
    pair_id: str,
    simulation: SimulationReference,
    package: RealEvidencePackage,
) -> SimRealPair:
    return SimRealPair(
        pair_id=pair_id,
        policy_id=package.real_episode.identity.policy_id,
        task_id=package.real_episode.outcome.task_id,
        shift_id="nominal",
        severity=0.0,
        simulation=simulation,
        real=RealReference(
            package_path=real_evidence_conformance_fixture_path().as_posix(),
            package_identity=package.identity,
            real_episode_id=package.real_episode.identity.episode_id,
            variant_id="nominal_reconstruction",
            trial_id=package.real_episode.identity.trial_id,
        ),
        sim_step_seconds=0.1,
        real_event_step_seconds=0.1,
    )


def _make_simulation_reference(
    run_dir: Path, package: RealEvidencePackage
) -> SimulationReference:
    identity = package.real_episode.identity
    ledger = FailureEventLedger(
        task_id=package.real_episode.outcome.task_id,
        episode_index=0,
        episode_seed=0,
        engine_name="mujoco",
    )
    ledger.emitter("task_logic", "unit", annotation_source="test").emit(
        FailureEventDraft(
            role="symptom",
            category="unstable_contact",
            subtype="unstable_contact",
            onset_step=2,
            summary_label="unstable_contact",
        )
    )
    episode = EpisodeResult(
        task_id=package.real_episode.outcome.task_id,
        episode_index=0,
        seed=0,
        success=False,
        failure_label="unstable_contact",
        metrics={},
        steps=[
            StepRecord(
                observation={"raw": [0.0]},
                action=[0.0],
                reward=0.0,
                terminated=index == 2,
                truncated=False,
                info={"success": False},
            )
            for index in range(3)
        ],
        failure_ledger=ledger.snapshot(),
    )
    summary = aggregate_episodes([episode])
    summary["compute"] = {"wall_time_seconds": 0.1}
    summary["metric_vector"] = build_metric_vector(summary, [episode])
    metadata = {
        "run_id": "sim-run-1",
        "policy_name": identity.policy_id,
        "policy_metadata": {
            "checkpoint_id": "shared-checkpoint",
            "checkpoint_sha256": identity.checkpoint_sha256,
            "preprocessing_sha256": "b" * 64,
        },
    }
    run_dir.mkdir(parents=True)
    (run_dir / "run.yaml").write_text(
        yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8"
    )
    (run_dir / "dataset_manifest.json").write_text("{}\n", encoding="utf-8")
    (run_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (run_dir / "episodes.json").write_text(
        json.dumps([episode.to_dict()], indent=2), encoding="utf-8"
    )
    hashes = {
        name: file_sha256(run_dir / name)
        for name in ("run.yaml", "dataset_manifest.json", "metrics.json", "episodes.json")
    }
    return SimulationReference(
        run_dir=run_dir.as_posix(),
        run_id="sim-run-1",
        artifacts_sha256=hashes,
        policy_name=identity.policy_id,
        checkpoint_id="shared-checkpoint",
        checkpoint_sha256=identity.checkpoint_sha256,
        preprocessing_sha256="b" * 64,
        task_id=package.real_episode.outcome.task_id,
        episode_seed=0,
        episode_index=0,
    )


def _predictive_row(
    sim_success: bool,
    real_success: bool,
    *,
    severity: float,
    failure: bool,
) -> dict[str, Any]:
    return {
        "sim_success": sim_success,
        "real_success": real_success,
        "severity": severity,
        "sim_failure_category": "failure" if failure else None,
        "sim_failure_time_seconds": 1.0 if failure else None,
        "sim_recovery_gain": None,
    }
