from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from nyssa_bench.cli import main
from nyssa_bench.core.episode import EpisodeResult, StepRecord
from nyssa_bench.failures import FailureEventDraft, FailureEventLedger
from nyssa_bench.failures.detectors import FailureDetectorContract
from nyssa_bench.metrics.success import aggregate_episodes
from nyssa_bench.metrics.vector import build_metric_vector
from nyssa_bench.regression import (
    ConfirmedBoundaryReference,
    PolicyCheckpointIdentity,
    RegressionCellSpec,
    RegressionEpisodeKey,
    RegressionEvidenceRequirements,
    RegressionRule,
    RegressionStudyEvaluator,
    RegressionStudySpec,
    RunArtifactReference,
    file_sha256,
    load_regression_report,
    write_regression_report,
)
from nyssa_bench.reports.comparison import (
    comparison_contract_hash,
    load_comparison_contract,
)
from nyssa_bench.stress_search import (
    SearchVariable,
    StressObservation,
    StressSearchSpace,
    StressSearchStudy,
    StressSearchStudySpec,
    write_stress_search_study,
)
from nyssa_bench.validity import AuditResult, BenchmarkValidityReport


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _policy(name: str, checkpoint: str) -> PolicyCheckpointIdentity:
    return PolicyCheckpointIdentity(
        policy_name=name,
        checkpoint_id=checkpoint,
        checkpoint_sha256=_digest(checkpoint),
        preprocessing_sha256=_digest(f"{checkpoint}-preprocessing"),
    )


def _validity() -> dict[str, Any]:
    audit = AuditResult(
        audit_id="unit_validity",
        category="unit",
        status="passed",
        severity="info",
        inputs={},
        evidence={"fixture": True},
        remediation="No remediation required.",
        claim_impact="none",
        summary="Synthetic validity evidence passed.",
    )
    return BenchmarkValidityReport(
        benchmark_id="regression_fixture",
        benchmark_version="1.0.0",
        claim_tier="unit",
        spec_sha256="a" * 64,
        audits=(audit,),
        metadata={"required_audits": [audit.audit_id]},
    ).to_dict()


def _episode(
    index: int,
    *,
    success: bool,
    safety: float,
    detector_evidence: bool,
    condition_id: str,
    stressor_spec: dict[str, Any] | None,
) -> EpisodeResult:
    ledger = FailureEventLedger(
        task_id="task",
        episode_index=index,
        episode_seed=index,
        engine_name="unit",
    )
    if not success:
        ledger.emitter(
            "task_logic", "unit_task", annotation_source="unit_test"
        ).emit(
            FailureEventDraft(
                role="symptom",
                category="task",
                subtype="missed_target",
                onset_step=0,
                end_step=0,
                summary_label="missed_target",
            )
        )
    detector_context = {
        "format": "nyssa-failure-detector-manifest-v1",
        "engine_name": "unit",
        "task_id": "task",
        "detectors": [
            {
                "contract": FailureDetectorContract(
                    detector_id="unit_detector",
                    detector_version="1.0.0",
                ).to_dict(),
                "support": {"status": "supported"},
                "emitted_event_count": int(not success),
            }
        ]
        if detector_evidence
        else [],
    }
    applications = []
    composition_order = []
    if stressor_spec is not None:
        composition_order = [stressor_spec["stressor_id"]]
        applications = [
            {
                "stressor_id": stressor_spec["stressor_id"],
                "category": "sensor",
                "composition_index": 0,
                "application_points": ["transform_observation"],
                "requested": stressor_spec,
                "seed": index,
                "status": "applied",
                "applied_parameters": stressor_spec.get("parameters", {}),
                "reason": None,
                "backend_evidence": {},
            }
        ]
    return EpisodeResult(
        task_id="task",
        episode_index=index,
        seed=index,
        success=success,
        failure_label=None if success else "missed_target",
        failure_label_source=None if success else "env",
        metrics={
            "safety_violation_rate": safety,
            "damage_event_count": safety,
            "expert_intervention_rate": 0.0,
        },
        steps=[
            StepRecord(
                observation={"raw": [float(index)]},
                action=[0.0],
                reward=float(success),
                terminated=True,
                truncated=False,
                info={"success": success},
            )
        ],
        stressor_context={
            "format": "nyssa-stressor-context-v1",
            "condition_id": condition_id,
            "composition_order": composition_order,
            "episode_seed": index,
            "applications": applications,
        },
        failure_detector_context=detector_context,
        failure_ledger=ledger.snapshot(),
    )


def _make_run(
    root: Path,
    *,
    policy: PolicyCheckpointIdentity,
    successes: list[bool],
    safety: list[float] | None = None,
    engine: str = "unit",
    detector_evidence: bool = True,
    condition_id: str = "clean",
    stressor_spec: dict[str, Any] | None = None,
) -> RunArtifactReference:
    safety = safety or [0.0] * len(successes)
    episodes = [
        _episode(
            index,
            success=success,
            safety=safety[index],
            detector_evidence=detector_evidence,
            condition_id=condition_id,
            stressor_spec=stressor_spec,
        )
        for index, success in enumerate(successes)
    ]
    run_id = f"{policy.checkpoint_id}-{condition_id}"
    seed_protocol = {
        "format": "nyssa-episode-seed-v2",
        "run_seed": 0,
        "episode_seed_stride": 1_000_000,
        "formula": "run_seed * episode_seed_stride + episode_index",
        "shared_across_tasks": True,
    }
    stressor_config = (
        {
            "format": "nyssa-stressor-config-v1",
            "condition_id": condition_id,
            "unsupported_policy": "error",
            "stressors": [stressor_spec],
        }
        if stressor_spec is not None
        else None
    )
    metadata = {
        "run_id": run_id,
        "suite_id": "regression_suite",
        "task_ids": ["task"],
        "policy_name": policy.policy_name,
        "policy_metadata": {
            "checkpoint_id": policy.checkpoint_id,
            "checkpoint_sha256": policy.checkpoint_sha256,
            "preprocessing_sha256": policy.preprocessing_sha256,
        },
        "engine_name": engine,
        "episodes_per_task": len(episodes),
        "seed": 0,
        "seed_protocol": seed_protocol,
        "stressor_config": stressor_config,
        "started_at": "2026-09-06T00:00:00Z",
    }
    config = {
        "suite": {"suite_id": "regression_suite", "tasks": ["task"]},
        "engine": engine,
        "episodes_per_task": len(episodes),
        "seed_protocol": seed_protocol,
        "stressor_config": stressor_config,
    }
    manifest = {
        "format": "nyssa-dataset-manifest-v1",
        "run": metadata,
        "suite": config["suite"],
        "tasks": [
            {
                "task_id": "task",
                "success": {"success_info_keys": ["success"]},
                "randomization": {"seed": True},
                "ood_splits": {"pose": "held_out"},
            }
        ],
    }
    summary = aggregate_episodes(episodes)
    summary["format"] = "nyssa-run-metrics-v2"
    summary["compute"] = {"wall_time_seconds": 1.0}
    summary["metric_vector"] = build_metric_vector(summary, episodes)
    summary["public_claim_validation"] = {
        "status": "validated",
        "public_claim": True,
        "checks": {"unit_fixture": True},
        "failures": [],
    }
    summary["benchmark_validity"] = _validity()
    root.mkdir(parents=True)
    (root / "run.yaml").write_text(
        yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8"
    )
    (root / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (root / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (root / "metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (root / "episodes.json").write_text(
        json.dumps([episode.to_dict() for episode in episodes], indent=2),
        encoding="utf-8",
    )
    artifacts = {
        name: file_sha256(root / name)
        for name in (
            "run.yaml",
            "dataset_manifest.json",
            "metrics.json",
            "episodes.json",
        )
    }
    return RunArtifactReference(
        run_dir=root.as_posix(),
        run_id=run_id,
        artifact_binding="pinned",
        artifacts_sha256=artifacts,
    )


def _study(
    tmp_path: Path,
    *,
    baseline_success: list[bool],
    candidate_success: list[bool],
    baseline_safety: list[float] | None = None,
    candidate_safety: list[float] | None = None,
    candidate_engine: str = "unit",
    detector_evidence: bool = True,
    rule: RegressionRule | None = None,
    condition_kind: str = "clean",
    condition_id: str = "clean",
    stressor_spec: dict[str, Any] | None = None,
    boundary_references: tuple[ConfirmedBoundaryReference, ...] = (),
) -> RegressionStudySpec:
    baseline_policy = _policy("baseline", "baseline-v1")
    candidate_policy = _policy("candidate", "candidate-v2")
    baseline = _make_run(
        tmp_path / "baseline",
        policy=baseline_policy,
        successes=baseline_success,
        safety=baseline_safety,
        detector_evidence=detector_evidence,
        condition_id=condition_id,
        stressor_spec=stressor_spec,
    )
    baseline = replace(baseline, run_dir="baseline")
    candidate = _make_run(
        tmp_path / "candidate",
        policy=candidate_policy,
        successes=candidate_success,
        safety=candidate_safety,
        engine=candidate_engine,
        detector_evidence=detector_evidence,
        condition_id=condition_id,
        stressor_spec=stressor_spec,
    )
    candidate = replace(
        candidate,
        run_dir="candidate",
        artifact_binding="observe_and_record",
        artifacts_sha256={},
    )
    contract_hash = comparison_contract_hash(
        load_comparison_contract(tmp_path / "baseline")
    )
    severities = (
        {stressor_spec["stressor_id"]: float(stressor_spec["severity"])}
        if stressor_spec is not None
        else {}
    )
    cell = RegressionCellSpec(
        cell_id="primary",
        condition_kind=condition_kind,  # type: ignore[arg-type]
        condition_id=condition_id,
        severity_levels=severities,
        comparison_contract_sha256=contract_hash,
        baseline_run=baseline,
        candidate_run=candidate,
        episode_keys=tuple(
            RegressionEpisodeKey("task", index, index)
            for index in range(len(baseline_success))
        ),
        boundary_references=boundary_references,
    )
    rule = rule or RegressionRule(
        rule_id="success_non_inferiority",
        source="paired_success",
        metric_id="success",
        cell_ids=("primary",),
        kind="non_inferiority",
        direction="higher",
        non_inferiority_margin=0.05,
        minimum_pairs=2,
    )
    return RegressionStudySpec(
        study_id="candidate_release_gate",
        study_version="1.0.0",
        baseline_policy=baseline_policy,
        candidate_policy=candidate_policy,
        cells=(cell,),
        rules=(rule,),
        evidence_requirements=RegressionEvidenceRequirements(
            minimum_pair_coverage=1.0,
            require_failure_ledger=True,
            require_detector_evidence=True,
            require_replays=False,
            require_run_validity=True,
            require_benchmark_validity=True,
            required_metric_vector=(
                "clean_success_rate"
                if condition_kind == "clean"
                else "shifted_success_rate",
            ),
        ),
        prespecified_at="2026-09-05T00:00:00Z",
        metadata={"purpose": "unit regression fixture"},
    )


def test_regression_study_contract_round_trip_and_hash(tmp_path: Path) -> None:
    spec = _study(
        tmp_path,
        baseline_success=[True] * 4,
        candidate_success=[True] * 4,
    )

    loaded = RegressionStudySpec.from_dict(spec.to_dict())

    assert loaded == spec
    assert len(spec.sha256) == 64
    with pytest.raises(ValueError, match="unknown cells"):
        RegressionStudySpec(
            **{
                **spec.__dict__,
                "rules": (
                    RegressionRule(
                        rule_id="bad",
                        source="paired_success",
                        metric_id="success",
                        cell_ids=("missing",),
                        kind="non_inferiority",
                        direction="higher",
                    ),
                ),
            }
        )
    with pytest.raises(ValueError, match="must stay within the pack"):
        RunArtifactReference(
            run_dir="run",
            run_id="run",
            artifact_binding="pinned",
            artifacts_sha256={
                **spec.cells[0].baseline_run.artifacts_sha256,
                "../outside.mp4": "b" * 64,
            },
        )


def test_clean_non_inferiority_pass_and_artifact_round_trip(tmp_path: Path) -> None:
    spec = _study(
        tmp_path,
        baseline_success=[True] * 4,
        candidate_success=[True] * 4,
    )

    report = RegressionStudyEvaluator(spec, spec_root=tmp_path).evaluate()
    paths = write_regression_report(report, tmp_path / "report")

    assert report["decision"] == "pass", json.dumps(report, indent=2)
    assert report["exit_code"] == 0
    assert report["rules"][0]["status"] == "passed"
    assert report["rules"][0]["measurement"]["paired_difference"]["ci95"] == [
        0.0,
        0.0,
    ]
    assert (
        report["cells"][0]["candidate_run_reference"]["artifact_binding"]
        == "observe_and_record"
    )
    evaluated = report["cells"][0]["candidate_evaluated_reference"]
    assert evaluated["artifact_binding"] == "pinned"
    assert set(evaluated["artifacts_sha256"]) >= {
        "run.yaml",
        "dataset_manifest.json",
        "metrics.json",
        "episodes.json",
    }
    assert load_regression_report(paths["json"]) == report
    assert "Policy checkpoint regression study" in paths["html"].read_text(
        encoding="utf-8"
    )
    tampered = json.loads(paths["json"].read_text(encoding="utf-8"))
    tampered["decision"] = "fail"
    tampered["exit_code"] = 1
    unhashed = {
        key: value for key, value in tampered.items() if key != "report_sha256"
    }
    tampered["report_sha256"] = hashlib.sha256(
        json.dumps(
            unhashed,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    paths["json"].write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match cell and rule states"):
        load_regression_report(paths["json"])


def test_clear_performance_regression_fails(tmp_path: Path) -> None:
    spec = _study(
        tmp_path,
        baseline_success=[True] * 4,
        candidate_success=[False] * 4,
    )

    report = RegressionStudyEvaluator(spec, spec_root=tmp_path).evaluate()

    assert report["decision"] == "fail"
    assert report["rules"][0]["status"] == "failed"
    assert report["rules"][0]["measurement"]["paired_difference"]["value"] == -1.0


def test_safety_blocking_regression_fails(tmp_path: Path) -> None:
    rule = RegressionRule(
        rule_id="safety_block",
        source="episode_metric",
        metric_id="safety_violation_rate",
        cell_ids=("primary",),
        kind="safety_block",
        direction="lower",
        minimum_pairs=2,
        candidate_limit=0.1,
    )
    spec = _study(
        tmp_path,
        baseline_success=[True] * 4,
        candidate_success=[True] * 4,
        baseline_safety=[0.0] * 4,
        candidate_safety=[1.0] * 4,
        rule=rule,
    )

    report = RegressionStudyEvaluator(spec, spec_root=tmp_path).evaluate()

    assert report["decision"] == "fail"
    assert report["rules"][0]["status"] == "failed"
    assert "blocking safety limit" in report["rules"][0]["reason"]


def test_missing_required_evidence_is_inconclusive(tmp_path: Path) -> None:
    spec = _study(
        tmp_path,
        baseline_success=[True] * 4,
        candidate_success=[True] * 4,
        detector_evidence=False,
    )

    report = RegressionStudyEvaluator(spec, spec_root=tmp_path).evaluate()

    assert report["decision"] == "inconclusive"
    assert report["cells"][0]["status"] == "inconclusive"
    assert report["rules"][0]["status"] == "inconclusive"


def test_incompatible_result_packs_are_invalid(tmp_path: Path) -> None:
    spec = _study(
        tmp_path,
        baseline_success=[True] * 4,
        candidate_success=[True] * 4,
        candidate_engine="other",
    )

    report = RegressionStudyEvaluator(spec, spec_root=tmp_path).evaluate()

    assert report["decision"] == "invalid"
    assert report["cells"][0]["status"] == "invalid"
    assert "comparison-incompatible" in report["cells"][0]["reason"]


def test_tampered_run_artifact_is_invalid(tmp_path: Path) -> None:
    spec = _study(
        tmp_path,
        baseline_success=[True] * 4,
        candidate_success=[True] * 4,
    )
    baseline_metrics = tmp_path / "baseline" / "metrics.json"
    baseline_metrics.write_text("{}\n", encoding="utf-8")

    report = RegressionStudyEvaluator(spec, spec_root=tmp_path).evaluate()

    assert report["decision"] == "invalid"
    assert "artifact hash mismatch" in report["cells"][0]["reason"]


def test_mismatched_checkpoint_identity_is_invalid(tmp_path: Path) -> None:
    spec = _study(
        tmp_path,
        baseline_success=[True] * 4,
        candidate_success=[True] * 4,
    )
    spec = replace(
        spec,
        candidate_policy=replace(
            spec.candidate_policy, checkpoint_id="different-candidate"
        ),
    )

    report = RegressionStudyEvaluator(spec, spec_root=tmp_path).evaluate()

    assert report["decision"] == "invalid"
    assert "policy identity mismatch" in report["cells"][0]["reason"]


def test_posthoc_study_timestamp_is_invalid(tmp_path: Path) -> None:
    spec = _study(
        tmp_path,
        baseline_success=[True] * 4,
        candidate_success=[True] * 4,
    )
    spec = replace(spec, prespecified_at="2026-09-07T00:00:00Z")

    report = RegressionStudyEvaluator(spec, spec_root=tmp_path).evaluate()

    assert report["decision"] == "invalid"
    assert "after the candidate run started" in report["cells"][0]["reason"]


@pytest.mark.parametrize(
    ("source", "metric_id", "direction"),
    [
        ("failure_category_rate", "task", "lower"),
        ("failure_onset_steps", "failure_onset_steps", "higher"),
        ("failure_duration_steps", "failure_duration_steps", "higher"),
    ],
)
def test_temporal_failure_rules_use_paired_evidence(
    tmp_path: Path, source: str, metric_id: str, direction: str
) -> None:
    rule = RegressionRule(
        rule_id=f"{source}_rule",
        source=source,  # type: ignore[arg-type]
        metric_id=metric_id,
        cell_ids=("primary",),
        kind="non_inferiority",
        direction=direction,  # type: ignore[arg-type]
        non_inferiority_margin=0.0,
        minimum_pairs=2,
    )
    spec = _study(
        tmp_path,
        baseline_success=[False] * 4,
        candidate_success=[False] * 4,
        rule=rule,
    )

    report = RegressionStudyEvaluator(spec, spec_root=tmp_path).evaluate()

    assert report["decision"] == "pass", json.dumps(report, indent=2)
    measurement = report["rules"][0]["measurement"]
    assert measurement["paired_difference"]["sample_size"] == 4
    assert measurement["paired_difference"]["value"] == 0.0


def test_metric_vector_rule_uses_prespecified_run_intervals(tmp_path: Path) -> None:
    rule = RegressionRule(
        rule_id="shifted_success_vector",
        source="metric_vector",
        metric_id="shifted_success_rate",
        cell_ids=("primary",),
        kind="non_inferiority",
        direction="higher",
        non_inferiority_margin=1.0,
        minimum_pairs=2,
    )
    stressor = {
        "format": "nyssa-stressor-spec-v1",
        "schema_version": 1,
        "stressor_id": "observation_gaussian_noise",
        "severity": 0.5,
        "parameters": {"max_std": 0.2},
        "seed": None,
    }
    spec = _study(
        tmp_path,
        baseline_success=[True] * 4,
        candidate_success=[True] * 4,
        rule=rule,
        condition_kind="shifted",
        condition_id="noise-s05",
        stressor_spec=stressor,
    )

    report = RegressionStudyEvaluator(spec, spec_root=tmp_path).evaluate()

    assert report["decision"] == "pass", json.dumps(report, indent=2)
    difference = report["rules"][0]["measurement"]["paired_difference"]
    assert difference["interval_method"] == "conservative_difference_of_run_intervals"


def test_cli_returns_ci_friendly_decision_code(tmp_path: Path) -> None:
    spec = _study(
        tmp_path,
        baseline_success=[True] * 4,
        candidate_success=[True] * 4,
    )
    spec_path = tmp_path / "regression.json"
    spec_path.write_text(json.dumps(spec.to_dict(), indent=2), encoding="utf-8")

    exit_code = main(
        ["regression-gate", str(spec_path), "--out", str(tmp_path / "gate")]
    )

    assert exit_code == 0
    assert (tmp_path / "gate" / "regression_report.json").exists()
    fingerprint_path = tmp_path / "baseline_fingerprint.json"
    assert (
        main(
            [
                "regression-fingerprint",
                str(tmp_path / "baseline"),
                "--out",
                str(fingerprint_path),
            ]
        )
        == 0
    )
    fingerprint = json.loads(fingerprint_path.read_text(encoding="utf-8"))
    assert fingerprint["run_reference"]["artifact_binding"] == "pinned"
    assert (
        fingerprint["comparison_contract_sha256"]
        == spec.cells[0].comparison_contract_sha256
    )


def test_confirmed_boundary_case_preserves_provenance(tmp_path: Path) -> None:
    boundary_path, point = _confirmed_boundary(tmp_path)
    stressor_spec = {
        "format": "nyssa-stressor-spec-v1",
        "schema_version": 1,
        "stressor_id": "observation_gaussian_noise",
        "severity": float(point["severity"]),
        "parameters": {"max_std": float(point["max_std"])},
        "seed": None,
    }
    reference = ConfirmedBoundaryReference(
        study_path=boundary_path.as_posix(),
        artifact_sha256=file_sha256(boundary_path),
        point=point,
    )
    spec = _study(
        tmp_path / "runs",
        baseline_success=[True] * 4,
        candidate_success=[True] * 4,
        condition_kind="confirmed_boundary",
        condition_id="confirmed-boundary",
        stressor_spec=stressor_spec,
        boundary_references=(reference,),
    )

    report = RegressionStudyEvaluator(spec, spec_root=tmp_path / "runs").evaluate()

    assert report["decision"] == "pass", json.dumps(report, indent=2)
    check = report["cells"][0]["boundary_references"][0]
    assert check["status"] == "passed"
    assert check["confirmation"]["confirmed_boundary"] is True


def _confirmed_boundary(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    space = StressSearchSpace(
        space_id="regression_boundary",
        engine_name="unit",
        task_id="task",
        variables=(
            SearchVariable(
                "severity",
                "observation_gaussian_noise",
                "severity",
                "continuous",
                lower=0.0,
                upper=1.0,
            ),
            SearchVariable(
                "max_std",
                "observation_gaussian_noise",
                "parameters.max_std",
                "continuous",
                lower=0.1,
                upper=0.5,
            ),
        ),
    )
    study = StressSearchStudy(
        StressSearchStudySpec(
            study_id="regression_boundary",
            search_space=space,
            sampler_id="random",
            study_seed=11,
            discovery_budget=2,
            batch_size=2,
            confirmation_budget=2,
            confirmation_repeats=2,
            provenance={"producer_id": "unit", "study_purpose": "regression"},
        )
    )
    proposals = study.propose(2)
    study.observe(
        (
            _stress_observation(proposals[0].proposal_id, True),
            _stress_observation(proposals[1].proposal_id, False),
        )
    )
    confirmation = study.select_confirmation_conditions()
    study.observe_confirmation(
        tuple(
            _stress_observation(proposal.proposal_id, index == 0)
            for index, proposal in enumerate(confirmation)
        )
    )
    assert study.summary()["confirmation"]["confirmed_boundary_count"] == 1
    path = write_stress_search_study(study, tmp_path / "boundary.json")
    point = dict(study.summary()["confirmation"]["conditions"][0]["point"])
    return path, point


def _stress_observation(proposal_id: str, success: bool) -> StressObservation:
    episode = EpisodeResult(
        task_id="task",
        episode_index=0,
        seed=0,
        success=success,
        failure_label=None if success else "missed_target",
        metrics={},
    )
    summary = aggregate_episodes([episode])
    summary["compute"] = {"wall_time_seconds": 0.01}
    vector = build_metric_vector(summary, [episode])
    events: tuple[dict[str, Any], ...] = ()
    if not success:
        ledger = FailureEventLedger(
            task_id="task",
            episode_index=0,
            episode_seed=0,
            engine_name="unit",
        )
        event = ledger.emitter(
            "task_logic", "unit", annotation_source="unit_test"
        ).emit(
            FailureEventDraft(
                role="symptom",
                category="task",
                subtype="missed_target",
                onset_step=0,
            )
        )
        events = (event.to_dict(),)
    return StressObservation(
        proposal_id=proposal_id,
        status="success" if success else "policy_failure",
        success=success,
        metric_vector=vector,
        failure_events=events,
        provenance={"source": "unit", "source_id": proposal_id},
        application_evidence={"all_requests_resolved": True},
    )
