from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from nyssa_bench.cli import main
from nyssa_bench.core.episode import EpisodeResult
from nyssa_bench.core.suite import Suite
from nyssa_bench.failures import FailureEventDraft, FailureEventLedger
from nyssa_bench.metrics.success import aggregate_episodes
from nyssa_bench.metrics.vector import build_metric_vector
from nyssa_bench.stress_search import (
    BoundaryStressSampler,
    LatinHypercubeStressSampler,
    RandomStressSampler,
    SearchConstraint,
    SearchVariable,
    StressObservation,
    StressSearchSpace,
    StressSearchStudy,
    StressSearchStudySpec,
    compare_stress_search_studies,
    load_stress_search_study,
    make_stress_sampler,
    write_stress_search_report,
    write_stress_search_study,
    observation_from_run,
)
from nyssa_bench.stressors import StressorContext, StressorPipeline
from nyssa_bench.validity import AuditResult, BenchmarkValidityReport


def _space(*, stressor_id: str = "observation_gaussian_noise") -> StressSearchSpace:
    variables = [
        SearchVariable(
            "severity",
            stressor_id,
            "severity",
            "continuous",
            lower=0.0,
            upper=1.0,
        )
    ]
    fixed = {}
    if stressor_id == "observation_gaussian_noise":
        variables.append(
            SearchVariable(
                "max_std",
                stressor_id,
                "parameters.max_std",
                "continuous",
                lower=0.1,
                upper=0.5,
            )
        )
    if stressor_id == "friction_scale":
        variables.append(
            SearchVariable(
                "target_scale",
                stressor_id,
                "parameters.target_scale",
                "continuous",
                lower=0.1,
                upper=1.0,
            )
        )
    return StressSearchSpace(
        space_id=f"{stressor_id}_space",
        engine_name="mujoco",
        task_id="unit_task",
        variables=tuple(variables),
        fixed_parameters=fixed,
    )


def _metric_vector(success: bool) -> dict:
    episode = EpisodeResult(
        task_id="unit_task",
        episode_index=0,
        seed=0,
        success=success,
        failure_label=None if success else "missed_target",
        metrics={},
    )
    summary = aggregate_episodes([episode])
    summary["compute"] = {"wall_time_seconds": 0.01}
    return build_metric_vector(summary, [episode])


def _failure_event() -> dict:
    ledger = FailureEventLedger(
        task_id="unit_task",
        episode_index=0,
        episode_seed=0,
        engine_name="mujoco",
    )
    event = ledger.emitter(
        "task_logic", "unit_task", annotation_source="test"
    ).emit(
        FailureEventDraft(
            role="symptom",
            category="task",
            subtype="missed_target",
            onset_step=4,
        )
    )
    return event.to_dict()


def _study_provenance() -> dict[str, str]:
    return {
        "producer_id": "test-suite",
        "study_purpose": "synthetic stress-search validation",
    }


def _observation(proposal_id: str, success: bool) -> StressObservation:
    success = bool(success)
    return StressObservation(
        proposal_id=proposal_id,
        status="success" if success else "policy_failure",
        success=success,
        metric_vector=_metric_vector(success),
        failure_events=() if success else (_failure_event(),),
        provenance={"source": "synthetic_test", "source_id": proposal_id},
        application_evidence={
            "source": "synthetic_test",
            "all_requests_resolved": True,
        },
    )


def test_search_space_uses_registered_executable_stressors_and_constraints() -> None:
    space = StressSearchSpace(
        space_id="constrained",
        engine_name="mujoco",
        task_id="unit_task",
        variables=(
            SearchVariable(
                "severity",
                "action_gaussian_noise",
                "severity",
                "continuous",
                0.0,
                1.0,
            ),
            SearchVariable(
                "max_std",
                "action_gaussian_noise",
                "parameters.max_std",
                "continuous",
                0.1,
                0.5,
            ),
        ),
        constraints=(SearchConstraint("combined", "sum_le", ("severity", "max_std"), 1.2),),
    )

    normalized = space.validate_point({"severity": 0.5, "max_std": 0.3})
    specs = space.stressor_specs(normalized, seed=7)

    assert specs[0].stressor_id == "action_gaussian_noise"
    assert specs[0].severity == 0.5
    assert specs[0].parameters["max_std"] == 0.3
    assert specs[0].seed is not None
    with pytest.raises(ValueError, match="violates constraint"):
        space.validate_point({"severity": 1.0, "max_std": 0.5})
    with pytest.raises(ValueError, match="Unknown stressor"):
        StressSearchSpace(
            "bad",
            "mujoco",
            "unit_task",
            (
                SearchVariable(
                    "severity", "not_registered", "severity", "continuous", 0.0, 1.0
                ),
            ),
        )


def test_random_sampler_is_deterministic_and_resume_preserves_future_proposals() -> None:
    space = _space()
    first = RandomStressSampler(space, study_seed=19, budget=8)
    second = RandomStressSampler(space, study_seed=19, budget=8)

    first_batch = first.propose(3)
    second_batch = second.propose(3)
    assert [item.to_dict() for item in first_batch] == [
        item.to_dict() for item in second_batch
    ]
    observations = tuple(_observation(item.proposal_id, index % 2 == 0) for index, item in enumerate(first_batch))
    first.update(observations)
    second.update(observations)
    state = first.state_dict()
    resumed = make_stress_sampler(
        "random", space, study_seed=19, budget=8, state=state
    )

    expected = first.propose(2)
    actual = resumed.propose(2)

    assert [item.to_dict() for item in actual] == [item.to_dict() for item in expected]
    assert resumed.state_dict() == first.state_dict()


def test_latin_hypercube_uses_each_stratum_once_per_variable() -> None:
    sampler = LatinHypercubeStressSampler(_space(), study_seed=3, budget=8)

    proposals = sampler.propose(8)

    assert len(proposals) == 8
    for variable_id in ("severity", "max_std"):
        assert {
            proposal.acquisition["strata"][variable_id] for proposal in proposals
        } == set(range(8))


def test_adaptive_resume_preserves_boundary_proposal_and_rejects_stale_pending() -> None:
    space = _space()
    sampler = BoundaryStressSampler(
        space, study_seed=23, budget=8, config={"warmup": 4}
    )
    for _ in range(4):
        proposal = sampler.propose(1)[0]
        sampler.update(
            (
                _observation(
                    proposal.proposal_id,
                    float(proposal.point["severity"]) < 0.5,
                ),
            )
        )
    resumed = make_stress_sampler(
        "boundary_adaptive",
        space,
        study_seed=23,
        budget=8,
        config={"warmup": 4},
        state=sampler.state_dict(),
    )

    expected = sampler.propose(1)
    actual = resumed.propose(1)

    assert [item.to_dict() for item in actual] == [item.to_dict() for item in expected]
    with pytest.raises(RuntimeError, match="unobserved proposals"):
        resumed.propose(1)
    with pytest.raises(ValueError, match="unknown configuration"):
        BoundaryStressSampler(
            space, study_seed=0, budget=2, config={"unknown": True}
        )


def test_adaptive_sampler_can_stop_on_declared_boundary_tolerance() -> None:
    space = StressSearchSpace(
        space_id="one_dimensional",
        engine_name="mujoco",
        task_id="unit_task",
        variables=(
            SearchVariable(
                "severity",
                "observation_gaussian_noise",
                "severity",
                "continuous",
                0.0,
                1.0,
            ),
        ),
        fixed_parameters={"observation_gaussian_noise": {"max_std": 0.2}},
    )
    sampler = BoundaryStressSampler(
        space,
        study_seed=9,
        budget=20,
        config={
            "warmup": 4,
            "target_boundary_width": 1.0,
            "min_valid_observations": 4,
        },
    )
    while sampler.stopping_reason is None:
        proposal = sampler.propose(1)[0]
        sampler.update(
            (
                _observation(
                    proposal.proposal_id,
                    float(proposal.point["severity"]) < 0.5,
                ),
            )
        )

    assert sampler.stopping_reason == "boundary_tolerance_reached"
    assert len(sampler.proposals) < sampler.budget


def test_observation_statuses_remain_distinct_from_policy_failures() -> None:
    sampler = RandomStressSampler(_space(), study_seed=2, budget=6)
    proposals = sampler.propose(6)
    statuses = ["unsupported", "censored", "application_error", "invalid"]
    sampler.update(
        tuple(
            StressObservation(
                proposal_id=proposal.proposal_id,
                status=status,  # type: ignore[arg-type]
                success=None,
                metric_vector=None,
                reason=f"unit {status}",
            )
            for proposal, status in zip(proposals[:4], statuses)
        )
    )
    sampler.update(
        (
            _observation(proposals[4].proposal_id, True),
            _observation(proposals[5].proposal_id, False),
        )
    )

    counts = {item.status for item in sampler.observations.values()}
    assert counts == {
        "unsupported",
        "censored",
        "application_error",
        "invalid",
        "success",
        "policy_failure",
    }


def test_policy_failure_requires_metric_vector_and_temporal_event() -> None:
    with pytest.raises(ValueError, match="metric vector"):
        StressObservation("p", "policy_failure", False, None)
    with pytest.raises(ValueError, match="temporal failure-event"):
        StressObservation(
            "p",
            "policy_failure",
            False,
            _metric_vector(False),
            provenance={"source": "test", "source_id": "p"},
            application_evidence={"source": "test"},
        )


def test_benchmark_claim_mode_requires_claim_ready_validity_report() -> None:
    common = {
        "study_id": "claim_study",
        "search_space": _space(),
        "sampler_id": "random",
        "study_seed": 0,
        "discovery_budget": 2,
        "confirmation_budget": 2,
        "confirmation_repeats": 2,
        "provenance": _study_provenance(),
    }
    with pytest.raises(ValueError, match="BenchmarkValidity"):
        StressSearchStudySpec(**common, claim_mode="benchmark_claim")

    audit = AuditResult(
        audit_id="unit_validity",
        category="construct_validity",
        status="passed",
        severity="info",
        inputs={},
        evidence={"fixture": True},
        remediation="No remediation required.",
        claim_impact="none",
        summary="Synthetic validity fixture passed.",
    )
    validity = BenchmarkValidityReport(
        benchmark_id="unit_benchmark",
        benchmark_version="1.0.0",
        claim_tier="stress_search_test",
        spec_sha256="a" * 64,
        audits=(audit,),
        metadata={"required_audits": [audit.audit_id]},
    )
    claim_common = {
        **common,
        "provenance": {
            **common["provenance"],
            "benchmark_id": validity.benchmark_id,
        },
    }
    spec = StressSearchStudySpec(
        **claim_common,
        claim_mode="benchmark_claim",
        benchmark_validity=validity.to_dict(),
    )
    assert StressSearchStudy(spec).summary()["claim_eligible"] is True


def _run_synthetic_study(
    sampler_id: str, *, seed: int, boundary: float = 0.55
) -> StressSearchStudy:
    spec = StressSearchStudySpec(
        study_id=f"{sampler_id}_{seed}",
        search_space=_space(),
        sampler_id=sampler_id,
        study_seed=seed,
        discovery_budget=20,
        batch_size=1 if sampler_id == "boundary_adaptive" else 20,
        confirmation_budget=10,
        confirmation_repeats=10,
        sampler_config={"warmup": 4} if sampler_id == "boundary_adaptive" else {},
        provenance=_study_provenance(),
    )
    study = StressSearchStudy(spec)
    while True:
        proposals = study.propose()
        if not proposals:
            break
        study.observe(
            tuple(
                _observation(
                    proposal.proposal_id,
                    float(proposal.point["severity"]) < boundary,
                )
                for proposal in proposals
            )
        )
    confirmation = study.select_confirmation_conditions()
    study.observe_confirmation(
        tuple(
            _observation(
                proposal.proposal_id,
                np.random.default_rng(proposal.discovery_seed).random()
                < 1.0
                / (
                    1.0
                    + np.exp(
                        (float(proposal.point["severity"]) - boundary) / 0.08
                    )
                ),
            )
            for proposal in confirmation
        )
    )
    return study


def test_adaptive_sampler_recovers_known_boundary_and_confirms_held_out_trials(
    tmp_path: Path,
) -> None:
    study = _run_synthetic_study("boundary_adaptive", seed=11)
    adaptive = study.sampler

    assert isinstance(adaptive, BoundaryStressSampler)
    assert any(
        proposal.acquisition.get("strategy") == "nearest_opposite_midpoint"
        for proposal in adaptive.proposals
    )
    assert study.summary()["candidate_boundary_pairs"] > 0
    assert study.summary()["confirmation"]["coverage"] == 1.0
    confirmation_condition = study.summary()["confirmation"]["conditions"][0]
    assert abs(float(confirmation_condition["point"]["severity"]) - 0.55) < 0.2
    assert confirmation_condition["confirmed_boundary"] is True
    assert not study.pending_confirmation_ids
    discovery_seeds = {proposal.discovery_seed for proposal in adaptive.proposals}
    confirmation_seeds = {
        proposal.discovery_seed for proposal in study.confirmation_proposals
    }
    assert discovery_seeds.isdisjoint(confirmation_seeds)

    path = write_stress_search_study(study, tmp_path / "study.json")
    loaded = load_stress_search_study(path)
    assert loaded.to_dict() == study.to_dict()

    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["summary"]["candidate_boundary_pairs"] = -1
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="summary does not match"):
        load_stress_search_study(path)


def test_efficiency_report_compares_adaptive_with_both_baselines(tmp_path: Path) -> None:
    studies = [
        _run_synthetic_study("random", seed=5),
        _run_synthetic_study("latin_hypercube", seed=5),
        _run_synthetic_study("boundary_adaptive", seed=5),
    ]

    report = compare_stress_search_studies(studies)
    paths = write_stress_search_report(report, tmp_path)

    assert {row["sampler_id"] for row in report["studies"]} == {
        "random",
        "latin_hypercube",
        "boundary_adaptive",
    }
    assert all(path.is_file() for path in paths.values())
    assert "confirmation_intervals" in report["studies"][0]
    assert report["comparison_complete"] is True
    assert report["claim_eligible"] is False
    assert report["matched_study_seeds"] == [5]
    assert "5" in report["baseline_samples_to_boundary_by_seed"]
    assert len(report["report_sha256"]) == 64
    assert "Wilson 95%" in report["uncertainty"]["boundary_confirmation"]

    mismatched = [*studies[:2], _run_synthetic_study("boundary_adaptive", seed=6)]
    with pytest.raises(ValueError, match="matched study-seed"):
        compare_stress_search_studies(mismatched)


def test_physical_friction_proposal_executes_through_stressor_pipeline() -> None:
    class Engine:
        def apply_stressor(self, stressor_id, parameters):
            return {
                "status": "applied",
                "stressor_id": stressor_id,
                "scale": parameters["scale"],
            }

    space = _space(stressor_id="friction_scale")
    proposal = RandomStressSampler(space, study_seed=7, budget=1).propose(1)[0]
    specs = space.stressor_specs(proposal.point, seed=proposal.discovery_seed)
    pipeline = StressorPipeline(
        specs,
        context=StressorContext(engine_name="mujoco", task_id="unit_task"),
        episode_seed=proposal.discovery_seed,
    )

    pipeline.after_reset(Engine(), {})

    assert pipeline.applications[0].status == "applied"
    assert pipeline.applications[0].backend_evidence["stressor_id"] == "friction_scale"


def test_real_mujoco_friction_search_proposal_when_backend_is_installed() -> None:
    pytest.importorskip("gymnasium")
    pytest.importorskip("mujoco")
    from nyssa_bench.engines.mujoco_adapter import MuJoCoEngine

    task = Suite.load("mujoco_control_v0").filter_tasks(["mujoco_reacher"]).tasks[0]
    engine = MuJoCoEngine()
    pipeline = None
    try:
        engine.load_task(task)
        observation, _ = engine.reset(seed=7)
        space = StressSearchSpace(
            space_id="real_mujoco_friction",
            engine_name="mujoco",
            task_id=task.task_id,
            variables=(
                SearchVariable(
                    "severity",
                    "friction_scale",
                    "severity",
                    "continuous",
                    0.0,
                    1.0,
                ),
                SearchVariable(
                    "target_scale",
                    "friction_scale",
                    "parameters.target_scale",
                    "continuous",
                    0.1,
                    1.0,
                ),
            ),
        )
        proposal = RandomStressSampler(space, study_seed=7, budget=1).propose(1)[0]
        pipeline = StressorPipeline(
            space.stressor_specs(proposal.point, seed=proposal.discovery_seed),
            context=StressorContext(engine_name="mujoco", task_id=task.task_id),
            episode_seed=proposal.discovery_seed,
        )
        pipeline.after_reset(engine, observation)
    finally:
        engine.close()

    assert pipeline is not None
    assert pipeline.applications[0].status == "applied"
    assert pipeline.applications[0].backend_evidence["backend"] == "mujoco"


def test_cli_supports_init_propose_observe_confirm_and_validate(
    tmp_path: Path,
) -> None:
    spec = StressSearchStudySpec(
        study_id="cli_study",
        search_space=_space(),
        sampler_id="random",
        study_seed=3,
        discovery_budget=4,
        batch_size=4,
        confirmation_budget=4,
        confirmation_repeats=4,
        provenance=_study_provenance(),
    )
    spec_path = tmp_path / "spec.yaml"
    study_path = tmp_path / "study.json"
    proposals_path = tmp_path / "proposals.json"
    spec_path.write_text(yaml.safe_dump(spec.to_dict()), encoding="utf-8")

    assert main(["stress-search-init", str(spec_path), "--out", str(study_path)]) == 0
    assert (
        main(
            [
                "stress-search-propose",
                str(study_path),
                "--out",
                str(study_path),
                "--proposals-out",
                str(proposals_path),
            ]
        )
        == 0
    )
    proposals = json.loads(proposals_path.read_text(encoding="utf-8"))["proposals"]
    observations_path = tmp_path / "observations.json"
    observations_path.write_text(
        json.dumps(
            [
                _observation(item["proposal_id"], index % 2 == 0).to_dict()
                for index, item in enumerate(proposals)
            ]
        ),
        encoding="utf-8",
    )
    assert (
        main(
            [
                "stress-search-observe",
                str(study_path),
                str(observations_path),
                "--out",
                str(study_path),
            ]
        )
        == 0
    )
    confirmation_path = tmp_path / "confirmation.json"
    assert (
        main(
            [
                "stress-search-confirm",
                str(study_path),
                "--out",
                str(study_path),
                "--proposals-out",
                str(confirmation_path),
            ]
        )
        == 0
    )
    assert main(["validate", str(spec_path)]) == 0
    assert main(["validate", str(study_path)]) == 0
    assert len(
        json.loads(confirmation_path.read_text(encoding="utf-8"))["proposals"]
    ) == 4


def _write_run_artifacts(
    run_dir: Path,
    proposal,
    search_space: StressSearchSpace,
    *,
    success: bool,
    unsupported: bool = False,
) -> None:
    run_dir.mkdir(parents=True)
    metric_vector = _metric_vector(success)
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "success_rate": 1.0 if success else 0.0,
                "metric_vector": metric_vector,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "episodes.json").write_text(
        json.dumps(
            [
                {
                    "episode_index": 0,
                    "success": success,
                    "failure_ledger": {
                        "events": [] if success else [_failure_event()]
                    },
                    "steps": [
                        {
                            "truncated": False,
                            "info": {"safety_violation": False},
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "stressor_manifest.json").write_text(
        json.dumps(
            {
                "configured": search_space.stressor_config(proposal).to_dict(),
                "summary": {
                    "unsupported_stressors": ["action_gaussian_noise"]
                    if unsupported
                    else []
                }
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "run.yaml").write_text(
        yaml.safe_dump(
            {
                "engine_name": search_space.engine_name,
                "task_ids": [search_space.task_id],
                "seed": proposal.discovery_seed,
            }
        ),
        encoding="utf-8",
    )


def test_run_artifact_ingestion_preserves_policy_and_application_statuses(
    tmp_path: Path,
) -> None:
    study = StressSearchStudy(
        StressSearchStudySpec(
            study_id="ingest",
            search_space=_space(),
            sampler_id="random",
            study_seed=4,
            discovery_budget=2,
            batch_size=2,
            confirmation_budget=2,
            confirmation_repeats=2,
            provenance=_study_provenance(),
        )
    )
    proposals = study.propose(2)
    failure_run = tmp_path / "failure_run"
    unsupported_run = tmp_path / "unsupported_run"
    _write_run_artifacts(
        failure_run, proposals[0], study.spec.search_space, success=False
    )
    _write_run_artifacts(
        unsupported_run,
        proposals[1],
        study.spec.search_space,
        success=False,
        unsupported=True,
    )

    failure = observation_from_run(
        proposals[0],
        failure_run,
        search_space=study.spec.search_space,
        success_threshold=0.5,
    )
    unsupported = observation_from_run(
        proposals[1],
        unsupported_run,
        search_space=study.spec.search_space,
        success_threshold=0.5,
    )

    assert failure.status == "policy_failure"
    assert failure.failure_events
    assert failure.metric_vector is not None
    assert unsupported.status == "unsupported"
    assert unsupported.success is None
    assert unsupported.metric_vector is None

    run_metadata = yaml.safe_load((failure_run / "run.yaml").read_text(encoding="utf-8"))
    run_metadata["seed"] += 1
    (failure_run / "run.yaml").write_text(
        yaml.safe_dump(run_metadata), encoding="utf-8"
    )
    mismatched = observation_from_run(
        proposals[0],
        failure_run,
        search_space=study.spec.search_space,
        success_threshold=0.5,
    )
    assert mismatched.status == "invalid"
    assert "run_seed" in str(mismatched.reason)


def test_cli_ingests_run_into_correct_study_phase(tmp_path: Path) -> None:
    spec = StressSearchStudySpec(
        study_id="ingest_cli",
        search_space=_space(),
        sampler_id="random",
        study_seed=4,
        discovery_budget=1,
        confirmation_budget=1,
        confirmation_repeats=1,
        provenance=_study_provenance(),
    )
    study = StressSearchStudy(spec)
    proposal = study.propose(1)[0]
    study_path = write_stress_search_study(study, tmp_path / "study.json")
    run_dir = tmp_path / "run"
    _write_run_artifacts(run_dir, proposal, study.spec.search_space, success=True)
    observation_path = tmp_path / "observation.json"

    assert (
        main(
            [
                "stress-search-ingest-run",
                str(study_path),
                proposal.proposal_id,
                str(run_dir),
                "--out",
                str(study_path),
                "--observation-out",
                str(observation_path),
            ]
        )
        == 0
    )
    loaded = load_stress_search_study(study_path)
    assert loaded.sampler.observations[proposal.proposal_id].status == "success"
    assert json.loads(observation_path.read_text(encoding="utf-8"))["observations"]
