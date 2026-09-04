import csv
import json

import pytest

from nyssa_bench.arena import (
    DuplicateEpisodeKeysError,
    EpisodeKey,
    IncompletePairwiseComparisonError,
    PairwiseConditionMismatchError,
    PreferenceRecord,
    assess_episode_pairing,
    compare_episode_pairs,
    save_arena_report,
    save_pairwise_results,
    save_preference_table,
)
from nyssa_bench.core.episode import EpisodeResult, StepRecord
from nyssa_bench.recovery import (
    BranchOutcome,
    BranchPoint,
    BranchStep,
    CounterfactualRecoveryRecord,
    RestoreCapability,
)


def test_compare_episode_pairs_counts_wins_and_complete_coverage():
    policy_a = [
        _episode("pick", 0, 0, success=True),
        _episode("pick", 1, 0, success=False, failure_label="timeout"),
        _episode("stack", 0, 0, success=False, failure_label="bad_grasp"),
    ]
    policy_b = [
        _episode("pick", 0, 0, success=False, failure_label="timeout"),
        _episode("pick", 1, 0, success=False, failure_label="timeout"),
        _episode("stack", 0, 0, success=True),
    ]

    summary = compare_episode_pairs(
        policy_a, policy_b, policy_a_label="a", policy_b_label="b"
    )

    assert summary.total_pairs == 3
    assert summary.wins == {"a": 1, "tie_failure": 1, "b": 1}
    assert summary.failure_deltas == {"b:timeout": 1, "a:bad_grasp": 1}
    assert summary.outcomes[0].winner == "a"
    assert summary.comparison_mode == "complete"
    assert summary.pairing_claim_eligible is True
    assert summary.caveats == ()
    assert summary.coverage.complete is True
    assert summary.coverage.policy_a_coverage == 1.0
    assert summary.coverage.policy_b_coverage == 1.0
    assert summary.coverage.joint_coverage == 1.0


def test_compare_episode_pairs_requires_complete_matching_by_default():
    policy_a = [_episode("pick", 0, 0, success=True)]
    policy_b = [
        _episode("pick", 0, 0, success=False, failure_label="timeout"),
        _episode("stack", 99, 0, success=True),
    ]

    with pytest.raises(
        IncompletePairwiseComparisonError, match="allow_partial=True"
    ) as caught:
        compare_episode_pairs(policy_a, policy_b)

    coverage = caught.value.coverage
    assert coverage.matched_count == 1
    assert coverage.unmatched_a_count == 0
    assert coverage.unmatched_b_keys == (EpisodeKey("stack", 99, 0),)
    assert coverage.policy_a_coverage == 1.0
    assert coverage.policy_b_coverage == 0.5
    assert coverage.joint_coverage == 0.5


def test_compare_episode_pairs_partial_mode_is_caveated_and_not_claim_eligible():
    summary = compare_episode_pairs(
        [_episode("pick", seed, 0, success=True) for seed in range(5)],
        [
            _episode("pick", seed, 0, success=False, failure_label="timeout")
            for seed in [0, 5, 6, 7, 8]
        ],
        allow_partial=True,
    )

    assert summary.total_pairs == 1
    assert summary.comparison_mode == "partial_exploratory"
    assert summary.pairing_claim_eligible is False
    assert "not eligible for benchmark claims" in summary.caveats[0]
    assert summary.coverage.unmatched_a_count == 4
    assert summary.coverage.unmatched_b_count == 4
    assert summary.coverage.policy_a_coverage == 0.2
    assert summary.coverage.policy_b_coverage == 0.2
    assert summary.coverage.joint_coverage == pytest.approx(1 / 9)


def test_duplicate_episode_keys_are_reported_for_both_inputs_and_rejected():
    duplicate_a = _episode("pick", 0, 0, success=True)
    duplicate_b = _episode("stack", 1, 0, success=False, failure_label="timeout")
    policy_a = [duplicate_a, duplicate_a, _episode("stack", 1, 0, success=True)]
    policy_b = [
        _episode("pick", 0, 0, success=True),
        duplicate_b,
        duplicate_b,
        duplicate_b,
    ]

    coverage = assess_episode_pairing(
        policy_a, policy_b, policy_a_label="a", policy_b_label="b"
    )

    assert coverage.policy_a_requested_count == 3
    assert coverage.policy_a_unique_count == 2
    assert coverage.duplicate_a_keys[0].key == EpisodeKey("pick", 0, 0)
    assert coverage.duplicate_a_keys[0].count == 2
    assert coverage.duplicate_b_keys[0].key == EpisodeKey("stack", 1, 0)
    assert coverage.duplicate_b_keys[0].count == 3

    with pytest.raises(
        DuplicateEpisodeKeysError, match=r"a: .* x2; b: .* x3"
    ) as caught:
        compare_episode_pairs(
            policy_a,
            policy_b,
            policy_a_label="a",
            policy_b_label="b",
            allow_partial=True,
        )

    assert caught.value.coverage == coverage


def test_empty_pairwise_inputs_are_not_complete_evidence():
    with pytest.raises(IncompletePairwiseComparisonError) as caught:
        compare_episode_pairs([], [])

    assert caught.value.coverage.matched_count == 0
    assert caught.value.coverage.complete is False


def test_preference_schema_and_complete_arena_artifacts(tmp_path):
    summary = compare_episode_pairs(
        [_episode("pick", 0, 0, success=True)],
        [_episode("pick", 0, 0, success=False, failure_label="timeout")],
    )
    preference = PreferenceRecord(
        task_id="pick",
        seed=0,
        episode_index=0,
        choice="policy_a",
        reason="cleaner completion",
        evaluator_id="eval_1",
    )

    assert PreferenceRecord.from_dict(preference.to_dict()) == preference
    results_path = save_pairwise_results(summary, tmp_path)
    table_path = save_preference_table([preference], tmp_path)
    report_path = save_arena_report(summary, tmp_path)

    first_line = results_path.read_text(encoding="utf-8").splitlines()[0]
    summary_payload = json.loads(
        (tmp_path / "pairwise_summary.json").read_text(encoding="utf-8")
    )
    with (tmp_path / "pairwise_coverage.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        coverage_row = next(csv.DictReader(handle))
    with (tmp_path / "pairwise_metrics.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        metric_rows = {row["metric"]: row for row in csv.DictReader(handle)}
    report = report_path.read_text(encoding="utf-8")

    assert json.loads(first_line)["winner"] == "policy_a"
    assert summary_payload["format"] == "nyssa-pairwise-summary-v2"
    assert summary_payload["coverage"]["matched_count"] == 1
    assert summary_payload["pairing_claim_eligible"] is True
    assert coverage_row["matched_count"] == "1"
    assert coverage_row["joint_coverage"] == "1.0"
    assert coverage_row["comparison_contract_sha256"] == summary_payload[
        "comparison_contract_sha256"
    ]
    assert metric_rows["success_difference"]["value"] == "1.0"
    assert "cleaner completion" in table_path.read_text(encoding="utf-8")
    assert "Complete paired comparison" in report
    assert "Total pairs: 1" in report
    assert (tmp_path / "pairwise_metrics.csv").is_file()
    assert summary_payload["comparison_contract_sha256"]
    assert summary_payload["paired_metrics"]["success_difference"]["value"] == 1.0


def test_partial_coverage_is_visible_in_all_arena_artifacts(tmp_path):
    summary = compare_episode_pairs(
        [_episode("pick", 0, 0, success=True), _episode("pick", 1, 0, success=True)],
        [_episode("pick", 0, 0, success=False, failure_label="timeout")],
        allow_partial=True,
    )

    save_pairwise_results(summary, tmp_path)
    report = save_arena_report(summary, tmp_path).read_text(encoding="utf-8")
    summary_payload = json.loads(
        (tmp_path / "pairwise_summary.json").read_text(encoding="utf-8")
    )
    with (tmp_path / "pairwise_coverage.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        coverage_row = next(csv.DictReader(handle))

    assert summary_payload["comparison_mode"] == "partial_exploratory"
    assert summary_payload["pairing_claim_eligible"] is False
    assert summary_payload["coverage"]["unmatched_a_keys"] == [
        {"episode_index": 0, "seed": 1, "task_id": "pick"}
    ]
    assert coverage_row["unmatched_a_count"] == "1"
    assert coverage_row["policy_a_coverage"] == "0.5"
    assert json.loads(coverage_row["unmatched_a_keys"])[0]["seed"] == 1
    assert "NON-COMPARABLE PARTIAL OUTPUT" in report
    assert "Unmatched policy_a episodes" in report
    assert "50.0%" in report


def test_equal_terminal_outcomes_preserve_distinct_failure_mechanisms():
    policy_a = [
        _episode(
            "pick",
            0,
            0,
            success=False,
            failure_label="bad_grasp",
            failure_events=[_event("mechanism", "grasp", "insufficient_contact", 3)],
        )
    ]
    policy_b = [
        _episode(
            "pick",
            0,
            0,
            success=False,
            failure_label="collision",
            failure_events=[_event("mechanism", "contact", "collision", 5)],
        )
    ]

    summary = compare_episode_pairs(policy_a, policy_b)

    assert summary.wins == {"tie_failure": 1}
    assert summary.paired_metrics["success_difference"]["value"] == 0.0
    mechanisms = summary.paired_metrics["failure_event_deltas"]["mechanisms"]
    assert mechanisms == {"collision": -1, "insufficient_contact": 1}
    evidence = summary.outcomes[0].evidence
    assert evidence["policy_a"]["time_to_failure"]["step"] == 3
    assert evidence["policy_b"]["time_to_failure"]["step"] == 5
    assert evidence["time_to_failure"]["delta_a_minus_b"] == -2


def test_identical_policies_have_zero_paired_deltas():
    episodes = [
        _episode(
            "pick",
            seed,
            0,
            success=seed == 0,
            failure_label=None if seed == 0 else "timeout",
            failure_events=[]
            if seed == 0
            else [_event("consequence", "task", "timeout", 7)],
            metrics={"collision_count": float(seed)},
        )
        for seed in range(2)
    ]

    summary = compare_episode_pairs(episodes, episodes)

    assert summary.paired_metrics["success_difference"]["value"] == 0.0
    assert (
        summary.paired_metrics["numeric_deltas"]["collision_count"]["value"]
        == 0.0
    )
    assert summary.paired_metrics["failure_event_deltas"]["categories"] == {}
    assert summary.paired_metrics["success_difference"]["discordance"] == {
        "both_failed": 1,
        "both_succeeded": 1,
    }


def test_time_to_failure_reports_censoring_instead_of_imputing_success():
    summary = compare_episode_pairs(
        [_episode("pick", 0, 0, success=True, step_count=8)],
        [
            _episode(
                "pick",
                0,
                0,
                success=False,
                failure_label="collision",
                failure_events=[_event("symptom", "contact", "collision", 4)],
                step_count=6,
            )
        ],
    )

    paired_time = summary.outcomes[0].evidence["time_to_failure"]
    assert paired_time["status"] == "censored_or_missing"
    assert paired_time["policy_a"]["status"] == "right_censored_success"
    aggregate = summary.paired_metrics["time_to_failure_difference"]
    assert aggregate["status"] == "unavailable"
    assert aggregate["sample_size"] == 0
    assert aggregate["missing_count"] == 1


def test_missing_failure_ledger_coverage_is_reported():
    summary = compare_episode_pairs(
        [_episode("pick", 0, 0, success=False, failure_label="timeout")],
        [
            _episode(
                "pick",
                0,
                0,
                success=False,
                failure_label="timeout",
                failure_events=[_event("consequence", "task", "timeout", 9)],
            )
        ],
    )

    coverage = summary.paired_metrics["evidence_coverage"]
    assert coverage["failure_ledger_pairs"] == 0
    assert coverage["failure_ledger_rate"] == 0.0


def test_detector_runtime_support_is_evidence_not_a_condition_mismatch():
    contract = {"detector_id": "contact", "detector_version": "1.0.0"}
    policy_a = _episode(
        "pick",
        0,
        0,
        success=False,
        failure_label="collision",
        failure_detector_context={
            "format": "nyssa-failure-detector-manifest-v1",
            "detectors": [
                {
                    "contract": contract,
                    "support": {"status": "supported"},
                    "emitted_event_count": 1,
                }
            ],
        },
    )
    policy_b = _episode(
        "pick",
        0,
        0,
        success=False,
        failure_label="timeout",
        failure_detector_context={
            "format": "nyssa-failure-detector-manifest-v1",
            "detectors": [
                {
                    "contract": contract,
                    "support": {"status": "unsupported"},
                    "emitted_event_count": 0,
                }
            ],
        },
    )

    summary = compare_episode_pairs([policy_a], [policy_b])

    assert summary.outcomes[0].condition_compatible is True
    assert summary.paired_metrics["evidence_coverage"][
        "failure_detector_evidence_pairs"
    ] == 1
    profiles = summary.outcomes[0].evidence
    assert profiles["policy_a"]["failure_detectors"]["detectors"][0][
        "support_status"
    ] == "supported"
    assert profiles["policy_b"]["failure_detectors"]["detectors"][0][
        "support_status"
    ] == "unsupported"


def test_condition_mismatch_is_rejected_or_explicitly_exploratory():
    policy_a = [
        _episode(
            "pick",
            0,
            0,
            success=True,
            stressor_context={"condition_id": "clean"},
        )
    ]
    policy_b = [
        _episode(
            "pick",
            0,
            0,
            success=False,
            failure_label="timeout",
            stressor_context={"condition_id": "shifted"},
        )
    ]

    with pytest.raises(PairwiseConditionMismatchError) as caught:
        compare_episode_pairs(policy_a, policy_b)
    assert caught.value.mismatches[0]["differing_fields"] == [
        "stressor_execution"
    ]

    summary = compare_episode_pairs(
        policy_a, policy_b, allow_condition_mismatch=True
    )
    assert summary.pairing_claim_eligible is False
    assert summary.outcomes[0].condition_identity is None
    assert summary.paired_metrics["condition_incompatible_pairs"] == 1
    assert summary.paired_metrics["success_difference"]["status"] == "unavailable"


def test_safety_intervention_and_recovery_deltas_keep_missingness():
    summary = compare_episode_pairs(
        [
            _episode(
                "pick",
                0,
                0,
                success=True,
                metrics={
                    "safety_violation_rate": 0.0,
                    "expert_intervention_rate": 0.2,
                    "counterfactual_recovery_gain": 0.5,
                },
            )
        ],
        [
            _episode(
                "pick",
                0,
                0,
                success=True,
                metrics={
                    "safety_violation_rate": 1.0,
                    "expert_intervention_rate": 0.1,
                },
            )
        ],
    )

    metrics = summary.paired_metrics["numeric_deltas"]
    assert metrics["safety_violation_rate"]["value"] == -1.0
    assert metrics["expert_intervention_rate"]["value"] == pytest.approx(0.1)
    assert metrics["counterfactual_recovery_gain"]["status"] == "unavailable"
    assert metrics["counterfactual_recovery_gain"]["missing_count"] == 1


def test_counterfactual_recovery_delta_requires_and_uses_branch_evidence():
    policy_a = _episode(
        "pick",
        0,
        0,
        success=True,
        metrics={
            "counterfactual_recovery_gain": 1.0,
            "counterfactual_branch_coverage": 1.0,
            "counterfactual_eligible_branch_point_count": 1.0,
        },
        counterfactual_recovery=[_branch_record(False, True)],
    )
    policy_b = _episode(
        "pick",
        0,
        0,
        success=True,
        metrics={
            "counterfactual_recovery_gain": 0.0,
            "counterfactual_branch_coverage": 1.0,
            "counterfactual_eligible_branch_point_count": 1.0,
        },
        counterfactual_recovery=[_branch_record(False, False)],
    )

    summary = compare_episode_pairs([policy_a], [policy_b])

    recovery = summary.outcomes[0].evidence["metrics"][
        "counterfactual_recovery_gain"
    ]
    assert recovery["status"] == "available"
    assert recovery["delta_a_minus_b"] == 1.0
    assert recovery["policy_a_evidence_tier"] == "exact_counterfactual"
    assert summary.paired_metrics["numeric_deltas"][
        "counterfactual_recovery_gain"
    ]["value"] == 1.0


def test_comparison_contract_is_hashed_and_label_mismatch_is_rejected():
    episodes = [_episode("pick", 0, 0, success=True)]
    summary = compare_episode_pairs(
        episodes,
        episodes,
        policy_a_label="alpha",
        policy_b_label="beta",
    )

    assert len(summary.comparison_contract_sha256) == 64
    assert (
        summary.outcomes[0].comparison_contract_sha256
        == summary.comparison_contract_sha256
    )
    incompatible = {**summary.comparison_contract, "policy_a_label": "wrong"}
    with pytest.raises(ValueError, match="policy_a_label"):
        compare_episode_pairs(
            episodes,
            episodes,
            policy_a_label="alpha",
            policy_b_label="beta",
            comparison_contract=incompatible,
        )


def _episode(
    task_id: str,
    seed: int,
    episode_index: int,
    *,
    success: bool,
    failure_label: str | None = None,
    failure_events: list[dict] | None = None,
    metrics: dict[str, float] | None = None,
    stressor_context: dict | None = None,
    step_count: int = 0,
    counterfactual_recovery: list[CounterfactualRecoveryRecord] | None = None,
    failure_detector_context: dict | None = None,
):
    return EpisodeResult(
        task_id=task_id,
        episode_index=episode_index,
        seed=seed,
        success=success,
        failure_label=failure_label,
        failure_label_source="mapper" if failure_label else None,
        metrics=metrics or {},
        stressor_context=stressor_context or {},
        failure_ledger={"events": failure_events}
        if failure_events is not None
        else None,  # type: ignore[arg-type]
        steps=[
            StepRecord(
                observation={"raw": [0.0]},
                action=[0.0],
                reward=0.0,
                terminated=False,
                truncated=False,
                info={},
            )
            for _ in range(step_count)
        ],
        counterfactual_recovery=counterfactual_recovery or [],
        failure_detector_context=failure_detector_context or {},
    )


def _event(role: str, category: str, subtype: str, onset_step: int) -> dict:
    return {
        "event_id": f"{role}:{category}:{subtype}:{onset_step}",
        "role": role,
        "category": category,
        "subtype": subtype,
        "onset_step": onset_step,
        "end_step": onset_step,
        "confidence": 1.0,
        "evidence": {"policy_observable": [], "privileged": [], "external": []},
    }


def _branch_record(
    continue_success: bool, recovery_success: bool
) -> CounterfactualRecoveryRecord:
    capabilities = tuple(
        RestoreCapability(
            component=component,
            component_id=f"unit_{component}",
            required=True,
            supported=True,
            fidelity=f"exact_{component}",
            captures_rng=component in {"engine", "stressors", "process_rng"},
            exact=True,
        )
        for component in ("engine", "policy", "stressors", "process_rng")
    )
    point = BranchPoint(
        branch_point_id="pick:episode-0:step-0:recovery-1",
        task_id="pick",
        episode_index=0,
        episode_seed=0,
        step_index=0,
        recovery_attempt_id=1,
        requested_repeats=1,
        requested_branches=("continue", "recovery"),
        trigger_kind="recovery_decision",
        trigger_reason="test",
        trigger_event_id=None,
        snapshot_sha256="a" * 64,
        restoration_grade="exact",
        restore_capabilities=capabilities,
        matched_randomness=True,
        repeat_seed_strategy="test",
        reseeded_components=("engine",),
        strongest_causal_claim_eligible=True,
    )

    def outcome(kind: str, success: bool) -> BranchOutcome:
        step = BranchStep(
            offset=0,
            action=[0.0],
            reward=float(success),
            terminated=success,
            truncated=False,
            success=success,
            safety_violation=False,
            damage_event_count=0.0,
        )
        return BranchOutcome(
            branch_point_id=point.branch_point_id,
            branch_kind=kind,  # type: ignore[arg-type]
            repeat_index=0,
            branch_seed=0,
            status="completed",
            success=success,
            terminated=success,
            truncated=False,
            total_reward=float(success),
            terminal_reason="task_success" if success else "horizon_exhausted",
            initial_action_count=1,
            trajectory_sha256=("b" if kind == "continue" else "c") * 64,
            matched_rng_sha256="d" * 64,
            steps=(step,),
        )

    return CounterfactualRecoveryRecord(
        branch_point=point,
        outcomes=(
            outcome("continue", continue_success),
            outcome("recovery", recovery_success),
        ),
    )
