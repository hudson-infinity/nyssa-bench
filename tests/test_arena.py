import csv
import json

import pytest

from nyssa_bench.arena import (
    DuplicateEpisodeKeysError,
    EpisodeKey,
    IncompletePairwiseComparisonError,
    PreferenceRecord,
    assess_episode_pairing,
    compare_episode_pairs,
    save_arena_report,
    save_pairwise_results,
    save_preference_table,
)
from nyssa_bench.core.episode import EpisodeResult


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
    report = report_path.read_text(encoding="utf-8")

    assert json.loads(first_line)["winner"] == "policy_a"
    assert summary_payload["format"] == "nyssa-pairwise-summary-v1"
    assert summary_payload["coverage"]["matched_count"] == 1
    assert summary_payload["pairing_claim_eligible"] is True
    assert coverage_row["matched_count"] == "1"
    assert coverage_row["joint_coverage"] == "1.0"
    assert "cleaner completion" in table_path.read_text(encoding="utf-8")
    assert "Complete paired comparison" in report
    assert "Total pairs: 1" in report


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


def _episode(
    task_id: str,
    seed: int,
    episode_index: int,
    *,
    success: bool,
    failure_label: str | None = None,
):
    return EpisodeResult(
        task_id=task_id,
        episode_index=episode_index,
        seed=seed,
        success=success,
        failure_label=failure_label,
        failure_label_source="mapper" if failure_label else None,
        metrics={},
    )
