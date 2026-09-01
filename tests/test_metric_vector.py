from __future__ import annotations

from typing import Any

import pytest

from nyssa_bench.core.episode import EpisodeResult, StepRecord
from nyssa_bench.core.suite import Suite
from nyssa_bench.failures import FailureEventDraft, FailureEventLedger
from nyssa_bench.metrics.run_claims import RunClaimValidator
from nyssa_bench.metrics.sim_to_real import prototype_reliability_score
from nyssa_bench.metrics.vector import (
    DEFINITIONS_BY_ID,
    METRIC_VECTOR_FORMAT,
    RUN_METRICS_FORMAT,
    build_metric_vector,
    migrate_metric_summary,
    sim_real_metrics_are_supported,
    validate_metric_vector,
)


def _episode(
    *,
    seed: int,
    success: bool,
    shifted: bool = False,
    task_id: str = "task",
    episode_index: int | None = None,
    failure_step: int | None = None,
) -> EpisodeResult:
    index = seed if episode_index is None else episode_index
    steps = [StepRecord({}, 0.0, 0.0, False, False, {}) for _ in range(6)]
    ledger = None
    if failure_step is not None:
        mutable = FailureEventLedger(
            task_id=task_id,
            episode_index=index,
            episode_seed=seed,
            engine_name="unit",
        )
        condition = mutable.emitter("stressor", "test_stressor")
        condition.emit(
            FailureEventDraft(
                role="contributing_condition",
                category="stressor",
                subtype="shift",
                onset_step=0,
            )
        )
        detector = mutable.emitter("external_monitor", "test_detector")
        detector.emit(
            FailureEventDraft(
                role="mechanism",
                category="control",
                subtype="stall",
                onset_step=failure_step,
            )
        )
        ledger = mutable.snapshot()
    return EpisodeResult(
        task_id=task_id,
        episode_index=index,
        seed=seed,
        success=success,
        failure_label=None if success else "planner_stuck",
        failure_label_source=None if success else "mapper",
        metrics={
            "expert_intervention_count": float(seed % 2),
            "safety_violation_rate": float(not success),
        },
        steps=steps,
        stressor_context={
            "applications": [
                {
                    "status": "applied",
                    "requested": {"severity": 1.0},
                }
            ]
        }
        if shifted
        else {"applications": []},
        failure_ledger=ledger,
    )


def _summary(*, episodes: int = 4, successes: int = 3) -> dict[str, Any]:
    return {
        "episodes": episodes,
        "success_count": successes,
        "success_rate": successes / episodes,
        "success_rate_ci95": [0.30, 0.95],
        "failure_counts": {"planner_stuck": episodes - successes},
        "metrics": {},
        "metric_ci95": {},
        "compute": {"wall_time_seconds": 2.5},
    }


def test_metric_vector_has_complete_definitions_and_no_scalar_composite():
    vector = build_metric_vector(
        _summary(),
        [_episode(seed=0, success=True), _episode(seed=1, success=False)],
    )

    assert vector["format"] == METRIC_VECTOR_FORMAT
    assert set(vector["definitions"]) == set(DEFINITIONS_BY_ID)
    assert set(vector["values"]) == set(DEFINITIONS_BY_ID)
    assert vector["scalar_composite"] is None
    for definition in vector["definitions"].values():
        assert {
            "population",
            "denominator",
            "aggregation",
            "missing_data",
            "direction",
            "uncertainty",
        } <= set(definition)


def test_clean_and_shifted_success_use_explicit_populations_and_intervals():
    episodes = [
        _episode(seed=0, success=True),
        _episode(seed=1, success=True),
        _episode(seed=0, success=True, shifted=True),
        _episode(seed=1, success=False, shifted=True),
    ]

    vector = build_metric_vector(_summary(), episodes)
    clean = vector["values"]["clean_success_rate"]
    shifted = vector["values"]["shifted_success_rate"]
    degradation = vector["values"]["robustness_degradation"]

    assert clean["value"] == 1.0
    assert clean["denominator"] == 2
    assert len(clean["ci95"]) == 2
    assert shifted["value"] == 0.5
    assert degradation["status"] == "available"
    assert degradation["value"] == 0.5
    assert degradation["sample_size"] == 2


def test_degradation_rejects_incompatible_episode_populations():
    episodes = [
        _episode(seed=0, success=True),
        _episode(seed=1, success=False, shifted=True),
    ]

    measurement = build_metric_vector(_summary(), episodes)["values"][
        "robustness_degradation"
    ]

    assert measurement["status"] == "incompatible"
    assert measurement["value"] is None
    assert "one-to-one" in measurement["reason"]


def test_missing_metrics_are_not_coerced_to_zero():
    vector = build_metric_vector(_summary(), [_episode(seed=0, success=True)])

    for metric_id in (
        "shifted_success_rate",
        "robustness_auc",
        "failure_prediction_ece",
        "counterfactual_recovery_gain",
        "false_intervention_rate",
        "damage_event_rate",
        "sim_real_rank_correlation",
    ):
        measurement = vector["values"][metric_id]
        assert measurement["status"] != "available"
        assert measurement["value"] is None
        assert measurement["reason"]


def test_invalid_stressor_severity_is_excluded_instead_of_treated_as_clean():
    episode = _episode(seed=0, success=True, shifted=True)
    episode.stressor_context["applications"][0]["requested"]["severity"] = "bad"

    vector = build_metric_vector(_summary(episodes=1, successes=1), [episode])

    assert vector["population"]["excluded_condition_episodes"] == 1
    assert vector["values"]["clean_success_rate"]["status"] == "unavailable"
    assert vector["values"]["shifted_success_rate"]["status"] == "unavailable"


def test_impossible_rate_counts_are_reported_as_incompatible():
    episode = _episode(seed=0, success=True)
    episode.metrics["expert_intervention_count"] = 10.0

    measurement = build_metric_vector(_summary(), [episode])["values"][
        "expert_intervention_rate"
    ]

    assert measurement["status"] == "incompatible"
    assert measurement["value"] is None


def test_time_to_failure_ignores_stressor_condition_events_and_reports_censoring():
    episodes = [
        _episode(seed=0, success=False, failure_step=4),
        _episode(seed=1, success=True),
    ]

    measurement = build_metric_vector(_summary(), episodes)["values"][
        "mean_time_to_failure_steps"
    ]

    assert measurement["value"] == 4.0
    assert measurement["ci95"] == [4.0, 4.0]
    assert measurement["censored_count"] == 1


def test_robustness_auc_records_interpolation_and_observed_span():
    robustness = {
        "auc_convention": "trapezoidal_success_rate_integral_normalized_by_observed_severity_span",
        "robustness_auc": 0.75,
        "robustness_auc_ci95": [0.6, 0.9],
        "paired_episode_coverage": 20,
        "severity_domain": [0.0, 1.0],
    }

    measurement = build_metric_vector(_summary(), robustness_summary=robustness)[
        "values"
    ]["robustness_auc"]

    assert measurement["value"] == 0.75
    assert measurement["ci95"] == [0.6, 0.9]
    assert measurement["interpolation"] == "piecewise_linear_trapezoidal"
    assert measurement["normalization"] == "observed_severity_span"


def test_legacy_scalar_fields_migrate_without_reinterpretation():
    legacy = {
        **_summary(),
        "prototype_reliability_score": 0.73,
        "score_kind": "prototype_reliability_heuristic",
        "sim_to_real_score": 0.73,
        "sim_to_real_score_deprecated": True,
    }

    migrated = migrate_metric_summary(legacy)

    assert migrated["format"] == RUN_METRICS_FORMAT
    assert "prototype_reliability_score" not in migrated
    assert "sim_to_real_score" not in migrated
    assert migrated["legacy_metrics"]["values"]["sim_to_real_score"] == 0.73
    assert (
        migrated["metric_migration"]["legacy_scalar_policy"]
        == "preserved_for_audit_only_not_mapped_or_ranked"
    )
    assert migrated["metric_vector"]["scalar_composite"] is None


def test_legacy_scalar_python_api_warns_and_is_not_part_of_new_schema():
    with pytest.warns(DeprecationWarning, match="legacy heuristic"):
        value = prototype_reliability_score({"success_rate": 1.0})

    assert value == 1.0
    assert "prototype_reliability_score" not in build_metric_vector(_summary())


def test_validated_hardware_evidence_is_required_for_sim_real_metrics():
    no_hardware = build_metric_vector(_summary())
    assert no_hardware["values"]["sim_real_rank_correlation"]["status"] == "unavailable"
    assert sim_real_metrics_are_supported(no_hardware) is True

    summary = {
        **_summary(),
        "hardware_calibration": {
            "validated": True,
            "study_id": "hardware-study-1",
            "contract_sha256": "a" * 64,
            "metrics": {
                "rank_correlation": {
                    "value": 0.8,
                    "ci95": [0.5, 0.95],
                    "sample_size": 6,
                },
                "failure_distribution_similarity": {
                    "value": 0.7,
                    "ci95": [0.4, 0.9],
                    "sample_size": 12,
                },
            },
        },
    }
    calibrated = build_metric_vector(summary)

    assert calibrated["values"]["sim_real_rank_correlation"]["value"] == 0.8
    assert sim_real_metrics_are_supported(calibrated) is True


def test_claim_validation_rejects_unsubstantiated_sim_real_label():
    vector = build_metric_vector(_summary())
    vector["values"]["sim_real_rank_correlation"] = {
        "status": "available",
        "value": 0.9,
        "ci95": [0.8, 1.0],
        "sample_size": 5,
        "numerator": None,
        "denominator": 5,
        "source": "unverified",
        "reason": None,
    }
    suite = Suite.load("mujoco_control_v0")

    validation = RunClaimValidator().validate(
        suite=suite,
        engine_name="mujoco",
        episodes_per_task=0,
        episodes=[],
        out_dir=None,
        metric_vector=vector,
    )

    assert validation.checks["sim_real_metrics_have_hardware_calibration"] is False
    assert "sim_real_metrics_have_hardware_calibration" in validation.failures


def test_metric_vector_validation_rejects_unknown_schema_and_missing_metrics():
    vector = build_metric_vector(_summary())
    bad_format = {**vector, "format": "nyssa-metric-vector-v99"}
    with pytest.raises(ValueError, match="Unsupported metric vector format"):
        validate_metric_vector(bad_format)

    incomplete = {**vector, "values": dict(vector["values"])}
    incomplete["values"].pop("clean_success_rate")
    with pytest.raises(ValueError, match="every registered metric"):
        validate_metric_vector(incomplete)

    composite = {**vector, "scalar_composite": 0.5}
    with pytest.raises(ValueError, match="cannot contain a composite scalar"):
        validate_metric_vector(composite)

    non_finite = {**vector, "values": dict(vector["values"])}
    non_finite["values"]["wall_time_seconds"] = {
        **non_finite["values"]["wall_time_seconds"],
        "value": float("nan"),
    }
    with pytest.raises(ValueError, match="finite JSON-compatible"):
        validate_metric_vector(non_finite)
