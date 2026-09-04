from __future__ import annotations

import json
from pathlib import Path

import pytest

from nyssa_bench.cli import main
from nyssa_bench.arena import assess_episode_pairing
from nyssa_bench.core.episode import EpisodeResult, StepRecord
from nyssa_bench.core.suite import Suite
from nyssa_bench.core.task import TaskSpec
from nyssa_bench.engines.base import NyssaEngine
from nyssa_bench.metrics.run_claims import RunClaimValidator
from nyssa_bench.plugins import get_plugin_registry
from nyssa_bench.runner import PolicyRunner
from nyssa_bench.validity import (
    ABLATION_AUDIT,
    DEFAULT_REQUIRED_AUDITS,
    HIDDEN_TEST_AUDIT,
    LEAKAGE_AUDIT,
    PAIRING_AUDIT,
    RANK_STABILITY_AUDIT,
    SHORTCUT_AUDIT,
    SIM_REAL_AUDIT,
    STATISTICS_AUDIT,
    BenchmarkValidityEvaluator,
    BenchmarkValiditySpec,
    load_benchmark_validity_report,
    load_benchmark_validity_spec,
    write_benchmark_validity_report,
    paired_design_audit_inputs,
)


def _valid_inputs() -> dict[str, dict]:
    return {
        SHORTCUT_AUDIT: {
            "max_trivial_success_rate": 0.05,
            "baselines": [
                {
                    "policy_id": "zero_action",
                    "kind": "trivial",
                    "success_rate": 0.0,
                    "episodes": 100,
                }
            ],
        },
        LEAKAGE_AUDIT: {
            "training": {
                "seeds": ["0"],
                "assets": ["train_asset"],
                "tasks": ["train_task"],
                "demonstrations": ["demo_train"],
                "language": ["instruction_train"],
            },
            "evaluation": {
                "seeds": ["100"],
                "assets": ["test_asset"],
                "tasks": ["test_task"],
                "demonstrations": [],
                "language": ["instruction_test"],
            },
        },
        ABLATION_AUDIT: {
            "full_success_rate": 0.8,
            "language_ablated_success_rate": 0.2,
            "observation_ablated_success_rate": 0.1,
            "max_retained_fraction": 0.5,
        },
        STATISTICS_AUDIT: {
            "min_sample_size": 50,
            "max_ci95_width": 0.2,
            "estimates": [
                {"metric_id": "success", "sample_size": 100, "ci95": [0.7, 0.85]}
            ],
        },
        PAIRING_AUDIT: {
            "comparison": {
                "comparable": True,
                "mismatches": [],
                "comparison_contract_sha256": "c" * 64,
            },
            "coverage": {
                "complete": True,
                "joint_coverage": 1.0,
                "duplicate_a_count": 0,
                "duplicate_b_count": 0,
            },
        },
        RANK_STABILITY_AUDIT: {
            "min_pairwise_agreement": 0.8,
            "ranking_ids": ["seed_0", "seed_1"],
            "rankings": [["a", "b", "c"], ["a", "b", "c"]],
        },
        HIDDEN_TEST_AUDIT: {
            "splits": [
                {
                    "split_id": "hidden_v1",
                    "partition": "hidden_test",
                    "content_sha256": "a" * 64,
                    "protected": True,
                    "contents_published": False,
                    "contamination_status": "clean",
                    "producer_id": "dataset_owner",
                    "evaluator_id": "evaluation_service",
                }
            ]
        },
        SIM_REAL_AUDIT: {"required": False, "hardware_available": False},
    }


def _spec(
    inputs: dict[str, dict] | None = None,
    *,
    benchmark_id: str = "unit_benchmark",
) -> BenchmarkValiditySpec:
    return BenchmarkValiditySpec(
        benchmark_id=benchmark_id,
        benchmark_version="1.0.0",
        required_audits=DEFAULT_REQUIRED_AUDITS,
        audit_inputs=inputs if inputs is not None else _valid_inputs(),
    )


def test_complete_validity_report_passes_without_sim_real_claim() -> None:
    report = BenchmarkValidityEvaluator().evaluate(_spec())

    assert report.status == "validated"
    assert report.claim_ready is True
    assert report.blocking_audits == ()
    assert {audit.audit_id: audit.status for audit in report.audits}[
        SIM_REAL_AUDIT
    ] == "not_applicable"
    assert len(report.to_dict()["report_sha256"]) == 64


@pytest.mark.parametrize(
    ("audit_id", "mutate"),
    [
        (
            SHORTCUT_AUDIT,
            lambda value: value["baselines"][0].update(success_rate=0.9),
        ),
        (
            LEAKAGE_AUDIT,
            lambda value: value["evaluation"]["assets"].append("train_asset"),
        ),
        (
            ABLATION_AUDIT,
            lambda value: value.update(language_ablated_success_rate=0.75),
        ),
        (
            STATISTICS_AUDIT,
            lambda value: value["estimates"][0].update(sample_size=5),
        ),
        (
            PAIRING_AUDIT,
            lambda value: value["coverage"].update(complete=False),
        ),
        (
            RANK_STABILITY_AUDIT,
            lambda value: value["rankings"].__setitem__(1, ["c", "b", "a"]),
        ),
        (
            HIDDEN_TEST_AUDIT,
            lambda value: value["splits"][0].update(contents_published=True),
        ),
    ],
)
def test_invalid_fixtures_block_claims(audit_id: str, mutate) -> None:
    inputs = _valid_inputs()
    mutate(inputs[audit_id])

    report = BenchmarkValidityEvaluator().evaluate(_spec(inputs))
    audit = {item.audit_id: item for item in report.audits}[audit_id]

    assert audit.status == "failed"
    expected_impact = "downgrade" if audit_id == RANK_STABILITY_AUDIT else "block"
    assert audit.claim_impact == expected_impact
    assert audit.inputs
    assert audit.evidence
    assert audit.remediation
    assert report.status == (
        "downgraded" if audit_id == RANK_STABILITY_AUDIT else "blocked"
    )
    assert report.claim_ready is False


def test_missing_required_audit_is_not_interpreted_as_pass() -> None:
    inputs = _valid_inputs()
    inputs.pop(LEAKAGE_AUDIT)

    report = BenchmarkValidityEvaluator().evaluate(_spec(inputs))
    leakage = {item.audit_id: item for item in report.audits}[LEAKAGE_AUDIT]

    assert leakage.status == "missing"
    assert leakage.claim_impact == "block"
    assert LEAKAGE_AUDIT in report.blocking_audits


def test_public_claim_tier_cannot_omit_required_audits_from_spec() -> None:
    inputs = _valid_inputs()
    spec = BenchmarkValiditySpec(
        benchmark_id="unit_benchmark",
        benchmark_version="1.0.0",
        claim_tier="public_simulation",
        required_audits=(SHORTCUT_AUDIT,),
        audit_inputs={SHORTCUT_AUDIT: inputs[SHORTCUT_AUDIT]},
    )

    report = BenchmarkValidityEvaluator().evaluate(spec)

    assert report.claim_ready is False
    assert LEAKAGE_AUDIT in report.blocking_audits
    assert set(report.metadata["required_audits"]) == set(DEFAULT_REQUIRED_AUDITS)


def test_sim_real_claim_tier_forces_hardware_audit_requirement() -> None:
    spec = _spec()
    sim_real_spec = BenchmarkValiditySpec(
        benchmark_id=spec.benchmark_id,
        benchmark_version=spec.benchmark_version,
        claim_tier="sim_real_predictive",
        required_audits=spec.required_audits,
        audit_inputs=spec.audit_inputs,
    )

    report = BenchmarkValidityEvaluator().evaluate(sim_real_spec)

    assert report.claim_ready is False
    assert SIM_REAL_AUDIT in report.blocking_audits


def test_sim_real_claim_requires_validated_hardware_study() -> None:
    inputs = _valid_inputs()
    inputs[SIM_REAL_AUDIT] = {"required": True, "hardware_available": False}
    missing = BenchmarkValidityEvaluator().evaluate(_spec(inputs))
    assert SIM_REAL_AUDIT in missing.blocking_audits

    inputs[SIM_REAL_AUDIT] = {
        "required": True,
        "hardware_available": True,
        "study": {
            "validated": True,
            "study_id": "hardware_v1",
            "contract_sha256": "b" * 64,
        },
    }
    incomplete = BenchmarkValidityEvaluator().evaluate(_spec(inputs))
    sim_real = {item.audit_id: item for item in incomplete.audits}[SIM_REAL_AUDIT]
    assert sim_real.status == "failed"
    assert sim_real.evidence["metric_issues"] == ["metrics_missing"]

    inputs[SIM_REAL_AUDIT] = {
        "required": True,
        "hardware_available": True,
        "study": {
            "validated": True,
            "study_id": "hardware_v1",
            "contract_sha256": "b" * 64,
            "metrics": {
                "rank_correlation": {
                    "value": 0.8,
                    "ci95": [0.5, 0.95],
                    "sample_size": 8,
                },
                "failure_distribution_similarity": {
                    "value": 0.7,
                    "ci95": [0.5, 0.9],
                    "sample_size": 8,
                },
                "incremental_predictive_value": {
                    "value": 0.1,
                    "ci95": [0.01, 0.2],
                    "sample_size": 8,
                    "held_out": True,
                },
            },
        },
    }
    validated = BenchmarkValidityEvaluator().evaluate(_spec(inputs))
    assert validated.claim_ready is True


def test_spec_and_report_artifacts_round_trip_and_reject_tampering(
    tmp_path: Path,
) -> None:
    spec = _spec()
    spec_path = tmp_path / "benchmark_validity.yaml"
    import yaml

    spec_path.write_text(yaml.safe_dump(spec.to_dict()), encoding="utf-8")
    assert load_benchmark_validity_spec(spec_path) == spec

    report = BenchmarkValidityEvaluator().evaluate(spec)
    report_path = write_benchmark_validity_report(
        report, tmp_path / "benchmark_validity.json"
    )
    assert load_benchmark_validity_report(report_path) == report

    tampered = json.loads(report_path.read_text(encoding="utf-8"))
    tampered["audits"][0]["status"] = "failed"
    report_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="hash does not match"):
        load_benchmark_validity_report(report_path)


def test_rank_audit_rejects_duplicate_or_incomplete_policy_lists() -> None:
    inputs = _valid_inputs()
    inputs[RANK_STABILITY_AUDIT]["rankings"] = [["a", "b"], ["a", "a"]]
    report = BenchmarkValidityEvaluator().evaluate(_spec(inputs))
    rank = {item.audit_id: item for item in report.audits}[RANK_STABILITY_AUDIT]
    assert rank.status == "missing"
    assert rank.claim_impact == "block"


def test_hidden_test_audit_detects_content_commitment_collision() -> None:
    inputs = _valid_inputs()
    inputs[HIDDEN_TEST_AUDIT]["splits"].append(
        {
            "split_id": "train_v1",
            "partition": "train",
            "content_sha256": "a" * 64,
        }
    )

    report = BenchmarkValidityEvaluator().evaluate(_spec(inputs))
    hidden = {item.audit_id: item for item in report.audits}[HIDDEN_TEST_AUDIT]

    assert hidden.status == "failed"
    assert "content_commitment_collision" in hidden.evidence["failures"][0][
        "reasons"
    ]
    assert hidden.evidence["failures"][0]["colliding_split_ids"] == ["train_v1"]


def test_malformed_audit_input_becomes_missing_evidence_not_exception() -> None:
    inputs = _valid_inputs()
    inputs[PAIRING_AUDIT]["coverage"]["duplicate_a_count"] = "not-an-integer"

    report = BenchmarkValidityEvaluator().evaluate(_spec(inputs))
    pairing = {item.audit_id: item for item in report.audits}[PAIRING_AUDIT]

    assert pairing.status == "missing"
    assert pairing.claim_impact == "block"
    assert "invalid_counts" in pairing.evidence


def test_pairing_adapter_consumes_comparison_and_coverage_contracts() -> None:
    episode = EpisodeResult(
        task_id="task",
        episode_index=0,
        seed=0,
        success=True,
        failure_label=None,
        metrics={},
    )
    coverage = assess_episode_pairing([episode], [episode])
    inputs = paired_design_audit_inputs(
        {
            "comparable": True,
            "mismatches": [],
            "comparison_mode": "strict",
            "comparison_contract_sha256": "d" * 64,
        },
        coverage,
    )
    all_inputs = _valid_inputs()
    all_inputs[PAIRING_AUDIT] = inputs

    report = BenchmarkValidityEvaluator().evaluate(_spec(all_inputs))
    pairing = {item.audit_id: item for item in report.audits}[PAIRING_AUDIT]

    assert pairing.status == "passed"
    assert pairing.inputs["coverage"]["matched_count"] == 1


def test_cli_audit_and_validate_commands(tmp_path: Path) -> None:
    import yaml

    spec_path = tmp_path / "spec.yaml"
    report_path = tmp_path / "report.json"
    spec_path.write_text(yaml.safe_dump(_spec().to_dict()), encoding="utf-8")

    assert main(["audit-benchmark", str(spec_path), "--out", str(report_path)]) == 0
    assert main(["validate", str(spec_path)]) == 0
    assert main(["validate", str(report_path)]) == 0
    report = load_benchmark_validity_report(report_path)
    assert report.claim_ready is True

    invalid_inputs = _valid_inputs()
    invalid_inputs[PAIRING_AUDIT]["coverage"]["complete"] = False
    invalid_path = tmp_path / "invalid.yaml"
    invalid_report = tmp_path / "invalid.json"
    invalid_path.write_text(
        yaml.safe_dump(_spec(invalid_inputs).to_dict()), encoding="utf-8"
    )
    assert (
        main(
            ["audit-benchmark", str(invalid_path), "--out", str(invalid_report)]
        )
        == 2
    )
    assert load_benchmark_validity_report(invalid_report).status == "blocked"


def test_run_claim_validator_requires_separate_benchmark_validity(
    tmp_path: Path,
) -> None:
    task = TaskSpec(
        task_id="validity_task",
        engine="mujoco",
        robot="unit",
        scene="unit",
        description="Validity fixture",
        success={
            "engine_env_ids": {"mujoco": "Unit-v0"},
            "success_info_keys": ["success"],
        },
    )
    suite = Suite("validity_suite", "Validity suite", (task,))
    episodes = []
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    for index in range(100):
        replay = video_dir / f"episode_{index:03d}.mp4"
        replay.write_bytes(b"video")
        episodes.append(
            EpisodeResult(
                task_id=task.task_id,
                episode_index=index,
                seed=index,
                success=True,
                failure_label=None,
                metrics={},
                replay_path=replay.relative_to(tmp_path).as_posix(),
                steps=[
                    StepRecord(
                        observation={},
                        action=0.0,
                        reward=1.0,
                        terminated=True,
                        truncated=False,
                        info={"success": True},
                    )
                ],
            )
        )
    common = {
        "suite": suite,
        "engine_name": "mujoco",
        "episodes_per_task": 100,
        "episodes": episodes,
        "out_dir": tmp_path,
        "package_versions": {"mujoco": "3.3.1"},
        "git_info": {"commit": "a" * 40, "dirty": False},
    }

    missing = RunClaimValidator().validate(**common)
    valid_report = BenchmarkValidityEvaluator().evaluate(
        _spec(benchmark_id=suite.suite_id)
    )
    valid = RunClaimValidator().validate(
        **common, benchmark_validity=valid_report
    )

    assert missing.public_claim is False
    assert "benchmark_validity_present" in missing.failures
    assert "benchmark_validity_claim_ready" in missing.failures
    assert valid.public_claim is True
    assert valid.benchmark_tier == "real"
    assert valid.benchmark_validity is not None
    assert valid.benchmark_validity["report_sha256"]

    mismatched = RunClaimValidator().validate(
        **common,
        benchmark_validity=BenchmarkValidityEvaluator().evaluate(_spec()),
    )
    assert "benchmark_validity_matches_suite" in mismatched.failures

    unstable_inputs = _valid_inputs()
    unstable_inputs[RANK_STABILITY_AUDIT]["rankings"][1] = ["c", "b", "a"]
    downgraded_report = BenchmarkValidityEvaluator().evaluate(
        _spec(unstable_inputs, benchmark_id=suite.suite_id)
    )
    downgraded = RunClaimValidator().validate(
        **common, benchmark_validity=downgraded_report
    )
    assert downgraded_report.status == "downgraded"
    assert downgraded.public_claim is False
    assert downgraded.benchmark_tier == "benchmark_validity_downgraded"


def test_runner_hashes_benchmark_validity_into_result_pack(tmp_path: Path) -> None:
    class Engine(NyssaEngine):
        max_steps = 1

        def load_task(self, task_spec):
            self.task = task_spec

        def reset(self, seed=None):
            return {"raw": [0.0]}, {"seed": seed}

        def step(self, action):
            return {"raw": [action]}, 1.0, True, False, {"success": True}

        def render(self):
            return None

        def get_state(self):
            return {}

        def close(self):
            return None

    class Policy:
        def act(self, observation):
            return 0.0

    get_plugin_registry().engines["validity_unit"] = Engine
    task = TaskSpec(
        task_id="validity_pack_task",
        engine="validity_unit",
        robot="unit",
        scene="unit",
        description="Validity pack fixture",
        success={
            "engine_factory": {"validity_unit": "tests:Engine"},
            "success_info_keys": ["success"],
            "max_steps": 1,
        },
    )
    report = BenchmarkValidityEvaluator().evaluate(_spec())
    runner = PolicyRunner(
        policy=Policy(),
        engine="validity_unit",
        episodes=1,
        out=tmp_path,
        capture_replay=False,
        benchmark_validity=report,
    )

    result = runner.evaluate(Suite("validity_pack", "Validity pack", (task,)))

    artifact = load_benchmark_validity_report(tmp_path / "benchmark_validity.json")
    manifest = json.loads(
        (tmp_path / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    assert artifact == report
    assert "benchmark_validity.json" in manifest["artifacts"]
    assert manifest["artifacts"]["benchmark_validity.json"]["sha256"]
    assert result.summary["benchmark_validity"]["claim_ready"] is True
    assert (
        result.summary["public_claim_validation"]["checks"][
            "benchmark_validity_matches_suite"
        ]
        is False
    )
    assert "Benchmark Validity" in (tmp_path / "report.html").read_text(
        encoding="utf-8"
    )
