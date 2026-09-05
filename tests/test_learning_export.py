from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from nyssa_bench.cli import main
from nyssa_bench.core.suite import Suite
from nyssa_bench.core.task import TaskSpec
from nyssa_bench.engines.base import NyssaEngine
from nyssa_bench.experts import ExpertActionScore, ExpertProvider
from nyssa_bench.learning_export import (
    ExportSplit,
    LearningExportConfig,
    export_learning_evidence,
    load_learning_evidence,
    query_learning_evidence,
    validate_learning_evidence_use,
)
from nyssa_bench.plugins import get_plugin_registry
from nyssa_bench.runner import PolicyRunner
from nyssa_bench.stress_search import (
    SearchVariable,
    StressObservation,
    StressSearchSpace,
    StressSearchStudy,
    StressSearchStudySpec,
    write_stress_search_study,
)


class _ExportEngine(NyssaEngine):
    max_steps = 1

    def load_task(self, task_spec):
        self.task = task_spec

    def reset(self, seed=None):
        return {"raw": [0.0]}, {"seed": seed}

    def step(self, action):
        value = float(action)
        success = value > 0.0
        info = {
            "success": success,
            "completion_time": 1.0,
            "path_efficiency": float(success),
        }
        if not success:
            info.update(
                {"failure_label": "missed_target", "failure_label_source": "env"}
            )
        return (
            {"raw": [value]},
            value,
            True,
            False,
            info,
        )

    def render(self):
        return None

    def get_state(self):
        return {}

    def close(self):
        return None


class _ExportPolicy:
    def act(self, observation):
        del observation
        return -1.0


class _RecoveryExpert(ExpertProvider):
    provider_id = "export-recovery"

    def score_action(self, observation, action, *, task, engine=None):
        del observation, action, task, engine
        return ExpertActionScore(False, 1.0, "unsafe_action")

    def recover(self, *, state, failure, task, engine=None):
        del state, failure, task, engine
        return [1.0]

    def act(self, observation, *, task, engine=None):
        del observation, task, engine
        return 1.0

    def metadata(self):
        return {
            "provider_id": self.provider_id,
            "capabilities": ["score_action", "recover"],
        }


def _suite() -> Suite:
    task = TaskSpec(
        task_id="learning_export_task",
        engine="learning_export_unit",
        robot="unit",
        scene="unit",
        description="Learning export fixture",
        success={
            "engine_factory": {"learning_export_unit": "tests:_ExportEngine"},
            "success_info_keys": ["success"],
            "max_steps": 1,
        },
        failure_labels=["missed_target"],
    )
    return Suite("learning_export_suite", "Learning export suite", (task,))


def _run(out: Path, *, recovery: bool, verifier_only: bool = False) -> Path:
    get_plugin_registry().engines["learning_export_unit"] = _ExportEngine
    runner = PolicyRunner(
        policy=_ExportPolicy(),
        engine="learning_export_unit",
        episodes=1,
        seed=7 if recovery else 8,
        out=out,
        capture_replay=False,
        expert_provider=_RecoveryExpert() if recovery or verifier_only else None,
        enable_recovery=recovery,
        enable_verifier=recovery or verifier_only,
    )
    runner.evaluate(_suite())
    return out


def _config(**overrides: Any) -> LearningExportConfig:
    values = {
        "benchmark_id": "learning_export_benchmark",
        "split": ExportSplit("public_test_v1", "public_test", "a" * 64),
        "policy_families": {"_ExportPolicy": "unit_policy_family"},
        "licenses": ("Apache-2.0",),
    }
    values.update(overrides)
    return LearningExportConfig(**values)


def test_export_round_trip_preserves_failures_actions_and_exclusions(
    tmp_path: Path,
) -> None:
    failed_run = _run(tmp_path / "failed", recovery=False)
    recovered_run = _run(tmp_path / "recovered", recovery=True)

    package = export_learning_evidence(
        [failed_run, recovered_run], tmp_path / "export", config=_config()
    )
    loaded = load_learning_evidence(package.root)

    assert loaded.manifest.to_dict() == package.manifest.to_dict()
    assert [episode.to_dict() for episode in loaded.episodes] == [
        episode.to_dict() for episode in package.episodes
    ]
    assert loaded.manifest.episode_count == 2
    assert all(item.excluded_from_evaluation for item in loaded.exclusions)
    assert all(
        episode.exclusion.source_split_id == "public_test_v1"
        for episode in loaded.episodes
    )

    failed = next(episode for episode in loaded.episodes if not episode.success)
    assert failed.failure_label == "missed_target"
    assert failed.failure_ledger is not None
    assert failed.failure_cluster is not None
    assert failed.failure_cluster["method"] == (
        "temporal_role_category_subtype_signature_v1"
    )
    assert failed.failure_cluster["terminal_label"] == "missed_target"

    recovered = next(episode for episode in loaded.episodes if episode.success)
    step = recovered.steps[0]
    assert step.proposed_action == -1.0
    assert step.rejected_action == -1.0
    assert step.action_rejected is True
    assert step.executed_action_before_stressors == 1.0
    assert step.executed_action == 1.0
    assert step.executed_action_source == "recovery"
    assert step.recovery_action == 1.0
    assert step.oracle_action is None

    queried = query_learning_evidence(
        loaded,
        failure_type="missed_target",
        task="learning_export_task",
        policy_family="unit_policy_family",
    )
    assert queried == (failed,)
    assert query_learning_evidence(loaded, boundary=False) == loaded.episodes
    assert loaded.facets["facets"]["recoverability"]
    authorization = validate_learning_evidence_use(loaded, purpose="training")
    assert authorization["authorized"] is True
    assert authorization["source_split_ids"] == ["public_test_v1"]
    with pytest.raises(ValueError, match="excluded from evaluation reuse"):
        validate_learning_evidence_use(loaded, purpose="evaluation")


def test_export_can_reference_large_media_without_copying_it(tmp_path: Path) -> None:
    run_dir = _run(tmp_path / "run", recovery=False)
    video = run_dir / "videos" / "episode.mp4"
    video.parent.mkdir(exist_ok=True)
    video.write_bytes(b"large-media-reference")
    episodes_path = run_dir / "episodes.json"
    episodes = json.loads(episodes_path.read_text(encoding="utf-8"))
    episodes[0]["replay_path"] = "videos/episode.mp4"
    episodes_path.write_text(json.dumps(episodes), encoding="utf-8")
    _refresh_episode_manifest_hash(run_dir)

    package = export_learning_evidence([run_dir], tmp_path / "export", config=_config())
    loaded = load_learning_evidence(package.root, verify_external_artifacts=True)
    reference = loaded.episodes[0].artifacts[0]

    assert reference.embedded is False
    assert reference.uri.startswith("nyssa-run://")
    assert reference.sha256 == hashlib.sha256(video.read_bytes()).hexdigest()
    assert not (package.root / "episode.mp4").exists()


def test_export_preserves_oracle_replacement_separately(tmp_path: Path) -> None:
    run_dir = _run(tmp_path / "oracle", recovery=False, verifier_only=True)
    package = export_learning_evidence([run_dir], tmp_path / "export", config=_config())
    step = package.episodes[0].steps[0]

    assert step.proposed_action == -1.0
    assert step.rejected_action == -1.0
    assert step.executed_action_source == "expert"
    assert step.oracle_action == 1.0
    assert step.recovery_action is None


def test_large_observation_uses_content_addressed_source_pointer(
    tmp_path: Path,
) -> None:
    run_dir = _run(tmp_path / "run", recovery=False)
    episodes_path = run_dir / "episodes.json"
    episodes = json.loads(episodes_path.read_text(encoding="utf-8"))
    episodes[0]["steps"][0]["observation"] = {"pixels": list(range(100))}
    episodes_path.write_text(json.dumps(episodes), encoding="utf-8")
    _refresh_episode_manifest_hash(run_dir)
    config = _config(max_inline_observation_bytes=32)

    package = export_learning_evidence([run_dir], tmp_path / "export", config=config)
    loaded = load_learning_evidence(package.root, verify_external_artifacts=True)
    observation = loaded.episodes[0].steps[0].observation

    assert observation["storage"] == "source_json_pointer"
    assert observation["json_pointer"] == "/0/steps/0/observation"
    assert len(observation["value_sha256"]) == 64
    source_reference = next(
        item
        for item in loaded.episodes[0].artifacts
        if item.artifact_id.endswith(":source_episodes")
    )
    assert source_reference.embedded is False


def test_export_detects_tampered_embedded_artifact(tmp_path: Path) -> None:
    run_dir = _run(tmp_path / "run", recovery=False)
    package = export_learning_evidence([run_dir], tmp_path / "export", config=_config())
    with (package.root / "episodes.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")

    with pytest.raises(ValueError, match="artifact hash mismatch"):
        load_learning_evidence(package.root)


def test_export_rejects_manifest_tampering_and_legacy_intervention_loss(
    tmp_path: Path,
) -> None:
    run_dir = _run(tmp_path / "run", recovery=True)
    episodes_path = run_dir / "episodes.json"
    episodes = json.loads(episodes_path.read_text(encoding="utf-8"))
    info = episodes[0]["steps"][0]["info"]
    info.pop("proposed_action")
    info.pop("rejected_action")
    episodes_path.write_text(json.dumps(episodes), encoding="utf-8")
    _refresh_episode_manifest_hash(run_dir)

    with pytest.raises(ValueError, match="lacks original proposed action"):
        export_learning_evidence([run_dir], tmp_path / "export", config=_config())


def test_hidden_test_export_requires_nonpublic_governance() -> None:
    hidden = ExportSplit("hidden_v1", "hidden_test", "b" * 64)
    with pytest.raises(ValueError, match="cannot be exported as public"):
        _config(split=hidden)
    restricted = _config(
        split=hidden,
        privacy_level="restricted",
        privacy_restrictions=("approved researchers only",),
    )
    assert restricted.privacy_level == "restricted"


def test_source_episode_hash_must_match_dataset_manifest(tmp_path: Path) -> None:
    run_dir = _run(tmp_path / "run", recovery=False)
    with (run_dir / "episodes.json").open("a", encoding="utf-8") as handle:
        handle.write(" ")

    with pytest.raises(ValueError, match="do not match dataset manifest"):
        export_learning_evidence([run_dir], tmp_path / "export", config=_config())


def test_cli_exports_and_validates_learning_package(tmp_path: Path) -> None:
    run_dir = _run(tmp_path / "run", recovery=False)
    out = tmp_path / "export"

    assert (
        main(
            [
                "export-learning-evidence",
                str(run_dir),
                "--out",
                str(out),
                "--benchmark-id",
                "learning_export_benchmark",
                "--split-id",
                "public_test_v1",
                "--split-partition",
                "public_test",
                "--split-sha256",
                "a" * 64,
                "--policy-family",
                "_ExportPolicy=unit_policy_family",
                "--license",
                "Apache-2.0",
            ]
        )
        == 0
    )
    assert main(["validate", str(out)]) == 0
    assert main(["validate", str(out / "manifest.json")]) == 0
    assert load_learning_evidence(out).manifest.episode_count == 1


def test_export_attaches_boundary_search_context_when_available(tmp_path: Path) -> None:
    run_dir = _run(tmp_path / "run", recovery=False)
    run = yaml.safe_load((run_dir / "run.yaml").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    episodes = json.loads((run_dir / "episodes.json").read_text(encoding="utf-8"))
    space = StressSearchSpace(
        space_id="learning_boundary",
        engine_name="learning_export_unit",
        task_id="learning_export_task",
        variables=(
            SearchVariable(
                "severity",
                "action_gaussian_noise",
                "severity",
                "continuous",
                0.0,
                1.0,
            ),
        ),
        fixed_parameters={"action_gaussian_noise": {"max_std": 0.2}},
    )
    study = StressSearchStudy(
        StressSearchStudySpec(
            study_id="learning_boundary_study",
            search_space=space,
            sampler_id="random",
            study_seed=0,
            discovery_budget=1,
            confirmation_budget=1,
            confirmation_repeats=1,
            provenance={
                "producer_id": "test-suite",
                "study_purpose": "learning export boundary fixture",
            },
        )
    )
    proposal = study.propose(1)[0]
    failure_events = tuple(episodes[0]["failure_ledger"]["events"])
    study.observe(
        (
            StressObservation(
                proposal_id=proposal.proposal_id,
                status="policy_failure",
                success=False,
                metric_vector=metrics["metric_vector"],
                failure_events=failure_events,
                provenance={
                    "source": "nyssa_result_pack",
                    "source_id": run["run_id"],
                },
                application_evidence={"fixture": True},
            ),
        )
    )
    study_path = write_stress_search_study(study, tmp_path / "boundary.json")
    config = _config(boundary_studies=(study_path,))

    package = export_learning_evidence([run_dir], tmp_path / "export", config=config)
    episode = package.episodes[0]

    assert episode.boundary_context is not None
    assert episode.boundary_context["study_id"] == "learning_boundary_study"
    assert episode.boundary_context["observation"]["status"] == "policy_failure"
    assert query_learning_evidence(package, boundary=True) == (episode,)


def _refresh_episode_manifest_hash(run_dir: Path) -> None:
    episodes_path = run_dir / "episodes.json"
    manifest_path = run_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["episodes.json"]["sha256"] = hashlib.sha256(
        episodes_path.read_bytes()
    ).hexdigest()
    manifest["artifacts"]["episodes.json"]["bytes"] = episodes_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
