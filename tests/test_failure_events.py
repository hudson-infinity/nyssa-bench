import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from nyssa_bench import PolicyRunner, Suite
from nyssa_bench.engines.base import NyssaEngine
from nyssa_bench.experts import ExpertActionScore, ExpertProvider
from nyssa_bench.failures import (
    FAILURE_EVENT_FORMAT,
    FAILURE_LEDGER_FORMAT,
    CausalHypothesis,
    FailureEventDraft,
    FailureEventLedger,
    FailureEvidence,
    FailureLedgerRecord,
    derive_failure_label,
    failure_ledger_from_episode_dict,
)
from nyssa_bench.plugins import get_plugin_registry
from nyssa_bench.replay.timeline import episode_timeline
from nyssa_bench.stressors import StressorConfig, StressorSpec


FIXTURES = Path(__file__).parent / "fixtures"


def test_failure_event_schema_round_trip_partitions_evidence_visibility():
    ledger = FailureEventLedger(
        task_id="pick_cube",
        episode_index=3,
        episode_seed=17,
        engine_name="maniskill",
        stressor_context={"condition_id": "occlusion_s05"},
    )
    emitter = ledger.emitter(
        "human_annotation", "reviewer-01", annotation_source="expert_review"
    )
    event = emitter.emit(
        FailureEventDraft(
            event_id="event-reviewed",
            role="candidate_cause",
            category="perception",
            subtype="camera_occlusion",
            onset_step=4,
            end_step=7,
            temporal_precision="step_interval",
            confidence=0.8,
            evidence=(
                _evidence("visible", "policy_observable", 4),
                _evidence("state", "privileged", 4),
                _evidence("note", "external", 5),
            ),
            recovery_eligibility="eligible",
        )
    )

    payload = ledger.snapshot().to_dict()
    restored = FailureLedgerRecord.from_dict(payload)

    assert payload["format"] == FAILURE_LEDGER_FORMAT
    assert payload["events"][0]["format"] == FAILURE_EVENT_FORMAT
    assert len(payload["events"][0]["evidence"]["policy_observable"]) == 1
    assert len(payload["events"][0]["evidence"]["privileged"]) == 1
    assert len(payload["events"][0]["evidence"]["external"]) == 1
    assert event.active_stressor_context["condition_id"] == "occlusion_s05"
    assert restored == ledger.snapshot()


def test_ledger_orders_allows_overlap_and_deduplicates_only_explicit_matches():
    ledger = FailureEventLedger(
        task_id="task",
        episode_index=0,
        episode_seed=0,
        engine_name="mujoco",
    )
    emitter = ledger.emitter("simulator_state", "unit-engine")
    emitter.emit(
        FailureEventDraft(
            event_id="late",
            role="symptom",
            category="interaction",
            subtype="slip",
            onset_step=8,
        )
    )
    emitter.emit(
        FailureEventDraft(
            event_id="stall-a",
            role="mechanism",
            category="control",
            subtype="stall",
            onset_step=2,
            end_step=4,
            temporal_precision="step_interval",
            confidence=0.4,
            evidence=(_evidence("stall-a", "privileged", 2),),
            deduplication_key="control-stall",
        )
    )
    merged = emitter.emit(
        FailureEventDraft(
            event_id="stall-b",
            role="mechanism",
            category="control",
            subtype="stall",
            onset_step=4,
            end_step=6,
            temporal_precision="step_interval",
            confidence=0.9,
            evidence=(_evidence("stall-b", "privileged", 4),),
            deduplication_key="control-stall",
        )
    )
    emitter.emit(
        FailureEventDraft(
            event_id="stall-recurs",
            role="mechanism",
            category="control",
            subtype="stall",
            onset_step=7,
            deduplication_key="control-stall",
        )
    )

    assert [event.event_id for event in ledger.events] == [
        "stall-a",
        "stall-recurs",
        "late",
    ]
    assert (merged.onset_step, merged.end_step, merged.confidence) == (2, 6, 0.9)
    assert {item.evidence_id for item in merged.evidence} == {"stall-a", "stall-b"}
    assert ledger.snapshot().to_dict()["overlap_semantics"] == (
        "allowed_without_implied_causality"
    )


def test_ledger_rejects_invalid_confidence_and_unknown_causal_parents():
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        FailureEventDraft(
            role="symptom",
            category="interaction",
            subtype="slip",
            onset_step=1,
            confidence=1.2,
        )

    ledger = FailureEventLedger(
        task_id="task",
        episode_index=0,
        episode_seed=0,
        engine_name="mujoco",
    )
    ledger.emitter("human_annotation", "reviewer").emit(
        FailureEventDraft(
            event_id="child",
            role="symptom",
            category="interaction",
            subtype="slip",
            onset_step=1,
            causal_hypotheses=(CausalHypothesis("missing-parent"),),
        )
    )

    with pytest.raises(ValueError, match="unknown candidate parents"):
        ledger.snapshot()


def test_generated_event_ids_do_not_collide_with_explicit_ids():
    ledger = FailureEventLedger(
        task_id="task",
        episode_index=0,
        episode_seed=0,
        engine_name="mujoco",
    )
    emitter = ledger.emitter("task_logic", "unit-task")
    emitter.emit(
        FailureEventDraft(
            event_id="event-000001",
            role="symptom",
            category="interaction",
            subtype="first",
            onset_step=0,
        )
    )
    generated = emitter.emit(
        FailureEventDraft(
            role="symptom",
            category="interaction",
            subtype="second",
            onset_step=1,
        )
    )

    assert generated.event_id == "event-000002"


def test_multi_stage_fixture_preserves_hypotheses_without_causal_assertions():
    fixture = yaml.safe_load(
        (FIXTURES / "multi_stage_failure_events.yaml").read_text(encoding="utf-8")
    )
    ledger = FailureEventLedger(
        task_id="pick_cube",
        episode_index=0,
        episode_seed=0,
        engine_name="maniskill",
        stressor_context={"condition_id": "camera_occlusion_s05"},
    )
    emitter = ledger.emitter(
        "human_annotation", "multi-stage-fixture", annotation_source="curated_fixture"
    )
    emitter.emit_many(fixture["events"], default_step=0)

    record = ledger.snapshot()
    payload = record.to_dict()

    assert [event.subtype for event in record.events] == [
        "camera_occlusion",
        "grasp_pose_offset",
        "object_slip",
        "missed_target",
    ]
    assert record.events[1].causal_hypotheses[0].parent_event_id == "occlusion"
    assert payload["causal_semantics"] == "candidate_hypotheses_only"
    assert all(
        hypothesis["semantics"] == "hypothesis_only_not_established_causality"
        for event in payload["events"]
        for hypothesis in event["causal_hypotheses"]
    )
    assert derive_failure_label(record, fallback=None) == "missed_target"


def test_legacy_flat_failure_label_migrates_without_changing_summary():
    legacy = {
        "task_id": "pick_cube",
        "episode_index": 2,
        "seed": 9,
        "success": False,
        "failure_label": "object_slip",
        "failure_label_source": "mapper",
        "steps": [{}, {}, {}],
    }

    record = failure_ledger_from_episode_dict(legacy, engine_name="maniskill")

    assert record is not None
    assert record.events[0].subtype == "object_slip"
    assert record.events[0].temporal_precision == "terminal_only"
    assert record.events[0].provenance.source == "legacy_mapper"
    assert (
        derive_failure_label(record, fallback=legacy["failure_label"])
        == (legacy["failure_label"])
    )

    environment_labeled = {
        **legacy,
        "failure_label": "wrong_object",
        "failure_label_source": "env",
    }
    environment_record = failure_ledger_from_episode_dict(environment_labeled)
    assert environment_record is not None
    assert environment_record.events[0].provenance.source == "task_logic"


def test_cli_loader_migrates_an_old_result_pack_without_changing_flat_fields(
    tmp_path: Path,
):
    from nyssa_bench.cli import _load_episodes

    legacy = {
        "task_id": "pick_cube",
        "episode_index": 1,
        "seed": 4,
        "success": False,
        "failure_label": "bad_grasp",
        "failure_label_source": "env",
        "metrics": {},
        "steps": [],
    }
    (tmp_path / "episodes.json").write_text(json.dumps([legacy]), encoding="utf-8")
    (tmp_path / "run.yaml").write_text("engine_name: maniskill\n", encoding="utf-8")

    episode = _load_episodes(tmp_path)[0]

    assert episode.failure_label == "bad_grasp"
    assert episode.failure_label_source == "env"
    assert episode.failure_ledger is not None
    assert episode.failure_ledger.engine_name == "maniskill"
    assert episode.failure_ledger.events[0].summary_label == "bad_grasp"


class _FailureEventEngine(NyssaEngine):
    max_steps = 1

    def __init__(self) -> None:
        self._events: list[FailureEventDraft | dict[str, Any]] = []

    def load_task(self, task_spec: Any) -> None:
        self.task_spec = task_spec

    def reset(self, seed: int | None = None):
        self._events = [
            {
                "role": "contributing_condition",
                "category": "dynamics",
                "subtype": "initial_instability",
                "onset_step": 0,
            }
        ]
        return _observation(), {"seed": seed}

    def step(self, action: Any):
        return (
            _observation(),
            0.0,
            True,
            False,
            {
                "success": False,
                "object_slip": True,
                "failure_events": [
                    {
                        "role": "symptom",
                        "category": "interaction",
                        "subtype": "object_slip",
                        "onset_step": 0,
                        "confidence": 1.0,
                        "recovery_eligibility": "eligible",
                    }
                ],
            },
        )

    def drain_failure_events(self):
        events, self._events = self._events, []
        return events

    def render(self):
        return None

    def get_state(self):
        return {}

    def close(self) -> None:
        return None


class _QueuedEventPolicy:
    def reset(self, task: Any | None = None, seed: int | None = None) -> None:
        self._events = [
            {
                "role": "candidate_cause",
                "category": "policy",
                "subtype": "low_action_confidence",
                "onset_step": 0,
                "confidence": 0.6,
            }
        ]

    def act(self, observation: dict[str, Any]):
        return [0.5]

    def drain_failure_events(self):
        events, self._events = self._events, []
        return events


class _QueuedEventExpert(ExpertProvider):
    provider_id = "queued-event-expert"

    def reset(self, **kwargs: Any) -> None:
        self._events: list[dict[str, Any]] = []

    def score_action(self, observation: dict[str, Any], action: Any, **kwargs: Any):
        self._events.append(
            {
                "role": "candidate_cause",
                "category": "verification",
                "subtype": "verifier_uncertainty",
                "onset_step": 0,
                "confidence": 0.7,
            }
        )
        return ExpertActionScore(
            accepted=False,
            confidence=0.9,
            reason="unsafe_action",
        )

    def recover(self, **kwargs: Any):
        self._events.append(
            {
                "role": "contributing_condition",
                "category": "recovery",
                "subtype": "planner_recovery_candidate",
                "onset_step": 0,
            }
        )
        return [[0.25]]

    def drain_failure_events(self):
        events, self._events = self._events, []
        return events

    def metadata(self):
        return {"provider_id": self.provider_id, "capabilities": ["score", "recover"]}


def test_runner_collects_component_events_and_writes_renderable_timeline(
    tmp_path: Path,
):
    get_plugin_registry().engines["failure_event_unit"] = _FailureEventEngine
    runner = PolicyRunner(
        policy=_QueuedEventPolicy(),
        engine="failure_event_unit",
        episodes=1,
        out=tmp_path,
        capture_replay=False,
        expert_provider=_QueuedEventExpert(),
        enable_verifier=True,
        enable_recovery=True,
        stressor_config=StressorConfig(
            condition_id="delay_s1",
            stressors=(StressorSpec("action_delay", 1.0, {"max_delay_steps": 1}),),
        ),
    )

    report = runner.evaluate(
        Suite.load("maniskill_smoke_v0").filter_tasks(["maniskill_pick_cube"])
    )

    episode = runner.episode_results[0]
    assert episode.failure_label == "object_slip"
    assert episode.failure_label_source == "mapper"
    assert episode.failure_ledger is not None
    sources = {event.provenance.source for event in episode.failure_ledger.events}
    assert {
        "simulator_state",
        "policy_output",
        "verifier_output",
        "stressor",
        "recovery",
        "legacy_mapper",
    } <= sources
    assert all(
        event.active_stressor_context["condition_id"] == "delay_s1"
        for event in episode.failure_ledger.events
    )
    assert report.summary["failure_event_summary"]["event_count"] >= 8
    assert (tmp_path / "failure_ledger.json").is_file()
    assert "failure_events" not in episode.steps[0].info
    assert episode.steps[0].info["engine_failure_event_ids"]
    assert episode.steps[0].info["failure_event_ids"]
    replay = json.loads((tmp_path / "replay_manifest.json").read_text())
    assert replay["episodes"][0]["failure_ledger"]["events"]
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "Failure Timelines" in html
    assert "object_slip" in html
    assert "privileged:terminal_failure_classification" in html
    dataset_manifest = json.loads(
        (tmp_path / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    assert "failure_ledger.json" in dataset_manifest["artifacts"]
    timeline = episode_timeline(episode)
    assert timeline[0]["failure_events"]


def _evidence(evidence_id: str, visibility: str, step: int) -> FailureEvidence:
    return FailureEvidence(
        evidence_id=evidence_id,
        evidence_type="unit_evidence",
        payload={"value": evidence_id},
        source="unit_test",
        annotation_source="unit_test",
        confidence=1.0,
        visibility=visibility,  # type: ignore[arg-type]
        captured_step=step,
    )


def _observation() -> dict[str, Any]:
    return {
        "raw": np.asarray([0.0], dtype=np.float32),
        "action_space": {
            "type": "box",
            "shape": [1],
            "low": [-1.0],
            "high": [1.0],
            "dtype": "float32",
        },
    }
