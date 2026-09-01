import json
from collections.abc import Mapping
from typing import Any

import pytest

from nyssa_bench.failures import (
    FailureEvidence,
    FailureEventDraft,
    FailureEventLedger,
)
from nyssa_bench.failures.detectors import (
    ContactDetector,
    DetectorSignalRequirement,
    FailureDetector,
    FailureDetectorContract,
    FailureDetectorManager,
    FailureDetectorRuntimeError,
    GraspDetector,
    StallDetector,
    summarize_failure_detectors,
    write_failure_detector_manifest,
)


def _draft(
    *,
    category: str = "test",
    subtype: str = "unit",
    step: int = 0,
) -> FailureEventDraft:
    return FailureEventDraft(
        role="symptom",
        category=category,
        subtype=subtype,
        onset_step=step,
        confidence=1.0,
        evidence=(
            FailureEvidence(
                evidence_id=f"{category}:{step}",
                evidence_type="test",
                payload={},
                source="test_detector",
                annotation_source="unit_test",
                confidence=1.0,
                visibility="privileged",
                captured_step=step,
            ),
        ),
    )


def _draft_payload(
    *, category: str = "test", subtype: str = "unit", step: int = 0
) -> dict[str, Any]:
    evidence = {
        "policy_observable": [
            {
                "format": "nyssa-failure-evidence-v1",
                "evidence_id": f"{category}:{step}",
                "evidence_type": "test",
                "payload": {},
                "source": "test_detector",
                "annotation_source": "unit_test",
                "confidence": 1.0,
                "visibility": "policy_observable",
                "captured_step": step,
            }
        ],
        "privileged": [],
        "external": [],
    }
    return {
        "role": "symptom",
        "category": category,
        "subtype": subtype,
        "onset_step": step,
        "confidence": 1.0,
        "evidence": evidence,
    }


class _ProbeDetector(FailureDetector):
    detector_id = "probe_detector"

    def detect(
        self,
        *,
        step_index: int,
        observation: Mapping[str, Any] | None,
        action: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Mapping[str, Any],
        task: Any,
        engine: Any,
        stressor_context: Mapping[str, Any] | None = None,
    ) -> list[FailureEventDraft | dict[str, Any]]:
        return [
            _draft(step=step_index),
            {"event_id": f"raw:{step_index}", **_draft_payload(step=step_index)},
        ]


def test_detector_manager_flattens_and_collects_payloads():
    ledger = FailureEventLedger(
        task_id="unit-task",
        episode_index=0,
        episode_seed=0,
        engine_name="unit",
    )
    manager = FailureDetectorManager(detectors=(_ProbeDetector(),), engine_name="unit")
    manager.reset(
        task=None,
        engine=None,
        observation={},
        stressor_context=None,
        reset_info={},
    )

    payloads = manager.detect(
        step_index=3,
        observation={},
        action=None,
        reward=0.0,
        terminated=False,
        truncated=False,
        info={},
        task=None,
        engine=None,
    )

    events = manager.emit(ledger, payloads, default_step=3)
    assert len(payloads) == 2
    assert len(events) == 2
    assert events[0].event_id != events[1].event_id
    assert {event.provenance.component_id for event in events} == {"probe_detector"}


def test_detector_contract_is_versioned_and_records_configuration():
    contract = ContactDetector(collision_threshold=2.5).contract().to_dict()

    assert contract["format"] == "nyssa-failure-detector-v1"
    assert contract["protocol_version"] == 1
    assert contract["detector_version"] == "1.0.0"
    assert contract["mode"] == "passive"
    assert contract["configuration"] == {"collision_threshold": 2.5}
    assert contract["signal_requirements"][0]["visibility"] == "privileged"


def test_detector_contract_rejects_nonportable_configuration():
    with pytest.raises(ValueError, match="JSON-compatible"):
        FailureDetectorContract(
            detector_id="invalid_config",
            detector_version="1.0.0",
            configuration={"callback": lambda: None},
        )


def test_manager_activates_detector_when_runtime_signal_appears():
    manager = FailureDetectorManager(
        detectors=(ContactDetector(),), engine_name="third_party"
    )
    manager.reset(
        task=None,
        engine=None,
        observation={},
        stressor_context=None,
        reset_info={},
    )
    assert manager.manifest()["detectors"][0]["support"]["status"] == "pending"

    emissions = manager.detect(
        step_index=0,
        observation={},
        action=None,
        reward=0.0,
        terminated=False,
        truncated=False,
        info={"collision": True},
        task=None,
        engine=None,
    )

    assert len(emissions) == 1
    assert manager.manifest()["detectors"][0]["support"]["status"] == "supported"


def test_detector_manager_preserves_lifecycle_order():
    calls: list[str] = []

    class LifecycleDetector(FailureDetector):
        detector_id = "lifecycle_detector"

        def reset(self, **kwargs: Any) -> None:
            calls.append("reset")

        def observe_before_action(self, **kwargs: Any):
            calls.append("before")

        def observe_after_action(self, **kwargs: Any):
            calls.append("after")

        def detect(self, **kwargs: Any):
            calls.append("detect")
            return []

        def finalize(self, **kwargs: Any):
            calls.append("finalize")
            return []

    manager = FailureDetectorManager(
        detectors=(LifecycleDetector(),), engine_name="unit"
    )
    manager.reset(
        task=None,
        engine=None,
        observation={},
        stressor_context=None,
        reset_info={},
    )
    manager.observe_before_action(
        step_index=0,
        observation={},
        action=None,
        task=None,
        engine=None,
    )
    manager.observe_after_action(
        step_index=0,
        pre_observation={},
        post_observation={},
        action=None,
        reward=0.0,
        terminated=False,
        truncated=False,
        info={},
        task=None,
        engine=None,
    )
    manager.detect(
        step_index=0,
        observation={},
        action=None,
        reward=0.0,
        terminated=False,
        truncated=False,
        info={},
        task=None,
        engine=None,
    )
    manager.finalize(
        step_index=0,
        final_observation={},
        reward=0.0,
        terminated=False,
        truncated=False,
        success=True,
        info={},
        task=None,
        engine=None,
    )

    assert calls == ["reset", "before", "after", "detect", "finalize"]


def test_default_detector_instances_do_not_share_episode_state():
    first = ContactDetector()
    second = ContactDetector()
    first.reset(task=None, engine=None, observation={}, stressor_context=None)
    second.reset(task=None, engine=None, observation={}, stressor_context=None)

    assert first.detect(
        step_index=0,
        observation={},
        action=None,
        reward=0.0,
        terminated=False,
        truncated=False,
        info={"collision": True},
        task=None,
        engine=None,
    )
    assert second.detect(
        step_index=0,
        observation={},
        action=None,
        reward=0.0,
        terminated=False,
        truncated=False,
        info={"collision": True},
        task=None,
        engine=None,
    )


def test_manager_skips_incompatible_detector_without_calling_reset():
    class ManiSkillOnly(_ProbeDetector):
        detector_id = "maniskill_only"
        supported_engines = ("maniskill",)

        def __init__(self) -> None:
            self.reset_called = False

        def reset(self, **kwargs: Any) -> None:
            self.reset_called = True

    detector = ManiSkillOnly()
    manager = FailureDetectorManager(detectors=(detector,), engine_name="mujoco")
    manager.reset(
        task=None,
        engine=None,
        observation={},
        stressor_context=None,
        reset_info={},
    )

    entry = manager.manifest()["detectors"][0]
    assert entry["support"]["status"] == "unsupported"
    assert "engine 'mujoco'" in entry["support"]["reason"]
    assert detector.reset_called is False


def test_passive_detector_cannot_mutate_info_payload():
    class MutatingDetector(FailureDetector):
        detector_id = "mutating_detector"

        def detect(self, **kwargs: Any):
            kwargs["info"]["changed"] = True
            return []

    manager = FailureDetectorManager(
        detectors=(MutatingDetector(),), engine_name="unit"
    )
    manager.reset(
        task=None,
        engine=None,
        observation={},
        stressor_context=None,
        reset_info={},
    )
    info: dict[str, Any] = {"original": True}

    with pytest.raises(FailureDetectorRuntimeError, match="mutating_detector.*detect"):
        manager.detect(
            step_index=0,
            observation={},
            action=None,
            reward=0.0,
            terminated=False,
            truncated=False,
            info=info,
            task=None,
            engine=None,
        )
    assert info == {"original": True}


def test_instrumented_detector_declares_new_capability():
    class InstrumentedDetector(_ProbeDetector):
        detector_id = "instrumented_detector"
        mode = "instrumented"
        signal_requirements = (
            DetectorSignalRequirement(
                any_of=("engine.instrumented",), visibility="privileged"
            ),
        )

        def request_instrumentation(self, *, task: Any, engine: Any) -> set[str]:
            return {"engine.instrumented"}

    manager = FailureDetectorManager(
        detectors=(InstrumentedDetector(),), engine_name="unit"
    )
    manager.reset(
        task=None,
        engine=None,
        observation={},
        stressor_context=None,
        reset_info={},
    )

    assert manager.manifest()["detectors"][0]["support"]["status"] == "supported"


def test_overlapping_detector_events_keep_distinct_provenance():
    class OtherProbe(_ProbeDetector):
        detector_id = "other_probe"

        def detect(self, **kwargs: Any):
            step_index = int(kwargs["step_index"])
            return [
                _draft(step=step_index),
                {
                    "event_id": f"other:raw:{step_index}",
                    **_draft_payload(step=step_index),
                },
            ]

    manager = FailureDetectorManager(
        detectors=(_ProbeDetector(), OtherProbe()), engine_name="unit"
    )
    manager.reset(
        task=None,
        engine=None,
        observation={},
        stressor_context=None,
        reset_info={},
    )
    emissions = manager.detect(
        step_index=0,
        observation={},
        action=None,
        reward=0.0,
        terminated=False,
        truncated=False,
        info={},
        task=None,
        engine=None,
    )
    ledger = FailureEventLedger(
        task_id="unit-task", episode_index=0, episode_seed=0, engine_name="unit"
    )
    manager.emit(ledger, emissions, default_step=0)

    assert len(ledger.events) == 4
    assert {event.provenance.component_id for event in ledger.events} == {
        "probe_detector",
        "other_probe",
    }


def test_contact_detector_tracks_collision_edges_and_recovery():
    detector = ContactDetector()
    detector.reset(task=None, engine=None, observation={}, stressor_context=None)

    first = detector.detect(
        step_index=0,
        observation={},
        action=None,
        reward=0.0,
        terminated=False,
        truncated=False,
        info={"collision_count": 1},
        task=None,
        engine=None,
    )
    second = detector.detect(
        step_index=1,
        observation={},
        action=None,
        reward=0.0,
        terminated=False,
        truncated=False,
        info={"collision_count": 2},
        task=None,
        engine=None,
    )
    reset_like = detector.detect(
        step_index=2,
        observation={},
        action=None,
        reward=0.0,
        terminated=False,
        truncated=False,
        info={"collision": False},
        task=None,
        engine=None,
    )
    third = detector.detect(
        step_index=3,
        observation={},
        action=None,
        reward=0.0,
        terminated=False,
        truncated=False,
        info={"collision": True},
        task=None,
        engine=None,
    )

    assert len(first) == 1
    assert len(second) == 0
    assert len(reset_like) == 0
    assert len(third) == 1


def test_grasp_detector_emits_distinct_mechanisms_once():
    detector = GraspDetector()
    detector.reset(task=None, engine=None, observation={}, stressor_context=None)

    events = []
    events.extend(
        detector.detect(
            step_index=0,
            observation={},
            action=None,
            reward=0.0,
            terminated=False,
            truncated=False,
            info={
                "wrong_object_selected": True,
                "object_slip": True,
                "bad_grasp": True,
            },
            task=None,
            engine=None,
        )
    )
    events.extend(
        detector.detect(
            step_index=1,
            observation={},
            action=None,
            reward=0.0,
            terminated=False,
            truncated=False,
            info={
                "wrong_object_selected": True,
                "object_slip": True,
                "bad_grasp": True,
            },
            task=None,
            engine=None,
        )
    )
    subtypes = {
        event.subtype for event in events if isinstance(event, FailureEventDraft)
    }
    assert subtypes == {"wrong_object", "object_slip", "bad_grasp"}


def test_grasp_detector_localizes_contact_loss_after_confirmed_grasp():
    detector = GraspDetector()
    detector.reset(task=None, engine=None, observation={}, stressor_context=None)

    held = detector.detect(
        step_index=4,
        observation={},
        action=None,
        reward=0.0,
        terminated=False,
        truncated=False,
        info={"is_grasped": True},
        task=None,
        engine=None,
    )
    lost = detector.detect(
        step_index=5,
        observation={},
        action=None,
        reward=0.0,
        terminated=False,
        truncated=False,
        info={"is_grasped": False},
        task=None,
        engine=None,
    )

    assert held == []
    assert len(lost) == 1
    assert lost[0].subtype == "object_slip"
    assert lost[0].evidence[0].payload["contact_lost"] is True


def test_stall_detector_detects_reward_plateau():
    detector = StallDetector(stall_window=2, reward_tolerance=1e-6, min_steps=2)
    detector.reset(task=None, engine=None, observation={}, stressor_context=None)

    assert (
        detector.detect(
            step_index=0,
            observation={},
            action=None,
            reward=1.0,
            terminated=False,
            truncated=False,
            info={},
            task=None,
            engine=None,
        )
        == []
    )
    assert (
        detector.detect(
            step_index=1,
            observation={},
            action=None,
            reward=1.00000000001,
            terminated=False,
            truncated=False,
            info={},
            task=None,
            engine=None,
        )
        == []
    )
    events = detector.detect(
        step_index=2,
        observation={},
        action=None,
        reward=1.00000000001,
        terminated=False,
        truncated=False,
        info={},
        task=None,
        engine=None,
    )
    assert len(events) == 1
    assert isinstance(events[0], FailureEventDraft)
    assert events[0].subtype == "planner_stuck"


def test_stall_detector_finalize_returns_stuck_if_plateau_unresolved():
    detector = StallDetector(stall_window=2, reward_tolerance=1e-6, min_steps=2)
    detector.reset(task=None, engine=None, observation={}, stressor_context=None)
    detector._stagnant_steps = 2
    detector._last_reward = 1.0
    events = detector.finalize(
        step_index=3,
        final_observation={},
        reward=1.0,
        terminated=True,
        truncated=False,
        success=False,
        info={},
        task=None,
        engine=None,
    )
    assert len(events) == 1
    assert events[0].role == "mechanism"
    assert events[0].onset_step == 2
    assert events[0].end_step == 3
    assert events[0].evidence[0].captured_step == 3


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ContactDetector(collision_threshold=-1),
        lambda: ContactDetector(collision_threshold=float("nan")),
        lambda: StallDetector(stall_window=0),
        lambda: StallDetector(min_steps=0),
        lambda: StallDetector(reward_tolerance=float("inf")),
    ],
)
def test_detector_configuration_rejects_invalid_values(factory):
    with pytest.raises(ValueError):
        factory()


def test_manager_rejects_duplicate_detector_ids():
    with pytest.raises(ValueError, match="IDs must be unique"):
        FailureDetectorManager(detectors=(_ProbeDetector(), _ProbeDetector()))


def test_pending_detector_becomes_explicitly_unsupported_at_finalize():
    manager = FailureDetectorManager(detectors=(GraspDetector(),), engine_name="unit")
    manager.reset(
        task=None,
        engine=None,
        observation={},
        stressor_context=None,
        reset_info={},
    )
    manager.finalize(
        step_index=0,
        final_observation={},
        reward=0.0,
        terminated=False,
        truncated=True,
        success=False,
        info={},
        task=None,
        engine=None,
    )

    support = manager.manifest()["detectors"][0]["support"]
    assert support["status"] == "unsupported"
    assert "not observed" in support["reason"]
    with pytest.raises(RuntimeError, match="already been finalized"):
        manager.finalize(
            step_index=0,
            final_observation={},
            reward=0.0,
            terminated=False,
            truncated=True,
            success=False,
            info={},
            task=None,
            engine=None,
        )


def test_failure_detector_manifest_preserves_contracts_and_counts(tmp_path):
    class Episode:
        task_id = "task"
        episode_index = 0
        seed = 7
        failure_detector_context = {
            "format": "nyssa-failure-detector-manifest-v1",
            "detectors": [
                {
                    "contract": ContactDetector().contract().to_dict(),
                    "support": {
                        "status": "supported",
                        "available_signals": ["info.collision"],
                        "missing_requirements": [],
                        "reason": None,
                    },
                    "emitted_event_count": 2,
                }
            ],
        }

    episodes = [Episode()]
    summary = summarize_failure_detectors(episodes)
    path = write_failure_detector_manifest(episodes, tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert summary["status_counts"] == {"supported": 1}
    assert summary["emitted_event_counts"] == {"contact_detector": 2}
    assert payload["summary"] == summary
    assert payload["episodes"][0]["seed"] == 7
