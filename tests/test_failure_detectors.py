from typing import Any

from nyssa_bench.failures import (
    FailureEvidence,
    FailureEventDraft,
    FailureEventLedger,
)
from nyssa_bench.failures.detectors import (
    ContactDetector,
    FailureDetector,
    FailureDetectorManager,
    GraspDetector,
    StallDetector,
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
        observation: dict[str, Any] | None,
        action: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
        task: Any,
        engine: Any,
        stressor_context: dict[str, Any] | None = None,
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
    emitter = ledger.emitter("simulator_state", "unit-detector")
    manager = FailureDetectorManager(detectors=(_ProbeDetector(),))

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

    events = emitter.emit_many(payloads, default_step=3)
    assert len(payloads) == 2
    assert len(events) == 2
    assert events[0].event_id != events[1].event_id


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
