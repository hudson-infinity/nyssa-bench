from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from nyssa_bench import PolicyRunner, Suite
from nyssa_bench.cli import main
from nyssa_bench.engines.base import NyssaEngine
from nyssa_bench.experts import ExpertActionScore, ExpertProvider
from nyssa_bench.failures.protocol import (
    EventProvenance,
    FailureEvent,
    FailureLedgerRecord,
)
from nyssa_bench.monitors import (
    ActionMagnitudeFailureMonitor,
    FailureMonitor,
    FailureMonitorContract,
    FailureMonitorManager,
    MonitorInput,
    MonitorInputSpec,
    MonitorOutcome,
    MonitorPrediction,
    MonitorPredictionRecord,
    compare_monitor_records,
    contract_sha256,
    load_failure_monitor,
    load_monitor_manifest,
    prediction_id,
    summarize_monitor_records,
    write_monitor_manifest,
)
from nyssa_bench.plugins import get_plugin_registry
from nyssa_bench.policies.base import Policy


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _contract(
    monitor_id: str = "unit_monitor",
    *,
    inputs: tuple[MonitorInputSpec, ...] | None = None,
    outputs: tuple[str, ...] = ("failure_risk",),
    horizon: int | None = 3,
    state_semantics: str = "stateless",
) -> FailureMonitorContract:
    return FailureMonitorContract(
        monitor_id=monitor_id,
        monitor_version="1.2.3",
        inputs=inputs
        or (
            MonitorInputSpec(
                input_id="action",
                source="proposed_action",
                visibility="policy_observable",
            ),
        ),
        outputs=outputs,
        checkpoint_id=f"{monitor_id}-checkpoint",
        checkpoint_sha256=_digest(f"{monitor_id}-checkpoint"),
        preprocessing_sha256=_digest(f"{monitor_id}-preprocessing"),
        state_semantics=state_semantics,  # type: ignore[arg-type]
        deterministic=True,
        prediction_horizon_steps=horizon,
        alert_threshold=0.5,
        calibration_bins=5,
    )


def _prediction(
    contract: FailureMonitorContract,
    *,
    step: int,
    risk: float,
    episode_index: int = 0,
    seed: int = 7,
    category: str | None = None,
    mechanism: str | None = None,
    event_ids: tuple[str, ...] = (),
    action_timestamp: int | None = None,
) -> MonitorPrediction:
    return MonitorPrediction(
        prediction_id=prediction_id(
            contract,
            task_id="task",
            episode_seed=seed,
            episode_index=episode_index,
            step_index=step,
        ),
        monitor_id=contract.monitor_id,
        contract_sha256=contract_sha256(contract),
        task_id="task",
        episode_index=episode_index,
        episode_seed=seed,
        environment_step=step,
        observation_timestamp=step,
        policy_action_timestamp=step
        if action_timestamp is None
        else action_timestamp,
        failure_risk=risk,
        failure_category=category,
        failure_mechanism=mechanism,
        failure_event_ids_before_prediction=event_ids,
    )


def _event(
    *,
    step: int = 2,
    event_id: str = "event-1",
    role: str = "symptom",
    category: str = "interaction",
    subtype: str = "object_slip",
) -> FailureEvent:
    return FailureEvent(
        event_id=event_id,
        role=role,  # type: ignore[arg-type]
        category=category,
        subtype=subtype,
        onset_step=step,
        end_step=None,
        temporal_precision="exact_step",
        confidence=1.0,
        evidence=(),
        provenance=EventProvenance(
            source="simulator_state",
            component_id="unit",
            annotation_source="unit_test",
        ),
        recovery_eligibility="eligible",
    )


def _ledger(*events: FailureEvent) -> FailureLedgerRecord:
    return FailureLedgerRecord(
        task_id="task",
        episode_index=0,
        episode_seed=7,
        engine_name="unit",
        events=tuple(events),
    )


class _FixedMonitor(FailureMonitor):
    def __init__(
        self,
        contract: FailureMonitorContract,
        *,
        risk: float = 0.75,
        wrong_event_ids: bool = False,
    ) -> None:
        self._contract = contract
        self.risk = risk
        self.wrong_event_ids = wrong_event_ids
        self.reset_calls: list[tuple[str, int, int]] = []

    def contract(self) -> FailureMonitorContract:
        return self._contract

    def reset(self, *, task: Any, episode_index: int, seed: int) -> None:
        self.reset_calls.append((task.task_id, episode_index, seed))

    def predict(self, monitor_input: MonitorInput) -> MonitorPrediction:
        return _prediction(
            self._contract,
            step=monitor_input.environment_step,
            risk=self.risk,
            episode_index=monitor_input.episode_index,
            seed=monitor_input.episode_seed,
            event_ids=("incorrect",)
            if self.wrong_event_ids
            else monitor_input.failure_event_ids_before_prediction,
            action_timestamp=monitor_input.policy_action_timestamp,
        )


class _MutatingMonitor(_FixedMonitor):
    def predict(self, monitor_input: MonitorInput) -> MonitorPrediction:
        monitor_input.proposed_action[0] = 99.0
        return super().predict(monitor_input)


def test_contract_round_trip_hash_and_observability_tiers() -> None:
    deployable = _contract()
    privileged = _contract(
        "privileged",
        inputs=(
            MonitorInputSpec(
                input_id="state",
                source="privileged_state",
                visibility="privileged",
            ),
        ),
    )

    assert FailureMonitorContract.from_dict(deployable.to_dict()) == deployable
    assert deployable.observability_tier == "deployable_monitor"
    assert privileged.observability_tier == "privileged_monitor"
    assert len(contract_sha256(deployable)) == 64

    optional_privileged = _contract(
        "optional_privileged",
        inputs=(
            MonitorInputSpec(
                input_id="action",
                source="proposed_action",
                visibility="policy_observable",
            ),
            MonitorInputSpec(
                input_id="optional_state",
                source="privileged_state",
                visibility="privileged",
                required=False,
            ),
        ),
    )
    assert optional_privileged.observability_tier == "privileged_monitor"


def test_contract_rejects_false_visibility_and_invalid_checkpoint_hash() -> None:
    with pytest.raises(ValueError, match="must use privileged"):
        MonitorInputSpec(
            input_id="state",
            source="privileged_state",
            visibility="policy_observable",
        )
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        replace(_contract(), checkpoint_sha256="invalid")


def test_outcome_rejects_labels_on_censored_or_negative_predictions() -> None:
    with pytest.raises(ValueError, match="cannot carry observed labels"):
        MonitorOutcome(
            status="censored",
            failure_within_horizon=None,
            eventual_episode_failure=True,
            failure_onset_step=None,
            failure_category=None,
            failure_mechanism=None,
            recovery_eligible=None,
        )
    with pytest.raises(ValueError, match="cannot contain failure attributes"):
        MonitorOutcome(
            status="observed",
            failure_within_horizon=False,
            eventual_episode_failure=True,
            failure_onset_step=None,
            failure_category="control",
            failure_mechanism=None,
            recovery_eligible=None,
        )


def test_manager_aligns_predictions_and_labels_failure_horizon() -> None:
    monitor = _FixedMonitor(_contract())
    manager = FailureMonitorManager((monitor,))
    task = type("Task", (), {"task_id": "task"})()
    manager.reset(task=task, episode_index=0, seed=7, policy=object(), engine=object())

    predictions = manager.predict(
        step_index=0,
        observation={"raw": [0.0]},
        proposed_action=[0.9],
        policy=object(),
        engine=object(),
        failure_event_ids=(),
    )
    records = manager.finalize(
        success=False,
        truncated=False,
        episode_steps=3,
        failure_ledger=_ledger(_event()),
    )

    assert monitor.reset_calls == [("task", 0, 7)]
    assert predictions[0].environment_step == 0
    assert predictions[0].latency_ms >= 0.0
    assert records[0].outcome.failure_within_horizon is True
    assert records[0].outcome.failure_onset_step == 2
    assert records[0].outcome.failure_category == "interaction"


def test_manager_censors_incomplete_horizon_and_rejects_evidence_mismatch() -> None:
    task = type("Task", (), {"task_id": "task"})()
    manager = FailureMonitorManager((_FixedMonitor(_contract(horizon=5)),))
    manager.reset(task=task, episode_index=0, seed=7, policy=object(), engine=object())
    manager.predict(
        step_index=0,
        observation={},
        proposed_action=[0.0],
        policy=object(),
        engine=object(),
    )
    records = manager.finalize(
        success=False,
        truncated=True,
        episode_steps=2,
        failure_ledger=_ledger(),
    )
    assert records[0].outcome.status == "censored"

    invalid = FailureMonitorManager(
        (_FixedMonitor(_contract(), wrong_event_ids=True),)
    )
    invalid.reset(task=task, episode_index=0, seed=7, policy=object(), engine=object())
    with pytest.raises(ValueError, match="inputs do not match"):
        invalid.predict(
            step_index=0,
            observation={},
            proposed_action=[0.0],
            policy=object(),
            engine=object(),
        )


def test_outcome_excludes_failure_events_known_before_prediction() -> None:
    manager = FailureMonitorManager((_FixedMonitor(_contract()),))
    task = type("Task", (), {"task_id": "task"})()
    manager.reset(task=task, episode_index=0, seed=7, policy=object(), engine=object())
    manager.predict(
        step_index=2,
        observation={},
        proposed_action=[0.0],
        policy=object(),
        engine=object(),
        failure_event_ids=("event-1",),
    )

    records = manager.finalize(
        success=True,
        truncated=False,
        episode_steps=3,
        failure_ledger=_ledger(_event(step=2)),
    )

    assert records[0].prediction.failure_event_ids_before_prediction == ("event-1",)
    assert records[0].outcome.failure_within_horizon is False


def test_manager_marks_unavailable_privileged_input_unsupported() -> None:
    contract = _contract(
        inputs=(
            MonitorInputSpec(
                input_id="state",
                source="privileged_state",
                visibility="privileged",
            ),
        )
    )
    manager = FailureMonitorManager((_FixedMonitor(contract),))
    task = type("Task", (), {"task_id": "task"})()
    manager.reset(task=task, episode_index=0, seed=7, policy=object(), engine=object())

    assert manager.support[contract.monitor_id]["status"] == "unsupported"
    assert manager.predict(
        step_index=0,
        observation={},
        proposed_action=[0.0],
        policy=object(),
        engine=object(),
    ) == ()


def test_manager_snapshots_shared_privileged_inputs_once_per_step() -> None:
    inputs = (
        MonitorInputSpec(
            input_id="state",
            source="privileged_state",
            visibility="privileged",
        ),
    )
    manager = FailureMonitorManager(
        (
            _FixedMonitor(_contract("monitor_a", inputs=inputs)),
            _FixedMonitor(_contract("monitor_b", inputs=inputs)),
        )
    )
    task = type("Task", (), {"task_id": "task"})()

    class Engine:
        calls = 0

        def get_state(self) -> dict[str, int]:
            self.calls += 1
            return {"snapshot": self.calls}

    engine = Engine()
    manager.reset(task=task, episode_index=0, seed=7, policy=object(), engine=engine)
    manager.predict(
        step_index=0,
        observation={},
        proposed_action=[0.0],
        policy=object(),
        engine=engine,
    )

    assert engine.calls == 1


def test_manager_preserves_cached_policy_action_age() -> None:
    manager = FailureMonitorManager((_FixedMonitor(_contract()),))
    task = type("Task", (), {"task_id": "task"})()
    manager.reset(task=task, episode_index=0, seed=7, policy=object(), engine=object())

    prediction = manager.predict(
        step_index=2,
        policy_action_timestamp=0,
        observation={},
        proposed_action=[0.0],
        policy=object(),
        engine=object(),
    )[0]

    assert prediction.policy_action_timestamp == 0
    assert prediction.policy_action_age_steps == 2
    assert MonitorPrediction.from_dict(prediction.to_dict()) == prediction


def test_manager_isolates_external_monitor_inputs_from_live_action() -> None:
    manager = FailureMonitorManager((_MutatingMonitor(_contract()),))
    task = type("Task", (), {"task_id": "task"})()
    manager.reset(task=task, episode_index=0, seed=7, policy=object(), engine=object())
    action = [0.2]

    manager.predict(
        step_index=0,
        observation={},
        proposed_action=action,
        policy=object(),
        engine=object(),
    )

    assert action == [0.2]


def test_restorable_contract_requires_state_implementation() -> None:
    with pytest.raises(ValueError, match="does not implement get_state"):
        FailureMonitorManager(
            (_FixedMonitor(_contract(state_semantics="restorable")),)
        )


def test_reference_monitor_is_deterministic_integration_control() -> None:
    monitor = ActionMagnitudeFailureMonitor(alert_threshold=0.8)
    contract = monitor.contract()
    monitor_input = MonitorInput(
        task_id="task",
        episode_index=0,
        episode_seed=7,
        environment_step=0,
        observation_timestamp=0,
        policy_action_timestamp=0,
        observation=None,
        proposed_action=[-0.3, 0.9],
    )

    first = monitor.predict(monitor_input)
    second = monitor.predict(monitor_input)

    assert first == second
    assert first.failure_risk == 0.9
    assert first.intervention_recommended is True
    assert contract.declared_compute["kind"] == "analytic_reference"


def test_metrics_and_paired_comparison_keep_recovery_separate() -> None:
    contract_a = _contract("monitor_a")
    contract_b = _contract("monitor_b")
    contracts = {"monitor_a": contract_a, "monitor_b": contract_b}
    outcome_positive = MonitorOutcome(
        status="observed",
        failure_within_horizon=True,
        eventual_episode_failure=True,
        failure_onset_step=2,
        failure_category="interaction",
        failure_mechanism="object_slip",
        recovery_eligible=True,
    )
    outcome_negative = MonitorOutcome(
        status="observed",
        failure_within_horizon=False,
        eventual_episode_failure=False,
        failure_onset_step=None,
        failure_category=None,
        failure_mechanism=None,
        recovery_eligible=None,
    )
    records = [
        MonitorPredictionRecord(
            _prediction(contract_a, step=0, risk=0.9), outcome_positive
        ),
        MonitorPredictionRecord(
            _prediction(contract_b, step=0, risk=0.6), outcome_positive
        ),
        MonitorPredictionRecord(
            _prediction(contract_a, step=1, risk=0.2), outcome_negative
        ),
        MonitorPredictionRecord(
            _prediction(contract_b, step=1, risk=0.7), outcome_negative
        ),
    ]

    summary = summarize_monitor_records(records, contracts)
    comparison = compare_monitor_records(
        records, contracts, "monitor_a", "monitor_b"
    )

    assert summary["monitors"]["monitor_a"]["classification"] == {
        "true_positive": 1,
        "false_positive": 0,
        "true_negative": 1,
        "false_negative": 0,
        "precision": 1.0,
        "recall": 1.0,
        "false_alarm_rate": 0.0,
        "missed_failure_rate": 0.0,
    }
    assert comparison["matched_predictions"] == 2
    assert comparison["brier_delta_a_minus_b"] < 0.0
    assert comparison["recovery_effects_included"] is False


def test_monitor_metrics_cover_temporal_and_structured_predictions() -> None:
    contract = _contract(
        outputs=(
            "failure_risk",
            "failure_category",
            "failure_mechanism",
            "expected_time_to_failure",
            "recovery_eligibility",
        )
    )
    prediction = replace(
        _prediction(
            contract,
            step=0,
            risk=0.9,
            category="interaction",
            mechanism="object_slip",
        ),
        expected_time_to_failure=2.0,
        recovery_eligibility="eligible",
        latency_ms=4.0,
        compute={"flops": 100.0},
    )
    outcome = MonitorOutcome(
        status="observed",
        failure_within_horizon=True,
        eventual_episode_failure=True,
        failure_onset_step=2,
        failure_category="interaction",
        failure_mechanism="object_slip",
        recovery_eligible=True,
    )

    result = summarize_monitor_records(
        [MonitorPredictionRecord(prediction, outcome)],
        {contract.monitor_id: contract},
    )["monitors"][contract.monitor_id]

    assert result["lead_time"]["mean_steps"] == 2
    assert result["category_accuracy"] == 1.0
    assert result["mechanism_accuracy"] == 1.0
    assert result["expected_time_to_failure_mae"] == 0.0
    assert result["recovery_eligibility_accuracy"] == 1.0
    assert result["latency"]["mean_ms"] == 4.0
    assert result["compute"]["observed_numeric"]["flops"]["total"] == 100.0


def test_paired_comparison_rejects_missing_predictions() -> None:
    contract_a = _contract("monitor_a")
    contract_b = _contract("monitor_b")
    outcome = MonitorOutcome(
        status="observed",
        failure_within_horizon=False,
        eventual_episode_failure=False,
        failure_onset_step=None,
        failure_category=None,
        failure_mechanism=None,
        recovery_eligible=None,
    )
    records = [
        MonitorPredictionRecord(
            _prediction(contract_a, step=0, risk=0.1), outcome
        ),
        MonitorPredictionRecord(
            _prediction(contract_b, step=1, risk=0.1), outcome
        ),
    ]

    with pytest.raises(ValueError, match="identical prediction identities"):
        compare_monitor_records(
            records,
            {"monitor_a": contract_a, "monitor_b": contract_b},
            "monitor_a",
            "monitor_b",
        )


def test_monitor_manifest_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    contract = _contract()
    outcome = MonitorOutcome(
        status="observed",
        failure_within_horizon=False,
        eventual_episode_failure=False,
        failure_onset_step=None,
        failure_category=None,
        failure_mechanism=None,
        recovery_eligible=None,
    )
    record = MonitorPredictionRecord(_prediction(contract, step=0, risk=0.1), outcome)
    path = write_monitor_manifest(
        [record],
        {contract.monitor_id: contract},
        {contract.monitor_id: {"status": "supported"}},
        tmp_path / "monitor.json",
    )

    payload, contracts, records = load_monitor_manifest(path)
    assert payload["summary"]["prediction_count"] == 1
    assert contracts[contract.monitor_id] == contract
    assert records == (record,)

    changed = json.loads(path.read_text(encoding="utf-8"))
    changed["records"][0]["prediction"]["failure_risk"] = 0.2
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_monitor_manifest(path)


def test_compare_failure_monitors_cli_writes_paired_result(tmp_path: Path) -> None:
    contract_a = _contract("monitor_a")
    contract_b = _contract("monitor_b")
    outcome = MonitorOutcome(
        status="observed",
        failure_within_horizon=False,
        eventual_episode_failure=False,
        failure_onset_step=None,
        failure_category=None,
        failure_mechanism=None,
        recovery_eligible=None,
    )
    manifest = write_monitor_manifest(
        [
            MonitorPredictionRecord(
                _prediction(contract_a, step=0, risk=0.1), outcome
            ),
            MonitorPredictionRecord(
                _prediction(contract_b, step=0, risk=0.2), outcome
            ),
        ],
        {"monitor_a": contract_a, "monitor_b": contract_b},
        {
            "monitor_a": {"status": "supported"},
            "monitor_b": {"status": "supported"},
        },
        tmp_path / "failure_monitor_predictions.json",
    )
    out = tmp_path / "comparison.json"

    assert (
        main(
            [
                "compare-failure-monitors",
                str(manifest),
                "--monitor-a",
                "monitor_a",
                "--monitor-b",
                "monitor_b",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    assert json.loads(out.read_text(encoding="utf-8"))["matched_predictions"] == 1


def test_external_monitor_loader_requires_factory_contract(tmp_path: Path) -> None:
    module_path = tmp_path / "monitor_plugin.py"
    module_path.write_text(
        "from nyssa_bench.monitors import ActionMagnitudeFailureMonitor\n"
        "def create_failure_monitor():\n"
        "    return ActionMagnitudeFailureMonitor()\n",
        encoding="utf-8",
    )

    assert isinstance(load_failure_monitor(module_path), ActionMagnitudeFailureMonitor)
    with pytest.raises(FileNotFoundError, match="module not found"):
        load_failure_monitor(tmp_path / "missing.py")


class _MonitorIntegrationEngine(NyssaEngine):
    max_steps = 1

    def __init__(self) -> None:
        self.position = 0.0
        self.elapsed = 0

    def load_task(self, task_spec: Any) -> None:
        self.task_spec = task_spec

    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        self.position = 0.0
        self.elapsed = 0
        return self._observation(), {"seed": seed}

    def step(
        self, action: Any
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self.position += float(np.asarray(action).reshape(-1)[0])
        self.elapsed += 1
        success = self.position > 0.0
        return self._observation(), self.position, True, False, {"success": success}

    def render(self) -> Any:
        return None

    def get_state(self) -> dict[str, Any]:
        return {"position": self.position, "elapsed": self.elapsed}

    def set_state(self, state: Any) -> dict[str, Any]:
        self.position = float(state["position"])
        self.elapsed = int(state["elapsed"])
        return self._observation()

    def state_restore_capability(self) -> dict[str, Any]:
        return {
            "supported": True,
            "fidelity": "exact_unit_state",
            "captures_rng": False,
            "exact": True,
            "reason": None,
        }

    def close(self) -> None:
        return None

    def _observation(self) -> dict[str, Any]:
        return {
            "raw": [self.position],
            "action_space": {
                "type": "box",
                "shape": [1],
                "low": [-1.0],
                "high": [1.0],
            },
        }


class _MonitorIntegrationPolicy(Policy):
    def __init__(self) -> None:
        self.calls = 0

    def reset(self, task: Any | None = None, seed: int | None = None) -> None:
        self.calls = 0

    def act(self, observation: dict[str, Any]) -> list[float]:
        self.calls += 1
        return [-1.0]

    def get_state(self) -> dict[str, int]:
        return {"calls": self.calls}

    def set_state(self, state: Any) -> None:
        self.calls = int(state["calls"])

    def state_restore_capability(self) -> dict[str, Any]:
        return {
            "supported": True,
            "fidelity": "exact_unit_state",
            "captures_rng": False,
            "exact": True,
            "reason": None,
        }


class _MonitorRecoveryExpert(ExpertProvider):
    provider_id = "monitor-recovery-unit"

    def score_action(
        self,
        observation: dict[str, Any],
        action: Any,
        *,
        task: Any,
        engine: Any | None = None,
    ) -> ExpertActionScore:
        return ExpertActionScore(accepted=True, confidence=1.0, reason="accepted")

    def recover(
        self,
        *,
        state: dict[str, Any],
        failure: str | None,
        task: Any,
        engine: Any | None = None,
    ) -> list[Any]:
        assert failure == "action_magnitude_reference:risk=1.000000"
        return [[1.0]]

    def state_restore_capability(self) -> dict[str, Any]:
        return {
            "supported": True,
            "fidelity": "exact_declared_stateless",
            "captures_rng": False,
            "exact": True,
            "reason": None,
        }

    def metadata(self) -> dict[str, Any]:
        return {"provider_id": self.provider_id, "capabilities": ["recover"]}


def test_runner_records_monitor_evidence_and_counterfactual_link(
    tmp_path: Path,
) -> None:
    get_plugin_registry().engines["monitor_integration_unit"] = (
        _MonitorIntegrationEngine
    )
    suite = Suite.load("tabletop_manipulation_v0").filter_tasks(["pick_cube"])
    runner = PolicyRunner(
        policy=_MonitorIntegrationPolicy(),
        engine="monitor_integration_unit",
        episodes=1,
        out=tmp_path,
        capture_replay=False,
        expert_provider=_MonitorRecoveryExpert(),
        enable_recovery=True,
        failure_monitors=("action-magnitude",),
        enable_monitor_intervention=True,
        counterfactual_repeats=1,
        counterfactual_horizon=1,
    )

    report = runner.evaluate(suite)
    episode = runner.episode_results[0]
    record = episode.failure_monitor_records[0]

    assert episode.success is True
    assert episode.steps[0].info["action_rejected"] is False
    assert episode.steps[0].info["verifier_rejected"] is False
    assert episode.steps[0].info["failure_monitor_intervention_triggered"] is True
    assert episode.steps[0].info["action_source"] == "recovery"
    assert record.intervention_branch_point_id is not None
    assert (
        episode.counterfactual_recovery[0].branch_point.trigger_kind
        == "failure_monitor_recommendation"
    )
    assert report.summary["failure_monitor_metrics"]["prediction_count"] == 1
    assert report.summary["metric_vector"]["values"]["failure_prediction_ece"][
        "status"
    ] == "available"
    manifest_path = tmp_path / "failure_monitor_predictions.json"
    assert manifest_path.exists()
    _, contracts, records = load_monitor_manifest(manifest_path)
    assert set(contracts) == {"action_magnitude_reference"}
    assert records[0].intervention_branch_point_id == record.intervention_branch_point_id
    dataset_manifest = json.loads(
        (tmp_path / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    assert "failure_monitor_predictions.json" in dataset_manifest["artifacts"]


def test_runner_rejects_confounded_multi_monitor_intervention() -> None:
    with pytest.raises(ValueError, match="exactly one failure monitor"):
        PolicyRunner(
            policy=_MonitorIntegrationPolicy(),
            enable_recovery=True,
            failure_monitors=("action-magnitude", "action-magnitude"),
            enable_monitor_intervention=True,
        )
