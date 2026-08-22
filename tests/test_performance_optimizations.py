from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from nyssa_bench import PolicyRunner, Suite
from nyssa_bench.baselines.features import flatten_observation
from nyssa_bench.core.episode import EpisodeResult
from nyssa_bench.engines.base import NyssaEngine
from nyssa_bench.experts import ExpertProvider
from nyssa_bench.failures import compact_stressor_context
from nyssa_bench.metrics.success import aggregate_episodes
from nyssa_bench.plugins import get_plugin_registry
from nyssa_bench.stressors import (
    StressorContext,
    StressorPipeline,
    StressorSpec,
)
from nyssa_bench.stressors.robustness import _bootstrap_auc, _normalized_auc


class _ExplodingTail(list[Any]):
    def __iter__(self):
        yield 3.5
        raise AssertionError(
            "flattening read values beyond the requested feature limit"
        )


def test_flatten_observation_stops_at_requested_feature_limit():
    result = flatten_observation({"raw": _ExplodingTail()}, max_dim=1)

    assert result.tolist() == [3.5]


def test_flatten_observation_preserves_numeric_array_prefix_and_padding():
    image = np.arange(4 * 4 * 3, dtype=np.float32).reshape(4, 4, 3)

    prefix = flatten_observation({"raw": image}, max_dim=7)
    padded = flatten_observation({"raw": [1.0, 2.0]}, max_dim=4)

    assert prefix.tolist() == image.reshape(-1)[:7].astype(float).tolist()
    assert padded.tolist() == [1.0, 2.0, 0.0, 0.0]


def test_vectorized_bootstrap_matches_scalar_reference():
    severities = [0.0, 0.5, 1.0]
    outcomes = np.asarray(
        [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0],
        ],
        dtype=bool,
    )
    samples = 257
    seed = 19
    rng = np.random.default_rng(seed)
    expected = np.empty(samples, dtype=float)
    for sample_index in range(samples):
        indices = rng.integers(0, outcomes.shape[1], size=outcomes.shape[1])
        rates = outcomes[:, indices].mean(axis=1).tolist()
        expected[sample_index] = _normalized_auc(severities, rates)

    actual = _bootstrap_auc(
        severities,
        outcomes,
        samples=samples,
        seed=seed,
        max_sampled_outcomes_per_batch=16,
    )

    assert np.allclose(actual, expected, rtol=0.0, atol=1e-15)


def test_episode_aggregation_preserves_group_and_missing_metric_semantics():
    episodes = [
        _episode("task_a", 0, True, {"score": 2.0, "recovery_attempt_count": 1.0}),
        _episode("task_a", 1, False, {"recovery_attempt_count": 2.0}),
        _episode("task_b", 0, True, {"score": 4.0, "recovery_attempt_count": 3.0}),
    ]

    summary = aggregate_episodes(episodes)

    assert summary["episodes"] == 3
    assert summary["success_count"] == 2
    assert summary["success_rate"] == 2 / 3
    assert summary["failure_counts"] == {"missed_target": 1}
    assert summary["metrics"]["score"] == 2.0
    assert summary["metrics"]["recovery_attempt_count"] == 6.0
    assert summary["per_task"]["task_a"]["metrics"]["score"] == 1.0
    assert summary["per_seed"]["0"]["episodes"] == 2


def test_stressor_application_context_avoids_runtime_state_serialization():
    pipeline = StressorPipeline(
        [
            StressorSpec("action_delay", 1.0, {"max_delay_steps": 2}),
            StressorSpec("action_gaussian_noise", 0.5),
        ],
        context=StressorContext(engine_name="mujoco", task_id="task"),
        episode_seed=7,
        condition_id="combined_shift",
    )
    observation = {
        "raw": [0.0],
        "action_space": {"low": [-1.0], "high": [1.0]},
    }
    pipeline.after_reset(object(), observation)
    pipeline.transform_action([0.25], observation=observation, step_index=0)

    application_context = pipeline.application_context()
    manifest = pipeline.manifest()

    assert "final_state" not in application_context
    assert compact_stressor_context(application_context) == compact_stressor_context(
        manifest
    )
    assert "final_state" in manifest


class _ThreeStepEngine(NyssaEngine):
    max_steps = 3

    def load_task(self, task_spec: Any) -> None:
        self.task_spec = task_spec

    def reset(self, seed: int | None = None):
        self.step_index = 0
        return _observation(), {"seed": seed}

    def step(self, action: Any):
        self.step_index += 1
        success = self.step_index == self.max_steps
        return _observation(), 1.0, success, False, {"success": success}

    def render(self):
        return None

    def get_state(self):
        return {"step_index": self.step_index}

    def close(self) -> None:
        return None


class _ConstantPolicy:
    def act(self, observation: dict[str, Any]):
        return [0.0]


class _CountingExpert(ExpertProvider):
    provider_id = "counting-expert"

    def __init__(self) -> None:
        self.metadata_calls = 0

    def metadata(self):
        self.metadata_calls += 1
        return {"provider_id": self.provider_id, "capabilities": []}


def test_runner_caches_expert_provider_metadata_within_each_episode():
    get_plugin_registry().engines["metadata_count_unit"] = _ThreeStepEngine
    expert = _CountingExpert()
    runner = PolicyRunner(
        policy=_ConstantPolicy(),
        engine="metadata_count_unit",
        episodes=2,
        capture_replay=False,
        expert_provider=expert,
    )

    runner.evaluate(
        Suite.load("maniskill_smoke_v0").filter_tasks(["maniskill_pick_cube"])
    )

    assert expert.metadata_calls == 3  # once per episode plus final run metadata


def test_runner_reuses_serialized_episode_payloads(
    tmp_path: Path,
    monkeypatch: Any,
):
    get_plugin_registry().engines["serialization_count_unit"] = _ThreeStepEngine
    calls = 0
    original = EpisodeResult.to_dict

    def counted_to_dict(episode: EpisodeResult):
        nonlocal calls
        calls += 1
        return original(episode)

    monkeypatch.setattr(EpisodeResult, "to_dict", counted_to_dict)
    runner = PolicyRunner(
        policy=_ConstantPolicy(),
        engine="serialization_count_unit",
        episodes=2,
        out=tmp_path,
        capture_replay=False,
    )

    runner.evaluate(
        Suite.load("maniskill_smoke_v0").filter_tasks(["maniskill_pick_cube"])
    )

    json_episodes = json.loads((tmp_path / "episodes.json").read_text(encoding="utf-8"))
    jsonl_episodes = [
        json.loads(line)
        for line in (tmp_path / "episodes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert calls == 2
    assert jsonl_episodes == json_episodes


def _episode(
    task_id: str,
    seed: int,
    success: bool,
    metrics: dict[str, float],
) -> EpisodeResult:
    return EpisodeResult(
        task_id=task_id,
        episode_index=seed,
        seed=seed,
        success=success,
        failure_label=None if success else "missed_target",
        metrics=metrics,
    )


def _observation() -> dict[str, Any]:
    return {
        "raw": [0.0],
        "action_space": {
            "type": "box",
            "shape": [1],
            "low": [-1.0],
            "high": [1.0],
            "dtype": "float32",
        },
    }
