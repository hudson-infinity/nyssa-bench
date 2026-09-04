import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from nyssa_bench import PolicyRunner, Suite
from nyssa_bench.engines.base import NyssaEngine
from nyssa_bench.engines.maniskill_adapter import ManiSkillEngine
from nyssa_bench.engines.mujoco_adapter import MuJoCoEngine
from nyssa_bench.plugins import get_plugin_registry
from nyssa_bench.stressors import (
    STRESSOR_REGISTRY,
    Stressor,
    StressorCompositionError,
    StressorConfig,
    StressorContext,
    StressorPipeline,
    StressorSpec,
    StressorUnsupportedError,
    UnsupportedStressorError,
    load_robustness_sweep,
    robustness_sweep_metrics,
    save_robustness_report,
)
from nyssa_bench.stressors.builtin import _add_gaussian_noise


def test_stressor_spec_and_config_round_trip(tmp_path: Path):
    config = StressorConfig(
        condition_id="sensor_s05",
        stressors=(
            StressorSpec(
                stressor_id="observation_gaussian_noise",
                severity=0.5,
                parameters={"max_std": 0.2},
                seed=7,
            ),
        ),
        unsupported_policy="record",
    )
    path = tmp_path / "stressors.yaml"
    path.write_text(
        """format: nyssa-stressor-config-v1
condition_id: sensor_s05
unsupported_policy: record
stressors:
  - stressor_id: observation_gaussian_noise
    severity: 0.5
    seed: 7
    parameters:
      max_std: 0.2
""",
        encoding="utf-8",
    )

    assert StressorConfig.from_dict(config.to_dict()) == config
    assert StressorConfig.load(path) == config
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        StressorSpec("action_delay", severity=1.1)


def test_noise_is_seed_deterministic_and_severity_resolves_distinct_parameters():
    context = StressorContext(engine_name="mujoco", task_id="reacher")
    low = StressorPipeline(
        [
            StressorSpec(
                "observation_gaussian_noise", severity=0.25, parameters={"max_std": 0.4}
            )
        ],
        context=context,
        episode_seed=13,
    )
    high = StressorPipeline(
        [
            StressorSpec(
                "observation_gaussian_noise", severity=0.75, parameters={"max_std": 0.4}
            )
        ],
        context=context,
        episode_seed=13,
    )
    repeated = StressorPipeline(
        [
            StressorSpec(
                "observation_gaussian_noise", severity=0.25, parameters={"max_std": 0.4}
            )
        ],
        context=context,
        episode_seed=13,
    )
    observation = {
        "raw": np.asarray([1.0, 2.0]),
        "action_space": {"low": [-1.0], "high": [1.0]},
    }
    for pipeline in (low, high, repeated):
        pipeline.after_reset(object(), observation)

    low_value = low.transform_observation(observation, step_index=0)["raw"]
    high_value = high.transform_observation(observation, step_index=0)["raw"]
    repeated_value = repeated.transform_observation(observation, step_index=0)["raw"]

    assert low.applications[0].applied_parameters["std"] == 0.1
    assert high.applications[0].applied_parameters["std"] == pytest.approx(0.3)
    assert np.array_equal(low_value, repeated_value)
    assert not np.array_equal(low_value, high_value)
    assert low_value.dtype == observation["raw"].dtype


@pytest.mark.parametrize(
    ("clip_min", "clip_max", "expected"),
    [
        (None, None, [-2.0, 2.0]),
        (-1.0, None, [-1.0, 2.0]),
        (None, 1.0, [-2.0, 1.0]),
        (-1.0, 1.0, [-1.0, 1.0]),
    ],
)
def test_array_noise_supports_unbounded_and_one_sided_clipping(
    clip_min: float | None,
    clip_max: float | None,
    expected: list[float],
) -> None:
    value = np.asarray([-2.0, 2.0], dtype=np.float32)

    transformed = _add_gaussian_noise(
        value,
        rng=np.random.default_rng(0),
        std=0.0,
        clip_min=clip_min,
        clip_max=clip_max,
    )

    assert transformed.dtype == np.float32
    np.testing.assert_array_equal(transformed, np.asarray(expected, dtype=np.float32))


def test_visual_brightness_only_changes_declared_rgb_image_fields():
    pipeline = StressorPipeline(
        [StressorSpec("image_brightness", 1.0, {"target_scale": 0.5})],
        context=StressorContext(
            engine_name="maniskill",
            task_id="pick_cube",
            observation_mode="rgb",
        ),
        episode_seed=3,
    )
    rgb = np.full((8, 8, 3), 200, dtype=np.uint8)
    state = np.asarray([1.0, 2.0], dtype=np.float32)
    observation = {"raw": {"sensor_data": {"rgb": rgb}, "state": state}}
    pipeline.after_reset(object(), observation)

    transformed = pipeline.transform_observation(observation, step_index=0)

    assert np.array_equal(transformed["raw"]["sensor_data"]["rgb"], rgb // 2)
    assert np.array_equal(transformed["raw"]["state"], state)
    application = pipeline.applications[0]
    assert application.category == "visual"
    assert application.severity_domain == (0.0, 1.0)
    assert application.lifetime == "episode"


def test_composition_order_is_explicit_and_changes_action_semantics():
    context = StressorContext(engine_name="maniskill", task_id="pick")
    delay_then_noise = StressorPipeline(
        [
            StressorSpec(
                "action_delay", severity=1.0, parameters={"max_delay_steps": 1}
            ),
            StressorSpec(
                "action_gaussian_noise", severity=1.0, parameters={"max_std": 0.2}
            ),
        ],
        context=context,
        episode_seed=5,
    )
    noise_then_delay = StressorPipeline(
        [
            StressorSpec(
                "action_gaussian_noise", severity=1.0, parameters={"max_std": 0.2}
            ),
            StressorSpec(
                "action_delay", severity=1.0, parameters={"max_delay_steps": 1}
            ),
        ],
        context=context,
        episode_seed=5,
    )
    observation = {"raw": [0.0], "action_space": {"low": [-1.0], "high": [1.0]}}
    for pipeline in (delay_then_noise, noise_then_delay):
        pipeline.after_reset(object(), observation)

    first = delay_then_noise.transform_action(
        [0.5], observation=observation, step_index=0
    )
    second = noise_then_delay.transform_action(
        [0.5], observation=observation, step_index=0
    )

    assert delay_then_noise.manifest()["composition_order"] == [
        "action_delay",
        "action_gaussian_noise",
    ]
    assert not np.allclose(first, [0.0])
    assert np.allclose(second, [0.0])


def test_duplicate_stressors_are_rejected_as_incompatible():
    with pytest.raises(StressorCompositionError, match="Duplicate stressors"):
        StressorPipeline(
            [StressorSpec("action_delay", 0.5), StressorSpec("action_delay", 1.0)],
            context=StressorContext(engine_name="mujoco", task_id="reacher"),
            episode_seed=0,
        )


def test_stressor_state_snapshot_restores_rng_and_delay_buffer():
    specs = [
        StressorSpec("action_gaussian_noise", 0.5),
        StressorSpec("action_delay", 1.0, {"max_delay_steps": 2}),
    ]
    context = StressorContext(engine_name="mujoco", task_id="reacher")
    original = StressorPipeline(specs, context=context, episode_seed=42)
    restored = StressorPipeline(specs, context=context, episode_seed=42)
    observation = {"raw": [0.0], "action_space": {"low": [-1.0], "high": [1.0]}}
    for pipeline in (original, restored):
        pipeline.after_reset(object(), observation)
    original.transform_action([0.2], observation=observation, step_index=0)
    restored.set_state(original.get_state())

    expected = original.transform_action([0.4], observation=observation, step_index=1)
    actual = restored.transform_action([0.4], observation=observation, step_index=1)

    assert np.allclose(expected, actual)
    assert original.get_state() == restored.get_state()


def test_stressor_state_restore_rejects_a_different_execution_context():
    original = StressorPipeline(
        [StressorSpec("action_delay", 1.0, {"max_delay_steps": 1})],
        context=StressorContext(engine_name="mujoco", task_id="reacher"),
        episode_seed=42,
        condition_id="delay_s1",
    )
    different_task = StressorPipeline(
        [StressorSpec("action_delay", 1.0, {"max_delay_steps": 1})],
        context=StressorContext(engine_name="mujoco", task_id="pusher"),
        episode_seed=42,
        condition_id="delay_s1",
    )
    observation = {"raw": [0.0], "action_space": {"low": [-1.0], "high": [1.0]}}
    original.after_reset(object(), observation)
    different_task.after_reset(object(), observation)

    with pytest.raises(StressorCompositionError, match="engine or task context"):
        different_task.set_state(original.get_state())


def test_record_policy_captures_late_lifecycle_rejection(
    monkeypatch: pytest.MonkeyPatch,
):
    class LateUnsupportedStressor(Stressor):
        stressor_id = "late_unsupported_test"
        category = "system"
        application_points = ("before_step",)
        supported_engines = frozenset({"*"})

        def before_step(self, engine: Any, *, step_index: int) -> None:
            raise StressorUnsupportedError("runtime capability unavailable")

    monkeypatch.setitem(
        STRESSOR_REGISTRY,
        LateUnsupportedStressor.stressor_id,
        LateUnsupportedStressor,
    )
    pipeline = StressorPipeline(
        [StressorSpec(LateUnsupportedStressor.stressor_id, 1.0)],
        context=StressorContext(engine_name="mujoco", task_id="reacher"),
        episode_seed=0,
        unsupported_policy="record",
    )
    pipeline.after_reset(object(), _observation())

    pipeline.before_step(object(), step_index=0)

    assert pipeline.applications[0].status == "unsupported"
    assert pipeline.applications[0].reason == "runtime capability unavailable"


def test_unsupported_stressors_fail_or_record_explicitly():
    spec = StressorSpec("friction_scale", severity=0.5)
    context = StressorContext(engine_name="unknown", task_id="task")

    with pytest.raises(UnsupportedStressorError, match="engine 'unknown'"):
        StressorPipeline([spec], context=context, episode_seed=0)

    pipeline = StressorPipeline(
        [spec], context=context, episode_seed=0, unsupported_policy="record"
    )

    assert pipeline.has_unsupported is True
    assert pipeline.applications[0].status == "unsupported"
    assert "engine 'unknown'" in str(pipeline.applications[0].reason)


def test_mujoco_friction_stressor_changes_and_restores_model_values():
    class Model:
        geom_friction = np.asarray([[1.0, 0.1, 0.01], [0.5, 0.2, 0.02]], dtype=float)

    class Unwrapped:
        model = Model()

    class Env:
        unwrapped = Unwrapped()

    engine = MuJoCoEngine()
    engine.env = Env()
    engine._capture_friction_baseline()

    evidence = engine.apply_stressor("friction_scale", {"scale": 0.25})

    assert evidence["status"] == "applied"
    assert evidence["geom_count"] == 2
    assert np.allclose(
        engine.env.unwrapped.model.geom_friction,
        [[0.25, 0.025, 0.0025], [0.125, 0.05, 0.005]],
    )
    engine._restore_friction_baseline()
    assert np.allclose(
        engine.env.unwrapped.model.geom_friction, [[1.0, 0.1, 0.01], [0.5, 0.2, 0.02]]
    )


def test_maniskill_cpu_friction_stressor_changes_and_restores_physx_materials():
    class Material:
        static_friction = 0.6
        dynamic_friction = 0.4

    class Shape:
        material = Material()

    class Body:
        def get_collision_shapes(self):
            return [Shape()]

    class View:
        _bodies = [Body()]
        links: list[Any] = []

    class Scene:
        gpu_sim_enabled = False
        actor_views = {"cube": View()}
        articulation_views: dict[str, Any] = {}

    class Target:
        scene = Scene()

    class Env:
        unwrapped = Target()

    engine = ManiSkillEngine()
    engine.env = Env()

    evidence = engine.apply_stressor("friction_scale", {"scale": 0.5})

    material = Scene.actor_views["cube"]._bodies[0].get_collision_shapes()[0].material
    assert evidence["status"] == "applied"
    assert evidence["material_count"] == 1
    assert material.static_friction == pytest.approx(0.3)
    assert material.dynamic_friction == pytest.approx(0.2)
    engine._restore_friction_baseline()
    assert material.static_friction == pytest.approx(0.6)
    assert material.dynamic_friction == pytest.approx(0.4)


class _StressorUnitEngine(NyssaEngine):
    max_steps = 1

    def load_task(self, task_spec: Any) -> None:
        self.task_spec = task_spec

    def reset(self, seed: int | None = None):
        self.actions: list[float] = []
        return _observation(), {"seed": seed}

    def step(self, action: Any):
        value = float(np.asarray(action).reshape(-1)[0])
        self.actions.append(value)
        return _observation(), value, True, False, {"success": value > 0.2}

    def render(self):
        return None

    def get_state(self):
        return {"actions": list(self.actions)}

    def close(self) -> None:
        return None


class _ConstantPolicy:
    def act(self, observation: dict[str, Any]):
        return [0.5]


def test_runner_executes_stressors_and_writes_reproducible_context(tmp_path: Path):
    get_plugin_registry().engines["stressor_unit"] = _StressorUnitEngine
    config = StressorConfig(
        condition_id="delay_s1",
        stressors=(StressorSpec("action_delay", 1.0, {"max_delay_steps": 1}),),
    )
    runner = PolicyRunner(
        policy=_ConstantPolicy(),
        engine="stressor_unit",
        episodes=1,
        out=tmp_path,
        capture_replay=False,
        stressor_config=config,
    )

    report = runner.evaluate(
        Suite.load("maniskill_smoke_v0").filter_tasks(["maniskill_pick_cube"])
    )

    episode = runner.episode_results[0]
    manifest = json.loads(
        (tmp_path / "stressor_manifest.json").read_text(encoding="utf-8")
    )
    replay = json.loads((tmp_path / "replay_manifest.json").read_text(encoding="utf-8"))
    assert episode.steps[0].action == [0.0]
    assert episode.steps[0].info["action_before_stressors"] == [0.5]
    assert episode.steps[0].info["stressor_action_modified"] is True
    assert (
        episode.steps[0].info["stressor_state"]["format"] == "nyssa-stressor-context-v1"
    )
    assert episode.stressor_context["applications"][0]["status"] == "applied"
    assert manifest["summary"]["applied_stressors"] == ["action_delay"]
    assert replay["episodes"][0]["stressor_context"]["condition_id"] == "delay_s1"
    assert report.summary["stressor_execution"]["all_requests_resolved"] is True
    assert report.summary["success_rate"] == 0.0
    assert (tmp_path / "dataset_manifest.json").read_text(encoding="utf-8").find(
        "stressor_manifest.json"
    ) >= 0


def test_runner_record_mode_downgrades_claim_for_unsupported_stressor(tmp_path: Path):
    get_plugin_registry().engines["stressor_unit"] = _StressorUnitEngine
    config = StressorConfig(
        condition_id="friction_s05",
        stressors=(StressorSpec("friction_scale", 0.5),),
        unsupported_policy="record",
    )
    runner = PolicyRunner(
        policy=_ConstantPolicy(),
        engine="stressor_unit",
        episodes=1,
        out=tmp_path,
        capture_replay=False,
        stressor_config=config,
    )

    report = runner.evaluate(
        Suite.load("maniskill_smoke_v0").filter_tasks(["maniskill_pick_cube"])
    )

    assert report.summary["stressor_execution"]["unsupported_stressors"] == [
        "friction_scale"
    ]
    assert report.summary["public_claim"] is False
    assert (
        report.summary["public_claim_validation"]["checks"][
            "stressor_requests_resolved"
        ]
        is False
    )
    assert (
        "stressor_requests_resolved"
        in report.summary["public_claim_validation"]["failures"]
    )


def test_robustness_sweep_metrics_and_artifacts_include_uncertainty(tmp_path: Path):
    episodes_by_severity = {
        0.0: [_episode_dict(seed, True) for seed in range(4)],
        1.0: [_episode_dict(seed, seed < 2) for seed in range(4)],
    }

    summary = robustness_sweep_metrics(
        stressor_id="action_delay",
        episodes_by_severity=episodes_by_severity,
        bootstrap_samples=100,
        bootstrap_seed=7,
    )
    paths = save_robustness_report(summary, tmp_path)

    assert summary["clean_success_rate"] == 1.0
    assert summary["max_severity_success_rate"] == 0.5
    assert summary["max_severity_degradation"] == 0.5
    assert summary["robustness_auc"] == 0.75
    assert len(summary["robustness_auc_ci95"]) == 2
    with paths["robustness_csv"].open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[1]["degradation_from_clean"] == "0.5"
    assert "Robustness AUC" in paths["robustness_report"].read_text(encoding="utf-8")


def test_load_robustness_sweep_from_runner_result_packs(tmp_path: Path):
    get_plugin_registry().engines["stressor_sweep_unit"] = _StressorUnitEngine
    suite = Suite.load("maniskill_smoke_v0").filter_tasks(["maniskill_pick_cube"])
    run_dirs = []
    for severity in (0.0, 1.0):
        run_dir = tmp_path / f"severity_{severity}"
        PolicyRunner(
            policy=_ConstantPolicy(),
            engine="stressor_sweep_unit",
            episodes=3,
            seed=2,
            out=run_dir,
            capture_replay=False,
            stressor_config=StressorConfig(
                condition_id=f"delay_s{severity}",
                stressors=(
                    StressorSpec("action_delay", severity, {"max_delay_steps": 1}),
                ),
            ),
        ).evaluate(suite)
        run_dirs.append(run_dir)

    summary = load_robustness_sweep(run_dirs, bootstrap_samples=50, bootstrap_seed=3)

    assert summary["stressor_id"] == "action_delay"
    assert summary["paired_episode_coverage"] == 3
    assert summary["clean_success_rate"] == 1.0
    assert summary["max_severity_success_rate"] == 0.0
    assert summary["robustness_auc"] == 0.5
    assert summary["points"][1]["applied_parameters"] == [
        {"delay_steps": 1, "initial_action": "zeros", "max_delay_steps": 1}
    ]


def test_real_mujoco_backend_executes_dynamics_and_system_stressors_when_installed():
    gym = pytest.importorskip("gymnasium")
    pytest.importorskip("mujoco")
    try:
        env = gym.make("InvertedPendulum-v4")
    except Exception as exc:
        pytest.skip(f"MuJoCo integration environment unavailable: {exc}")
    engine = MuJoCoEngine()
    engine.env = env
    engine._capture_friction_baseline()
    baseline = env.unwrapped.model.geom_friction.copy()
    try:
        raw_observation, _ = env.reset(seed=0)
        observation = {
            "raw": raw_observation,
            "action_space": {
                "low": env.action_space.low.tolist(),
                "high": env.action_space.high.tolist(),
            },
        }
        pipeline = StressorPipeline(
            [
                StressorSpec("friction_scale", 1.0, {"target_scale": 0.5}),
                StressorSpec("action_delay", 1.0, {"max_delay_steps": 1}),
            ],
            context=StressorContext(engine_name="mujoco", task_id="inverted_pendulum"),
            episode_seed=0,
        )

        pipeline.after_reset(engine, observation)
        executed_action = pipeline.transform_action(
            np.full(env.action_space.shape, 0.5, dtype=np.float32),
            observation=observation,
            step_index=0,
        )
        env.step(executed_action)

        assert np.allclose(env.unwrapped.model.geom_friction, baseline * 0.5)
        assert np.allclose(executed_action, np.zeros(env.action_space.shape))
        assert [item.status for item in pipeline.applications] == [
            "applied",
            "applied",
        ]
    finally:
        env.close()


@pytest.mark.skipif(
    os.environ.get("NYSSA_RUN_MANISKILL_INTEGRATION") != "1",
    reason="set NYSSA_RUN_MANISKILL_INTEGRATION=1 on a ManiSkill-capable host",
)
def test_real_maniskill_cpu_backend_executes_dynamics_and_system_stressors():
    gym = pytest.importorskip("gymnasium")
    pytest.importorskip("mani_skill")
    env = gym.make(
        "PickCube-v1",
        obs_mode="state_dict",
        control_mode="pd_ee_delta_pose",
        sim_backend="cpu",
    )
    engine = ManiSkillEngine()
    engine.env = env
    try:
        engine._capture_friction_baseline()
        raw_observation, _ = env.reset(seed=0)
        observation = {
            "raw": raw_observation,
            "action_space": {
                "low": env.action_space.low.tolist(),
                "high": env.action_space.high.tolist(),
            },
        }
        pipeline = StressorPipeline(
            [
                StressorSpec("friction_scale", 1.0, {"target_scale": 0.5}),
                StressorSpec("action_delay", 1.0, {"max_delay_steps": 1}),
            ],
            context=StressorContext(
                engine_name="maniskill", task_id="maniskill_pick_cube"
            ),
            episode_seed=0,
        )

        pipeline.after_reset(engine, observation)
        executed_action = pipeline.transform_action(
            np.full(env.action_space.shape, 0.5, dtype=np.float32),
            observation=observation,
            step_index=0,
        )
        env.step(executed_action)

        assert pipeline.applications[0].backend_evidence["material_count"] > 0
        assert np.allclose(executed_action, np.zeros(env.action_space.shape))
        assert [item.status for item in pipeline.applications] == [
            "applied",
            "applied",
        ]
    finally:
        env.close()


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


def _episode_dict(seed: int, success: bool) -> dict[str, Any]:
    return {"task_id": "task", "seed": seed, "episode_index": seed, "success": success}
