from __future__ import annotations

import sys
from pathlib import Path

import pytest

from nyssa_bench.experts import make_expert_provider
from nyssa_bench.monitors.loaders import load_failure_monitor
from nyssa_bench.policies.base import load_policy_from_path
from nyssa_bench.policies.bc_policy import BCPolicy
from nyssa_bench.policies.loaders import load_callable_from_env, load_object
from nyssa_bench.policies.scripted_oracle_policy import ScriptedOraclePolicy
from nyssa_bench.policies.task_bc_policy import TaskBCPolicy


MODULE = '''from __future__ import annotations
from dataclasses import dataclass
from nyssa_bench.experts import ExpertProvider
from nyssa_bench.monitors.reference import ActionMagnitudeFailureMonitor

@dataclass
class PolicyAdapter:
    value: float = 0.5

    def act(self, observation):
        return [self.value]

def create_policy():
    return PolicyAdapter()

def create_expert_provider():
    return ExpertProvider()

def create_failure_monitor():
    return ActionMagnitudeFailureMonitor()
'''


@pytest.mark.parametrize("loader", [load_policy_from_path, load_failure_monitor, make_expert_provider])
def test_file_loaders_support_dataclasses_with_future_annotations(tmp_path: Path, loader):
    path = tmp_path / "adapter.py"
    path.write_text(MODULE, encoding="utf-8")
    assert loader(str(path)) is not None


def test_load_object_accepts_absolute_file_reference(tmp_path: Path):
    path = tmp_path / "adapter.py"
    path.write_text(MODULE, encoding="utf-8")
    assert load_object(f"{path}:PolicyAdapter")().act({}) == [0.5]


def test_policy_modules_with_same_filename_do_not_collide(tmp_path: Path):
    paths = [tmp_path / name / "adapter.py" for name in ("first", "second")]
    policies = []
    for index, path in enumerate(paths):
        path.parent.mkdir()
        path.write_text(MODULE.replace("0.5", str(index + 1.0)), encoding="utf-8")
        policies.append(load_policy_from_path(path))
    module_names = [type(policy).__module__ for policy in policies]
    assert module_names[0] != module_names[1]
    for path, module_name, policy in zip(paths, module_names, policies):
        assert Path(sys.modules[module_name].__file__) == path
        assert sys.modules[module_name].PolicyAdapter is type(policy)


def test_env_loader_instantiates_function_factory(tmp_path: Path, monkeypatch):
    path = tmp_path / "factory.py"
    path.write_text("def create_policy():\n    return lambda observation: [0.25]\n")
    monkeypatch.setenv("NYSSA_TEST_POLICY", f"{path}:create_policy")
    assert load_callable_from_env("NYSSA_TEST_POLICY")({}) == [0.25]


@pytest.mark.parametrize("parameter", ["observation", "observation=None"])
def test_env_loader_keeps_observation_callable(tmp_path: Path, monkeypatch, parameter):
    path = tmp_path / "callable.py"
    path.write_text(f"def policy({parameter}):\n    return observation['action']\n")
    monkeypatch.setenv("NYSSA_TEST_POLICY", f"{path}:policy")
    assert load_callable_from_env("NYSSA_TEST_POLICY")({"action": [0.5]}) == [0.5]


@pytest.mark.parametrize("adapter", [BCPolicy, TaskBCPolicy, ScriptedOraclePolicy])
def test_policy_adapters_support_models_with_parameterless_reset(adapter):
    class Model:
        def reset(self):
            self.value = 0.75

        def act(self, observation):
            return [self.value]

    policy = adapter(Model())
    policy.reset(task=object(), seed=42)
    assert policy.act({}) == [0.75]


def test_failed_module_load_restores_previous_registration(tmp_path: Path):
    path = tmp_path / "adapter.py"
    path.write_text(MODULE, encoding="utf-8")
    policy = load_policy_from_path(path)
    name = type(policy).__module__
    previous = sys.modules[name]
    path.write_text("raise RuntimeError('broken import')\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="broken import"):
        load_policy_from_path(path)
    assert sys.modules[name] is previous
