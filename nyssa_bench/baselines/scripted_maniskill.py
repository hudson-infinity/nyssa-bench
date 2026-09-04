from __future__ import annotations

from typing import Any

import numpy as np

from nyssa_bench.baselines.features import action_bounds, find_vector


class ManiSkillScriptedHeuristic:
    """Simple task-aware controller for common ManiSkill manipulation observations.

    This is a reference scripted baseline, not a guaranteed privileged solver.
    It uses common observation fields when present: tcp/ee pose, object pose,
    and goal/target pose. If those fields are unavailable it emits a zero action
    clipped to the action space.
    """

    def __init__(self, gain: float = 6.0, lift_height: float = 0.18) -> None:
        self.gain = gain
        self.lift_height = lift_height
        self.task_id = ""
        self.stage = 0
        self.close_steps = 0

    def reset(self, task: Any | None = None, seed: int | None = None) -> None:
        self.task_id = str(getattr(task, "task_id", ""))
        self.stage = 0
        self.close_steps = 0

    def get_state(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "stage": self.stage,
            "close_steps": self.close_steps,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            raise TypeError("scripted controller state must be a mapping")
        self.task_id = str(state["task_id"])
        self.stage = int(state["stage"])
        self.close_steps = int(state["close_steps"])

    def state_restore_capability(self) -> dict[str, Any]:
        return {
            "supported": True,
            "fidelity": "exact_scripted_controller_state",
            "captures_rng": False,
            "exact": True,
            "reason": None,
        }

    def act(self, observation: dict[str, Any]) -> Any:
        low, high, shape = action_bounds(observation)
        size = int(np.prod(shape))
        action = np.zeros(size, dtype=float)

        tcp = _first_vector(
            observation,
            ("tcp_pose", "ee_pose", "end_effector_pose", "tcp_pos", "ee_pos", "eef_pos"),
        )
        target, gripper = self._target_and_gripper(observation, tcp=tcp)
        if tcp is None or target is None:
            return np.clip(action.reshape(shape), low, high)

        delta = np.clip((target - tcp[:3]) * self.gain, -1.0, 1.0)
        action[: min(3, size)] = delta[: min(3, size)]
        if size >= 4:
            action[-1] = gripper
        return np.clip(action.reshape(shape), low, high)

    def _target_and_gripper(self, observation: dict[str, Any], *, tcp: np.ndarray | None) -> tuple[np.ndarray | None, float]:
        if tcp is None:
            return None, 1.0
        if "push" in self.task_id:
            return self._push_target_and_gripper(observation, tcp=tcp)
        if "stack" in self.task_id:
            return self._stack_target_and_gripper(observation, tcp=tcp)
        return self._pick_target_and_gripper(observation, tcp=tcp)

    def _pick_target_and_gripper(
        self,
        observation: dict[str, Any],
        *,
        tcp: np.ndarray,
    ) -> tuple[np.ndarray | None, float]:
        obj = _first_vector(observation, ("obj_pose", "object_pose", "cube_pose", "cube_pos", "obj_pos"))
        goal = _first_vector(observation, ("goal_pose", "target_pose", "goal_pos", "target_pos"))
        if obj is None:
            return None, 1.0
        obj_pos = obj[:3]
        grasped = _first_bool(observation, ("is_grasped", "grasped"))
        if grasped:
            self.stage = 2
            return (goal[:3] if goal is not None else obj_pos + np.array([0.0, 0.0, self.lift_height])), -1.0

        xy_error = float(np.linalg.norm(tcp[:2] - obj_pos[:2]))
        grasp_target = obj_pos + np.array([0.0, 0.0, 0.018])
        hover_target = obj_pos + np.array([0.0, 0.0, 0.075])
        if self.stage == 0 and (xy_error > 0.018 or tcp[2] > obj_pos[2] + 0.055):
            return hover_target, 1.0

        self.stage = 1
        if np.linalg.norm(tcp[:3] - grasp_target) < 0.035:
            self.close_steps += 1
        return grasp_target, -1.0

    def _push_target_and_gripper(
        self,
        observation: dict[str, Any],
        *,
        tcp: np.ndarray,
    ) -> tuple[np.ndarray | None, float]:
        obj = _first_vector(observation, ("obj_pose", "object_pose", "cube_pose", "cube_pos", "obj_pos"))
        goal = _first_vector(observation, ("goal_pose", "target_pose", "goal_pos", "target_pos"))
        if obj is None or goal is None:
            return None, 1.0
        obj_pos = obj[:3]
        goal_pos = goal[:3]
        direction = goal_pos[:2] - obj_pos[:2]
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-8:
            return obj_pos + np.array([0.0, 0.0, 0.025]), 1.0
        direction = direction / norm
        push_z = obj_pos[2] + 0.018
        behind = np.asarray([obj_pos[0] - 0.075 * direction[0], obj_pos[1] - 0.075 * direction[1], push_z])
        push_through = np.asarray([obj_pos[0] + 0.13 * direction[0], obj_pos[1] + 0.13 * direction[1], push_z])
        if np.linalg.norm(tcp[:2] - behind[:2]) > 0.025 or abs(float(tcp[2] - push_z)) > 0.025:
            return behind, 1.0
        return push_through, 1.0

    def _stack_target_and_gripper(
        self,
        observation: dict[str, Any],
        *,
        tcp: np.ndarray,
    ) -> tuple[np.ndarray | None, float]:
        cube_a = _first_vector(observation, ("cubea_pose", "cube_a_pose", "cubea_pos", "cube_a_pos"))
        cube_b = _first_vector(observation, ("cubeb_pose", "cube_b_pose", "cubeb_pos", "cube_b_pos"))
        if cube_a is None or cube_b is None:
            return None, 1.0
        cube_a_pos = cube_a[:3]
        cube_b_pos = cube_b[:3]
        grasped = _first_bool(observation, ("is_cubea_grasped", "is_cube_a_grasped", "cubea_grasped"))
        if grasped:
            self.stage = 2
            stack_target = cube_b_pos + np.array([0.0, 0.0, 0.075])
            if np.linalg.norm(tcp[:3] - stack_target) < 0.04:
                self.stage = 3
                return stack_target, 1.0
            return stack_target, -1.0

        xy_error = float(np.linalg.norm(tcp[:2] - cube_a_pos[:2]))
        grasp_target = cube_a_pos + np.array([0.0, 0.0, 0.018])
        hover_target = cube_a_pos + np.array([0.0, 0.0, 0.075])
        if self.stage == 0 and (xy_error > 0.018 or tcp[2] > cube_a_pos[2] + 0.055):
            return hover_target, 1.0

        self.stage = 1
        if np.linalg.norm(tcp[:3] - grasp_target) < 0.035:
            self.close_steps += 1
        return grasp_target, -1.0


def create_scripted_oracle() -> ManiSkillScriptedHeuristic:
    return ManiSkillScriptedHeuristic()


def _first_vector(observation: dict[str, Any], names: tuple[str, ...]) -> np.ndarray | None:
    vector = find_vector(observation, names)
    if vector is None:
        return None
    return vector


def _first_bool(observation: dict[str, Any], names: tuple[str, ...]) -> bool:
    value = _find_named_value(observation.get("raw", observation), names)
    if value is None:
        return False
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return bool(value[0]) if value else False
    return bool(value)


def _find_named_value(value: Any, names: tuple[str, ...]) -> Any | None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).lower()
            if any(name in normalized for name in names):
                return item
        for item in value.values():
            found = _find_named_value(item, names)
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for item in value:
            found = _find_named_value(item, names)
            if found is not None:
                return found
    return None
