from __future__ import annotations

import copy
import hashlib
import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from nyssa_bench.recovery.protocol import RestoreCapability, RestorationGrade


class BranchStateError(RuntimeError):
    pass


@dataclass
class BranchSnapshot:
    observation: dict[str, Any]
    engine_state: Any
    policy_state: Any
    expert_state: Any
    stressor_state: Any
    process_rng_state: dict[str, Any]
    capabilities: tuple[RestoreCapability, ...]
    snapshot_sha256: str
    randomness_sha256: str
    restoration_grade: RestorationGrade
    matched_randomness: bool
    unsupported_reason: str | None

    @classmethod
    def capture(
        cls,
        *,
        observation: dict[str, Any],
        engine: Any,
        policy: Any,
        expert: Any,
        stressors: Any,
        require_expert: bool,
    ) -> "BranchSnapshot":
        components = (
            ("engine", engine, True),
            ("policy", policy, True),
            ("expert", expert, require_expert),
            ("stressors", stressors, True),
        )
        states: dict[str, Any] = {}
        capabilities: list[RestoreCapability] = []
        for component, instance, required in components:
            capability = _component_capability(component, instance, required=required)
            if capability.supported:
                try:
                    states[component] = copy.deepcopy(instance.get_state())
                except Exception as exc:
                    capability = RestoreCapability(
                        component=component,
                        component_id=capability.component_id,
                        required=required,
                        supported=False,
                        fidelity="capture_failed",
                        captures_rng=False,
                        exact=False,
                        reason=f"state capture failed: {type(exc).__name__}: {exc}",
                    )
                    states[component] = None
            else:
                states[component] = None
            capabilities.append(capability)

        process_state, process_capability = _capture_process_rng_state()
        capabilities.append(process_capability)
        required_capabilities = [item for item in capabilities if item.required]
        unsupported = [item for item in required_capabilities if not item.supported]
        if unsupported:
            restoration_grade: RestorationGrade = "unsupported"
            unsupported_reason = "; ".join(
                f"{item.component}: {item.reason or item.fidelity}"
                for item in unsupported
            )
        else:
            exact = all(item.exact for item in required_capabilities)
            restoration_grade = "exact" if exact else "qualified"
            unsupported_reason = None

        engine_capability = next(
            item for item in capabilities if item.component == "engine"
        )
        stressor_capability = next(
            item for item in capabilities if item.component == "stressors"
        )
        matched_randomness = bool(
            restoration_grade != "unsupported"
            and process_capability.exact
            and engine_capability.captures_rng
            and stressor_capability.captures_rng
        )
        payload = {
            "observation": observation,
            "engine": states["engine"],
            "policy": states["policy"],
            "expert": states["expert"],
            "stressors": states["stressors"],
            "process_rng": process_state,
        }
        return cls(
            observation=copy.deepcopy(observation),
            engine_state=states["engine"],
            policy_state=states["policy"],
            expert_state=states["expert"],
            stressor_state=states["stressors"],
            process_rng_state=process_state,
            capabilities=tuple(capabilities),
            snapshot_sha256=_state_sha256(payload),
            randomness_sha256=_state_sha256(
                {
                    "engine": states["engine"],
                    "policy": states["policy"],
                    "expert": states["expert"] if require_expert else None,
                    "stressors": states["stressors"],
                    "process_rng": process_state,
                }
            ),
            restoration_grade=restoration_grade,
            matched_randomness=matched_randomness,
            unsupported_reason=unsupported_reason,
        )

    def restore(
        self,
        *,
        engine: Any,
        policy: Any,
        expert: Any,
        stressors: Any,
    ) -> dict[str, Any]:
        if self.restoration_grade == "unsupported":
            raise BranchStateError(self.unsupported_reason or "snapshot is unsupported")
        restored_observation: dict[str, Any] | None = None
        try:
            candidate = engine.set_state(copy.deepcopy(self.engine_state))
            if isinstance(candidate, dict):
                restored_observation = candidate
            stressors.set_state(copy.deepcopy(self.stressor_state), engine=engine)
            policy.set_state(copy.deepcopy(self.policy_state))
            expert_capability = next(
                item for item in self.capabilities if item.component == "expert"
            )
            if expert_capability.required or expert_capability.supported:
                expert.set_state(copy.deepcopy(self.expert_state))
            _restore_process_rng_state(copy.deepcopy(self.process_rng_state))
        except Exception as exc:
            raise BranchStateError(
                f"failed to restore branch snapshot: {type(exc).__name__}: {exc}"
            ) from exc
        return copy.deepcopy(restored_observation or self.observation)


def reseed_branch_streams(
    *,
    seed: int,
    engine: Any,
    policy: Any,
    expert: Any,
    stressors: Any,
    include_expert: bool,
) -> tuple[str, ...]:
    if seed < 0:
        raise ValueError("branch seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    seeded = ["process_rng"]
    components = [
        ("engine", engine),
        ("policy", policy),
        ("stressors", stressors),
    ]
    if include_expert:
        components.append(("expert", expert))
    for index, (name, component) in enumerate(components):
        method = getattr(component, "seed_branch_rng", None)
        if not callable(method):
            continue
        component_seed = _derived_component_seed(seed, index, name)
        try:
            result = method(component_seed)
        except NotImplementedError:
            continue
        if result is not False:
            seeded.append(name)
    return tuple(seeded)


def state_sha256(value: Any) -> str:
    return _state_sha256(value)


def _component_capability(
    component: str, instance: Any, *, required: bool
) -> RestoreCapability:
    component_id = instance.__class__.__name__
    get_state = getattr(instance, "get_state", None)
    set_state = getattr(instance, "set_state", None)
    method = getattr(instance, "state_restore_capability", None)
    if not callable(get_state) or not callable(set_state):
        return RestoreCapability(
            component=component,
            component_id=component_id,
            required=required,
            supported=False,
            fidelity="unsupported",
            captures_rng=False,
            exact=False,
            reason="component does not expose get_state and set_state",
        )
    raw = method() if callable(method) else {}
    if not isinstance(raw, dict):
        raw = {}
    supported = bool(raw.get("supported", callable(method)))
    fidelity = str(raw.get("fidelity", "declared_state_restore"))
    exact = bool(raw.get("exact", supported and fidelity.startswith("exact")))
    return RestoreCapability(
        component=component,
        component_id=str(raw.get("component_id", component_id)),
        required=required,
        supported=supported,
        fidelity=fidelity,
        captures_rng=bool(raw.get("captures_rng", False)),
        exact=exact,
        reason=str(raw["reason"]) if raw.get("reason") is not None else None,
    )


def _capture_process_rng_state() -> tuple[dict[str, Any], RestoreCapability]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    try:
        import torch

        state["torch_cpu"] = torch.random.get_rng_state().clone()
        if torch.cuda.is_available():
            state["torch_cuda"] = [
                item.clone() for item in torch.cuda.get_rng_state_all()
            ]
    except ImportError:
        pass
    except Exception as exc:
        return state, RestoreCapability(
            component="process_rng",
            component_id="python_numpy_torch",
            required=True,
            supported=True,
            fidelity="qualified_process_rng",
            captures_rng=True,
            exact=False,
            reason=f"optional torch RNG capture failed: {type(exc).__name__}: {exc}",
        )
    return state, RestoreCapability(
        component="process_rng",
        component_id="python_numpy_torch",
        required=True,
        supported=True,
        fidelity="exact_process_rng",
        captures_rng=True,
        exact=True,
        reason=None,
    )


def _restore_process_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    if "torch_cpu" not in state:
        return
    try:
        import torch
    except ImportError as exc:
        raise BranchStateError(
            "snapshot contains torch RNG state but torch is unavailable"
        ) from exc
    torch.random.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state:
        if not torch.cuda.is_available():
            raise BranchStateError(
                "snapshot contains CUDA RNG state but CUDA is unavailable"
            )
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _derived_component_seed(seed: int, index: int, component: str) -> int:
    payload = (
        f"nyssa-counterfactual-component-seed-v1:{seed}:{index}:{component}".encode()
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def _state_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    _update_digest(digest, value, {})
    return digest.hexdigest()


def _update_digest(digest: Any, value: Any, seen: dict[int, int]) -> None:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    track_identity = isinstance(value, (dict, list, tuple, np.ndarray)) or hasattr(
        value, "__dict__"
    )
    if track_identity:
        identity = id(value)
        if identity in seen:
            digest.update(f"reference:{seen[identity]}".encode())
            return
        seen[identity] = len(seen)
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        digest.update(b"array:")
        digest.update(str(contiguous.dtype).encode())
        digest.update(repr(contiguous.shape).encode())
        digest.update(contiguous.tobytes())
        return
    if hasattr(value, "numpy"):
        try:
            _update_digest(digest, value.numpy(), seen)
            return
        except (TypeError, RuntimeError):
            pass
    bit_generator = getattr(value, "bit_generator", None)
    if bit_generator is not None and hasattr(bit_generator, "state"):
        digest.update(type(value).__qualname__.encode())
        _update_digest(digest, bit_generator.state, seen)
        return
    if type(value).__module__.startswith("torch") and callable(
        getattr(value, "get_state", None)
    ):
        digest.update(type(value).__qualname__.encode())
        _update_digest(digest, value.get_state(), seen)
        return
    if isinstance(value, dict):
        digest.update(b"mapping{")
        for key in sorted(value, key=lambda item: repr(item)):
            _update_digest(digest, key, seen)
            _update_digest(digest, value[key], seen)
        digest.update(b"}")
        return
    if isinstance(value, (list, tuple)):
        digest.update(b"sequence[")
        for item in value:
            _update_digest(digest, item, seen)
        digest.update(b"]")
        return
    if isinstance(value, bytes):
        digest.update(b"bytes:")
        digest.update(value)
        return
    if hasattr(value, "__dict__"):
        digest.update(type(value).__qualname__.encode())
        _update_digest(digest, vars(value), seen)
        return
    digest.update(type(value).__qualname__.encode())
    digest.update(b":")
    digest.update(repr(value).encode("utf-8", errors="backslashreplace"))
