from __future__ import annotations

import hashlib
from typing import Any

from nyssa_bench.stressors.base import Stressor, StressorUnsupportedError
from nyssa_bench.stressors.protocol import (
    STRESSOR_CONTEXT_FORMAT,
    StressorApplication,
    StressorContext,
    StressorSpec,
    UnsupportedPolicy,
)
from nyssa_bench.stressors.registry import make_stressor


class StressorCompositionError(ValueError):
    pass


class UnsupportedStressorError(RuntimeError):
    pass


class StressorPipeline:
    def __init__(
        self,
        specs: tuple[StressorSpec, ...] | list[StressorSpec],
        *,
        context: StressorContext,
        episode_seed: int,
        condition_id: str = "clean",
        unsupported_policy: UnsupportedPolicy = "error",
    ) -> None:
        if unsupported_policy not in {"error", "record"}:
            raise ValueError("unsupported_policy must be 'error' or 'record'")
        self.specs = tuple(specs)
        self.context = context
        self.episode_seed = int(episode_seed)
        self.condition_id = condition_id
        self.unsupported_policy = unsupported_policy
        self.stressors = [make_stressor(spec.stressor_id) for spec in self.specs]
        _validate_composition(self.stressors)
        self.applications: list[StressorApplication] = []
        self._prepare()

    @property
    def composition_order(self) -> tuple[str, ...]:
        return tuple(spec.stressor_id for spec in self.specs)

    @property
    def has_unsupported(self) -> bool:
        return any(
            application.status == "unsupported" for application in self.applications
        )

    def before_reset(self, engine: Any) -> None:
        for stressor, application in zip(
            self.stressors, self.applications, strict=True
        ):
            if application.status == "requested":
                try:
                    stressor.before_reset(engine)
                except StressorUnsupportedError as exc:
                    self._mark_unsupported(application, str(exc))

    def after_reset(self, engine: Any, observation: Any) -> None:
        for stressor, application in zip(
            self.stressors, self.applications, strict=True
        ):
            if application.status != "requested":
                continue
            try:
                evidence = stressor.after_reset(engine, observation)
            except StressorUnsupportedError as exc:
                self._mark_unsupported(application, str(exc))
                continue
            application.status = "applied"
            application.applied_parameters = dict(stressor.applied_parameters)
            application.backend_evidence = dict(evidence)

    def transform_observation(self, observation: Any, *, step_index: int) -> Any:
        transformed = observation
        for stressor, application in zip(
            self.stressors, self.applications, strict=True
        ):
            if (
                application.status != "applied"
                or "observation" not in stressor.application_points
            ):
                continue
            try:
                transformed = stressor.transform_observation(
                    transformed, step_index=step_index
                )
            except StressorUnsupportedError as exc:
                self._mark_unsupported(application, str(exc))
        return transformed

    def transform_action(
        self, action: Any, *, observation: Any, step_index: int
    ) -> Any:
        transformed = action
        for stressor, application in zip(
            self.stressors, self.applications, strict=True
        ):
            if (
                application.status != "applied"
                or "action" not in stressor.application_points
            ):
                continue
            try:
                transformed = stressor.transform_action(
                    transformed,
                    observation=observation,
                    step_index=step_index,
                )
            except StressorUnsupportedError as exc:
                self._mark_unsupported(application, str(exc))
        return transformed

    def before_step(self, engine: Any, *, step_index: int) -> None:
        for stressor, application in zip(
            self.stressors, self.applications, strict=True
        ):
            if application.status == "applied":
                try:
                    stressor.before_step(engine, step_index=step_index)
                except StressorUnsupportedError as exc:
                    self._mark_unsupported(application, str(exc))

    def after_step(self, engine: Any, info: dict[str, Any], *, step_index: int) -> None:
        for stressor, application in zip(
            self.stressors, self.applications, strict=True
        ):
            if application.status == "applied":
                try:
                    stressor.after_step(engine, info, step_index=step_index)
                except StressorUnsupportedError as exc:
                    self._mark_unsupported(application, str(exc))

    def get_state(self) -> dict[str, Any]:
        return {
            "format": STRESSOR_CONTEXT_FORMAT,
            "condition_id": self.condition_id,
            "episode_seed": self.episode_seed,
            "context": self.context.to_dict(),
            "composition_order": list(self.composition_order),
            "stressors": {
                stressor.stressor_id: stressor.get_state()
                for stressor, application in zip(
                    self.stressors, self.applications, strict=True
                )
                if application.status == "applied"
            },
        }

    def drain_failure_events(self) -> list[Any]:
        events: list[Any] = []
        for stressor in self.stressors:
            payloads = stressor.drain_failure_events()
            if payloads:
                events.extend(payloads)
        return events

    def set_state(self, state: dict[str, Any], *, engine: Any | None = None) -> None:
        if state.get("format") != STRESSOR_CONTEXT_FORMAT:
            raise ValueError(
                f"Unsupported stressor context format: {state.get('format')}"
            )
        if tuple(state.get("composition_order", [])) != self.composition_order:
            raise StressorCompositionError(
                "Cannot restore stressor state into a different composition order"
            )
        if state.get("condition_id") != self.condition_id:
            raise StressorCompositionError(
                "Cannot restore stressor state into a different condition"
            )
        if int(state.get("episode_seed", -1)) != self.episode_seed:
            raise StressorCompositionError(
                "Cannot restore stressor state into a different episode seed"
            )
        if state.get("context") != self.context.to_dict():
            raise StressorCompositionError(
                "Cannot restore stressor state into a different engine or task context"
            )
        stressor_states = state.get("stressors", {})
        if not isinstance(stressor_states, dict):
            raise ValueError("stressor state payload must be a mapping")
        for stressor, application in zip(
            self.stressors, self.applications, strict=True
        ):
            if application.status != "applied":
                continue
            payload = stressor_states.get(stressor.stressor_id)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Missing state for applied stressor '{stressor.stressor_id}'"
                )
            stressor.set_state(payload, engine=engine)

    def manifest(self) -> dict[str, Any]:
        return {
            "format": STRESSOR_CONTEXT_FORMAT,
            **self.application_context(),
            "episode_seed": self.episode_seed,
            "context": self.context.to_dict(),
            "unsupported_policy": self.unsupported_policy,
            "final_state": self.get_state(),
        }

    def application_context(self) -> dict[str, Any]:
        """Return applied/requested metadata without serializing runtime state."""

        return {
            "condition_id": self.condition_id,
            "composition_order": list(self.composition_order),
            "applications": [
                application.to_dict() for application in self.applications
            ],
        }

    def _prepare(self) -> None:
        for index, (spec, stressor) in enumerate(
            zip(self.specs, self.stressors, strict=True)
        ):
            seed = (
                spec.seed
                if spec.seed is not None
                else _derived_seed(self.episode_seed, index, spec.stressor_id)
            )
            stressor.reset(spec, seed=seed)
            application = StressorApplication(
                stressor_id=stressor.stressor_id,
                category=stressor.category,
                composition_index=index,
                application_points=stressor.application_points,
                severity_domain=stressor.severity_domain,
                lifetime=stressor.lifetime,
                observable_by_policy=stressor.observable_by_policy,
                privileged=stressor.privileged,
                requested=spec.to_dict(),
                seed=seed,
                applied_parameters=dict(stressor.applied_parameters),
            )
            self.applications.append(application)
            if spec.severity == 0.0:
                application.status = "skipped"
                application.reason = "severity_zero_clean_condition"
                continue
            reason = stressor.support_reason(self.context)
            if reason is not None:
                self._mark_unsupported(application, reason)

    def _mark_unsupported(self, application: StressorApplication, reason: str) -> None:
        application.status = "unsupported"
        application.reason = reason
        if self.unsupported_policy == "error":
            raise UnsupportedStressorError(
                f"Stressor '{application.stressor_id}' is unsupported: {reason}"
            )


def _validate_composition(stressors: list[Stressor]) -> None:
    ids = [stressor.stressor_id for stressor in stressors]
    duplicates = sorted(
        stressor_id for stressor_id in set(ids) if ids.count(stressor_id) > 1
    )
    if duplicates:
        raise StressorCompositionError(
            f"Duplicate stressors are incompatible: {', '.join(duplicates)}"
        )
    for stressor in stressors:
        conflicts = sorted(set(ids).intersection(stressor.conflicts_with))
        if conflicts:
            raise StressorCompositionError(
                f"Stressor '{stressor.stressor_id}' conflicts with: {', '.join(conflicts)}"
            )


def _derived_seed(episode_seed: int, index: int, stressor_id: str) -> int:
    payload = f"nyssa-stressor-seed-v1:{episode_seed}:{index}:{stressor_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)
