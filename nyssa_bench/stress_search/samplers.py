from __future__ import annotations

import hashlib
import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from nyssa_bench.stress_search.protocol import (
    SearchVariable,
    StressObservation,
    StressProposal,
    StressSearchSpace,
)


STRESS_SAMPLER_STATE_FORMAT = "nyssa-stress-sampler-state-v1"


@dataclass(frozen=True)
class SamplerCapabilities:
    sampler_id: str
    sampler_version: str
    supported_variable_kinds: tuple[str, ...]
    supported_constraints: tuple[str, ...]
    batch_semantics: str
    deterministic_seed_scheme: str
    objective: str
    uncertainty_model: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "sampler_id": self.sampler_id,
            "sampler_version": self.sampler_version,
            "supported_variable_kinds": list(self.supported_variable_kinds),
            "supported_constraints": list(self.supported_constraints),
            "batch_semantics": self.batch_semantics,
            "deterministic_seed_scheme": self.deterministic_seed_scheme,
            "objective": self.objective,
            "uncertainty_model": self.uncertainty_model,
        }


class StressSampler(ABC):
    capabilities: SamplerCapabilities

    def __init__(
        self,
        space: StressSearchSpace,
        *,
        study_seed: int,
        budget: int,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        if study_seed < 0 or budget <= 0:
            raise ValueError("study_seed must be non-negative and budget must be positive")
        unsupported_kinds = {
            variable.kind for variable in space.variables
        } - set(self.capabilities.supported_variable_kinds)
        unsupported_constraints = {
            constraint.kind for constraint in space.constraints
        } - set(self.capabilities.supported_constraints)
        if unsupported_kinds:
            raise ValueError(
                f"sampler {self.capabilities.sampler_id} does not support variable kinds: "
                + ", ".join(sorted(unsupported_kinds))
            )
        if unsupported_constraints:
            raise ValueError(
                f"sampler {self.capabilities.sampler_id} does not support constraints: "
                + ", ".join(sorted(unsupported_constraints))
            )
        self.space = space
        self.study_seed = int(study_seed)
        self.budget = int(budget)
        self.config = dict(config or {})
        json.dumps(self.config, allow_nan=False, sort_keys=True)
        self._validate_config()
        self.proposals: list[StressProposal] = []
        self.observations: dict[str, StressObservation] = {}
        self.candidate_cursor = 0
        self.stopping_reason: str | None = None

    def _validate_config(self) -> None:
        if self.config:
            raise ValueError(
                f"sampler {self.capabilities.sampler_id} does not accept configuration keys: "
                + ", ".join(sorted(self.config))
            )

    def propose(self, count: int = 1) -> tuple[StressProposal, ...]:
        if count <= 0:
            raise ValueError("proposal count must be positive")
        if self.stopping_reason is not None:
            return ()
        if self.capabilities.batch_semantics == "synchronous" and self.pending_ids:
            raise RuntimeError(
                f"sampler has unobserved proposals: {', '.join(self.pending_ids)}"
            )
        requested = min(count, self.budget - len(self.proposals))
        if requested <= 0:
            self.stopping_reason = "budget_exhausted"
            return ()
        emitted: list[StressProposal] = []
        known_points = {_point_key(proposal.point) for proposal in self.proposals}
        max_attempts = max(100, requested * 1000)
        attempts = 0
        while len(emitted) < requested and attempts < max_attempts:
            candidate = self._candidate(self.candidate_cursor)
            self.candidate_cursor += 1
            attempts += 1
            if candidate is None:
                self.stopping_reason = "design_exhausted"
                break
            point, acquisition, parents = candidate
            try:
                point = self.space.validate_point(point)
            except ValueError:
                continue
            key = _point_key(point)
            if key in known_points:
                continue
            proposal_index = len(self.proposals)
            seed = _derived_seed(
                self.study_seed,
                proposal_index,
                self.capabilities.sampler_id,
            )
            proposal = StressProposal(
                proposal_id=_proposal_id(
                    self.space.sha256,
                    self.capabilities.sampler_id,
                    proposal_index,
                    point,
                ),
                proposal_index=proposal_index,
                point=point,
                discovery_seed=seed,
                parent_proposal_ids=parents,
                acquisition=acquisition,
            )
            self.proposals.append(proposal)
            emitted.append(proposal)
            known_points.add(key)
        if len(emitted) < requested and self.stopping_reason is None:
            self.stopping_reason = "feasible_unique_points_exhausted"
        if len(self.proposals) >= self.budget and self.stopping_reason is None:
            self.stopping_reason = "budget_exhausted"
        return tuple(emitted)

    def update(self, observations: list[StressObservation] | tuple[StressObservation, ...]) -> None:
        known = {proposal.proposal_id for proposal in self.proposals}
        batch_ids = [observation.proposal_id for observation in observations]
        if len(batch_ids) != len(set(batch_ids)):
            raise ValueError("observation batch contains duplicate proposal IDs")
        for observation in observations:
            if observation.proposal_id not in known:
                raise ValueError(
                    f"observation references unknown proposal: {observation.proposal_id}"
                )
            if observation.proposal_id in self.observations:
                raise ValueError(
                    f"proposal already has an observation: {observation.proposal_id}"
                )
        self.observations.update(
            (observation.proposal_id, observation) for observation in observations
        )
        self._after_update()

    def _after_update(self) -> None:
        return None

    @property
    def pending_ids(self) -> tuple[str, ...]:
        return tuple(
            proposal.proposal_id
            for proposal in self.proposals
            if proposal.proposal_id not in self.observations
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "format": STRESS_SAMPLER_STATE_FORMAT,
            "capabilities": self.capabilities.to_dict(),
            "space": self.space.to_dict(),
            "space_sha256": self.space.sha256,
            "study_seed": self.study_seed,
            "budget": self.budget,
            "config": self.config,
            "candidate_cursor": self.candidate_cursor,
            "stopping_reason": self.stopping_reason,
            "proposals": [proposal.to_dict() for proposal in self.proposals],
            "observations": [
                self.observations[proposal.proposal_id].to_dict()
                for proposal in self.proposals
                if proposal.proposal_id in self.observations
            ],
            "pending_proposal_ids": list(self.pending_ids),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("format") != STRESS_SAMPLER_STATE_FORMAT:
            raise ValueError(f"unsupported sampler-state format: {state.get('format')}")
        unknown = sorted(
            set(state)
            - {
                "format",
                "capabilities",
                "space",
                "space_sha256",
                "study_seed",
                "budget",
                "config",
                "candidate_cursor",
                "stopping_reason",
                "proposals",
                "observations",
                "pending_proposal_ids",
            }
        )
        if unknown:
            raise ValueError(f"unknown sampler-state fields: {', '.join(unknown)}")
        capabilities = state.get("capabilities")
        if (
            not isinstance(capabilities, Mapping)
            or dict(capabilities) != self.capabilities.to_dict()
        ):
            raise ValueError("sampler state belongs to a different sampler")
        space_state = state.get("space")
        if not isinstance(space_state, Mapping):
            raise ValueError("sampler state search space must be a mapping")
        if StressSearchSpace.from_dict(space_state).sha256 != self.space.sha256:
            raise ValueError("sampler state embeds a different search space")
        if state.get("space_sha256") != self.space.sha256:
            raise ValueError("sampler state belongs to a different search space")
        if int(state.get("study_seed", -1)) != self.study_seed:
            raise ValueError("sampler state uses a different study seed")
        if int(state.get("budget", -1)) != self.budget:
            raise ValueError("sampler state uses a different budget")
        if state.get("config") != self.config:
            raise ValueError("sampler state uses different configuration")
        proposals_raw = state.get("proposals")
        observations_raw = state.get("observations")
        if not isinstance(proposals_raw, list) or not isinstance(observations_raw, list):
            raise ValueError("sampler state proposals and observations must be lists")
        proposals = [StressProposal.from_dict(item) for item in proposals_raw]
        if [proposal.proposal_index for proposal in proposals] != list(
            range(len(proposals))
        ):
            raise ValueError("sampler proposal indices must be contiguous")
        if len(proposals) > self.budget:
            raise ValueError("sampler state exceeds its discovery budget")
        if any(proposal.phase != "discovery" for proposal in proposals):
            raise ValueError("sampler state contains non-discovery proposals")
        known_parent_ids: set[str] = set()
        for proposal in proposals:
            point = self.space.validate_point(proposal.point)
            expected_seed = _derived_seed(
                self.study_seed,
                proposal.proposal_index,
                self.capabilities.sampler_id,
            )
            expected_id = _proposal_id(
                self.space.sha256,
                self.capabilities.sampler_id,
                proposal.proposal_index,
                point,
            )
            if proposal.discovery_seed != expected_seed or proposal.proposal_id != expected_id:
                raise ValueError("sampler proposal identity or seed is inconsistent")
            if not set(proposal.parent_proposal_ids) <= known_parent_ids:
                raise ValueError("sampler proposal references a non-prior parent")
            known_parent_ids.add(proposal.proposal_id)
        self.proposals = proposals
        self.observations = {}
        self.update(tuple(StressObservation.from_dict(item) for item in observations_raw))
        self.candidate_cursor = int(state.get("candidate_cursor", len(proposals)))
        if self.candidate_cursor < len(proposals):
            raise ValueError("candidate_cursor cannot precede emitted proposals")
        reason = state.get("stopping_reason")
        self.stopping_reason = str(reason) if reason is not None else None
        allowed_reasons = {
            None,
            "budget_exhausted",
            "design_exhausted",
            "feasible_unique_points_exhausted",
            "boundary_tolerance_reached",
        }
        if self.stopping_reason not in allowed_reasons:
            raise ValueError(f"unsupported sampler stopping reason: {self.stopping_reason}")
        if len(self.proposals) >= self.budget and self.stopping_reason not in {
            "budget_exhausted",
            "boundary_tolerance_reached",
        }:
            raise ValueError("completed sampler budget requires budget_exhausted status")
        if self.stopping_reason == "budget_exhausted" and len(self.proposals) < self.budget:
            raise ValueError("budget_exhausted cannot precede the discovery budget")
        if list(self.pending_ids) != state.get("pending_proposal_ids", []):
            raise ValueError("sampler pending proposal IDs do not match state")

    @abstractmethod
    def _candidate(
        self, cursor: int
    ) -> tuple[dict[str, Any], dict[str, Any], tuple[str, ...]] | None: ...


class RandomStressSampler(StressSampler):
    capabilities = SamplerCapabilities(
        sampler_id="random",
        sampler_version="1.0.0",
        supported_variable_kinds=("continuous", "integer", "categorical"),
        supported_constraints=("sum_le", "sum_ge", "forbidden_combination"),
        batch_semantics="independent_batch",
        deterministic_seed_scheme="sha256_study_seed_candidate_cursor",
        objective="coverage_baseline_without_adaptation",
        uncertainty_model="none",
    )

    def _candidate(self, cursor: int):
        rng = np.random.default_rng(
            _derived_seed(self.study_seed, cursor, self.capabilities.sampler_id)
        )
        return (
            {
                variable.variable_id: _sample_variable(variable, rng)
                for variable in self.space.variables
            },
            {"strategy": "uniform_random", "candidate_cursor": cursor},
            (),
        )


class LatinHypercubeStressSampler(StressSampler):
    capabilities = SamplerCapabilities(
        sampler_id="latin_hypercube",
        sampler_version="1.0.0",
        supported_variable_kinds=("continuous", "integer", "categorical"),
        supported_constraints=("sum_le", "sum_ge", "forbidden_combination"),
        batch_semantics="fixed_budget_design",
        deterministic_seed_scheme="per_variable_seeded_budget_permutation",
        objective="space_filling_baseline",
        uncertainty_model="none",
    )

    def __init__(
        self,
        space: StressSearchSpace,
        *,
        study_seed: int,
        budget: int,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            space, study_seed=study_seed, budget=budget, config=config
        )
        self._strata: dict[str, np.ndarray] = {}
        self._jitters: dict[str, np.ndarray] = {}
        for variable_index, variable in enumerate(self.space.variables):
            rng = np.random.default_rng(
                _derived_seed(
                    self.study_seed,
                    variable_index,
                    f"lhs:{variable.variable_id}",
                )
            )
            self._strata[variable.variable_id] = rng.permutation(self.budget)
            self._jitters[variable.variable_id] = rng.random(self.budget)

    def _candidate(self, cursor: int):
        if cursor >= self.budget:
            return None
        point = {}
        strata = {}
        for variable in self.space.variables:
            stratum = int(self._strata[variable.variable_id][cursor])
            jitter = float(self._jitters[variable.variable_id][cursor])
            point[variable.variable_id] = _stratified_variable(
                variable, stratum=stratum, jitter=jitter, budget=self.budget
            )
            strata[variable.variable_id] = stratum
        return point, {"strategy": "latin_hypercube", "strata": strata}, ()


class BoundaryStressSampler(StressSampler):
    capabilities = SamplerCapabilities(
        sampler_id="boundary_adaptive",
        sampler_version="1.0.0",
        supported_variable_kinds=("continuous", "integer"),
        supported_constraints=("sum_le", "sum_ge", "forbidden_combination"),
        batch_semantics="synchronous",
        deterministic_seed_scheme="sha256_seeded_warmup_then_boundary_midpoint_jitter",
        objective="locate_success_to_policy_failure_transition",
        uncertainty_model="nearest_opposite_outcome_distance_in_normalized_space",
    )

    def _validate_config(self) -> None:
        unknown = set(self.config) - {
            "warmup",
            "jitter_scale",
            "target_boundary_width",
            "min_valid_observations",
        }
        if unknown:
            raise ValueError(
                "boundary sampler has unknown configuration keys: "
                + ", ".join(sorted(unknown))
            )
        warmup = self.config.get(
            "warmup", max(4, 2 * len(self.space.variables) + 2)
        )
        if isinstance(warmup, bool) or int(warmup) != warmup or int(warmup) < 2:
            raise ValueError("boundary warmup must be an integer of at least 2")
        jitter = float(self.config.get("jitter_scale", 0.1))
        if not math.isfinite(jitter) or not 0.0 <= jitter <= 0.5:
            raise ValueError(
                "boundary jitter_scale must be finite and within [0, 0.5]"
            )
        target_width = self.config.get("target_boundary_width")
        if target_width is not None and (
            not math.isfinite(float(target_width))
            or not 0.0 < float(target_width) <= 1.0
        ):
            raise ValueError(
                "target_boundary_width must be finite and within (0, 1]"
            )
        minimum = self.config.get("min_valid_observations", warmup)
        if isinstance(minimum, bool) or int(minimum) != minimum or int(minimum) < 2:
            raise ValueError("min_valid_observations must be an integer of at least 2")

    def _after_update(self) -> None:
        target = self.config.get("target_boundary_width")
        if target is None or self.pending_ids:
            return
        valid = [
            (proposal, self.observations.get(proposal.proposal_id))
            for proposal in self.proposals
            if self.observations.get(proposal.proposal_id) is not None
            and self.observations[proposal.proposal_id].status
            in {"success", "policy_failure"}
        ]
        minimum = int(
            self.config.get(
                "min_valid_observations",
                self.config.get("warmup", max(4, 2 * len(self.space.variables) + 2)),
            )
        )
        if len(valid) < minimum or not _has_both_outcomes(valid):
            return
        _, distance = _closest_opposite_pair(valid, self.space)
        if distance <= float(target):
            self.stopping_reason = "boundary_tolerance_reached"

    def _candidate(self, cursor: int):
        warmup = int(self.config.get("warmup", max(4, 2 * len(self.space.variables) + 2)))
        valid = [
            (proposal, self.observations.get(proposal.proposal_id))
            for proposal in self.proposals
            if self.observations.get(proposal.proposal_id) is not None
            and self.observations[proposal.proposal_id].status
            in {"success", "policy_failure"}
        ]
        if len(self.proposals) < warmup or not _has_both_outcomes(valid):
            rng = np.random.default_rng(
                _derived_seed(self.study_seed, cursor, "boundary_warmup")
            )
            return (
                {
                    variable.variable_id: _sample_variable(variable, rng)
                    for variable in self.space.variables
                },
                {
                    "strategy": "boundary_warmup_or_exploration",
                    "valid_outcomes": len(valid),
                    "warmup": warmup,
                },
                (),
            )
        pair, distance = _closest_opposite_pair(valid, self.space)
        left, right = pair
        midpoint = _midpoint(left.point, right.point, self.space)
        rng = np.random.default_rng(
            _derived_seed(self.study_seed, cursor, "boundary_jitter")
        )
        jitter_scale = float(self.config.get("jitter_scale", 0.1)) / math.sqrt(
            1.0 + len(valid)
        )
        point = _jitter_point(midpoint, self.space, rng, jitter_scale)
        return (
            point,
            {
                "strategy": "nearest_opposite_midpoint",
                "normalized_pair_distance": distance,
                "jitter_scale": jitter_scale,
                "objective": self.capabilities.objective,
                "uncertainty_model": self.capabilities.uncertainty_model,
            },
            (left.proposal_id, right.proposal_id),
        )


def make_stress_sampler(
    sampler_id: str,
    space: StressSearchSpace,
    *,
    study_seed: int,
    budget: int,
    config: Mapping[str, Any] | None = None,
    state: Mapping[str, Any] | None = None,
) -> StressSampler:
    sampler_types: dict[str, Callable[..., StressSampler]] = {
        "random": RandomStressSampler,
        "latin_hypercube": LatinHypercubeStressSampler,
        "boundary_adaptive": BoundaryStressSampler,
    }
    try:
        sampler = sampler_types[sampler_id](
            space, study_seed=study_seed, budget=budget, config=config
        )
    except KeyError as exc:
        raise ValueError(f"unknown stress sampler: {sampler_id}") from exc
    if state is not None:
        sampler.load_state_dict(state)
    return sampler


def _sample_variable(variable: SearchVariable, rng: np.random.Generator) -> Any:
    if variable.kind == "categorical":
        return variable.choices[int(rng.integers(0, len(variable.choices)))]
    if variable.kind == "integer":
        lower, upper = _numeric_bounds(variable)
        return int(rng.integers(int(lower), int(upper) + 1))
    lower, upper = _numeric_bounds(variable)
    return float(rng.uniform(lower, upper))


def _stratified_variable(
    variable: SearchVariable, *, stratum: int, jitter: float, budget: int
) -> Any:
    position = (stratum + jitter) / budget
    if variable.kind == "categorical":
        return variable.choices[min(len(variable.choices) - 1, int(position * len(variable.choices)))]
    lower, upper = _numeric_bounds(variable)
    value = lower + position * (upper - lower)
    if variable.kind == "integer":
        return min(int(upper), max(int(lower), int(round(value))))
    return value


def _has_both_outcomes(values: list[tuple[StressProposal, StressObservation | None]]) -> bool:
    labels = {observation.success for _, observation in values if observation is not None}
    return labels == {False, True}


def _closest_opposite_pair(
    values: list[tuple[StressProposal, StressObservation | None]],
    space: StressSearchSpace,
) -> tuple[tuple[StressProposal, StressProposal], float]:
    successes = sorted(
        (
            proposal
            for proposal, observation in values
            if observation is not None and observation.success is True
        ),
        key=lambda proposal: proposal.proposal_id,
    )
    failures = sorted(
        (
            proposal
            for proposal, observation in values
            if observation is not None and observation.success is False
        ),
        key=lambda proposal: proposal.proposal_id,
    )
    if not successes or not failures:
        raise RuntimeError("boundary sampler has no opposite-outcome pair")
    success_matrix = _normalized_points(successes, space)
    failure_matrix = _normalized_points(failures, space)
    max_pairs_per_chunk = 1_000_000
    chunk_size = max(1, max_pairs_per_chunk // len(failures))
    best: tuple[float, str, str, StressProposal, StressProposal] | None = None
    for start in range(0, len(successes), chunk_size):
        stop = min(len(successes), start + chunk_size)
        squared = np.sum(
            (success_matrix[start:stop, None, :] - failure_matrix[None, :, :]) ** 2,
            axis=2,
        )
        flat_index = int(np.argmin(squared))
        row, column = np.unravel_index(flat_index, squared.shape)
        left = successes[start + int(row)]
        right = failures[int(column)]
        candidate = (
            math.sqrt(float(squared[row, column])),
            left.proposal_id,
            right.proposal_id,
            left,
            right,
        )
        if best is None or candidate[:3] < best[:3]:
            best = candidate
    assert best is not None
    return (best[3], best[4]), best[0]


def _normalized_points(
    proposals: list[StressProposal], space: StressSearchSpace
) -> np.ndarray:
    lower = np.asarray(
        [_numeric_bounds(variable)[0] for variable in space.variables], dtype=float
    )
    spans = np.asarray(
        [
            _numeric_bounds(variable)[1] - _numeric_bounds(variable)[0]
            for variable in space.variables
        ],
        dtype=float,
    )
    values = np.asarray(
        [
            [float(proposal.point[variable.variable_id]) for variable in space.variables]
            for proposal in proposals
        ],
        dtype=float,
    )
    return (values - lower) / spans


def _midpoint(
    left: Mapping[str, Any], right: Mapping[str, Any], space: StressSearchSpace
) -> dict[str, Any]:
    point = {}
    for variable in space.variables:
        value = (
            float(left[variable.variable_id]) + float(right[variable.variable_id])
        ) / 2.0
        point[variable.variable_id] = (
            int(round(value)) if variable.kind == "integer" else value
        )
    return point


def _jitter_point(
    point: Mapping[str, Any],
    space: StressSearchSpace,
    rng: np.random.Generator,
    scale: float,
) -> dict[str, Any]:
    result = {}
    for variable in space.variables:
        lower, upper = _numeric_bounds(variable)
        span = upper - lower
        value = float(point[variable.variable_id]) + float(rng.normal(0.0, scale)) * span
        value = min(upper, max(lower, value))
        result[variable.variable_id] = int(round(value)) if variable.kind == "integer" else value
    return result


def _point_key(point: Mapping[str, Any]) -> str:
    return json.dumps(point, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _proposal_id(
    space_sha256: str, sampler_id: str, index: int, point: Mapping[str, Any]
) -> str:
    payload = f"{space_sha256}:{sampler_id}:{index}:{_point_key(point)}".encode()
    return f"proposal-{hashlib.sha256(payload).hexdigest()[:20]}"


def _derived_seed(seed: int, index: int, value: str) -> int:
    payload = f"nyssa-stress-sampler-v1:{seed}:{index}:{value}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def _numeric_bounds(variable: SearchVariable) -> tuple[float, float]:
    if variable.lower is None or variable.upper is None:
        raise ValueError(f"{variable.variable_id} does not have numeric bounds")
    return float(variable.lower), float(variable.upper)
