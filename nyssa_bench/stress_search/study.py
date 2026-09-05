from __future__ import annotations

import csv
import hashlib
import html
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from nyssa_bench.stress_search.protocol import (
    STRESS_SEARCH_STUDY_FORMAT,
    StressObservation,
    StressProposal,
    StressSearchSpace,
)
from nyssa_bench.stress_search.samplers import StressSampler, make_stress_sampler
from nyssa_bench.validity.protocol import BenchmarkValidityReport


STRESS_SEARCH_SPEC_FORMAT = "nyssa-stress-search-study-spec-v1"
STRESS_SEARCH_REPORT_FORMAT = "nyssa-stress-search-report-v1"
STRESS_PROPOSAL_BATCH_FORMAT = "nyssa-stress-proposal-batch-v1"


@dataclass(frozen=True)
class StressSearchStudySpec:
    study_id: str
    search_space: StressSearchSpace
    sampler_id: str
    study_seed: int
    discovery_budget: int
    batch_size: int = 1
    confirmation_budget: int = 20
    confirmation_repeats: int = 10
    boundary_success_rate_range: tuple[float, float] = (0.1, 0.9)
    outcome_success_threshold: float = 0.5
    sampler_config: dict[str, Any] = field(default_factory=dict)
    claim_mode: str = "exploratory"
    benchmark_validity: dict[str, Any] | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    stopping_criteria: tuple[str, ...] = (
        "discovery_budget",
        "feasible_unique_points",
        "design_exhaustion",
        "optional_boundary_tolerance",
    )
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.study_id.strip():
            raise ValueError("study_id must be non-empty")
        if self.study_seed < 0 or self.discovery_budget <= 0 or self.batch_size <= 0:
            raise ValueError("study seed and discovery budget/batch size are invalid")
        if self.confirmation_budget <= 0 or self.confirmation_repeats <= 0:
            raise ValueError("confirmation budget and repeats must be positive")
        if self.confirmation_budget < self.confirmation_repeats:
            raise ValueError("confirmation budget must cover at least one condition")
        low, high = self.boundary_success_rate_range
        if not (0.0 <= low < high <= 1.0):
            raise ValueError("boundary success-rate range must lie within [0, 1]")
        if not math.isfinite(self.outcome_success_threshold) or not 0.0 <= self.outcome_success_threshold <= 1.0:
            raise ValueError("outcome_success_threshold must be finite and within [0, 1]")
        if self.schema_version != 1:
            raise ValueError(f"unsupported stress-search spec version: {self.schema_version}")
        json.dumps(self.sampler_config, allow_nan=False, sort_keys=True)
        if self.claim_mode not in {"exploratory", "benchmark_claim"}:
            raise ValueError("claim_mode must be exploratory or benchmark_claim")
        if not self.provenance.get("producer_id") or not self.provenance.get(
            "study_purpose"
        ):
            raise ValueError("study provenance requires producer_id and study_purpose")
        json.dumps(self.provenance, allow_nan=False, sort_keys=True)
        if not self.stopping_criteria or len(self.stopping_criteria) != len(
            set(self.stopping_criteria)
        ):
            raise ValueError("stopping_criteria must be non-empty and unique")
        validity_report = (
            BenchmarkValidityReport.from_dict(self.benchmark_validity)
            if self.benchmark_validity is not None
            else None
        )
        if self.claim_mode == "benchmark_claim":
            if validity_report is None or not validity_report.claim_ready:
                raise ValueError(
                    "benchmark_claim studies require a claim-ready BenchmarkValidity report"
                )
            if self.provenance.get("benchmark_id") != validity_report.benchmark_id:
                raise ValueError(
                    "stress-search benchmark_id does not match BenchmarkValidity evidence"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": STRESS_SEARCH_SPEC_FORMAT,
            "schema_version": self.schema_version,
            "study_id": self.study_id,
            "search_space": self.search_space.to_dict(),
            "sampler_id": self.sampler_id,
            "study_seed": self.study_seed,
            "discovery_budget": self.discovery_budget,
            "batch_size": self.batch_size,
            "confirmation_budget": self.confirmation_budget,
            "confirmation_repeats": self.confirmation_repeats,
            "boundary_success_rate_range": list(self.boundary_success_rate_range),
            "outcome_success_threshold": self.outcome_success_threshold,
            "sampler_config": self.sampler_config,
            "claim_mode": self.claim_mode,
            "benchmark_validity": self.benchmark_validity,
            "provenance": self.provenance,
            "stopping_criteria": list(self.stopping_criteria),
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StressSearchStudySpec":
        if data.get("format") != STRESS_SEARCH_SPEC_FORMAT:
            raise ValueError(f"unsupported stress-search spec format: {data.get('format')}")
        _reject_unknown(
            data,
            {
                "format",
                "schema_version",
                "study_id",
                "search_space",
                "sampler_id",
                "study_seed",
                "discovery_budget",
                "batch_size",
                "confirmation_budget",
                "confirmation_repeats",
                "boundary_success_rate_range",
                "outcome_success_threshold",
                "sampler_config",
                "claim_mode",
                "benchmark_validity",
                "provenance",
                "stopping_criteria",
            },
            "stress-search spec",
        )
        space = data.get("search_space")
        config = data.get("sampler_config", {})
        boundary_range = data.get("boundary_success_rate_range", [0.1, 0.9])
        if not isinstance(space, Mapping) or not isinstance(config, Mapping):
            raise ValueError("study search_space and sampler_config must be mappings")
        validity = data.get("benchmark_validity")
        provenance = data.get("provenance", {})
        stopping_criteria = data.get("stopping_criteria", [])
        if validity is not None and not isinstance(validity, Mapping):
            raise ValueError("benchmark_validity must be a mapping or null")
        if not isinstance(provenance, Mapping):
            raise ValueError("study provenance must be a mapping")
        if not isinstance(stopping_criteria, list) or not all(
            isinstance(item, str) for item in stopping_criteria
        ):
            raise ValueError("stopping_criteria must be a list of strings")
        if not isinstance(boundary_range, list) or len(boundary_range) != 2:
            raise ValueError("boundary_success_rate_range must have two values")
        return cls(
            study_id=str(data.get("study_id", "")),
            search_space=StressSearchSpace.from_dict(space),
            sampler_id=str(data.get("sampler_id", "")),
            study_seed=int(data.get("study_seed", -1)),
            discovery_budget=int(data.get("discovery_budget", 0)),
            batch_size=int(data.get("batch_size", 1)),
            confirmation_budget=int(data.get("confirmation_budget", 20)),
            confirmation_repeats=int(data.get("confirmation_repeats", 10)),
            boundary_success_rate_range=(
                float(boundary_range[0]),
                float(boundary_range[1]),
            ),
            outcome_success_threshold=float(data.get("outcome_success_threshold", 0.5)),
            sampler_config=dict(config),
            claim_mode=str(data.get("claim_mode", "exploratory")),
            benchmark_validity=dict(validity) if validity is not None else None,
            provenance=dict(provenance),
            stopping_criteria=tuple(stopping_criteria),
            schema_version=int(data.get("schema_version", 1)),
        )


class StressSearchStudy:
    def __init__(self, spec: StressSearchStudySpec) -> None:
        self.spec = spec
        self.sampler = make_stress_sampler(
            spec.sampler_id,
            spec.search_space,
            study_seed=spec.study_seed,
            budget=spec.discovery_budget,
            config=spec.sampler_config,
        )
        self.confirmation_proposals: list[StressProposal] = []
        self.confirmation_observations: dict[str, StressObservation] = {}
        self.confirmation_stopping_reason: str | None = None

    def propose(self, count: int | None = None) -> tuple[StressProposal, ...]:
        return self.sampler.propose(count or self.spec.batch_size)

    def observe(self, observations: Sequence[StressObservation]) -> None:
        self.sampler.update(tuple(observations))

    def select_confirmation_conditions(self) -> tuple[StressProposal, ...]:
        if self.sampler.pending_ids:
            raise RuntimeError("all discovery proposals must be observed before confirmation")
        if self.confirmation_proposals:
            return tuple(self.confirmation_proposals)
        self.confirmation_stopping_reason = None
        pairs = _opposite_pairs(self.sampler)
        if not pairs:
            self.confirmation_stopping_reason = "no_observed_success_failure_boundary"
            return ()
        condition_count = self.spec.confirmation_budget // self.spec.confirmation_repeats
        selected_points: list[tuple[dict[str, Any], tuple[str, str], float]] = []
        known = set()
        for distance, left, right in pairs:
            point = _midpoint(left.point, right.point, self.spec.search_space)
            try:
                point = self.spec.search_space.validate_point(point)
            except ValueError:
                continue
            key = _point_key(point)
            if key in known:
                continue
            selected_points.append((point, (left.proposal_id, right.proposal_id), distance))
            known.add(key)
            if len(selected_points) >= condition_count:
                break
        for condition_index, (point, parents, distance) in enumerate(selected_points):
            for repeat_index in range(self.spec.confirmation_repeats):
                index = len(self.confirmation_proposals)
                seed = _derived_confirmation_seed(
                    self.spec.study_seed, condition_index, repeat_index
                )
                proposal = StressProposal(
                    proposal_id=_confirmation_id(
                        self.spec.sha256, condition_index, repeat_index, point
                    ),
                    proposal_index=index,
                    point=point,
                    discovery_seed=seed,
                    phase="confirmation",
                    parent_proposal_ids=parents,
                    acquisition={
                        "strategy": "held_out_boundary_confirmation",
                        "condition_index": condition_index,
                        "repeat_index": repeat_index,
                        "normalized_parent_distance": distance,
                        "seed_namespace": "confirmation_v1",
                    },
                )
                self.confirmation_proposals.append(proposal)
        if not self.confirmation_proposals:
            self.confirmation_stopping_reason = "no_feasible_unique_boundary_midpoint"
        elif len(self.confirmation_proposals) < self.spec.confirmation_budget:
            self.confirmation_stopping_reason = "confirmation_condition_limit"
        return tuple(self.confirmation_proposals)

    def observe_confirmation(self, observations: Sequence[StressObservation]) -> None:
        known = {proposal.proposal_id for proposal in self.confirmation_proposals}
        batch_ids = [observation.proposal_id for observation in observations]
        if len(batch_ids) != len(set(batch_ids)):
            raise ValueError("confirmation observations contain duplicate IDs")
        for observation in observations:
            if observation.proposal_id not in known:
                raise ValueError(
                    f"unknown confirmation proposal: {observation.proposal_id}"
                )
            if observation.proposal_id in self.confirmation_observations:
                raise ValueError(
                    f"confirmation proposal already observed: {observation.proposal_id}"
                )
        self.confirmation_observations.update(
            (observation.proposal_id, observation) for observation in observations
        )

    @property
    def pending_confirmation_ids(self) -> tuple[str, ...]:
        return tuple(
            proposal.proposal_id
            for proposal in self.confirmation_proposals
            if proposal.proposal_id not in self.confirmation_observations
        )

    def summary(self) -> dict[str, Any]:
        discovery_statuses = Counter(
            observation.status for observation in self.sampler.observations.values()
        )
        pairs = _opposite_pairs(self.sampler)
        confirmation = _confirmation_summary(self)
        return {
            "sampler_id": self.spec.sampler_id,
            "discovery_budget": self.spec.discovery_budget,
            "discovery_proposals": len(self.sampler.proposals),
            "discovery_observations": len(self.sampler.observations),
            "discovery_status_counts": dict(sorted(discovery_statuses.items())),
            "pending_discovery": len(self.sampler.pending_ids),
            "candidate_boundary_pairs": len(pairs),
            "samples_to_first_boundary": _samples_to_first_boundary(pairs),
            "confirmation": confirmation,
            "stopping_reason": self.sampler.stopping_reason,
            "confirmation_stopping_reason": self.confirmation_stopping_reason,
            "claim_mode": self.spec.claim_mode,
            "claim_eligible": self.spec.claim_mode == "benchmark_claim",
            "benchmark_validity_report_sha256": (
                self.spec.benchmark_validity.get("report_sha256")
                if self.spec.benchmark_validity is not None
                else None
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "format": STRESS_SEARCH_STUDY_FORMAT,
            "spec": self.spec.to_dict(),
            "spec_sha256": self.spec.sha256,
            "sampler_state": self.sampler.state_dict(),
            "confirmation_proposals": [
                proposal.to_dict() for proposal in self.confirmation_proposals
            ],
            "confirmation_observations": [
                self.confirmation_observations[proposal.proposal_id].to_dict()
                for proposal in self.confirmation_proposals
                if proposal.proposal_id in self.confirmation_observations
            ],
            "pending_confirmation_ids": list(self.pending_confirmation_ids),
            "confirmation_stopping_reason": self.confirmation_stopping_reason,
            "summary": self.summary(),
        }
        payload["study_sha256"] = _sha256(payload)
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StressSearchStudy":
        if data.get("format") != STRESS_SEARCH_STUDY_FORMAT:
            raise ValueError(f"unsupported stress-search study format: {data.get('format')}")
        _reject_unknown(
            data,
            {
                "format",
                "spec",
                "spec_sha256",
                "sampler_state",
                "confirmation_proposals",
                "confirmation_observations",
                "pending_confirmation_ids",
                "confirmation_stopping_reason",
                "summary",
                "study_sha256",
            },
            "stress-search study",
        )
        spec_raw = data.get("spec")
        sampler_state = data.get("sampler_state")
        proposals_raw = data.get("confirmation_proposals")
        observations_raw = data.get("confirmation_observations")
        if not isinstance(spec_raw, Mapping) or not isinstance(sampler_state, Mapping):
            raise ValueError("study spec and sampler state must be mappings")
        if not isinstance(proposals_raw, list) or not isinstance(observations_raw, list):
            raise ValueError("confirmation proposals and observations must be lists")
        spec = StressSearchStudySpec.from_dict(spec_raw)
        if data.get("spec_sha256") != spec.sha256:
            raise ValueError("stress-search spec hash mismatch")
        study = cls(spec)
        study.sampler.load_state_dict(sampler_state)
        study.confirmation_proposals = [
            StressProposal.from_dict(item) for item in proposals_raw
        ]
        if [proposal.proposal_index for proposal in study.confirmation_proposals] != list(
            range(len(study.confirmation_proposals))
        ):
            raise ValueError("confirmation proposal indices must be contiguous")
        if any(
            proposal.phase != "confirmation"
            for proposal in study.confirmation_proposals
        ):
            raise ValueError("confirmation proposal list contains discovery proposals")
        if len({proposal.proposal_id for proposal in study.confirmation_proposals}) != len(
            study.confirmation_proposals
        ):
            raise ValueError("confirmation proposal IDs must be unique")
        discovery_ids = {item.proposal_id for item in study.sampler.proposals}
        for proposal in study.confirmation_proposals:
            study.spec.search_space.validate_point(proposal.point)
            condition_index = _int_value(
                proposal.acquisition.get("condition_index"), "condition_index"
            )
            repeat_index = _int_value(
                proposal.acquisition.get("repeat_index"), "repeat_index"
            )
            expected_seed = _derived_confirmation_seed(
                study.spec.study_seed, condition_index, repeat_index
            )
            expected_id = _confirmation_id(
                study.spec.sha256,
                condition_index,
                repeat_index,
                proposal.point,
            )
            if proposal.discovery_seed != expected_seed or proposal.proposal_id != expected_id:
                raise ValueError("confirmation proposal identity or seed is inconsistent")
            if len(proposal.parent_proposal_ids) != 2 or not set(
                proposal.parent_proposal_ids
            ) <= discovery_ids:
                raise ValueError(
                    "confirmation proposal must reference two discovery parents"
                )
            parent_observations = [
                study.sampler.observations.get(parent_id)
                for parent_id in proposal.parent_proposal_ids
            ]
            if (
                any(item is None for item in parent_observations)
                or {item.success for item in parent_observations if item is not None}
                != {False, True}
            ):
                raise ValueError(
                    "confirmation parents must have opposite policy outcomes"
                )
        if {
            proposal.discovery_seed for proposal in study.confirmation_proposals
        }.intersection(
            proposal.discovery_seed for proposal in study.sampler.proposals
        ):
            raise ValueError("confirmation and discovery seeds must be disjoint")
        study.observe_confirmation(
            tuple(StressObservation.from_dict(item) for item in observations_raw)
        )
        reason = data.get("confirmation_stopping_reason")
        study.confirmation_stopping_reason = str(reason) if reason is not None else None
        if list(study.pending_confirmation_ids) != data.get(
            "pending_confirmation_ids", []
        ):
            raise ValueError("pending confirmation IDs do not match study state")
        expected = study.to_dict()
        if data.get("summary") != expected["summary"]:
            raise ValueError("stress-search study summary does not match evidence")
        if data.get("study_sha256") != expected["study_sha256"]:
            raise ValueError("stress-search study hash mismatch")
        return study


def load_stress_search_study(path: str | Path) -> StressSearchStudy:
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw) if path.suffix.lower() == ".json" else yaml.safe_load(raw)
    if not isinstance(data, Mapping):
        raise ValueError("stress-search study must contain a mapping")
    return StressSearchStudy.from_dict(data)


def load_stress_search_spec(path: str | Path) -> StressSearchStudySpec:
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw) if path.suffix.lower() == ".json" else yaml.safe_load(raw)
    if not isinstance(data, Mapping):
        raise ValueError("stress-search spec must contain a mapping")
    return StressSearchStudySpec.from_dict(data)


def load_stress_observations(path: str | Path) -> tuple[StressObservation, ...]:
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw) if path.suffix.lower() == ".json" else yaml.safe_load(raw)
    if isinstance(data, Mapping) and "observations" in data:
        data = data["observations"]
    if not isinstance(data, list) or not all(isinstance(item, Mapping) for item in data):
        raise ValueError("stress observations must be a list of mappings")
    return tuple(StressObservation.from_dict(item) for item in data)


def write_stress_search_study(study: StressSearchStudy, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(study.to_dict(), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def write_stress_proposals(
    proposals: Sequence[StressProposal],
    path: str | Path,
    *,
    search_space: StressSearchSpace | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "format": STRESS_PROPOSAL_BATCH_FORMAT,
                "proposals": [
                    {
                        **proposal.to_dict(),
                        "stressor_config": search_space.stressor_config(
                            proposal
                        ).to_dict()
                        if search_space is not None
                        else None,
                    }
                    for proposal in proposals
                ]
            },
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def compare_stress_search_studies(
    studies: Sequence[StressSearchStudy],
) -> dict[str, Any]:
    if len(studies) < 2:
        raise ValueError("sample-efficiency comparison requires at least two studies")
    space_hashes = {study.spec.search_space.sha256 for study in studies}
    budgets = {study.spec.discovery_budget for study in studies}
    if len(space_hashes) != 1 or len(budgets) != 1:
        raise ValueError("stress-search studies must share search space and budget")
    sampler_ids = {study.spec.sampler_id for study in studies}
    required_samplers = {"random", "latin_hypercube", "boundary_adaptive"}
    if not required_samplers <= sampler_ids:
        raise ValueError(
            "comparison requires random, Latin-hypercube, and boundary-adaptive studies"
        )
    seed_sets = {
        sampler_id: {
            study.spec.study_seed
            for study in studies
            if study.spec.sampler_id == sampler_id
        }
        for sampler_id in required_samplers
    }
    if len({tuple(sorted(values)) for values in seed_sets.values()}) != 1:
        raise ValueError("sampler comparisons require matched study-seed sets")
    comparison_contracts = {
        (
            study.spec.confirmation_budget,
            study.spec.confirmation_repeats,
            study.spec.boundary_success_rate_range,
            study.spec.outcome_success_threshold,
        )
        for study in studies
    }
    if len(comparison_contracts) != 1:
        raise ValueError("stress-search confirmation contracts do not match")
    rows = []
    for study in studies:
        summary = study.summary()
        study_status = (
            "complete"
            if not study.sampler.pending_ids
            and bool(study.confirmation_proposals)
            and not study.pending_confirmation_ids
            else "incomplete"
        )
        rows.append(
            {
                "study_id": study.spec.study_id,
                "sampler_id": study.spec.sampler_id,
                "study_seed": study.spec.study_seed,
                "discovery_proposals": summary["discovery_proposals"],
                "samples_to_first_boundary": summary["samples_to_first_boundary"],
                "candidate_boundary_pairs": summary["candidate_boundary_pairs"],
                "confirmed_boundaries": summary["confirmation"][
                    "confirmed_boundary_count"
                ],
                "confirmation_coverage": summary["confirmation"]["coverage"],
                "confirmation_intervals": summary["confirmation"]["conditions"],
                "study_status": study_status,
                "claim_eligible": summary["claim_eligible"],
            }
        )
    baseline_by_seed: dict[int, int] = {}
    for row in rows:
        samples = row["samples_to_first_boundary"]
        if (
            row["sampler_id"] in {"random", "latin_hypercube"}
            and row["study_status"] == "complete"
            and samples is not None
        ):
            seed = int(row["study_seed"])
            baseline_by_seed[seed] = min(
                int(samples), baseline_by_seed.get(seed, int(samples))
            )
    for row in rows:
        samples = row["samples_to_first_boundary"]
        matched_baseline = baseline_by_seed.get(int(row["study_seed"]))
        row["sample_efficiency_ratio_vs_best_baseline"] = (
            matched_baseline / samples
            if matched_baseline is not None
            and samples not in {None, 0}
            and row["study_status"] == "complete"
            else None
        )
    report = {
        "format": STRESS_SEARCH_REPORT_FORMAT,
        "search_space_sha256": next(iter(space_hashes)),
        "discovery_budget": next(iter(budgets)),
        "baseline_samples_to_boundary_by_seed": {
            str(seed): samples for seed, samples in sorted(baseline_by_seed.items())
        },
        "comparison_complete": all(row["study_status"] == "complete" for row in rows),
        "claim_eligible": all(
            row["claim_eligible"] and row["study_status"] == "complete"
            for row in rows
        ),
        "matched_study_seeds": sorted(next(iter(seed_sets.values()))),
        "studies": rows,
        "uncertainty": {
            "boundary_confirmation": "Wilson 95% binomial interval per held-out condition",
            "samples_to_first_boundary": "descriptive per matched study seed; no interval is inferred from one seed",
        },
        "interpretation": "lower samples_to_first_boundary is more sample efficient; confirmation intervals retain uncertainty",
    }
    report["report_sha256"] = _sha256(report)
    return report


def write_stress_search_report(
    report: Mapping[str, Any], out_dir: str | Path
) -> dict[str, Path]:
    if report.get("format") != STRESS_SEARCH_REPORT_FORMAT:
        raise ValueError("unsupported stress-search report format")
    hash_payload = dict(report)
    observed_hash = hash_payload.pop("report_sha256", None)
    if observed_hash != _sha256(hash_payload):
        raise ValueError("stress-search report hash mismatch")
    if not isinstance(report.get("studies"), list):
        raise ValueError("stress-search report studies must be a list")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "stress_search_report.json"
    csv_path = out_dir / "stress_search_report.csv"
    html_path = out_dir / "stress_search_report.html"
    json_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    rows = list(report.get("studies", []))
    fields = [
        "study_id",
        "sampler_id",
        "study_seed",
        "study_status",
        "discovery_proposals",
        "samples_to_first_boundary",
        "candidate_boundary_pairs",
        "confirmed_boundaries",
        "confirmation_coverage",
        "sample_efficiency_ratio_vs_best_baseline",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    body_rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(row.get(field, '')))}</td>" for field in fields) + "</tr>"
        for row in rows
    )
    html_path.write_text(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<title>NyssaBench stress-search report</title></head><body>"
        "<h1>Stress-search sample efficiency</h1>"
        f"<p>Comparison complete: {html.escape(str(report.get('comparison_complete', False)))}; "
        f"claim eligible: {html.escape(str(report.get('claim_eligible', False)))}</p>"
        "<table><thead><tr>"
        + "".join(f"<th>{html.escape(field)}</th>" for field in fields)
        + f"</tr></thead><tbody>{body_rows}</tbody></table></body></html>\n",
        encoding="utf-8",
    )
    return {"json": json_path, "csv": csv_path, "html": html_path}


def _opposite_pairs(
    sampler: StressSampler,
) -> list[tuple[float, StressProposal, StressProposal]]:
    valid = [
        (proposal, sampler.observations.get(proposal.proposal_id))
        for proposal in sampler.proposals
        if sampler.observations.get(proposal.proposal_id) is not None
        and sampler.observations[proposal.proposal_id].status
        in {"success", "policy_failure"}
    ]
    pairs = []
    for index, (left, left_observation) in enumerate(valid):
        for right, right_observation in valid[index + 1 :]:
            if left_observation is None or right_observation is None:
                continue
            if left_observation.success == right_observation.success:
                continue
            distance = _normalized_distance(
                left.point, right.point, sampler.space
            )
            pairs.append((distance, left, right))
    return sorted(
        pairs, key=lambda item: (item[0], item[1].proposal_id, item[2].proposal_id)
    )


def _confirmation_summary(study: StressSearchStudy) -> dict[str, Any]:
    grouped: dict[str, list[StressProposal]] = {}
    for proposal in study.confirmation_proposals:
        grouped.setdefault(_point_key(proposal.point), []).append(proposal)
    conditions = []
    confirmed = 0
    for key, proposals in sorted(grouped.items()):
        observations = [
            study.confirmation_observations[proposal.proposal_id]
            for proposal in proposals
            if proposal.proposal_id in study.confirmation_observations
        ]
        valid = [
            observation
            for observation in observations
            if observation.status in {"success", "policy_failure"}
        ]
        successes = sum(observation.status == "success" for observation in valid)
        rate = successes / len(valid) if valid else None
        interval = _wilson(successes, len(valid)) if valid else None
        low, high = study.spec.boundary_success_rate_range
        is_confirmed = bool(
            rate is not None
            and len(valid) == len(proposals)
            and low <= rate <= high
        )
        confirmed += is_confirmed
        conditions.append(
            {
                "point": json.loads(key),
                "requested_repeats": len(proposals),
                "observed_repeats": len(observations),
                "valid_policy_outcomes": len(valid),
                "status_counts": dict(
                    sorted(Counter(observation.status for observation in observations).items())
                ),
                "success_rate": rate,
                "success_rate_ci95": interval,
                "confirmed_boundary": is_confirmed,
            }
        )
    return {
        "requested": len(study.confirmation_proposals),
        "observed": len(study.confirmation_observations),
        "pending": len(study.pending_confirmation_ids),
        "coverage": len(study.confirmation_observations)
        / len(study.confirmation_proposals)
        if study.confirmation_proposals
        else 0.0,
        "confirmed_boundary_count": confirmed,
        "conditions": conditions,
    }


def _samples_to_first_boundary(
    pairs: Sequence[tuple[float, StressProposal, StressProposal]],
) -> int | None:
    if not pairs:
        return None
    return min(max(left.proposal_index, right.proposal_index) + 1 for _, left, right in pairs)


def _normalized_distance(
    left: Mapping[str, Any], right: Mapping[str, Any], space: StressSearchSpace
) -> float:
    terms = [
            (
                (float(left[variable.variable_id]) - float(right[variable.variable_id]))
                / (_numeric_bounds(variable)[1] - _numeric_bounds(variable)[0])
            )
            ** 2
            for variable in space.variables
        ]
    return math.sqrt(math.fsum(terms))


def _midpoint(
    left: Mapping[str, Any], right: Mapping[str, Any], space: StressSearchSpace
) -> dict[str, Any]:
    result = {}
    for variable in space.variables:
        value = (float(left[variable.variable_id]) + float(right[variable.variable_id])) / 2.0
        result[variable.variable_id] = int(round(value)) if variable.kind == "integer" else value
    return result


def _point_key(point: Mapping[str, Any]) -> str:
    return json.dumps(point, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _derived_confirmation_seed(seed: int, condition: int, repeat: int) -> int:
    payload = f"nyssa-stress-confirmation-v1:{seed}:{condition}:{repeat}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def _confirmation_id(spec_hash: str, condition: int, repeat: int, point: Mapping[str, Any]) -> str:
    payload = f"{spec_hash}:{condition}:{repeat}:{_point_key(point)}".encode()
    return f"confirmation-{hashlib.sha256(payload).hexdigest()[:20]}"


def _sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _wilson(successes: int, total: int) -> list[float]:
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z**2 / total
    center = (proportion + z**2 / (2.0 * total)) / denominator
    margin = z * math.sqrt((proportion * (1.0 - proportion) + z**2 / (4.0 * total)) / total) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _numeric_bounds(variable: Any) -> tuple[float, float]:
    if variable.lower is None or variable.upper is None:
        raise ValueError(f"{variable.variable_id} does not have numeric bounds")
    return float(variable.lower), float(variable.upper)


def _reject_unknown(data: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"unknown {label} fields: {', '.join(unknown)}")


def _int_value(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    return result
