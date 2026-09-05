from __future__ import annotations

import hashlib
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from nyssa_bench.core.episode import EpisodeResult
from nyssa_bench.real_evidence import RealEvidencePackage, RealEvidenceValidator
from nyssa_bench.regression import (
    PolicyCheckpointIdentity,
    RunArtifactReference,
    load_run_evidence,
)
from nyssa_bench.simreal.metrics import (
    failure_distribution_similarity,
    kendall_tau_b,
    mean_maximum_rank_violation,
    pearson_correlation,
    spearman_correlation,
)
from nyssa_bench.simreal.protocol import SimRealPair, SimRealStudySpec


SIM_REAL_REPORT_FORMAT = "nyssa-sim-real-study-report-v1"


def evaluate_sim_real_study(
    spec: SimRealStudySpec, *, spec_root: str | Path
) -> dict[str, Any]:
    root = Path(spec_root).resolve()
    observations = []
    errors = []
    cache: dict[tuple[str, str], Any] = {}
    for pair in spec.pairs:
        if not pair.included:
            continue
        try:
            observations.append(_load_pair(pair, root, cache))
        except Exception as exc:
            errors.append(
                {
                    "pair_id": pair.pair_id,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
    if errors:
        return _report(spec, "invalid", observations, errors, {})
    metrics = {
        "policy_rank": _policy_rank(spec, observations),
        "failure_distribution": _failure_distribution(spec, observations),
        "shift_response": _shift_response(spec, observations),
        "time_to_failure": _time_to_failure(spec, observations),
        "recovery_effect": _recovery_effect(spec, observations),
        "incremental_predictive_value": _incremental_predictive_value(
            spec, observations
        ),
    }
    primary_unavailable = [
        metric_id
        for metric_id in spec.primary_metrics
        if metrics[metric_id]["status"] != "available"
    ]
    status = "complete" if not primary_unavailable else "inconclusive"
    return _report(spec, status, observations, [], metrics)


def _load_pair(
    pair: SimRealPair,
    root: Path,
    cache: dict[tuple[str, str], Any],
) -> dict[str, Any]:
    simulation_key = ("simulation", pair.simulation.run_id)
    if simulation_key not in cache:
        reference = RunArtifactReference(
            run_dir=pair.simulation.run_dir,
            run_id=pair.simulation.run_id,
            artifact_binding="pinned",
            artifacts_sha256=pair.simulation.artifacts_sha256,
        )
        policy = PolicyCheckpointIdentity(
            policy_name=pair.simulation.policy_name,
            checkpoint_id=pair.simulation.checkpoint_id,
            checkpoint_sha256=pair.simulation.checkpoint_sha256,
            preprocessing_sha256=pair.simulation.preprocessing_sha256,
        )
        cache[simulation_key] = load_run_evidence(reference, policy, spec_root=root)
    run = cache[simulation_key]
    matches = [
        episode
        for episode in run.episodes
        if episode.task_id == pair.simulation.task_id
        and episode.seed == pair.simulation.episode_seed
        and episode.episode_index == pair.simulation.episode_index
    ]
    if len(matches) != 1:
        raise ValueError("simulation episode identity is missing or ambiguous")
    episode = matches[0]

    package_path = Path(pair.real.package_path)
    package_path = (
        package_path.resolve()
        if package_path.is_absolute()
        else (root / package_path).resolve()
    )
    real_key = ("real", pair.real.package_identity)
    if real_key not in cache:
        package = RealEvidencePackage.load(package_path)
        validation = RealEvidenceValidator().validate(package)
        validation.raise_for_errors()
        if not validation.claim_ready:
            raise ValueError("real evidence package is not claim-ready")
        if package.identity != pair.real.package_identity:
            raise ValueError("real evidence package identity mismatch")
        cache[real_key] = package
    package = cache[real_key]
    real = package.real_episode
    if real.identity.episode_id != pair.real.real_episode_id:
        raise ValueError("real episode identity mismatch")
    if real.identity.trial_id != pair.real.trial_id:
        raise ValueError("real trial identity mismatch")
    if real.identity.policy_id != pair.policy_id:
        raise ValueError("real policy identity differs from pair policy")
    if real.identity.checkpoint_sha256 != pair.simulation.checkpoint_sha256:
        raise ValueError("simulation and real checkpoint hashes differ")
    if real.outcome.task_id != pair.task_id:
        raise ValueError("real task identity differs from pair task")
    variant_ids = {variant.variant_id for variant in package.reconstructed_variants}
    if pair.real.variant_id not in variant_ids or pair.real.variant_id not in package.mapping.variant_ids:
        raise ValueError("real/sim variant mapping is missing")
    return _observation(pair, episode, package)


def _observation(
    pair: SimRealPair, episode: EpisodeResult, package: RealEvidencePackage
) -> dict[str, Any]:
    sim_event = _first_event(episode.failure_ledger.events if episode.failure_ledger else ())
    real_events = package.real_episode.failure_events
    real_event = min(real_events, key=lambda item: (int(item["onset_step"]), str(item["event_id"]))) if real_events else None
    sim_failure_time = (
        float(sim_event.onset_step) * pair.sim_step_seconds if sim_event else None
    )
    real_failure_time = (
        float(real_event["onset_step"]) * pair.real_event_step_seconds
        if real_event
        else None
    )
    real_recovery = package.metadata.get("matched_recovery_gain_by_trial", {})
    real_recovery_gain = (
        _finite(real_recovery.get(pair.real.trial_id))
        if isinstance(real_recovery, Mapping)
        else None
    )
    return {
        "pair_id": pair.pair_id,
        "policy_id": pair.policy_id,
        "task_id": pair.task_id,
        "shift_id": pair.shift_id,
        "trial_id": pair.real.trial_id,
        "severity": pair.severity,
        "sim_success": episode.success,
        "real_success": package.real_episode.outcome.success,
        "sim_failure_category": sim_event.category if sim_event else None,
        "real_failure_category": str(real_event["category"])
        if real_event
        else None,
        "sim_failure_time_seconds": sim_failure_time,
        "real_failure_time_seconds": real_failure_time,
        "sim_time_censored": sim_failure_time is None,
        "real_time_censored": real_failure_time is None,
        "sim_censor_reason": None
        if sim_failure_time is not None
        else "success"
        if episode.success
        else "truncated"
        if episode.steps and episode.steps[-1].truncated
        else "failure_time_missing",
        "real_censor_reason": None
        if real_failure_time is not None
        else "success"
        if package.real_episode.outcome.success
        else "truncated"
        if package.real_episode.outcome.truncated
        else "failure_time_missing",
        "sim_duration_seconds": len(episode.steps) * pair.sim_step_seconds,
        "real_duration_seconds": package.real_episode.outcome.duration_seconds,
        "sim_recovery_gain": _finite(episode.metrics.get("counterfactual_recovery_gain")),
        "real_recovery_gain": real_recovery_gain,
        "real_intervention_count": len(package.real_episode.outcome.interventions),
    }


def _policy_rank(spec: SimRealStudySpec, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["policy_id"])].append(row)
    policies = sorted(grouped)
    simulated = [mean(float(item["sim_success"]) for item in grouped[key]) for key in policies]
    real = [mean(float(item["real_success"]) for item in grouped[key]) for key in policies]
    if len(policies) < 2:
        return _unavailable("at least two paired policies are required", len(policies))
    if len(set(simulated)) < 2 or len(set(real)) < 2:
        return _unavailable(
            "policy ranking is undefined when either domain has no performance variation",
            len(policies),
        )
    metric_functions = {
        "pearson": pearson_correlation,
        "spearman": spearman_correlation,
        "kendall_tau_b": kendall_tau_b,
        "mean_maximum_rank_violation": mean_maximum_rank_violation,
    }
    estimates = {
        name: function(simulated, real) for name, function in metric_functions.items()
    }
    uncertainty = {}
    for name, function in metric_functions.items():
        uncertainty[name] = _cluster_bootstrap(
            rows,
            spec,
            lambda sample: _rank_statistic(sample, function),
        )
    return {
        "status": "available",
        "policies": policies,
        "simulated_success": simulated,
        "real_success": real,
        "estimates": estimates,
        "sample_size": len(policies),
        "uncertainty": uncertainty,
    }


def _rank_statistic(
    rows: Sequence[Mapping[str, Any]],
    function: Callable[[Sequence[float], Sequence[float]], float | None],
) -> float | None:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["policy_id"])].append(row)
    if len(grouped) < 2:
        return None
    policies = sorted(grouped)
    return function(
        [mean(float(item["sim_success"]) for item in grouped[key]) for key in policies],
        [mean(float(item["real_success"]) for item in grouped[key]) for key in policies],
    )


def _failure_distribution(spec: SimRealStudySpec, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    sim = Counter(str(row["sim_failure_category"] or "none") for row in rows)
    real = Counter(str(row["real_failure_category"] or "none") for row in rows)
    estimate = failure_distribution_similarity(sim, real)
    if estimate is None:
        return _unavailable("failure distributions are empty", len(rows))
    return {
        "status": "available",
        "similarity": estimate,
        "simulated_counts": dict(sorted(sim.items())),
        "real_counts": dict(sorted(real.items())),
        "sample_size": len(rows),
        "ci95": _cluster_bootstrap(
            rows,
            spec,
            lambda sample: failure_distribution_similarity(
                Counter(str(row["sim_failure_category"] or "none") for row in sample),
                Counter(str(row["real_failure_category"] or "none") for row in sample),
            ),
        ),
    }


def _shift_response(spec: SimRealStudySpec, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    points = _degradation_points(rows)
    if len(points) < 2:
        return _unavailable("at least two policy-task degradation pairs are required", len(points))
    sim = [float(item["sim_degradation"]) for item in points]
    real = [float(item["real_degradation"]) for item in points]
    pearson = pearson_correlation(sim, real)
    spearman = spearman_correlation(sim, real)
    if pearson is None and spearman is None:
        return {
            **_unavailable("shift degradation has no rankable variation", len(points)),
            "points": points,
        }
    return {
        "status": "available",
        "points": points,
        "sample_size": len(points),
        "pearson": pearson,
        "pearson_ci95": _cluster_bootstrap(
            rows,
            spec,
            lambda sample: _shift_statistic(sample, pearson_correlation),
        ),
        "spearman": spearman,
        "spearman_ci95": _cluster_bootstrap(
            rows,
            spec,
            lambda sample: _shift_statistic(sample, spearman_correlation),
        ),
    }


def _degradation_points(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["policy_id"]), str(row["task_id"]))].append(row)
    points: list[dict[str, Any]] = []
    for key, values in sorted(groups.items()):
        clean = [item for item in values if float(item["severity"]) == 0.0]
        shifted = [item for item in values if float(item["severity"]) > 0.0]
        if not clean or not shifted:
            continue
        points.append(
            {
                "policy_id": key[0],
                "task_id": key[1],
                "sim_degradation": mean(float(item["sim_success"]) for item in clean)
                - mean(float(item["sim_success"]) for item in shifted),
                "real_degradation": mean(float(item["real_success"]) for item in clean)
                - mean(float(item["real_success"]) for item in shifted),
            }
        )
    return points


def _shift_statistic(
    rows: Sequence[Mapping[str, Any]],
    function: Callable[[Sequence[float], Sequence[float]], float | None],
) -> float | None:
    points = _degradation_points(rows)
    if len(points) < 2:
        return None
    return function(
        [float(item["sim_degradation"]) for item in points],
        [float(item["real_degradation"]) for item in points],
    )


def _time_to_failure(spec: SimRealStudySpec, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    observed = [
        row
        for row in rows
        if row["sim_failure_time_seconds"] is not None
        and row["real_failure_time_seconds"] is not None
    ]
    if len(observed) < 2:
        return {
            **_unavailable("at least two pairs need observed failure times", len(observed)),
            "censoring": _censoring(rows),
        }
    sim = [float(row["sim_failure_time_seconds"]) for row in observed]
    real = [float(row["real_failure_time_seconds"]) for row in observed]
    return {
        "status": "available",
        "sample_size": len(observed),
        "pearson": pearson_correlation(sim, real),
        "spearman": spearman_correlation(sim, real),
        "mean_absolute_error_seconds": mean(abs(x - y) for x, y in zip(sim, real)),
        "censoring": _censoring(rows),
        "ci95": _cluster_bootstrap(
            observed,
            spec,
            lambda sample: pearson_correlation(
                [float(row["sim_failure_time_seconds"]) for row in sample],
                [float(row["real_failure_time_seconds"]) for row in sample],
            ),
        ),
    }


def _recovery_effect(spec: SimRealStudySpec, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if spec.recovery_assumption == "disabled":
        return _unavailable("recovery comparison was prespecified as disabled", 0)
    if spec.recovery_assumption == "counterfactual_sim_only":
        return _unavailable(
            "counterfactual simulation recovery is not a matched real recovery effect",
            sum(row["sim_recovery_gain"] is not None for row in rows),
        )
    real_gains = []
    sim_gains = []
    for row in rows:
        sim_gain = row["sim_recovery_gain"]
        real_gain = row["real_recovery_gain"]
        if sim_gain is not None and real_gain is not None:
            sim_gains.append(float(sim_gain))
            real_gains.append(float(real_gain))
    if len(sim_gains) < 2:
        return _unavailable("matched recovery trials are insufficient", len(sim_gains))
    return {
        "status": "available",
        "sample_size": len(sim_gains),
        "pearson": pearson_correlation(sim_gains, real_gains),
        "mean_absolute_error": mean(
            abs(sim - real) for sim, real in zip(sim_gains, real_gains)
        ),
        "assumption": spec.recovery_assumption,
    }


def _incremental_predictive_value(
    spec: SimRealStudySpec, rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    holdout = set(spec.holdout_shift_ids)
    train = [row for row in rows if row["shift_id"] not in holdout]
    test = [row for row in rows if row["shift_id"] in holdout]
    if len(train) < 3 or len(test) < 2:
        return _unavailable("held-out predictive analysis has insufficient pairs", len(test))
    scores = _predictive_scores(train, test)
    assert scores is not None
    baseline_error, enhanced_error = scores
    improvement = baseline_error - enhanced_error
    return {
        "status": "available",
        "train_pairs": len(train),
        "holdout_pairs": len(test),
        "holdout_shift_ids": sorted(holdout),
        "baseline_features": ["intercept", "sim_success"],
        "enhanced_features": [
            "intercept",
            "sim_success",
            "severity",
            "sim_failure_present",
            "sim_failure_time_observed",
            "sim_recovery_gain",
        ],
        "baseline_brier": baseline_error,
        "enhanced_brier": enhanced_error,
        "incremental_brier_improvement": improvement,
        "incremental_brier_improvement_ci95": _cluster_bootstrap(
            rows,
            spec,
            lambda sample: _predictive_improvement(spec, sample),
        ),
        "interpretation": "positive values favor prespecified failure/recovery features",
    }


def _predictive_scores(
    train: Sequence[Mapping[str, Any]], test: Sequence[Mapping[str, Any]]
) -> tuple[float, float] | None:
    if len(train) < 3 or len(test) < 2:
        return None
    baseline_train = np.asarray([[1.0, float(row["sim_success"])] for row in train])
    enhanced_train = np.asarray([_enhanced_features(row) for row in train])
    target_train = np.asarray([float(row["real_success"]) for row in train])
    baseline_model = _ridge(baseline_train, target_train)
    enhanced_model = _ridge(enhanced_train, target_train)
    baseline_test = np.asarray([[1.0, float(row["sim_success"])] for row in test])
    enhanced_test = np.asarray([_enhanced_features(row) for row in test])
    target_test = np.asarray([float(row["real_success"]) for row in test])
    baseline_error = float(np.mean((baseline_test @ baseline_model - target_test) ** 2))
    enhanced_error = float(np.mean((enhanced_test @ enhanced_model - target_test) ** 2))
    return baseline_error, enhanced_error


def _predictive_improvement(
    spec: SimRealStudySpec, rows: Sequence[Mapping[str, Any]]
) -> float | None:
    holdout = set(spec.holdout_shift_ids)
    scores = _predictive_scores(
        [row for row in rows if row["shift_id"] not in holdout],
        [row for row in rows if row["shift_id"] in holdout],
    )
    return scores[0] - scores[1] if scores is not None else None


def _enhanced_features(row: Mapping[str, Any]) -> list[float]:
    return [
        1.0,
        float(row["sim_success"]),
        float(row["severity"]),
        float(row["sim_failure_category"] is not None),
        float(row["sim_failure_time_seconds"] is not None),
        float(row["sim_recovery_gain"] or 0.0),
    ]


def _ridge(features: np.ndarray, target: np.ndarray) -> np.ndarray:
    penalty = np.eye(features.shape[1]) * 1e-6
    penalty[0, 0] = 0.0
    return np.linalg.solve(features.T @ features + penalty, features.T @ target)


def _cluster_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    spec: SimRealStudySpec,
    statistic: Callable[[Sequence[Mapping[str, Any]]], float | None],
) -> list[float] | None:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(row[field] for field in spec.cluster_fields)
        groups[key].append(row)
    clusters = list(groups.values())
    if len(clusters) < 2:
        return None
    seed = int.from_bytes(
        hashlib.sha256(f"{spec.sha256}:{spec.bootstrap_seed}".encode()).digest()[:8],
        "big",
    )
    rng = random.Random(seed)
    estimates = []
    for _ in range(spec.bootstrap_samples):
        sample = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        value = statistic([row for cluster in sample for row in cluster])
        if value is not None and math.isfinite(value):
            estimates.append(value)
    if len(estimates) < max(20, spec.bootstrap_samples // 10):
        return None
    estimates.sort()
    return [
        estimates[round(0.025 * (len(estimates) - 1))],
        estimates[round(0.975 * (len(estimates) - 1))],
    ]


def _first_event(events: Sequence[Any]) -> Any | None:
    outcome = [event for event in events if event.role in {"symptom", "mechanism", "consequence"}]
    return min(outcome, key=lambda event: (event.onset_step, event.event_id)) if outcome else None


def _censoring(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {
        "both_observed": sum(
            not row["sim_time_censored"] and not row["real_time_censored"] for row in rows
        ),
        "simulation_censored": sum(bool(row["sim_time_censored"]) for row in rows),
        "real_censored": sum(bool(row["real_time_censored"]) for row in rows),
        "both_censored": sum(
            bool(row["sim_time_censored"]) and bool(row["real_time_censored"]) for row in rows
        ),
    }
    for domain in ("sim", "real"):
        reasons = Counter(
            str(row[f"{domain}_censor_reason"])
            for row in rows
            if row[f"{domain}_censor_reason"] is not None
        )
        for reason, count in sorted(reasons.items()):
            counts[f"{domain}_{reason}"] = count
    return counts


def _report(
    spec: SimRealStudySpec,
    status: str,
    rows: Sequence[Mapping[str, Any]],
    errors: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "format": SIM_REAL_REPORT_FORMAT,
        "status": status,
        "study_id": spec.study_id,
        "study_version": spec.study_version,
        "study_sha256": spec.sha256,
        "unit_of_analysis": spec.unit_of_analysis,
        "primary_metrics": list(spec.primary_metrics),
        "bootstrap": {
            "samples": spec.bootstrap_samples,
            "seed": spec.bootstrap_seed,
            "cluster_fields": list(spec.cluster_fields),
        },
        "pair_count": len(rows),
        "excluded_pairs": [
            {"pair_id": pair.pair_id, "reason": pair.exclusion_reason}
            for pair in spec.pairs
            if not pair.included
        ],
        "errors": list(errors),
        "pairs": list(rows),
        "metrics": dict(metrics),
        "recovery_assumption": spec.recovery_assumption,
        "claim_boundary": (
            "A complete analysis reports correspondence. Predictive sim-real claims "
            "still require the prespecified hardware study and BenchmarkValidity."
        ),
    }


def _unavailable(reason: str, sample_size: int) -> dict[str, Any]:
    return {"status": "unavailable", "reason": reason, "sample_size": sample_size}


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None
