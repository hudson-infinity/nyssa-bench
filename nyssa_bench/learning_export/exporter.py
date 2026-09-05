from __future__ import annotations

import hashlib
import json
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import unquote, urlparse

import yaml

from nyssa_bench.learning_export.protocol import (
    ArtifactReference,
    EvaluationExclusion,
    LearningEpisodeRecord,
    LearningExportManifest,
    LearningStepRecord,
)
from nyssa_bench.stress_search import load_stress_search_study
from nyssa_bench.utils.reproducibility import utc_now


@dataclass(frozen=True)
class ExportSplit:
    split_id: str
    partition: str
    content_sha256: str

    def __post_init__(self) -> None:
        if not self.split_id.strip():
            raise ValueError("export split_id must be non-empty")
        if self.partition not in {"train", "validation", "public_test", "hidden_test"}:
            raise ValueError(f"unsupported export split partition: {self.partition}")
        _require_sha256(self.content_sha256, "export split content hash")

    def to_dict(self) -> dict[str, Any]:
        return {
            "split_id": self.split_id,
            "partition": self.partition,
            "content_sha256": self.content_sha256,
        }


@dataclass(frozen=True)
class LearningExportConfig:
    benchmark_id: str
    split: ExportSplit
    policy_families: dict[str, str]
    licenses: tuple[str, ...]
    privacy_level: str = "public"
    privacy_restrictions: tuple[str, ...] = ()
    include_successes: bool = True
    boundary_studies: tuple[Path, ...] = ()
    max_inline_observation_bytes: int = 1_000_000

    def __post_init__(self) -> None:
        if not self.benchmark_id.strip():
            raise ValueError("benchmark_id must be non-empty")
        if not self.policy_families or any(
            not str(key).strip() or not str(value).strip()
            for key, value in self.policy_families.items()
        ):
            raise ValueError("policy_families must map policy IDs to non-empty families")
        if not self.licenses or any(not license_id.strip() for license_id in self.licenses):
            raise ValueError("at least one non-empty license declaration is required")
        if self.privacy_level not in {"public", "restricted", "private"}:
            raise ValueError(f"unsupported privacy level: {self.privacy_level}")
        if self.privacy_level != "public" and not self.privacy_restrictions:
            raise ValueError("restricted/private exports require privacy restrictions")
        if self.split.partition == "hidden_test" and self.privacy_level == "public":
            raise ValueError("hidden-test evidence cannot be exported as public")
        if self.max_inline_observation_bytes <= 0:
            raise ValueError("max_inline_observation_bytes must be positive")


@dataclass(frozen=True)
class LearningEvidencePackage:
    root: Path
    manifest: LearningExportManifest
    episodes: tuple[LearningEpisodeRecord, ...]
    exclusions: tuple[EvaluationExclusion, ...]
    facets: dict[str, Any]


def export_learning_evidence(
    run_dirs: Sequence[str | Path],
    out_dir: str | Path,
    *,
    config: LearningExportConfig,
) -> LearningEvidencePackage:
    if not run_dirs:
        raise ValueError("learning evidence export requires at least one source run")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    boundary_by_run = _boundary_contexts(config.boundary_studies)
    records: list[LearningEpisodeRecord] = []
    source_runs = []
    seen_episode_ids: set[str] = set()
    seen_run_ids: set[str] = set()
    for run_dir_value in run_dirs:
        run_dir = Path(run_dir_value)
        (
            run,
            dataset_manifest,
            episodes,
            source_hash,
            source_provenance,
        ) = _load_source_run(run_dir)
        run_id = str(run.get("run_id", ""))
        suite_id = str(run.get("suite_id", ""))
        policy_id = str(run.get("policy_name", ""))
        engine_name = str(run.get("engine_name", ""))
        if not all((run_id, suite_id, policy_id, engine_name)):
            raise ValueError(f"source run identity is incomplete: {run_dir}")
        if run_id in seen_run_ids:
            raise ValueError(f"duplicate source run identity: {run_id}")
        seen_run_ids.add(run_id)
        policy_family = config.policy_families.get(
            policy_id, config.policy_families.get("*")
        )
        if not policy_family:
            raise ValueError(
                f"no policy family declared for source policy {policy_id!r}"
            )
        source_runs.append(
            {
                "run_id": run_id,
                "suite_id": suite_id,
                "policy_id": policy_id,
                "policy_family": policy_family,
                "engine_name": engine_name,
                "source_uri": run_dir.resolve().as_uri(),
                "source_run_sha256": source_hash,
                "dataset_manifest_sha256": _file_sha256(
                    run_dir / "dataset_manifest.json"
                ),
                "provenance": source_provenance,
            }
        )
        for source_episode_position, raw_episode in enumerate(episodes):
            if not isinstance(raw_episode, dict):
                raise ValueError(f"source episode must be a mapping: {run_dir}")
            if not config.include_successes and bool(raw_episode.get("success")):
                continue
            record = _convert_episode(
                raw_episode,
                run_dir=run_dir,
                run=run,
                dataset_manifest=dataset_manifest,
                source_hash=source_hash,
                benchmark_id=config.benchmark_id,
                split=config.split,
                policy_family=policy_family,
                boundary_context=boundary_by_run.get(run_id),
                source_provenance=source_provenance,
                source_episode_position=source_episode_position,
                max_inline_observation_bytes=config.max_inline_observation_bytes,
            )
            if record.episode_id in seen_episode_ids:
                raise ValueError(f"duplicate exported episode identity: {record.episode_id}")
            seen_episode_ids.add(record.episode_id)
            records.append(record)
    records.sort(key=lambda item: (item.task_id, item.policy_id, item.seed, item.episode_index, item.episode_id))
    exclusions = tuple(record.exclusion for record in records)
    facets = build_facet_index(records)
    episode_path = out_dir / "episodes.jsonl"
    exclusion_path = out_dir / "evaluation_exclusions.json"
    facet_path = out_dir / "facets.json"
    _write_jsonl_atomic(episode_path, [record.to_dict() for record in records])
    _write_json_atomic(
        exclusion_path,
        {
            "format": "nyssa-evaluation-exclusion-set-v1",
            "exclusions": [item.to_dict() for item in exclusions],
        },
    )
    _write_json_atomic(facet_path, facets)
    episode_ref = _local_artifact_reference("episodes", episode_path, out_dir)
    exclusion_ref = _local_artifact_reference("evaluation_exclusions", exclusion_path, out_dir)
    facet_ref = _local_artifact_reference("facets", facet_path, out_dir)
    export_identity = _sha256(
        {
            "benchmark_id": config.benchmark_id,
            "split": config.split.to_dict(),
            "source_runs": source_runs,
            "episode_hashes": [record.content_sha256 for record in records],
            "licenses": list(config.licenses),
            "privacy_level": config.privacy_level,
        }
    )
    manifest = LearningExportManifest(
        export_id=f"learning-export-{export_identity[:20]}",
        created_at=utc_now(),
        source_runs=tuple(source_runs),
        episode_count=len(records),
        episode_file=episode_ref,
        exclusion_file=exclusion_ref,
        facet_file=facet_ref,
        facet_index=facets,
        licenses=config.licenses,
        privacy_level=config.privacy_level,  # type: ignore[arg-type]
        privacy_restrictions=config.privacy_restrictions,
    )
    _write_json_atomic(out_dir / "manifest.json", manifest.to_dict())
    return LearningEvidencePackage(
        root=out_dir,
        manifest=manifest,
        episodes=tuple(records),
        exclusions=exclusions,
        facets=facets,
    )


def load_learning_evidence(
    root: str | Path, *, verify_external_artifacts: bool = False
) -> LearningEvidencePackage:
    root = Path(root)
    manifest_raw = _load_json(root / "manifest.json")
    if not isinstance(manifest_raw, Mapping):
        raise ValueError("learning export manifest must be a mapping")
    manifest = LearningExportManifest.from_dict(manifest_raw)
    for reference in (
        manifest.episode_file,
        manifest.exclusion_file,
        manifest.facet_file,
    ):
        _verify_local_reference(reference, root)
    episode_path = root / manifest.episode_file.uri
    episodes = tuple(
        LearningEpisodeRecord.from_dict(item)
        for item in _load_jsonl(episode_path)
    )
    exclusions_raw = _load_json(root / manifest.exclusion_file.uri)
    if not isinstance(exclusions_raw, Mapping) or exclusions_raw.get(
        "format"
    ) != "nyssa-evaluation-exclusion-set-v1":
        raise ValueError("invalid evaluation exclusion set")
    exclusions_data = exclusions_raw.get("exclusions")
    if not isinstance(exclusions_data, list) or not all(
        isinstance(item, Mapping) for item in exclusions_data
    ):
        raise ValueError("evaluation exclusion set must contain mappings")
    exclusions = tuple(EvaluationExclusion.from_dict(item) for item in exclusions_data)
    facets = _load_json(root / manifest.facet_file.uri)
    if not isinstance(facets, dict):
        raise ValueError("facet index must be a mapping")
    if len(episodes) != manifest.episode_count:
        raise ValueError("learning export episode count does not match manifest")
    if tuple(episode.exclusion for episode in episodes) != exclusions:
        raise ValueError("episode exclusions do not match exclusion file")
    expected_facets = build_facet_index(episodes)
    if facets != expected_facets or manifest.facet_index != expected_facets:
        raise ValueError("learning export facet index does not match episodes")
    if verify_external_artifacts:
        for episode in episodes:
            for artifact in episode.artifacts:
                _verify_external_reference(artifact, manifest.source_runs)
    return LearningEvidencePackage(root, manifest, episodes, exclusions, facets)


def query_learning_evidence(
    package: LearningEvidencePackage,
    **filters: str | float | bool,
) -> tuple[LearningEpisodeRecord, ...]:
    allowed = {
        "task",
        "policy_family",
        "failure_type",
        "stressor",
        "severity",
        "recoverability",
        "boundary",
    }
    unknown = sorted(set(filters) - allowed)
    if unknown:
        raise ValueError(f"unknown learning-export filters: {', '.join(unknown)}")
    selected = {episode.episode_id for episode in package.episodes}
    for facet, raw_value in filters.items():
        value = _facet_value(raw_value)
        matching = set(package.facets.get("facets", {}).get(facet, {}).get(value, []))
        selected &= matching
    return tuple(episode for episode in package.episodes if episode.episode_id in selected)


def validate_learning_evidence_use(
    package: LearningEvidencePackage,
    *,
    purpose: str,
) -> dict[str, Any]:
    if purpose not in {"training", "data_selection", "evaluation"}:
        raise ValueError(f"unsupported learning-evidence purpose: {purpose}")
    if purpose == "evaluation":
        raise ValueError(
            "learning evidence is excluded from evaluation reuse; use an independent held-out split"
        )
    if package.manifest.evaluation_reuse_policy != "excluded" or not all(
        exclusion.excluded_from_evaluation for exclusion in package.exclusions
    ):
        raise ValueError("learning evidence package has invalid evaluation exclusions")
    return {
        "purpose": purpose,
        "authorized": True,
        "episode_count": len(package.episodes),
        "evaluation_reuse_policy": "excluded",
        "source_split_ids": sorted(
            {exclusion.source_split_id for exclusion in package.exclusions}
        ),
        "exclusion_ids": sorted(
            exclusion.exclusion_id for exclusion in package.exclusions
        ),
    }


def build_facet_index(
    episodes: Sequence[LearningEpisodeRecord],
) -> dict[str, Any]:
    facets: dict[str, dict[str, list[str]]] = {
        key: {}
        for key in (
            "task",
            "policy_family",
            "failure_type",
            "stressor",
            "severity",
            "recoverability",
            "boundary",
        )
    }
    for episode in episodes:
        values = {
            "task": [episode.task_id],
            "policy_family": [episode.policy_family],
            "failure_type": _failure_types(episode),
            "stressor": _stressor_ids(episode.stressor_context) or ["clean"],
            "severity": _stressor_severities(episode.stressor_context) or ["0"],
            "recoverability": _recoverability_values(episode),
            "boundary": ["true" if episode.boundary_context else "false"],
        }
        for facet, facet_values in values.items():
            for value in sorted(set(map(_facet_value, facet_values))):
                facets[facet].setdefault(value, []).append(episode.episode_id)
    for values in facets.values():
        for key in values:
            values[key].sort()
    return {
        "format": "nyssa-learning-evidence-facets-v1",
        "episode_count": len(episodes),
        "facets": facets,
    }


def _convert_episode(
    raw: dict[str, Any],
    *,
    run_dir: Path,
    run: dict[str, Any],
    dataset_manifest: dict[str, Any],
    source_hash: str,
    benchmark_id: str,
    split: ExportSplit,
    policy_family: str,
    boundary_context: dict[str, Any] | None,
    source_provenance: dict[str, Any],
    source_episode_position: int,
    max_inline_observation_bytes: int,
) -> LearningEpisodeRecord:
    run_id = str(run["run_id"])
    suite_id = str(run["suite_id"])
    task_id = str(raw.get("task_id", ""))
    episode_index = int(raw.get("episode_index", -1))
    seed = int(raw.get("seed", -1))
    source_episode_hash = _sha256(raw)
    episode_id = "episode-" + _sha256(
        {
            "run_id": run_id,
            "task_id": task_id,
            "episode_index": episode_index,
            "seed": seed,
            "source_episode_sha256": source_episode_hash,
        }
    )[:24]
    exclusion = EvaluationExclusion(
        exclusion_id=f"exclude-{episode_id.removeprefix('episode-')}",
        source_benchmark_id=benchmark_id,
        source_suite_id=suite_id,
        source_split_id=split.split_id,
        source_episode_id=episode_id,
        content_sha256=source_episode_hash,
    )
    steps_raw = raw.get("steps")
    if not isinstance(steps_raw, list):
        raise ValueError(f"source episode has invalid steps: {episode_id}")
    steps = tuple(
        _convert_step(
            step,
            step_index=index,
            episode_id=episode_id,
            run_id=run_id,
            run_dir=run_dir,
            source_episode_position=source_episode_position,
            max_inline_observation_bytes=max_inline_observation_bytes,
        )
        for index, step in enumerate(steps_raw)
    )
    failure_ledger = raw.get("failure_ledger")
    if failure_ledger is not None and not isinstance(failure_ledger, dict):
        raise ValueError(f"source episode has invalid failure ledger: {episode_id}")
    branches_raw = raw.get("counterfactual_recovery", [])
    if not isinstance(branches_raw, list) or not all(
        isinstance(item, dict) for item in branches_raw
    ):
        raise ValueError(f"source episode has invalid counterfactual records: {episode_id}")
    metrics = raw.get("metrics", {})
    if not isinstance(metrics, dict):
        raise ValueError(f"source episode has invalid metrics: {episode_id}")
    has_external_observation = any(
        isinstance(step.observation, dict)
        and step.observation.get("storage") == "source_json_pointer"
        for step in steps
    )
    artifacts = _episode_artifact_references(
        raw,
        run_dir,
        run_id,
        episode_id,
        include_source_episodes=has_external_observation,
    )
    failure_label = (
        str(raw["failure_label"]) if raw.get("failure_label") is not None else None
    )
    return LearningEpisodeRecord(
        episode_id=episode_id,
        task_id=task_id,
        policy_id=str(run["policy_name"]),
        policy_family=policy_family,
        engine_name=str(run["engine_name"]),
        episode_index=episode_index,
        seed=seed,
        success=bool(raw.get("success", False)),
        failure_label=failure_label,
        failure_ledger=failure_ledger,
        stressor_context=dict(raw.get("stressor_context", {})),
        split_lineage=split.to_dict(),
        provenance={
            "source_run_id": run_id,
            "source_run_sha256": source_hash,
            "source_episode_sha256": source_episode_hash,
            "dataset_manifest_sha256": _file_sha256(
                run_dir / "dataset_manifest.json"
            ),
            "git": source_provenance["git_info"],
            "package_versions_sha256": source_provenance[
                "package_versions_sha256"
            ],
            "environment_sha256": source_provenance["environment_sha256"],
        },
        steps=steps,
        counterfactual_recovery=tuple(branches_raw),
        recovery_metrics={
            key: value
            for key, value in metrics.items()
            if key.startswith("recovery_")
            or key.startswith("counterfactual_")
            or key in {
                "false_intervention_rate",
                "harmful_intervention_rate",
                "mean_intervention_cost_steps",
            }
        },
        failure_cluster=_failure_cluster(failure_label, failure_ledger),
        boundary_context=boundary_context,
        artifacts=artifacts,
        exclusion=exclusion,
    )


def _convert_step(
    raw: Any,
    *,
    step_index: int,
    episode_id: str,
    run_id: str,
    run_dir: Path,
    source_episode_position: int,
    max_inline_observation_bytes: int,
) -> LearningStepRecord:
    if not isinstance(raw, dict):
        raise ValueError(f"source step must be a mapping: {episode_id}:{step_index}")
    info = raw.get("info")
    if not isinstance(info, dict):
        raise ValueError(f"source step info must be a mapping: {episode_id}:{step_index}")
    executed_action = raw.get("action")
    executed_source = str(info.get("action_source") or "unknown").strip().lower()
    action_rejected = bool(info.get("action_rejected", False))
    if "proposed_action" in info:
        proposed_action = info["proposed_action"]
        proposed_source = str(info.get("proposed_action_source") or "unknown")
    elif action_rejected or executed_source in {"expert", "recovery"}:
        raise ValueError(
            f"intervention step lacks original proposed action: {episode_id}:{step_index}"
        )
    else:
        proposed_action = info.get("action_before_stressors", executed_action)
        proposed_source = executed_source
    rejected_action = info.get("rejected_action")
    if action_rejected and rejected_action is None:
        raise ValueError(
            f"rejected step lacks rejected_action evidence: {episode_id}:{step_index}"
        )
    before_stressors = info.get(
        "executed_action_before_stressors",
        info.get("action_before_stressors", executed_action),
    )
    oracle_action = info.get("oracle_action")
    recovery_action = info.get("recovery_action")
    if executed_source == "expert" and oracle_action is None:
        oracle_action = before_stressors
    if executed_source == "recovery" and recovery_action is None:
        recovery_action = before_stressors
    observation = raw.get("observation")
    serialized_observation = json.dumps(
        observation, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    if len(serialized_observation) > max_inline_observation_bytes:
        source_path = run_dir / "episodes.json"
        observation = {
            "storage": "source_json_pointer",
            "artifact_id": f"{episode_id}:source_episodes",
            "artifact_uri": f"nyssa-run://{run_id}/episodes.json",
            "json_pointer": (
                f"/{source_episode_position}/steps/{step_index}/observation"
            ),
            "value_sha256": hashlib.sha256(serialized_observation).hexdigest(),
            "serialized_bytes": len(serialized_observation),
            "source_artifact_sha256": _file_sha256(source_path),
        }
    return LearningStepRecord(
        step_index=step_index,
        observation=observation,
        proposed_action=proposed_action,
        proposed_action_source=proposed_source,
        rejected_action=rejected_action,
        action_rejected=action_rejected,
        executed_action_before_stressors=before_stressors,
        executed_action=executed_action,
        executed_action_source=executed_source,
        oracle_action=oracle_action,
        recovery_action=recovery_action,
        reward=float(raw.get("reward", 0.0)),
        terminated=bool(raw.get("terminated", False)),
        truncated=bool(raw.get("truncated", False)),
        info=info,
    )


def _load_source_run(
    run_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[Any], str, dict[str, Any]]:
    run = _load_yaml(run_dir / "run.yaml")
    manifest = _load_json(run_dir / "dataset_manifest.json")
    episodes = _load_json(run_dir / "episodes.json")
    metrics = _load_json(run_dir / "metrics.json")
    git_info = _load_json(run_dir / "git_info.json")
    package_versions = _load_json(run_dir / "package_versions.json")
    environment = _load_json(run_dir / "environment.json")
    if not isinstance(manifest, dict) or not isinstance(metrics, dict):
        raise ValueError(f"source run manifests must be mappings: {run_dir}")
    if not all(
        isinstance(value, dict)
        for value in (git_info, package_versions, environment)
    ):
        raise ValueError(f"source run provenance artifacts must be mappings: {run_dir}")
    if not isinstance(episodes, list):
        raise ValueError(f"source episodes must be a list: {run_dir}")
    manifest_run = manifest.get("run")
    if not isinstance(manifest_run, dict):
        raise ValueError(f"dataset manifest has no run metadata: {run_dir}")
    for key in ("run_id", "suite_id", "policy_name", "engine_name"):
        if run.get(key) != manifest_run.get(key):
            raise ValueError(
                f"source run and dataset manifest disagree on {key}: {run_dir}"
            )
    artifacts = manifest.get("artifacts", {})
    if not isinstance(artifacts, dict):
        raise ValueError(f"dataset manifest artifacts must be a mapping: {run_dir}")
    declared = artifacts.get("episodes.json", {})
    actual_episode_hash = _file_sha256(run_dir / "episodes.json")
    if not isinstance(declared, dict) or declared.get("sha256") != actual_episode_hash:
        raise ValueError(f"source episodes do not match dataset manifest: {run_dir}")
    declared_metrics = artifacts.get("metrics.json", {})
    actual_metrics_hash = _file_sha256(run_dir / "metrics.json")
    if (
        not isinstance(declared_metrics, dict)
        or declared_metrics.get("sha256") != actual_metrics_hash
    ):
        raise ValueError(f"source metrics do not match dataset manifest: {run_dir}")
    expected_episodes = int(run.get("episodes_per_task", 0) or 0) * len(
        run.get("task_ids", [])
    )
    if (
        expected_episodes != len(episodes)
        or int(metrics.get("episodes", -1)) != len(episodes)
    ):
        raise ValueError(f"source episode denominators are inconsistent: {run_dir}")
    source_hash = _sha256(
        {
            "run": run,
            "dataset_manifest_sha256": _file_sha256(
                run_dir / "dataset_manifest.json"
            ),
            "episodes_sha256": actual_episode_hash,
            "metrics_sha256": actual_metrics_hash,
            "git_info_sha256": _file_sha256(run_dir / "git_info.json"),
            "package_versions_sha256": _file_sha256(
                run_dir / "package_versions.json"
            ),
            "environment_sha256": _file_sha256(run_dir / "environment.json"),
        }
    )
    source_provenance = {
        "git_info": git_info,
        "git_info_sha256": _file_sha256(run_dir / "git_info.json"),
        "package_versions_sha256": _file_sha256(
            run_dir / "package_versions.json"
        ),
        "environment_sha256": _file_sha256(run_dir / "environment.json"),
        "run_claim_validation": metrics.get("public_claim_validation"),
        "benchmark_validity": metrics.get("benchmark_validity"),
    }
    return run, manifest, episodes, source_hash, source_provenance


def _boundary_contexts(paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    contexts = {}
    for path in paths:
        study = load_stress_search_study(path)
        proposals = {
            proposal.proposal_id: proposal
            for proposal in [
                *study.sampler.proposals,
                *study.confirmation_proposals,
            ]
        }
        observations = {
            **study.sampler.observations,
            **study.confirmation_observations,
        }
        for proposal_id, observation in observations.items():
            run_id = observation.provenance.get("source_id")
            proposal = proposals.get(proposal_id)
            if not run_id or proposal is None:
                continue
            if run_id in contexts:
                raise ValueError(f"source run appears in multiple boundary records: {run_id}")
            contexts[str(run_id)] = {
                "study_id": study.spec.study_id,
                "study_sha256": study.to_dict()["study_sha256"],
                "proposal": proposal.to_dict(),
                "observation_status": observation.status,
                "observation": observation.to_dict(),
                "confirmation": proposal.phase == "confirmation",
            }
    return contexts


def _episode_artifact_references(
    raw: Mapping[str, Any],
    run_dir: Path,
    run_id: str,
    episode_id: str,
    *,
    include_source_episodes: bool,
) -> tuple[ArtifactReference, ...]:
    references = []
    for key, media_type in (
        ("replay_path", "video/mp4"),
        ("failure_clip_path", "video/mp4"),
    ):
        value = raw.get(key)
        if not value:
            continue
        relative = Path(str(value))
        path = (run_dir / relative).resolve()
        root = run_dir.resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"declared episode artifact is missing or unsafe: {value}")
        references.append(
            ArtifactReference(
                artifact_id=f"{episode_id}:{key}",
                uri=f"nyssa-run://{run_id}/{relative.as_posix()}",
                sha256=_file_sha256(path),
                bytes=path.stat().st_size,
                media_type=mimetypes.guess_type(path.name)[0] or media_type,
                embedded=False,
            )
        )
    if include_source_episodes:
        source_path = run_dir / "episodes.json"
        references.append(
            ArtifactReference(
                artifact_id=f"{episode_id}:source_episodes",
                uri=f"nyssa-run://{run_id}/episodes.json",
                sha256=_file_sha256(source_path),
                bytes=source_path.stat().st_size,
                media_type="application/json",
                embedded=False,
            )
        )
    return tuple(references)


def _local_artifact_reference(
    artifact_id: str, path: Path, root: Path
) -> ArtifactReference:
    return ArtifactReference(
        artifact_id=artifact_id,
        uri=path.relative_to(root).as_posix(),
        sha256=_file_sha256(path),
        bytes=path.stat().st_size,
        media_type="application/x-ndjson"
        if path.suffix == ".jsonl"
        else "application/json",
        embedded=True,
    )


def _verify_local_reference(reference: ArtifactReference, root: Path) -> None:
    path = (root / reference.uri).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"local export artifact is missing or unsafe: {reference.uri}")
    if path.stat().st_size != reference.bytes or _file_sha256(path) != reference.sha256:
        raise ValueError(f"local export artifact hash mismatch: {reference.uri}")


def _verify_external_reference(
    reference: ArtifactReference, source_runs: Sequence[dict[str, Any]]
) -> None:
    prefix = "nyssa-run://"
    if not reference.uri.startswith(prefix):
        raise ValueError(f"unsupported external artifact URI: {reference.uri}")
    identity, _, relative = reference.uri[len(prefix) :].partition("/")
    source = next((item for item in source_runs if item.get("run_id") == identity), None)
    if source is None:
        raise ValueError(f"external artifact references unknown run: {identity}")
    parsed = urlparse(str(source["source_uri"]))
    if parsed.scheme != "file":
        raise ValueError(f"cannot resolve non-file source run: {identity}")
    decoded_path = unquote(parsed.path)
    if parsed.netloc:
        decoded_path = f"//{parsed.netloc}{decoded_path}"
    if len(decoded_path) >= 3 and decoded_path[0] == "/" and decoded_path[2] == ":":
        decoded_path = decoded_path[1:]
    source_path = Path(decoded_path)
    if not source_path.is_absolute():
        raise ValueError(f"cannot resolve external source run: {identity}")
    path = (source_path / relative).resolve()
    if not path.is_relative_to(source_path.resolve()) or not path.is_file():
        raise ValueError(f"external artifact is missing or unsafe: {reference.uri}")
    if path.stat().st_size != reference.bytes or _file_sha256(path) != reference.sha256:
        raise ValueError(f"external artifact hash mismatch: {reference.uri}")


def _stressor_ids(context: Mapping[str, Any]) -> list[str]:
    return [
        str(item.get("stressor_id"))
        for item in context.get("applications", [])
        if isinstance(item, Mapping) and item.get("status") == "applied"
    ]


def _stressor_severities(context: Mapping[str, Any]) -> list[str]:
    return [
        _facet_value(item.get("requested", {}).get("severity"))
        for item in context.get("applications", [])
        if isinstance(item, Mapping)
        and item.get("status") == "applied"
        and isinstance(item.get("requested"), Mapping)
    ]


def _recoverability_values(episode: LearningEpisodeRecord) -> list[str]:
    values = set()
    ledger = episode.failure_ledger or {}
    for event in ledger.get("events", []):
        if isinstance(event, Mapping):
            values.add(str(event.get("recovery_eligibility", "unknown")))
    if episode.counterfactual_recovery:
        values.add("counterfactually_evaluated")
    return sorted(values or {"unknown"})


def _failure_types(episode: LearningEpisodeRecord) -> list[str]:
    values = {episode.failure_label} if episode.failure_label else set()
    ledger = episode.failure_ledger or {}
    for event in ledger.get("events", []):
        if not isinstance(event, Mapping):
            continue
        if event.get("category"):
            values.add(str(event["category"]))
        if event.get("subtype"):
            values.add(str(event["subtype"]))
    return sorted(value for value in values if value) or ["none"]


def _failure_cluster(
    failure_label: str | None, ledger: Mapping[str, Any] | None
) -> dict[str, Any] | None:
    if failure_label is None:
        return None
    events = ledger.get("events", []) if isinstance(ledger, Mapping) else []
    signature = sorted(
        {
            (
                str(event.get("role", "unknown")),
                str(event.get("category", "unknown")),
                str(event.get("subtype", "unknown")),
            )
            for event in events
            if isinstance(event, Mapping)
        }
    )
    if signature:
        return {
            "cluster_id": "failure-cluster-"
            + _sha256(signature)[:16],
            "method": "temporal_role_category_subtype_signature_v1",
            "signature": [list(item) for item in signature],
            "terminal_label": failure_label,
        }
    return {
        "cluster_id": failure_label,
        "method": "terminal_failure_label_fallback",
        "terminal_label": failure_label,
    }


def _facet_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return format(value, ".12g")
    return str(value)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load learning-export source: {path}") from exc


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    values = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSONL record must be a mapping at {path}:{line_number}")
        values.append(value)
    return values


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot load learning-export source: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"source run metadata must be a mapping: {path}")
    return value


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_jsonl_atomic(path: Path, values: Sequence[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
