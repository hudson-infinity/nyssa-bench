from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Mapping
from urllib.parse import urlparse

import yaml

from nyssa_bench.stressors import STRESSOR_SPEC_FORMAT, StressorConfig, StressorSpec


SCENARIO_PACKAGE_FORMAT = "nyssa-external-scenario-package-v1"
SCENARIO_EXECUTION_FORMAT = "nyssa-scenario-execution-v1"
SCENARIO_SPLIT_FORMAT = "nyssa-scenario-split-lineage-v1"
SCENARIO_STRESSOR_AXIS_FORMAT = "nyssa-scenario-stressor-axis-v1"
SCENARIO_TASK_CONTRACT_FORMAT = "nyssa-task-contract-draft-v1"
SCENARIO_PROTOCOL_VERSION = 1
SCENARIO_EPISODE_SEED_FORMAT = "nyssa-episode-seed-v2"
SCENARIO_EPISODE_SEED_STRIDE = 1_000_000
SCENARIO_EPISODE_SEED_FORMULA = "run_seed * episode_seed_stride + episode_index"
DEFAULT_SCENARIO_MANIFEST = "scenario.yaml"

AssetRedistribution = Literal["redistributable", "protected", "metadata_only"]
SplitPartition = Literal["train", "validation", "public_test", "hidden_test"]
ContaminationStatus = Literal["clean", "known_overlap", "unknown"]


@dataclass(frozen=True)
class ScenarioProtocolReferences:
    nep_version: str
    task_contract_format: str
    stressor_contract_format: str = STRESSOR_SPEC_FORMAT
    split_contract_format: str = SCENARIO_SPLIT_FORMAT

    def __post_init__(self) -> None:
        _require_semver(self.nep_version, "protocol.nep_version")
        if self.task_contract_format != SCENARIO_TASK_CONTRACT_FORMAT:
            raise ValueError(
                f"protocol.task_contract_format must be {SCENARIO_TASK_CONTRACT_FORMAT}"
            )
        if self.stressor_contract_format != STRESSOR_SPEC_FORMAT:
            raise ValueError(
                f"protocol.stressor_contract_format must be {STRESSOR_SPEC_FORMAT}"
            )
        if self.split_contract_format != SCENARIO_SPLIT_FORMAT:
            raise ValueError(
                f"protocol.split_contract_format must be {SCENARIO_SPLIT_FORMAT}"
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ScenarioProtocolReferences":
        _reject_unknown(
            data,
            {
                "nep_version",
                "task_contract_format",
                "stressor_contract_format",
                "split_contract_format",
            },
            "protocol",
        )
        return cls(
            nep_version=str(data.get("nep_version", "")),
            task_contract_format=str(data.get("task_contract_format", "")),
            stressor_contract_format=str(
                data.get("stressor_contract_format", STRESSOR_SPEC_FORMAT)
            ),
            split_contract_format=str(
                data.get("split_contract_format", SCENARIO_SPLIT_FORMAT)
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "nep_version": self.nep_version,
            "task_contract_format": self.task_contract_format,
            "stressor_contract_format": self.stressor_contract_format,
            "split_contract_format": self.split_contract_format,
        }


@dataclass(frozen=True)
class ScenarioGenerator:
    generator_id: str
    generator_version: str
    algorithm_id: str
    revision: str
    repository_url: str

    def __post_init__(self) -> None:
        _require_text(self.generator_id, "generator.generator_id")
        _require_semver(self.generator_version, "generator.generator_version")
        _require_text(self.algorithm_id, "generator.algorithm_id")
        _require_text(self.revision, "generator.revision")
        _require_uri(
            self.repository_url,
            "generator.repository_url",
            allowed_schemes={"http", "https"},
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ScenarioGenerator":
        _reject_unknown(
            data,
            {
                "generator_id",
                "generator_version",
                "algorithm_id",
                "revision",
                "repository_url",
            },
            "generator",
        )
        return cls(
            generator_id=str(data.get("generator_id", "")),
            generator_version=str(data.get("generator_version", "")),
            algorithm_id=str(data.get("algorithm_id", "")),
            revision=str(data.get("revision", "")),
            repository_url=str(data.get("repository_url", "")),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "generator_id": self.generator_id,
            "generator_version": self.generator_version,
            "algorithm_id": self.algorithm_id,
            "revision": self.revision,
            "repository_url": self.repository_url,
        }


@dataclass(frozen=True)
class ScenarioEngineRequirement:
    engine_name: str
    version_spec: str
    task_id: str
    env_id: str | None = None
    factory: str | None = None
    runtime: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.engine_name, "engine.engine_name")
        _require_version_spec(self.version_spec, "engine.version_spec")
        _require_text(self.task_id, "engine.task_id")
        if bool(self.env_id) == bool(self.factory):
            raise ValueError("engine requires exactly one of env_id or factory")
        _require_json(self.runtime, "engine.runtime")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ScenarioEngineRequirement":
        _reject_unknown(
            data,
            {"engine_name", "version_spec", "task_id", "env_id", "factory", "runtime"},
            "engine",
        )
        runtime = data.get("runtime", {})
        if not isinstance(runtime, dict):
            raise ValueError("engine.runtime must be a mapping")
        return cls(
            engine_name=str(data.get("engine_name", "")),
            version_spec=str(data.get("version_spec", "")),
            task_id=str(data.get("task_id", "")),
            env_id=str(data["env_id"]) if data.get("env_id") is not None else None,
            factory=str(data["factory"]) if data.get("factory") is not None else None,
            runtime=dict(runtime),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine_name": self.engine_name,
            "version_spec": self.version_spec,
            "task_id": self.task_id,
            "env_id": self.env_id,
            "factory": self.factory,
            "runtime": self.runtime,
        }


@dataclass(frozen=True)
class ScenarioAsset:
    asset_id: str
    sha256: str
    license_id: str
    provenance_uri: str
    redistribution: AssetRedistribution
    path: str | None = None
    external_locator: str | None = None
    required: bool = True

    def __post_init__(self) -> None:
        _require_text(self.asset_id, "asset.asset_id")
        _require_sha256(self.sha256, f"asset {self.asset_id} sha256")
        _require_license_id(self.license_id, f"asset {self.asset_id} license_id")
        _require_uri(
            self.provenance_uri,
            f"asset {self.asset_id} provenance_uri",
            allowed_schemes={"http", "https", "doi", "urn"},
        )
        if self.redistribution not in {
            "redistributable",
            "protected",
            "metadata_only",
        }:
            raise ValueError(
                f"asset {self.asset_id} has unsupported redistribution policy"
            )
        if self.path is not None:
            _require_safe_relative_path(self.path, f"asset {self.asset_id} path")
        if self.redistribution == "redistributable" and not self.path:
            raise ValueError(
                f"redistributable asset {self.asset_id} requires a package-relative path"
            )
        if self.redistribution != "redistributable" and not (
            self.path or self.external_locator
        ):
            raise ValueError(
                f"protected asset {self.asset_id} requires a path or external_locator"
            )
        if self.external_locator is not None:
            _require_uri(
                self.external_locator,
                f"asset {self.asset_id} external_locator",
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ScenarioAsset":
        _reject_unknown(
            data,
            {
                "asset_id",
                "sha256",
                "license_id",
                "provenance_uri",
                "redistribution",
                "path",
                "external_locator",
                "required",
            },
            "asset",
        )
        return cls(
            asset_id=str(data.get("asset_id", "")),
            sha256=str(data.get("sha256", "")),
            license_id=str(data.get("license_id", "")),
            provenance_uri=str(data.get("provenance_uri", "")),
            redistribution=str(data.get("redistribution", "")),  # type: ignore[arg-type]
            path=str(data["path"]) if data.get("path") is not None else None,
            external_locator=str(data["external_locator"])
            if data.get("external_locator") is not None
            else None,
            required=bool(data.get("required", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "sha256": self.sha256,
            "license_id": self.license_id,
            "provenance_uri": self.provenance_uri,
            "redistribution": self.redistribution,
            "path": self.path,
            "external_locator": self.external_locator,
            "required": self.required,
        }


@dataclass(frozen=True)
class ScenarioInitialState:
    run_seed: int
    seed_protocol: dict[str, Any]
    state_sha256: str
    physical_parameters: dict[str, Any]
    observability: dict[str, str]

    def __post_init__(self) -> None:
        if self.run_seed < 0:
            raise ValueError("initial_state.run_seed must be non-negative")
        expected_seed_protocol = {
            "format": SCENARIO_EPISODE_SEED_FORMAT,
            "episode_seed_stride": SCENARIO_EPISODE_SEED_STRIDE,
            "formula": SCENARIO_EPISODE_SEED_FORMULA,
            "shared_across_tasks": True,
        }
        if self.seed_protocol != expected_seed_protocol:
            raise ValueError(
                "initial_state.seed_protocol must match nyssa-episode-seed-v2"
            )
        _require_sha256(self.state_sha256, "initial_state.state_sha256")
        if not self.physical_parameters:
            raise ValueError("initial_state.physical_parameters must not be empty")
        _require_json(self.physical_parameters, "initial_state.physical_parameters")
        if not self.observability:
            raise ValueError("initial_state.observability must not be empty")
        allowed = {"policy_observable", "privileged", "external"}
        if any(value not in allowed for value in self.observability.values()):
            raise ValueError(
                "initial_state.observability contains an invalid visibility"
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ScenarioInitialState":
        _reject_unknown(
            data,
            {
                "run_seed",
                "seed_protocol",
                "state_sha256",
                "physical_parameters",
                "observability",
            },
            "initial_state",
        )
        physical = data.get("physical_parameters", {})
        observability = data.get("observability", {})
        seed_protocol = data.get("seed_protocol", {})
        if not all(
            isinstance(value, dict)
            for value in (physical, observability, seed_protocol)
        ):
            raise ValueError(
                "initial_state seed_protocol, physical_parameters, and observability must be mappings"
            )
        return cls(
            run_seed=int(data.get("run_seed", -1)),
            seed_protocol=dict(seed_protocol),
            state_sha256=str(data.get("state_sha256", "")),
            physical_parameters=dict(physical),
            observability={
                str(key): str(value) for key, value in observability.items()
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_seed": self.run_seed,
            "seed_protocol": self.seed_protocol,
            "state_sha256": self.state_sha256,
            "physical_parameters": self.physical_parameters,
            "observability": self.observability,
        }


@dataclass(frozen=True)
class ScenarioStressorAxis:
    stressor_id: str
    severity_range: tuple[float, float]
    default_severity: float
    parameters: dict[str, Any] = field(default_factory=dict)
    composable_with: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    contract_format: str = STRESSOR_SPEC_FORMAT

    def __post_init__(self) -> None:
        _require_text(self.stressor_id, "stressor_axis.stressor_id")
        if self.contract_format != STRESSOR_SPEC_FORMAT:
            raise ValueError(
                f"stressor axis contract_format must be {STRESSOR_SPEC_FORMAT}"
            )
        lower, upper = self.severity_range
        if not all(_finite(value) for value in self.severity_range):
            raise ValueError("stressor severity_range must be finite")
        if lower < 0.0 or upper > 1.0 or lower > upper:
            raise ValueError("stressor severity_range must be ordered within [0, 1]")
        if not lower <= self.default_severity <= upper:
            raise ValueError("stressor default_severity must be within severity_range")
        if self.stressor_id in self.composable_with:
            raise ValueError("stressor axis cannot compose with itself")
        if len(set(self.composable_with)) != len(self.composable_with):
            raise ValueError("stressor composable_with values must be unique")
        _require_json(self.parameters, "stressor_axis.parameters")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ScenarioStressorAxis":
        _check_format(data, SCENARIO_STRESSOR_AXIS_FORMAT, "stressor axis")
        _reject_unknown(
            data,
            {
                "format",
                "contract_format",
                "stressor_id",
                "severity_range",
                "default_severity",
                "parameters",
                "composable_with",
                "constraints",
            },
            "stressor axis",
        )
        severity = data.get("severity_range", [])
        parameters = data.get("parameters", {})
        if not isinstance(severity, list) or len(severity) != 2:
            raise ValueError("stressor severity_range must contain two values")
        if not isinstance(parameters, dict):
            raise ValueError("stressor parameters must be a mapping")
        return cls(
            stressor_id=str(data.get("stressor_id", "")),
            severity_range=(float(severity[0]), float(severity[1])),
            default_severity=float(data.get("default_severity", severity[0])),
            parameters=dict(parameters),
            composable_with=_string_tuple(
                data.get("composable_with", []), "stressor composable_with"
            ),
            constraints=_string_tuple(data.get("constraints", []), "constraints"),
            contract_format=str(data.get("contract_format", STRESSOR_SPEC_FORMAT)),
        )

    def to_spec(
        self, *, severity: float | None = None, seed: int | None = None
    ) -> StressorSpec:
        selected = self.default_severity if severity is None else float(severity)
        lower, upper = self.severity_range
        if not lower <= selected <= upper:
            raise ValueError(
                f"severity {selected} is outside scenario axis {self.stressor_id} range [{lower}, {upper}]"
            )
        return StressorSpec(
            stressor_id=self.stressor_id,
            severity=selected,
            parameters=dict(self.parameters),
            seed=seed,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": SCENARIO_STRESSOR_AXIS_FORMAT,
            "contract_format": self.contract_format,
            "stressor_id": self.stressor_id,
            "severity_range": list(self.severity_range),
            "default_severity": self.default_severity,
            "parameters": self.parameters,
            "composable_with": list(self.composable_with),
            "constraints": list(self.constraints),
        }


@dataclass(frozen=True)
class ScenarioSplitLineage:
    split_id: str
    partition: SplitPartition
    content_sha256: str
    parent_split_ids: tuple[str, ...]
    member_count: int
    protected: bool
    contamination_status: ContaminationStatus
    contamination_sources: tuple[str, ...] = ()
    format: str = SCENARIO_SPLIT_FORMAT

    def __post_init__(self) -> None:
        _require_text(self.split_id, "split.split_id")
        if self.partition not in {"train", "validation", "public_test", "hidden_test"}:
            raise ValueError(f"split {self.split_id} has invalid partition")
        _require_sha256(self.content_sha256, f"split {self.split_id} content_sha256")
        if self.member_count <= 0:
            raise ValueError(f"split {self.split_id} member_count must be positive")
        if self.split_id in self.parent_split_ids:
            raise ValueError(f"split {self.split_id} cannot parent itself")
        if len(set(self.parent_split_ids)) != len(self.parent_split_ids):
            raise ValueError(f"split {self.split_id} parent_split_ids must be unique")
        if self.partition == "hidden_test" and not self.protected:
            raise ValueError("hidden_test splits must set protected: true")
        if self.contamination_status not in {"clean", "known_overlap", "unknown"}:
            raise ValueError(f"split {self.split_id} has invalid contamination_status")
        if self.contamination_status == "clean" and self.contamination_sources:
            raise ValueError(
                f"clean split {self.split_id} cannot declare contamination_sources"
            )
        if (
            self.contamination_status == "known_overlap"
            and not self.contamination_sources
        ):
            raise ValueError(
                f"known-overlap split {self.split_id} requires contamination_sources"
            )
        if len(set(self.contamination_sources)) != len(self.contamination_sources):
            raise ValueError(
                f"split {self.split_id} contamination_sources must be unique"
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ScenarioSplitLineage":
        _check_format(data, SCENARIO_SPLIT_FORMAT, "scenario split")
        _reject_unknown(
            data,
            {
                "format",
                "split_id",
                "partition",
                "content_sha256",
                "parent_split_ids",
                "member_count",
                "protected",
                "contamination_status",
                "contamination_sources",
            },
            "scenario split",
        )
        return cls(
            split_id=str(data.get("split_id", "")),
            partition=str(data.get("partition", "")),  # type: ignore[arg-type]
            content_sha256=str(data.get("content_sha256", "")),
            parent_split_ids=_string_tuple(
                data.get("parent_split_ids", []), "split parent_split_ids"
            ),
            member_count=int(data.get("member_count", 0)),
            protected=bool(data.get("protected", False)),
            contamination_status=str(  # type: ignore[arg-type]
                data.get("contamination_status", "unknown")
            ),
            contamination_sources=_string_tuple(
                data.get("contamination_sources", []),
                "split contamination_sources",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "split_id": self.split_id,
            "partition": self.partition,
            "content_sha256": self.content_sha256,
            "parent_split_ids": list(self.parent_split_ids),
            "member_count": self.member_count,
            "protected": self.protected,
            "contamination_status": self.contamination_status,
            "contamination_sources": list(self.contamination_sources),
        }


@dataclass(frozen=True)
class ScenarioEvaluation:
    success_predicate: dict[str, Any]
    horizon_steps: int
    safety_constraints: tuple[dict[str, Any], ...]
    solvability_checks: tuple[dict[str, Any], ...]

    def __post_init__(self) -> None:
        if not self.success_predicate:
            raise ValueError("evaluation.success_predicate must not be empty")
        if self.horizon_steps <= 0:
            raise ValueError("evaluation.horizon_steps must be positive")
        if not self.safety_constraints:
            raise ValueError("evaluation.safety_constraints must not be empty")
        if not self.solvability_checks:
            raise ValueError("evaluation.solvability_checks must not be empty")
        for index, check in enumerate(self.solvability_checks):
            if not check.get("check_id") or not check.get("policy"):
                raise ValueError(
                    f"evaluation.solvability_checks[{index}] requires check_id and policy"
                )
            expected = check.get("minimum_success_rate")
            if expected is None or not 0.0 <= float(expected) <= 1.0:
                raise ValueError(
                    f"evaluation.solvability_checks[{index}] has invalid minimum_success_rate"
                )
        _require_json(self.success_predicate, "evaluation.success_predicate")
        _require_json(list(self.safety_constraints), "evaluation.safety_constraints")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ScenarioEvaluation":
        _reject_unknown(
            data,
            {
                "success_predicate",
                "horizon_steps",
                "safety_constraints",
                "solvability_checks",
            },
            "evaluation",
        )
        predicate = data.get("success_predicate", {})
        safety = data.get("safety_constraints", [])
        solvability = data.get("solvability_checks", [])
        if not isinstance(predicate, dict):
            raise ValueError("evaluation.success_predicate must be a mapping")
        if not isinstance(safety, list) or not all(
            isinstance(item, dict) for item in safety
        ):
            raise ValueError("evaluation.safety_constraints must be a list of mappings")
        if not isinstance(solvability, list) or not all(
            isinstance(item, dict) for item in solvability
        ):
            raise ValueError("evaluation.solvability_checks must be a list of mappings")
        return cls(
            success_predicate=dict(predicate),
            horizon_steps=int(data.get("horizon_steps", 0)),
            safety_constraints=tuple(dict(item) for item in safety),
            solvability_checks=tuple(dict(item) for item in solvability),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "success_predicate": self.success_predicate,
            "horizon_steps": self.horizon_steps,
            "safety_constraints": list(self.safety_constraints),
            "solvability_checks": list(self.solvability_checks),
        }


@dataclass(frozen=True)
class ScenarioPackage:
    scenario_id: str
    scenario_version: str
    description: str
    content_sha256: str
    protocol: ScenarioProtocolReferences
    generator: ScenarioGenerator
    engine: ScenarioEngineRequirement
    assets: tuple[ScenarioAsset, ...]
    initial_state: ScenarioInitialState
    stressor_axes: tuple[ScenarioStressorAxis, ...]
    split_lineage: tuple[ScenarioSplitLineage, ...]
    evaluation: ScenarioEvaluation
    rare_event_provenance: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    source_path: Path | None = field(default=None, compare=False, repr=False)
    schema_version: int = SCENARIO_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        _require_text(self.scenario_id, "scenario_id")
        _require_semver(self.scenario_version, "scenario_version")
        _require_text(self.description, "description")
        _require_sha256(self.content_sha256, "content_sha256")
        if self.schema_version != SCENARIO_PROTOCOL_VERSION:
            raise ValueError(
                f"Unsupported scenario schema version: {self.schema_version}"
            )
        _require_unique([asset.asset_id for asset in self.assets], "asset IDs")
        _require_unique(
            [axis.stressor_id for axis in self.stressor_axes], "stressor axis IDs"
        )
        _require_unique([split.split_id for split in self.split_lineage], "split IDs")
        if not self.split_lineage:
            raise ValueError("scenario package requires split_lineage")
        if self.rare_event_provenance is not None:
            allowed = {
                "method_id",
                "method_version",
                "search_budget",
                "selected_condition",
                "objective",
                "study_sha256",
                "seed",
            }
            _reject_unknown(
                self.rare_event_provenance,
                allowed,
                "rare_event_provenance",
            )
            for field_name in (
                "method_id",
                "method_version",
                "search_budget",
                "selected_condition",
            ):
                field_value = self.rare_event_provenance.get(field_name)
                if field_value is None or field_value == "":
                    raise ValueError(f"rare_event_provenance requires {field_name}")
            _require_semver(
                str(self.rare_event_provenance["method_version"]),
                "rare_event_provenance.method_version",
            )
            if int(self.rare_event_provenance["search_budget"]) <= 0:
                raise ValueError("rare_event_provenance.search_budget must be positive")
            if self.rare_event_provenance.get("study_sha256") is not None:
                _require_sha256(
                    str(self.rare_event_provenance["study_sha256"]),
                    "rare_event_provenance.study_sha256",
                )
            if (
                self.rare_event_provenance.get("seed") is not None
                and int(self.rare_event_provenance["seed"]) < 0
            ):
                raise ValueError("rare_event_provenance.seed must be non-negative")
            _require_json(self.rare_event_provenance, "rare_event_provenance")
        _require_json(self.metadata, "metadata")

    @classmethod
    def load(cls, path: str | Path) -> "ScenarioPackage":
        source = resolve_scenario_manifest(path)
        raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Scenario manifest must contain a mapping: {source}")
        return cls.from_dict(raw, source_path=source)

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any], *, source_path: Path | None = None
    ) -> "ScenarioPackage":
        _check_format(data, SCENARIO_PACKAGE_FORMAT, "scenario package")
        _reject_unknown(
            data,
            {
                "format",
                "schema_version",
                "scenario_id",
                "scenario_version",
                "description",
                "content_sha256",
                "protocol",
                "generator",
                "engine",
                "assets",
                "initial_state",
                "stressor_axes",
                "split_lineage",
                "evaluation",
                "rare_event_provenance",
                "metadata",
            },
            "scenario package",
        )
        mappings = {
            name: data.get(name, {})
            for name in (
                "protocol",
                "generator",
                "engine",
                "initial_state",
                "evaluation",
            )
        }
        if any(not isinstance(value, dict) for value in mappings.values()):
            raise ValueError(
                "scenario protocol, generator, engine, initial_state, and evaluation must be mappings"
            )
        assets = _mapping_list(data.get("assets", []), "assets")
        axes = _mapping_list(data.get("stressor_axes", []), "stressor_axes")
        splits = _mapping_list(data.get("split_lineage", []), "split_lineage")
        rare_event = data.get("rare_event_provenance")
        metadata = data.get("metadata", {})
        if rare_event is not None and not isinstance(rare_event, dict):
            raise ValueError("rare_event_provenance must be a mapping")
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a mapping")
        return cls(
            scenario_id=str(data.get("scenario_id", "")),
            scenario_version=str(data.get("scenario_version", "")),
            description=str(data.get("description", "")),
            content_sha256=str(data.get("content_sha256", "")),
            protocol=ScenarioProtocolReferences.from_dict(mappings["protocol"]),
            generator=ScenarioGenerator.from_dict(mappings["generator"]),
            engine=ScenarioEngineRequirement.from_dict(mappings["engine"]),
            assets=tuple(ScenarioAsset.from_dict(item) for item in assets),
            initial_state=ScenarioInitialState.from_dict(mappings["initial_state"]),
            stressor_axes=tuple(ScenarioStressorAxis.from_dict(item) for item in axes),
            split_lineage=tuple(
                ScenarioSplitLineage.from_dict(item) for item in splits
            ),
            evaluation=ScenarioEvaluation.from_dict(mappings["evaluation"]),
            rare_event_provenance=dict(rare_event) if rare_event is not None else None,
            metadata=dict(metadata),
            source_path=source_path,
            schema_version=int(data.get("schema_version", SCENARIO_PROTOCOL_VERSION)),
        )

    @property
    def package_root(self) -> Path | None:
        return self.source_path.parent if self.source_path is not None else None

    @property
    def identity(self) -> str:
        return f"{self.scenario_id}@{self.scenario_version}:{self.content_sha256}"

    def compute_content_sha256(self) -> str:
        payload = self.to_dict()
        payload.pop("content_sha256", None)
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def stressor_config(
        self,
        *,
        severities: Mapping[str, float] | None = None,
        seed: int | None = None,
    ) -> StressorConfig:
        overrides = dict(severities or {})
        unknown = sorted(
            set(overrides) - {axis.stressor_id for axis in self.stressor_axes}
        )
        if unknown:
            raise ValueError(
                f"Unknown scenario stressor severity overrides: {', '.join(unknown)}"
            )
        specs = tuple(
            axis.to_spec(severity=overrides.get(axis.stressor_id), seed=seed)
            for axis in self.stressor_axes
        )
        return StressorConfig(
            condition_id=f"scenario:{self.scenario_id}@{self.scenario_version}",
            stressors=specs,
            unsupported_policy="error",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": SCENARIO_PACKAGE_FORMAT,
            "schema_version": self.schema_version,
            "scenario_id": self.scenario_id,
            "scenario_version": self.scenario_version,
            "description": self.description,
            "content_sha256": self.content_sha256,
            "protocol": self.protocol.to_dict(),
            "generator": self.generator.to_dict(),
            "engine": self.engine.to_dict(),
            "assets": [asset.to_dict() for asset in self.assets],
            "initial_state": self.initial_state.to_dict(),
            "stressor_axes": [axis.to_dict() for axis in self.stressor_axes],
            "split_lineage": [split.to_dict() for split in self.split_lineage],
            "evaluation": self.evaluation.to_dict(),
            "rare_event_provenance": self.rare_event_provenance,
            "metadata": self.metadata,
        }


def resolve_scenario_manifest(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / DEFAULT_SCENARIO_MANIFEST
    if not candidate.is_file():
        raise FileNotFoundError(f"Scenario manifest not found: {candidate}")
    return candidate.resolve()


def scenario_conformance_fixture_path(
    fixture_name: str = "valid_seeded_mujoco",
) -> Path:
    _require_safe_relative_path(fixture_name, "fixture_name")
    root = Path(__file__).resolve().parents[2] / "conformance" / "scenario" / "v1"
    fixture = (root / fixture_name).resolve()
    if not _is_relative_to(fixture, root.resolve()) or not fixture.is_dir():
        raise FileNotFoundError(
            f"Scenario conformance fixture not found: {fixture_name}"
        )
    return fixture


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_format(data: Mapping[str, Any], expected: str, label: str) -> None:
    if data.get("format") != expected:
        raise ValueError(f"Unsupported {label} format: {data.get('format')}")


def _reject_unknown(data: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"Unknown {label} fields: {', '.join(unknown)}")


def _require_text(value: str, label: str) -> None:
    if not value.strip():
        raise ValueError(f"{label} must be non-empty")


def _require_semver(value: str, label: str) -> None:
    if not re.fullmatch(
        r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:[-+][0-9A-Za-z.-]+)?",
        value,
    ):
        raise ValueError(f"{label} must be a semantic version")


def _require_version_spec(value: str, label: str) -> None:
    _require_text(value, label)
    clauses = value.split(",")
    pattern = re.compile(r"\s*(?:~=|==|!=|>=|<=|>|<)?\s*\d+(?:\.\d+){0,3}\s*")
    if not clauses or any(not pattern.fullmatch(clause) for clause in clauses):
        raise ValueError(f"{label} must be a comma-separated version constraint")


def _require_uri(
    value: str,
    label: str,
    *,
    allowed_schemes: set[str] | None = None,
) -> None:
    _require_text(value, label)
    parsed = urlparse(value)
    if not parsed.scheme or not (parsed.netloc or parsed.path):
        raise ValueError(f"{label} must be an absolute URI")
    if allowed_schemes is not None and parsed.scheme not in allowed_schemes:
        allowed = ", ".join(sorted(allowed_schemes))
        raise ValueError(f"{label} must use one of these URI schemes: {allowed}")


def _require_sha256(value: str, label: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def _require_license_id(value: str, label: str) -> None:
    _require_text(value, label)
    if not re.fullmatch(r"(?:LicenseRef-)?[A-Za-z0-9][A-Za-z0-9.+-]*", value):
        raise ValueError(f"{label} must be an SPDX ID or LicenseRef identifier")


def _require_safe_relative_path(value: str, label: str) -> None:
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        path.is_absolute()
        or re.match(r"^[A-Za-z]:/", normalized)
        or ".." in path.parts
        or not path.parts
    ):
        raise ValueError(f"{label} must be a safe package-relative path")


def _require_json(value: Any, label: str) -> None:
    try:
        json.dumps(value, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain finite JSON-compatible data") from exc


def _require_unique(values: list[str], label: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must be unique")


def _mapping_list(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{label} must be a list of mappings")
    return [dict(item) for item in value]


def _string_tuple(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    result = tuple(str(item) for item in value)
    if any(not item.strip() for item in result):
        raise ValueError(f"{label} values must be non-empty")
    return result


def _finite(value: float) -> bool:
    return value == value and value not in {float("inf"), float("-inf")}


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
