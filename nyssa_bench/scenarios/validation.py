from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from nyssa_bench.core.registry import ENGINE_REGISTRY
from nyssa_bench.core.task import TaskSpec
from nyssa_bench.plugins import get_plugin_registry
from nyssa_bench.stressors import STRESSOR_REGISTRY, StressorContext

from .protocol import ScenarioPackage, sha256_file


IssueSeverity = Literal["error", "warning"]


@dataclass(frozen=True)
class ScenarioValidationIssue:
    code: str
    message: str
    path: str
    severity: IssueSeverity = "error"

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "message": self.message,
            "path": self.path,
            "severity": self.severity,
        }


@dataclass(frozen=True)
class ScenarioValidationReport:
    scenario_identity: str
    schema_valid: bool
    execution_ready: bool
    claim_ready: bool
    expected_engine: str
    issues: tuple[ScenarioValidationIssue, ...]
    resolved_assets: tuple[str, ...]
    unresolved_protected_assets: tuple[str, ...]
    stressor_contracts: tuple[dict[str, Any], ...]

    @property
    def valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def raise_for_errors(self) -> None:
        errors = [issue for issue in self.issues if issue.severity == "error"]
        if not errors:
            return
        details = "; ".join(
            f"{issue.code} ({issue.path}): {issue.message}" for issue in errors
        )
        raise ScenarioValidationError(details, report=self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "nyssa-scenario-validation-v1",
            "scenario_identity": self.scenario_identity,
            "valid": self.valid,
            "schema_valid": self.schema_valid,
            "execution_ready": self.execution_ready,
            "claim_ready": self.claim_ready,
            "expected_engine": self.expected_engine,
            "issues": [issue.to_dict() for issue in self.issues],
            "resolved_assets": list(self.resolved_assets),
            "unresolved_protected_assets": list(self.unresolved_protected_assets),
            "stressor_contracts": list(self.stressor_contracts),
        }


class ScenarioValidationError(ValueError):
    def __init__(self, message: str, *, report: ScenarioValidationReport) -> None:
        self.report = report
        super().__init__(message)


class ScenarioPackageValidator:
    """Validate an external scenario without invoking its producer."""

    def validate(
        self,
        package: ScenarioPackage,
        *,
        expected_engine: str | None = None,
        require_execution_assets: bool = True,
    ) -> ScenarioValidationReport:
        engine_name = expected_engine or package.engine.engine_name
        issues: list[ScenarioValidationIssue] = []
        resolved_assets: list[str] = []
        unresolved_protected: list[str] = []

        if package.compute_content_sha256() != package.content_sha256:
            _error(
                issues,
                "package_hash_mismatch",
                "declared content_sha256 does not match canonical package content",
                "content_sha256",
            )
        if engine_name != package.engine.engine_name:
            _error(
                issues,
                "incompatible_engine",
                f"package requires '{package.engine.engine_name}', requested '{engine_name}'",
                "engine.engine_name",
            )
        if not _engine_registered(package.engine.engine_name):
            _error(
                issues,
                "engine_adapter_unavailable",
                f"no NyssaBench engine adapter is registered for '{package.engine.engine_name}'",
                "engine.engine_name",
            )

        task = self._validate_task_mapping(package, issues)
        if (
            package.initial_state.physical_parameters.get("mutation_policy")
            != "stressor_contracts_only"
        ):
            _error(
                issues,
                "physical_parameter_bypass",
                "physical parameter variation must use declared Stressor Contracts",
                "initial_state.physical_parameters.mutation_policy",
            )
        if (
            package.initial_state.physical_parameters.get("source")
            != "simulator_task_default"
        ):
            _error(
                issues,
                "physical_parameter_source_unsupported",
                "v1 execution supports task-default physical baselines only",
                "initial_state.physical_parameters.source",
            )
        unknown_physical_fields = sorted(
            set(package.initial_state.physical_parameters)
            - {"source", "mutation_policy"}
        )
        if unknown_physical_fields:
            _error(
                issues,
                "physical_parameter_unmapped",
                "physical baseline fields are not mapped by the task contract: "
                + ", ".join(unknown_physical_fields),
                "initial_state.physical_parameters",
            )
        self._validate_assets(
            package,
            issues,
            resolved_assets,
            unresolved_protected,
            require_execution_assets=require_execution_assets,
        )
        stressor_contracts = self._validate_stressors(package, task, issues)
        self._validate_splits(package, issues)

        execution_ready = (
            not any(issue.severity == "error" for issue in issues)
            and not unresolved_protected
        )
        if unresolved_protected and not require_execution_assets:
            execution_ready = False
        claim_ready = execution_ready and not any(
            issue.code == "evaluation_split_known_overlap" for issue in issues
        )
        return ScenarioValidationReport(
            scenario_identity=package.identity,
            schema_valid=True,
            execution_ready=execution_ready,
            claim_ready=claim_ready,
            expected_engine=engine_name,
            issues=tuple(issues),
            resolved_assets=tuple(sorted(resolved_assets)),
            unresolved_protected_assets=tuple(sorted(unresolved_protected)),
            stressor_contracts=tuple(stressor_contracts),
        )

    def _validate_task_mapping(
        self,
        package: ScenarioPackage,
        issues: list[ScenarioValidationIssue],
    ) -> TaskSpec | None:
        try:
            task = TaskSpec.load(package.engine.task_id)
        except (FileNotFoundError, ValueError) as exc:
            _error(
                issues,
                "task_mapping_unresolved",
                str(exc),
                "engine.task_id",
            )
            return None
        if task.engine != package.engine.engine_name:
            _error(
                issues,
                "task_engine_mismatch",
                f"task declares '{task.engine}', package declares '{package.engine.engine_name}'",
                "engine.task_id",
            )
        env_ids = task.success.get("engine_env_ids", {})
        task_env_id = (
            env_ids.get(package.engine.engine_name)
            if isinstance(env_ids, dict)
            else None
        )
        task_factory = task.success.get("engine_factory", {})
        task_factory_value = (
            task_factory.get(package.engine.engine_name)
            if isinstance(task_factory, dict)
            else None
        )
        if package.engine.env_id and task_env_id != package.engine.env_id:
            _error(
                issues,
                "task_environment_mismatch",
                f"task maps to '{task_env_id}', package declares '{package.engine.env_id}'",
                "engine.env_id",
            )
        if package.engine.factory and task_factory_value != package.engine.factory:
            _error(
                issues,
                "task_factory_mismatch",
                "package factory does not match the task contract",
                "engine.factory",
            )
        for key, value in package.evaluation.success_predicate.items():
            if task.success.get(key) != value:
                _error(
                    issues,
                    "success_predicate_mismatch",
                    f"package value for '{key}' does not match the task contract",
                    f"evaluation.success_predicate.{key}",
                )
        task_horizon = task.success.get("max_steps")
        if (
            task_horizon is not None
            and int(task_horizon) != package.evaluation.horizon_steps
        ):
            _error(
                issues,
                "horizon_mismatch",
                f"task horizon is {task_horizon}, package horizon is {package.evaluation.horizon_steps}",
                "evaluation.horizon_steps",
            )
        return task

    def _validate_assets(
        self,
        package: ScenarioPackage,
        issues: list[ScenarioValidationIssue],
        resolved_assets: list[str],
        unresolved_protected: list[str],
        *,
        require_execution_assets: bool,
    ) -> None:
        root = package.package_root
        for index, asset in enumerate(package.assets):
            path_label = f"assets[{index}]"
            if not asset.path or root is None:
                if asset.required:
                    self._unresolved_asset(
                        asset.asset_id,
                        asset.redistribution,
                        path_label,
                        issues,
                        unresolved_protected,
                        require_execution_assets=require_execution_assets,
                    )
                continue
            candidate = (root / asset.path).resolve()
            if not _is_within(candidate, root.resolve()):
                _error(
                    issues,
                    "unsafe_asset_path",
                    "asset path resolves outside the scenario package",
                    f"{path_label}.path",
                )
                continue
            if not candidate.is_file():
                if asset.required:
                    self._unresolved_asset(
                        asset.asset_id,
                        asset.redistribution,
                        path_label,
                        issues,
                        unresolved_protected,
                        require_execution_assets=require_execution_assets,
                    )
                continue
            if sha256_file(candidate) != asset.sha256:
                _error(
                    issues,
                    "asset_hash_mismatch",
                    f"asset '{asset.asset_id}' content does not match sha256",
                    f"{path_label}.sha256",
                )
                continue
            resolved_assets.append(asset.asset_id)

    def _unresolved_asset(
        self,
        asset_id: str,
        redistribution: str,
        path_label: str,
        issues: list[ScenarioValidationIssue],
        unresolved_protected: list[str],
        *,
        require_execution_assets: bool,
    ) -> None:
        protected = redistribution in {"protected", "metadata_only"}
        if protected:
            unresolved_protected.append(asset_id)
        severity: IssueSeverity = (
            "warning" if protected and not require_execution_assets else "error"
        )
        issues.append(
            ScenarioValidationIssue(
                code="protected_asset_unresolved" if protected else "asset_unresolved",
                message=f"required asset '{asset_id}' is not resolved",
                path=path_label,
                severity=severity,
            )
        )

    def _validate_stressors(
        self,
        package: ScenarioPackage,
        task: TaskSpec | None,
        issues: list[ScenarioValidationIssue],
    ) -> list[dict[str, Any]]:
        contracts: list[dict[str, Any]] = []
        axis_ids = {axis.stressor_id for axis in package.stressor_axes}
        for index, axis in enumerate(package.stressor_axes):
            label = f"stressor_axes[{index}]"
            stressor_cls = STRESSOR_REGISTRY.get(axis.stressor_id)
            if stressor_cls is None:
                _error(
                    issues,
                    "unknown_stressor_contract",
                    f"stressor '{axis.stressor_id}' is not registered",
                    f"{label}.stressor_id",
                )
                continue
            lower, upper = axis.severity_range
            contract_lower, contract_upper = stressor_cls.severity_domain
            if lower < contract_lower or upper > contract_upper:
                _error(
                    issues,
                    "stressor_severity_outside_contract",
                    f"axis range [{lower}, {upper}] exceeds [{contract_lower}, {contract_upper}]",
                    f"{label}.severity_range",
                )
            unknown_compositions = sorted(set(axis.composable_with) - axis_ids)
            if unknown_compositions:
                _error(
                    issues,
                    "unresolved_stressor_composition",
                    f"unknown composable axes: {', '.join(unknown_compositions)}",
                    f"{label}.composable_with",
                )
            conflicts = set(stressor_cls.conflicts_with) & set(axis.composable_with)
            if conflicts:
                _error(
                    issues,
                    "conflicting_stressor_composition",
                    f"contract conflicts with: {', '.join(sorted(conflicts))}",
                    f"{label}.composable_with",
                )
            if task is not None:
                try:
                    stressor = stressor_cls()
                    spec = axis.to_spec(seed=package.initial_state.run_seed)
                    stressor.reset(spec, seed=package.initial_state.run_seed)
                    reason = stressor.support_reason(
                        StressorContext(
                            engine_name=package.engine.engine_name,
                            task_id=task.task_id,
                            observation_mode=_task_mode(task.observation),
                            action_mode=_task_mode(task.action),
                        )
                    )
                except Exception as exc:
                    _error(
                        issues,
                        "invalid_stressor_parameters",
                        str(exc),
                        label,
                    )
                else:
                    if reason:
                        _error(
                            issues,
                            "stressor_not_executable",
                            reason,
                            label,
                        )
            contracts.append(
                {
                    "format": axis.contract_format,
                    "stressor_id": axis.stressor_id,
                    "category": stressor_cls.category,
                    "application_points": list(stressor_cls.application_points),
                    "severity_domain": list(stressor_cls.severity_domain),
                    "lifetime": stressor_cls.lifetime,
                    "observable_by_policy": stressor_cls.observable_by_policy,
                    "privileged": stressor_cls.privileged,
                }
            )
        return contracts

    def _validate_splits(
        self,
        package: ScenarioPackage,
        issues: list[ScenarioValidationIssue],
    ) -> None:
        splits = {split.split_id: split for split in package.split_lineage}
        if not any(
            split.partition in {"public_test", "hidden_test"}
            for split in package.split_lineage
        ):
            _error(
                issues,
                "evaluation_split_missing",
                "scenario packages require a public_test or hidden_test split",
                "split_lineage",
            )
        for split in package.split_lineage:
            missing = sorted(set(split.parent_split_ids) - set(splits))
            if missing:
                _error(
                    issues,
                    "split_parent_unresolved",
                    f"unknown parent splits: {', '.join(missing)}",
                    f"split_lineage.{split.split_id}.parent_split_ids",
                )
            if (
                split.partition in {"public_test", "hidden_test"}
                and split.contamination_status == "unknown"
            ):
                _error(
                    issues,
                    "evaluation_split_contamination_unknown",
                    "evaluation splits require explicit clean or known-overlap contamination status",
                    f"split_lineage.{split.split_id}.contamination_status",
                )
            if (
                split.partition in {"public_test", "hidden_test"}
                and split.contamination_status == "known_overlap"
            ):
                issues.append(
                    ScenarioValidationIssue(
                        code="evaluation_split_known_overlap",
                        message="evaluation split is executable but cannot support a clean public claim",
                        path=f"split_lineage.{split.split_id}.contamination_status",
                        severity="warning",
                    )
                )
        for split_id in splits:
            if _has_lineage_cycle(split_id, splits, set(), set()):
                _error(
                    issues,
                    "split_lineage_cycle",
                    f"split lineage contains a cycle involving '{split_id}'",
                    "split_lineage",
                )
                break
        by_hash: dict[str, list[Any]] = {}
        for split in package.split_lineage:
            by_hash.setdefault(split.content_sha256, []).append(split)
        for same_content in by_hash.values():
            partitions = {split.partition for split in same_content}
            if "train" in partitions and partitions & {"public_test", "hidden_test"}:
                if all(split.contamination_status == "clean" for split in same_content):
                    _error(
                        issues,
                        "split_content_overlap",
                        "train and evaluation splits share a content hash but claim to be clean",
                        "split_lineage",
                    )


def _error(
    issues: list[ScenarioValidationIssue], code: str, message: str, path: str
) -> None:
    issues.append(
        ScenarioValidationIssue(code=code, message=message, path=path, severity="error")
    )


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _task_mode(contract: dict[str, Any]) -> str | None:
    value = contract.get("mode") or contract.get("type")
    return str(value) if value is not None else None


def _engine_registered(engine_name: str) -> bool:
    return (
        engine_name in ENGINE_REGISTRY or engine_name in get_plugin_registry().engines
    )


def _has_lineage_cycle(
    split_id: str,
    splits: dict[str, Any],
    visiting: set[str],
    visited: set[str],
) -> bool:
    if split_id in visited:
        return False
    if split_id in visiting:
        return True
    visiting.add(split_id)
    split = splits[split_id]
    for parent in split.parent_split_ids:
        if parent in splits and _has_lineage_cycle(parent, splits, visiting, visited):
            return True
    visiting.remove(split_id)
    visited.add(split_id)
    return False
