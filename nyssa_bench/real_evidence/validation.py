from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from nyssa_bench.failures.protocol import FailureEvent, FailureLedgerRecord

from .protocol import RealEvidencePackage, sha256_file


IssueSeverity = Literal["error", "warning"]
ALLOWED_UNITS = {
    "m",
    "rad",
    "N",
    "N*m",
    "m/s",
    "rad/s",
    "s",
    "normalized",
    "unitless",
}


@dataclass(frozen=True)
class EvidenceValidationIssue:
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
class RealEvidenceValidationReport:
    package_identity: str
    valid: bool
    evidence_ready: bool
    calibration_ready: bool
    governance_ready: bool
    comparison_ready: bool
    claim_ready: bool
    issues: tuple[EvidenceValidationIssue, ...]
    resolved_artifacts: tuple[str, ...]
    unresolved_protected_artifacts: tuple[str, ...]
    real_ledger: FailureLedgerRecord | None
    variant_ledgers: dict[str, FailureLedgerRecord]

    def raise_for_errors(self) -> None:
        errors = [issue for issue in self.issues if issue.severity == "error"]
        if errors:
            details = "; ".join(
                f"{item.code} ({item.path}): {item.message}" for item in errors
            )
            raise RealEvidenceValidationError(details, report=self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "nyssa-real-evidence-validation-v1",
            "package_identity": self.package_identity,
            "valid": self.valid,
            "evidence_ready": self.evidence_ready,
            "calibration_ready": self.calibration_ready,
            "governance_ready": self.governance_ready,
            "comparison_ready": self.comparison_ready,
            "claim_ready": self.claim_ready,
            "issues": [item.to_dict() for item in self.issues],
            "resolved_artifacts": list(self.resolved_artifacts),
            "unresolved_protected_artifacts": list(self.unresolved_protected_artifacts),
            "real_ledger": self.real_ledger.to_dict()
            if self.real_ledger is not None
            else None,
            "variant_ledgers": {
                key: value.to_dict()
                for key, value in sorted(self.variant_ledgers.items())
            },
        }


class RealEvidenceValidationError(ValueError):
    def __init__(self, message: str, *, report: RealEvidenceValidationReport) -> None:
        self.report = report
        super().__init__(message)


class RealEvidenceValidator:
    def validate(
        self,
        package: RealEvidencePackage,
        *,
        require_artifacts: bool = True,
    ) -> RealEvidenceValidationReport:
        issues: list[EvidenceValidationIssue] = []
        resolved: list[str] = []
        unresolved_protected: list[str] = []
        artifact_payloads = self._validate_artifacts(
            package,
            issues,
            resolved,
            unresolved_protected,
            require_artifacts=require_artifacts,
        )
        if package.compute_content_sha256() != package.content_sha256:
            _error(
                issues,
                "package_hash_mismatch",
                "content_sha256 does not match canonical package content",
                "content_sha256",
            )
        self._validate_clocks_frames_streams(
            package,
            artifact_payloads,
            issues,
            require_payloads=require_artifacts,
        )
        self._validate_action_contract(
            package,
            artifact_payloads,
            issues,
            require_payloads=require_artifacts,
        )
        real_ledger = _failure_ledger(
            package.real_episode.failure_events,
            episode_id=package.real_episode.identity.episode_id,
            engine_name="real_robot",
            required_source="real_robot",
            issues=issues,
            path="real_episode.failure_events",
        )
        variant_ledgers = {
            variant.variant_id: _failure_ledger(
                variant.failure_events,
                episode_id=variant.variant_id,
                engine_name="reconstructed_simulation",
                required_source="reconstructed_simulation",
                issues=issues,
                path=f"reconstructed_variants.{variant.variant_id}.failure_events",
            )
            for variant in package.reconstructed_variants
        }
        variant_ledgers = {
            key: value for key, value in variant_ledgers.items() if value is not None
        }
        calibration_ready = self._validate_calibrations(package, issues)
        comparison_ready = self._validate_mapping(package, issues)
        governance_ready = self._validate_governance(package, issues)
        evidence_ready = (
            not any(issue.severity == "error" for issue in issues)
            and not unresolved_protected
            and real_ledger is not None
            and len(variant_ledgers) == len(package.reconstructed_variants)
        )
        valid = not any(issue.severity == "error" for issue in issues)
        claim_ready = (
            valid
            and evidence_ready
            and calibration_ready
            and comparison_ready
            and governance_ready
        )
        return RealEvidenceValidationReport(
            package_identity=package.identity,
            valid=valid,
            evidence_ready=evidence_ready,
            calibration_ready=calibration_ready,
            governance_ready=governance_ready,
            comparison_ready=comparison_ready,
            claim_ready=claim_ready,
            issues=tuple(issues),
            resolved_artifacts=tuple(sorted(resolved)),
            unresolved_protected_artifacts=tuple(sorted(unresolved_protected)),
            real_ledger=real_ledger,
            variant_ledgers=variant_ledgers,
        )

    def _validate_artifacts(
        self,
        package: RealEvidencePackage,
        issues: list[EvidenceValidationIssue],
        resolved: list[str],
        unresolved_protected: list[str],
        *,
        require_artifacts: bool,
    ) -> dict[str, Any]:
        root = package.package_root
        payloads: dict[str, Any] = {}
        for index, artifact in enumerate(package.artifacts):
            label = f"artifacts[{index}]"
            if not artifact.path or root is None:
                if artifact.required:
                    _unresolved(
                        artifact.artifact_id,
                        artifact.access,
                        label,
                        issues,
                        unresolved_protected,
                        require_artifacts=require_artifacts,
                    )
                continue
            candidate = (root / artifact.path).resolve()
            if not _within(candidate, root.resolve()):
                _error(issues, "unsafe_artifact_path", "path escapes package", label)
                continue
            if not candidate.is_file():
                if artifact.required:
                    _unresolved(
                        artifact.artifact_id,
                        artifact.access,
                        label,
                        issues,
                        unresolved_protected,
                        require_artifacts=require_artifacts,
                    )
                continue
            if sha256_file(candidate) != artifact.sha256:
                _error(issues, "artifact_hash_mismatch", "SHA-256 mismatch", label)
                continue
            resolved.append(artifact.artifact_id)
            if artifact.media_type == "application/json":
                try:
                    payloads[artifact.artifact_id] = json.loads(
                        candidate.read_text(encoding="utf-8")
                    )
                except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                    _error(issues, "artifact_json_invalid", str(exc), label)
        return payloads

    def _validate_clocks_frames_streams(
        self,
        package: RealEvidencePackage,
        payloads: dict[str, Any],
        issues: list[EvidenceValidationIssue],
        *,
        require_payloads: bool,
    ) -> None:
        episode = package.real_episode
        clocks = {clock.clock_id: clock for clock in episode.clocks}
        frames = {frame.frame_id: frame for frame in episode.frames}
        if not clocks:
            _error(
                issues,
                "clock_missing",
                "at least one clock is required",
                "real_episode.clocks",
            )
        if not frames:
            _error(
                issues,
                "frame_missing",
                "at least one coordinate frame is required",
                "real_episode.frames",
            )
        for frame in episode.frames:
            if frame.parent_frame_id and frame.parent_frame_id not in frames:
                _error(
                    issues,
                    "frame_parent_unresolved",
                    frame.parent_frame_id,
                    f"frames.{frame.frame_id}",
                )
        if _frame_cycle(frames):
            _error(
                issues,
                "frame_cycle",
                "coordinate frame graph contains a cycle",
                "real_episode.frames",
            )
        artifacts = {item.artifact_id for item in package.artifacts}
        for stream in episode.sensors:
            label = f"real_episode.sensors.{stream.stream_id}"
            if stream.clock_id not in clocks:
                _error(issues, "stream_clock_unresolved", stream.clock_id, label)
            if stream.frame_id and stream.frame_id not in frames:
                _error(issues, "stream_frame_unresolved", stream.frame_id, label)
            if stream.artifact_id not in artifacts:
                _error(issues, "stream_artifact_unresolved", stream.artifact_id, label)
            for unit in stream.units.values():
                if unit not in ALLOWED_UNITS:
                    _error(issues, "stream_unit_unsupported", unit, label)
            self._validate_series(
                payloads.get(stream.artifact_id),
                stream.timestamp_field,
                stream.value_field,
                stream.sample_count,
                clocks.get(stream.clock_id),
                label,
                issues,
                require_payload=require_payloads,
            )
            for marker in stream.missing_ranges:
                if marker.end_seconds > episode.outcome.duration_seconds:
                    _error(
                        issues, "missing_range_outside_episode", marker.reason, label
                    )
        stream_ids = {item.stream_id for item in episode.sensors} | {
            episode.actions.stream_id
        }
        for marker in episode.outcome.missing_data:
            if marker.stream_id not in stream_ids:
                _error(
                    issues,
                    "missing_marker_stream_unresolved",
                    marker.stream_id,
                    "real_episode.outcome.missing_data",
                )
            if marker.end_seconds < marker.start_seconds:
                _error(
                    issues,
                    "missing_marker_range_invalid",
                    marker.stream_id,
                    "real_episode.outcome.missing_data",
                )
        for event in episode.outcome.safety_events:
            unknown = sorted(set(event.evidence_artifact_ids) - artifacts)
            if unknown:
                _error(
                    issues,
                    "safety_evidence_unresolved",
                    ", ".join(unknown),
                    f"real_episode.outcome.safety_events.{event.event_id}",
                )

    def _validate_series(
        self,
        payload: Any,
        timestamp_field: str,
        value_field: str,
        sample_count: int,
        clock: Any,
        label: str,
        issues: list[EvidenceValidationIssue],
        *,
        require_payload: bool,
        dimension: int | None = None,
        lower: tuple[float, ...] | None = None,
        upper: tuple[float, ...] | None = None,
    ) -> None:
        if not isinstance(payload, dict):
            if require_payload:
                _error(
                    issues,
                    "stream_payload_missing",
                    "JSON stream artifact is unavailable",
                    label,
                )
            else:
                _warning(
                    issues,
                    "stream_payload_unavailable_metadata_only",
                    "stream payload was not inspected",
                    label,
                )
            return
        timestamps = payload.get(timestamp_field)
        values = payload.get(value_field)
        if not isinstance(timestamps, list) or not isinstance(values, list):
            _error(
                issues,
                "stream_fields_missing",
                "timestamp/value arrays are required",
                label,
            )
            return
        if len(timestamps) != sample_count or len(values) != sample_count:
            _error(
                issues,
                "stream_sample_count_mismatch",
                "declared and observed counts differ",
                label,
            )
            return
        scale = _clock_scale(getattr(clock, "timestamp_unit", "s"))
        seconds = []
        for value in timestamps:
            try:
                seconds.append(float(value) * scale)
            except (TypeError, ValueError):
                _error(issues, "timestamp_invalid", str(value), label)
                return
        if any(not _finite(item) for item in seconds) or any(
            right <= left for left, right in zip(seconds, seconds[1:])
        ):
            _error(
                issues,
                "timestamps_not_strictly_monotonic",
                "timestamps must increase",
                label,
            )
        if dimension is not None:
            for index, action in enumerate(values):
                if not isinstance(action, list) or len(action) != dimension:
                    _error(issues, "action_dimension_mismatch", str(index), label)
                    continue
                for axis, raw in enumerate(action):
                    try:
                        value = float(raw)
                    except (TypeError, ValueError):
                        _error(issues, "action_value_invalid", str(index), label)
                        continue
                    if not _finite(value) or value < lower[axis] or value > upper[axis]:  # type: ignore[index]
                        _error(
                            issues,
                            "action_out_of_bounds",
                            f"sample {index} axis {axis}",
                            label,
                        )

    def _validate_action_contract(
        self,
        package: RealEvidencePackage,
        payloads: dict[str, Any],
        issues: list[EvidenceValidationIssue],
        *,
        require_payloads: bool,
    ) -> None:
        action = package.real_episode.actions
        clocks = {item.clock_id: item for item in package.real_episode.clocks}
        frames = {item.frame_id for item in package.real_episode.frames}
        artifacts = {item.artifact_id for item in package.artifacts}
        label = "real_episode.actions"
        if action.clock_id not in clocks:
            _error(issues, "action_clock_unresolved", action.clock_id, label)
        if action.frame_id and action.frame_id not in frames:
            _error(issues, "action_frame_unresolved", action.frame_id, label)
        if action.artifact_id not in artifacts:
            _error(issues, "action_artifact_unresolved", action.artifact_id, label)
        for unit in action.units:
            if unit not in ALLOWED_UNITS:
                _error(issues, "action_unit_unsupported", unit, label)
        self._validate_series(
            payloads.get(action.artifact_id),
            action.timestamp_field,
            action.value_field,
            action.sample_count,
            clocks.get(action.clock_id),
            label,
            issues,
            require_payload=require_payloads,
            dimension=action.dimension,
            lower=action.lower_bounds,
            upper=action.upper_bounds,
        )

    def _validate_calibrations(
        self, package: RealEvidencePackage, issues: list[EvidenceValidationIssue]
    ) -> bool:
        calibrations = {item.calibration_id: item for item in package.calibrations}
        types = {
            item.calibration_type
            for item in package.calibrations
            if item.status != "missing"
        }
        required = {"clock", "latency", "geometry", "dynamics"}
        if any(
            item.modality in {"rgb", "depth", "point_cloud"}
            for item in package.real_episode.sensors
        ):
            required.add("camera")
        for calibration_type in sorted(required - types):
            _warning(issues, "calibration_missing", calibration_type, "calibrations")
        action_calibration = calibrations.get(
            package.real_episode.actions.latency_calibration_id
        )
        if (
            action_calibration is None
            or action_calibration.calibration_type != "latency"
        ):
            _error(
                issues,
                "action_latency_calibration_unresolved",
                package.real_episode.actions.latency_calibration_id,
                "real_episode.actions",
            )
        for clock in package.real_episode.clocks:
            calibration = calibrations.get(clock.calibration_id)
            if calibration is None or calibration.calibration_type != "clock":
                _error(
                    issues,
                    "clock_calibration_unresolved",
                    clock.calibration_id,
                    f"real_episode.clocks.{clock.clock_id}",
                )
        artifact_ids = {item.artifact_id for item in package.artifacts}
        for calibration in package.calibrations:
            label = f"calibrations.{calibration.calibration_id}"
            if calibration.status == "missing":
                _warning(
                    issues,
                    "calibration_declared_missing",
                    calibration.calibration_type,
                    label,
                )
                continue
            if calibration.artifact_id and calibration.artifact_id not in artifact_ids:
                _error(
                    issues,
                    "calibration_artifact_unresolved",
                    calibration.artifact_id,
                    label,
                )
            missing_units = sorted(set(calibration.estimate) - set(calibration.units))
            if missing_units:
                _warning(
                    issues,
                    "calibration_units_missing",
                    ", ".join(missing_units),
                    label,
                )
            if not calibration.uncertainty:
                _warning(
                    issues,
                    "calibration_uncertainty_missing",
                    calibration.calibration_type,
                    label,
                )
            if not calibration.fit_quality or any(
                not _finite(value) for value in calibration.fit_quality.values()
            ):
                _warning(
                    issues,
                    "calibration_fit_quality_missing",
                    calibration.calibration_type,
                    label,
                )
        return not any(
            (
                issue.code.startswith("calibration_")
                or issue.code
                in {
                    "clock_calibration_unresolved",
                    "action_latency_calibration_unresolved",
                }
            )
            and issue.severity in {"error", "warning"}
            for issue in issues
        )

    def _validate_mapping(
        self, package: RealEvidencePackage, issues: list[EvidenceValidationIssue]
    ) -> bool:
        mapping = package.mapping
        if mapping.real_episode_id != package.real_episode.identity.episode_id:
            _error(
                issues,
                "mapping_real_episode_mismatch",
                mapping.real_episode_id,
                "mapping.real_episode_id",
            )
        variant_ids = {item.variant_id for item in package.reconstructed_variants}
        if set(mapping.variant_ids) != variant_ids:
            _error(
                issues,
                "mapping_variant_family_incomplete",
                "mapping must name every variant exactly once",
                "mapping.variant_ids",
            )
        calibration_ids = {item.calibration_id for item in package.calibrations}
        for variant in package.reconstructed_variants:
            for parameter in variant.estimated_parameters:
                if parameter.calibration_id not in calibration_ids:
                    _error(
                        issues,
                        "variant_calibration_unresolved",
                        parameter.calibration_id,
                        f"variants.{variant.variant_id}",
                    )
            if not variant.fit_quality or any(
                not _finite(value) for value in variant.fit_quality.values()
            ):
                _warning(
                    issues,
                    "variant_fit_quality_missing",
                    variant.variant_id,
                    f"variants.{variant.variant_id}",
                )
        return not any(
            issue.code.startswith("mapping_") or issue.code.startswith("variant_")
            for issue in issues
        )

    def _validate_governance(
        self, package: RealEvidencePackage, issues: list[EvidenceValidationIssue]
    ) -> bool:
        governance = package.governance
        if not governance.operator_ids_pseudonymous:
            _warning(
                issues,
                "operator_identity_not_pseudonymous",
                "operator identity requires redaction",
                "governance",
            )
        if governance.privacy_classification != "public" and not governance.redactions:
            _warning(
                issues,
                "privacy_redactions_missing",
                "restricted evidence requires redactions",
                "governance",
            )
        if governance.redistribution != "allowed" and not any(
            item.access in {"protected", "external"} for item in package.artifacts
        ):
            _warning(
                issues,
                "artifact_access_policy_inconsistent",
                "restricted redistribution has no protected artifacts",
                "governance",
            )
        return not any(
            issue.code
            in {
                "operator_identity_not_pseudonymous",
                "privacy_redactions_missing",
                "artifact_access_policy_inconsistent",
            }
            for issue in issues
        )


def comparison_pairs(package: RealEvidencePackage) -> list[dict[str, Any]]:
    return [
        {
            "mapping_id": package.mapping.mapping_id,
            "real_episode_id": package.real_episode.identity.episode_id,
            "variant_id": variant.variant_id,
            "scenario_identity": variant.scenario_identity,
            "controlled_axes": list(package.mapping.controlled_axes),
            "matching_keys": list(package.mapping.matching_keys),
            "fit_quality": variant.fit_quality,
            "mismatches": [item.model_dump(mode="json") for item in variant.mismatches],
        }
        for variant in package.reconstructed_variants
    ]


def _failure_ledger(
    payloads: tuple[dict[str, Any], ...],
    *,
    episode_id: str,
    engine_name: str,
    required_source: str,
    issues: list[EvidenceValidationIssue],
    path: str,
) -> FailureLedgerRecord | None:
    events = []
    for index, payload in enumerate(payloads):
        try:
            event = FailureEvent.from_dict(payload)
        except (TypeError, ValueError) as exc:
            _error(issues, "failure_event_invalid", str(exc), f"{path}[{index}]")
            continue
        if event.provenance.source != required_source:
            _error(
                issues,
                "failure_event_provenance_mismatch",
                event.provenance.source,
                f"{path}[{index}]",
            )
        events.append(event)
    if len(events) != len(payloads):
        return None
    return FailureLedgerRecord(
        task_id=episode_id,
        episode_index=0,
        episode_seed=0,
        engine_name=engine_name,
        events=tuple(events),
    )


def _unresolved(
    artifact_id: str,
    access: str,
    path: str,
    issues: list[EvidenceValidationIssue],
    unresolved: list[str],
    *,
    require_artifacts: bool,
) -> None:
    protected = access in {"protected", "external"}
    if protected:
        unresolved.append(artifact_id)
    severity: IssueSeverity = (
        "warning" if protected and not require_artifacts else "error"
    )
    issues.append(
        EvidenceValidationIssue("artifact_unresolved", artifact_id, path, severity)
    )


def _error(
    issues: list[EvidenceValidationIssue], code: str, message: str, path: str
) -> None:
    issues.append(EvidenceValidationIssue(code, message, path, "error"))


def _warning(
    issues: list[EvidenceValidationIssue], code: str, message: str, path: str
) -> None:
    issues.append(EvidenceValidationIssue(code, message, path, "warning"))


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _frame_cycle(frames: dict[str, Any]) -> bool:
    for frame_id in frames:
        seen = set()
        current: str | None = frame_id
        while current is not None:
            if current in seen:
                return True
            seen.add(current)
            frame = frames.get(current)
            current = frame.parent_frame_id if frame is not None else None
    return False


def _clock_scale(unit: str) -> float:
    return {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}.get(unit, 1.0)


def _finite(value: float) -> bool:
    return value == value and value not in {float("inf"), float("-inf")}
