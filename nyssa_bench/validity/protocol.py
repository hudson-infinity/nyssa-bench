from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping


BENCHMARK_VALIDITY_SPEC_FORMAT = "nyssa-benchmark-validity-spec-v1"
BENCHMARK_VALIDITY_REPORT_FORMAT = "nyssa-benchmark-validity-report-v1"
BENCHMARK_AUDIT_FORMAT = "nyssa-benchmark-audit-v1"

AuditStatus = Literal["passed", "failed", "missing", "not_applicable"]
AuditSeverity = Literal["info", "warning", "error", "blocking"]
ClaimImpact = Literal["none", "warn", "downgrade", "block"]


@dataclass(frozen=True)
class AuditResult:
    audit_id: str
    category: str
    status: AuditStatus
    severity: AuditSeverity
    inputs: dict[str, Any]
    evidence: dict[str, Any]
    remediation: str
    claim_impact: ClaimImpact
    summary: str
    audit_version: int = 1

    def __post_init__(self) -> None:
        if not self.audit_id.strip() or not self.category.strip():
            raise ValueError("audit_id and category must be non-empty")
        if self.status not in {"passed", "failed", "missing", "not_applicable"}:
            raise ValueError(f"unsupported audit status: {self.status}")
        if self.severity not in {"info", "warning", "error", "blocking"}:
            raise ValueError(f"unsupported audit severity: {self.severity}")
        if self.claim_impact not in {"none", "warn", "downgrade", "block"}:
            raise ValueError(f"unsupported claim impact: {self.claim_impact}")
        if not self.remediation.strip() or not self.summary.strip():
            raise ValueError("audit remediation and summary must be non-empty")
        if self.audit_version != 1:
            raise ValueError(f"unsupported audit version: {self.audit_version}")
        _canonical_json(self.inputs)
        _canonical_json(self.evidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": BENCHMARK_AUDIT_FORMAT,
            "audit_version": self.audit_version,
            "audit_id": self.audit_id,
            "category": self.category,
            "status": self.status,
            "severity": self.severity,
            "inputs": self.inputs,
            "evidence": self.evidence,
            "remediation": self.remediation,
            "claim_impact": self.claim_impact,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AuditResult":
        if data.get("format") != BENCHMARK_AUDIT_FORMAT:
            raise ValueError(f"unsupported audit format: {data.get('format')}")
        required = {
            "audit_version",
            "audit_id",
            "category",
            "status",
            "severity",
            "inputs",
            "evidence",
            "remediation",
            "claim_impact",
            "summary",
        }
        _reject_unknown(data, required | {"format"}, "audit")
        missing = sorted(required - set(data))
        if missing:
            raise ValueError(f"missing audit fields: {', '.join(missing)}")
        inputs = data.get("inputs")
        evidence = data.get("evidence")
        if not isinstance(inputs, Mapping) or not isinstance(evidence, Mapping):
            raise ValueError("audit inputs and evidence must be mappings")
        return cls(
            audit_id=str(data["audit_id"]),
            category=str(data["category"]),
            status=str(data["status"]),  # type: ignore[arg-type]
            severity=str(data["severity"]),  # type: ignore[arg-type]
            inputs=dict(inputs),
            evidence=dict(evidence),
            remediation=str(data["remediation"]),
            claim_impact=str(data["claim_impact"]),  # type: ignore[arg-type]
            summary=str(data["summary"]),
            audit_version=int(data["audit_version"]),
        )


@dataclass(frozen=True)
class BenchmarkValiditySpec:
    benchmark_id: str
    benchmark_version: str
    required_audits: tuple[str, ...]
    audit_inputs: dict[str, dict[str, Any]]
    claim_tier: str = "public_simulation"
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.benchmark_id.strip() or not self.benchmark_version.strip():
            raise ValueError("benchmark identity must be non-empty")
        if not self.required_audits:
            raise ValueError("at least one required audit must be declared")
        if len(self.required_audits) != len(set(self.required_audits)):
            raise ValueError("required audits must be unique")
        if self.schema_version != 1:
            raise ValueError(f"unsupported validity spec version: {self.schema_version}")
        unknown = sorted(set(self.audit_inputs) - set(self.required_audits))
        if unknown:
            raise ValueError(
                "audit inputs contain undeclared audits: " + ", ".join(unknown)
            )
        _canonical_json(self.audit_inputs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": BENCHMARK_VALIDITY_SPEC_FORMAT,
            "schema_version": self.schema_version,
            "benchmark_id": self.benchmark_id,
            "benchmark_version": self.benchmark_version,
            "claim_tier": self.claim_tier,
            "required_audits": list(self.required_audits),
            "audit_inputs": self.audit_inputs,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BenchmarkValiditySpec":
        if data.get("format") != BENCHMARK_VALIDITY_SPEC_FORMAT:
            raise ValueError(f"unsupported validity spec format: {data.get('format')}")
        _reject_unknown(
            data,
            {
                "format",
                "schema_version",
                "benchmark_id",
                "benchmark_version",
                "claim_tier",
                "required_audits",
                "audit_inputs",
            },
            "validity spec",
        )
        required_audits = data.get("required_audits")
        audit_inputs = data.get("audit_inputs")
        if not isinstance(required_audits, list) or not all(
            isinstance(item, str) for item in required_audits
        ):
            raise ValueError("required_audits must be a list of strings")
        if not isinstance(audit_inputs, Mapping) or not all(
            isinstance(value, Mapping) for value in audit_inputs.values()
        ):
            raise ValueError("audit_inputs must map audit IDs to input mappings")
        return cls(
            benchmark_id=str(data.get("benchmark_id", "")),
            benchmark_version=str(data.get("benchmark_version", "")),
            claim_tier=str(data.get("claim_tier", "public_simulation")),
            required_audits=tuple(required_audits),
            audit_inputs={str(key): dict(value) for key, value in audit_inputs.items()},
            schema_version=int(data.get("schema_version", 1)),
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict()).encode()).hexdigest()


@dataclass(frozen=True)
class BenchmarkValidityReport:
    benchmark_id: str
    benchmark_version: str
    claim_tier: str
    spec_sha256: str
    audits: tuple[AuditResult, ...]
    schema_version: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"unsupported validity report version: {self.schema_version}")
        if not self.benchmark_id.strip() or not self.benchmark_version.strip() or not self.claim_tier.strip():
            raise ValueError("validity report benchmark identity and claim tier must be non-empty")
        if len(self.spec_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.spec_sha256
        ):
            raise ValueError("spec_sha256 must be a lowercase SHA-256 digest")
        ids = [audit.audit_id for audit in self.audits]
        if len(ids) != len(set(ids)):
            raise ValueError("benchmark audits must have unique IDs")
        _canonical_json(self.metadata)
        required = self.metadata.get("required_audits")
        if (
            not isinstance(required, list)
            or not all(isinstance(item, str) for item in required)
            or len(required) != len(set(required))
            or set(required) != set(ids)
        ):
            raise ValueError(
                "validity report required_audits must match its audit records"
            )

    @property
    def blocking_audits(self) -> tuple[str, ...]:
        return tuple(
            audit.audit_id
            for audit in self.audits
            if audit.claim_impact == "block" and audit.status != "passed"
        )

    @property
    def downgrade_audits(self) -> tuple[str, ...]:
        return tuple(
            audit.audit_id
            for audit in self.audits
            if audit.claim_impact == "downgrade" and audit.status != "passed"
        )

    @property
    def status(self) -> str:
        if self.blocking_audits:
            return "blocked"
        if self.downgrade_audits:
            return "downgraded"
        return "validated"

    @property
    def claim_ready(self) -> bool:
        return self.status == "validated"

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "format": BENCHMARK_VALIDITY_REPORT_FORMAT,
            "schema_version": self.schema_version,
            "benchmark_id": self.benchmark_id,
            "benchmark_version": self.benchmark_version,
            "claim_tier": self.claim_tier,
            "spec_sha256": self.spec_sha256,
            "status": self.status,
            "claim_ready": self.claim_ready,
            "blocking_audits": list(self.blocking_audits),
            "downgrade_audits": list(self.downgrade_audits),
            "audits": [audit.to_dict() for audit in self.audits],
            "metadata": self.metadata,
        }
        payload["report_sha256"] = hashlib.sha256(
            _canonical_json(payload).encode()
        ).hexdigest()
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BenchmarkValidityReport":
        if data.get("format") != BENCHMARK_VALIDITY_REPORT_FORMAT:
            raise ValueError(
                f"unsupported validity report format: {data.get('format')}"
            )
        _reject_unknown(
            data,
            {
                "format",
                "schema_version",
                "benchmark_id",
                "benchmark_version",
                "claim_tier",
                "spec_sha256",
                "status",
                "claim_ready",
                "blocking_audits",
                "downgrade_audits",
                "audits",
                "metadata",
                "report_sha256",
            },
            "validity report",
        )
        raw_audits = data.get("audits")
        metadata = data.get("metadata")
        if not isinstance(raw_audits, list) or not all(
            isinstance(item, Mapping) for item in raw_audits
        ):
            raise ValueError("validity report audits must be a list of mappings")
        if not isinstance(metadata, Mapping):
            raise ValueError("validity report metadata must be a mapping")
        report = cls(
            benchmark_id=str(data.get("benchmark_id", "")),
            benchmark_version=str(data.get("benchmark_version", "")),
            claim_tier=str(data.get("claim_tier", "")),
            spec_sha256=str(data.get("spec_sha256", "")),
            audits=tuple(AuditResult.from_dict(item) for item in raw_audits),
            schema_version=int(data.get("schema_version", 1)),
            metadata=dict(metadata),
        )
        expected = report.to_dict()
        if data.get("report_sha256") != expected["report_sha256"]:
            raise ValueError("validity report hash does not match its contents")
        for key in (
            "status",
            "claim_ready",
            "blocking_audits",
            "downgrade_audits",
        ):
            if data.get(key) != expected[key]:
                raise ValueError(f"validity report derived field mismatch: {key}")
        return report


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _reject_unknown(data: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"unknown {label} fields: {', '.join(unknown)}")
