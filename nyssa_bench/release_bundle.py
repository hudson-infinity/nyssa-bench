from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Mapping
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from nyssa_bench.claims import load_claim_evidence, validate_claim_evidence
from nyssa_bench.credibility import evaluate_credibility, load_credibility_spec
from nyssa_bench.nep import NEP_VERSION
from nyssa_bench.version import __version__


CONTAINER_RECORD_FORMAT = "nyssa-container-release-record-v1"
COMPATIBILITY_FORMAT = "nyssa-release-compatibility-v1"
RELEASE_BUNDLE_FORMAT = "nyssa-release-bundle-v1"
Profile = Literal["core", "mujoco", "maniskill"]
PROFILES: dict[str, dict[str, Any]] = {
    "core": {
        "python": "3.11",
        "platforms": ["linux/amd64", "linux/arm64"],
        "extra": None,
        "runtime": "Linux OCI runtime",
        "simulator_versions": {},
    },
    "mujoco": {
        "python": "3.11",
        "platforms": ["linux/amd64", "linux/arm64"],
        "extra": "mujoco",
        "runtime": "Linux with OSMesa headless rendering",
        "simulator_versions": {"gymnasium": "1.3.0", "mujoco": "3.12.0"},
    },
    "maniskill": {
        "python": "3.10",
        "platforms": ["linux/amd64"],
        "extra": "maniskill",
        "runtime": "Linux amd64, NVIDIA Container Toolkit, CUDA 12.4, Vulkan 1.3",
        "simulator_versions": {"mani-skill": "3.0.1", "torch": "2.5.1"},
        "host": (
            "NVIDIA driver compatible with CUDA 12.4 and a Vulkan-capable physical "
            "device exposed to the container"
        ),
    },
}
RELEASE_PATHS = (
    "CHANGELOG.md",
    "claims/claim_evidence.json",
    "claims/phase1_credibility.json",
    "docs/api_stability.md",
    "docs/docker.md",
    "docs/installation.md",
    "docs/nyssa_evaluation_protocol.md",
    "docs/phase1_credibility_gate.md",
    "docker/Dockerfile",
    "docker/Dockerfile.maniskill",
    "docker/Dockerfile.mujoco",
    "pyproject.toml",
    "uv.lock",
)
RELEASE_TREES = (
    "schemas/nep",
    "conformance/nep",
    "configs/reference",
    "configs/policy_tracks",
    "configs/suites",
    "configs/stressors",
)


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ContainerReleaseRecord(ContractModel):
    format: Literal["nyssa-container-release-record-v1"] = (
        "nyssa-container-release-record-v1"
    )
    profile: Profile
    image: str
    digest: str
    tags: tuple[str, ...]
    platforms: tuple[str, ...]
    version: str
    commit_sha: str
    python_version: str
    build_date: datetime
    compatibility: dict[str, Any]

    @field_validator("version")
    @classmethod
    def version_value(cls, value: str) -> str:
        return _version(value)

    @field_validator("commit_sha")
    @classmethod
    def commit_value(cls, value: str) -> str:
        return _commit(value)

    @field_validator("digest")
    @classmethod
    def digest_value(cls, value: str) -> str:
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
            raise ValueError("container digest must be a sha256 OCI digest")
        return value

    @model_validator(mode="after")
    def consistent_record(self) -> "ContainerReleaseRecord":
        expected = PROFILES[self.profile]
        if list(self.platforms) != expected["platforms"]:
            raise ValueError("container platforms differ from the compatibility policy")
        if self.python_version != expected["python"]:
            raise ValueError("container Python version differs from its profile")
        if not re.fullmatch(r"ghcr\.io/[a-z0-9._/-]+", self.image):
            raise ValueError("container image must be a lowercase GHCR path")
        immutable = {
            f"{self.image}:{self.version}",
            f"{self.image}:{self.version}-{self.commit_sha[:12]}",
        }
        if not immutable <= set(self.tags):
            raise ValueError("container record lacks immutable version and commit tags")
        if len(self.tags) != len(set(self.tags)):
            raise ValueError("container tags must be unique")
        if self.build_date.tzinfo is None:
            raise ValueError("container build date must include a timezone")
        if self.compatibility != expected:
            raise ValueError("container compatibility fields differ from policy")
        return self

    @property
    def identity(self) -> str:
        return f"{self.image}@{self.digest}"


def write_container_record(
    *,
    profile: Profile,
    image: str,
    digest: str,
    tags: list[str],
    version: str,
    commit_sha: str,
    build_date: str,
    out_path: str | Path,
) -> Path:
    record = ContainerReleaseRecord(
        profile=profile,
        image=image,
        digest=digest,
        tags=tuple(tags),
        platforms=tuple(PROFILES[profile]["platforms"]),
        version=version,
        commit_sha=commit_sha,
        python_version=str(PROFILES[profile]["python"]),
        build_date=_datetime(build_date),
        compatibility=dict(PROFILES[profile]),
    )
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, record.model_dump(mode="json"))
    return path


def build_release_bundle(
    *,
    repo_root: str | Path,
    out_dir: str | Path,
    version: str,
    commit_sha: str,
    build_date: str,
    container_records: list[str | Path],
    distributions: list[str | Path],
) -> dict[str, Path]:
    root = Path(repo_root).resolve()
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    version = _version(version)
    commit_sha = _commit(commit_sha)
    built_at = _datetime(build_date)
    if version != __version__:
        raise ValueError("release bundle and package versions differ")
    records = tuple(_load_container_record(path) for path in container_records)
    if {record.profile for record in records} != set(PROFILES):
        raise ValueError(
            "release bundle requires one record for every container profile"
        )
    if len(records) != len(PROFILES):
        raise ValueError("container release records contain duplicate profiles")
    for record in records:
        if record.version != version or record.commit_sha != commit_sha:
            raise ValueError("container and release identities differ")

    claim_path = root / "claims" / "claim_evidence.json"
    claim_matrix = load_claim_evidence(claim_path)
    claim_report = validate_claim_evidence(claim_matrix, repo_root=root)
    credibility_spec = load_credibility_spec(
        root / "claims" / "phase1_credibility.json"
    )
    credibility = evaluate_credibility(
        credibility_spec,
        spec_root=root,
        source_root=root,
    )
    source_files = _release_files(root)
    source_hashes = {
        path.relative_to(root).as_posix(): _sha256_file(path) for path in source_files
    }
    distribution_paths = [Path(path).resolve() for path in distributions]
    wheels = [path for path in distribution_paths if path.suffix == ".whl"]
    sdists = [path for path in distribution_paths if path.name.endswith(".tar.gz")]
    if len(wheels) != 1 or len(sdists) != 1 or len(distribution_paths) != 2:
        raise ValueError(
            "release bundle requires exactly one wheel and one .tar.gz distribution"
        )
    missing_distributions = [
        str(path) for path in distribution_paths if not path.is_file()
    ]
    if missing_distributions:
        raise ValueError(
            "release distribution is missing: " + ", ".join(missing_distributions)
        )
    compatibility = {
        "format": COMPATIBILITY_FORMAT,
        "version": version,
        "commit_sha": commit_sha,
        "build_date": built_at.isoformat(),
        "python_requires": ">=3.10",
        "nep_version": NEP_VERSION,
        "python_distributions": [
            {"filename": path.name, "sha256": _sha256_file(path)}
            for path in sorted(distribution_paths)
        ],
        "configuration_compatibility": {
            "core": ["tabletop_manipulation_v0"],
            "mujoco": ["mujoco_control_v0"],
            "maniskill": [
                "maniskill_smoke_v0",
                "maniskill_manipulation_v0",
                "maniskill_planner_bc_v0",
            ],
            "stressor_configs": [
                path.relative_to(root).as_posix()
                for path in source_files
                if path.parent == root / "configs" / "stressors"
            ],
            "evidence_boundary": (
                "configuration presence is not simulator execution validation"
            ),
        },
        "containers": [
            {
                **record.model_dump(mode="json"),
                "immutable_identity": record.identity,
            }
            for record in sorted(records, key=lambda item: item.profile)
        ],
        "source_artifacts_sha256": source_hashes,
        "headline_result_packs": _headline_evidence(root, claim_matrix),
        "credibility": {
            "highest_completed_gate": credibility["highest_completed_gate"],
            "phase1_complete": credibility["phase1_complete"],
            "public_wording": credibility["public_wording"],
        },
    }
    notes = _release_notes(version, commit_sha, claim_report, credibility, records)
    generated = {
        "compatibility-manifest.json": _json_bytes(compatibility),
        "phase1-credibility.json": _json_bytes(credibility),
        "RELEASE_NOTES.md": notes.encode("utf-8"),
    }
    bundle = out / f"nyssa-bench-{version}-release.zip"
    _write_deterministic_zip(bundle, root, source_files, generated)
    compatibility_path = out / "compatibility-manifest.json"
    credibility_path = out / "phase1-credibility.json"
    notes_path = out / "RELEASE_NOTES.md"
    _write_json(compatibility_path, compatibility)
    _write_json(credibility_path, credibility)
    notes_path.write_text(notes, encoding="utf-8")

    checked = distribution_paths + [
        bundle,
        compatibility_path,
        credibility_path,
        notes_path,
    ]
    if len({path.name for path in checked}) != len(checked):
        raise ValueError("release attachment filenames must be unique")
    missing = [str(path) for path in checked if not path.is_file()]
    if missing:
        raise ValueError("release attachment is missing: " + ", ".join(missing))
    checksums = out / "SHA256SUMS"
    checksums.write_text(
        "".join(f"{_sha256_file(path)}  {path.name}\n" for path in sorted(checked)),
        encoding="ascii",
    )
    manifest = {
        "format": RELEASE_BUNDLE_FORMAT,
        "version": version,
        "commit_sha": commit_sha,
        "bundle": {"path": bundle.name, "sha256": _sha256_file(bundle)},
        "compatibility": {
            "path": compatibility_path.name,
            "sha256": _sha256_file(compatibility_path),
        },
        "credibility": {
            "path": credibility_path.name,
            "sha256": _sha256_file(credibility_path),
        },
        "checksums": checksums.name,
    }
    manifest_path = out / "release-bundle-manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "bundle": bundle,
        "compatibility": compatibility_path,
        "credibility": credibility_path,
        "notes": notes_path,
        "checksums": checksums,
        "manifest": manifest_path,
    }


def _release_files(root: Path) -> list[Path]:
    files = [root / path for path in RELEASE_PATHS]
    for tree in RELEASE_TREES:
        directory = root / tree
        files.extend(path for path in directory.rglob("*") if path.is_file())
    missing = [
        path.relative_to(root).as_posix() for path in files if not path.is_file()
    ]
    if missing:
        raise ValueError("release source artifact is missing: " + ", ".join(missing))
    return sorted(set(files))


def _headline_evidence(root: Path, matrix: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = []
    for raw in matrix.get("headline_result_packs", []):
        if not isinstance(raw, Mapping):
            raise ValueError("headline result record must be a mapping")
        record = {key: raw.get(key) for key in ("result_id", "path")}
        for field in ("run_validity_artifact", "benchmark_validity_artifact"):
            relative = raw.get(field)
            if not isinstance(relative, str):
                raise ValueError(f"headline result lacks {field}")
            path = (root / relative).resolve()
            try:
                path.relative_to(root)
            except ValueError as exc:
                raise ValueError("headline result path escapes repository") from exc
            record[field] = {"path": relative, "sha256": _sha256_file(path)}
        records.append(record)
    return records


def _write_deterministic_zip(
    destination: Path,
    root: Path,
    files: list[Path],
    generated: Mapping[str, bytes],
) -> None:
    entries = {path.relative_to(root).as_posix(): path.read_bytes() for path in files}
    entries.update(generated)
    with ZipFile(
        destination, "w", compression=ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name, content in sorted(entries.items()):
            safe = PurePosixPath(name)
            if safe.is_absolute() or ".." in safe.parts:
                raise ValueError(f"unsafe release archive path: {name}")
            info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, content)


def _release_notes(
    version: str,
    commit_sha: str,
    claim_report: Mapping[str, Any],
    credibility: Mapping[str, Any],
    records: tuple[ContainerReleaseRecord, ...],
) -> str:
    counts = claim_report["claim_status_counts"]
    images = "\n".join(
        f"- `{record.identity}` ({', '.join(record.platforms)})"
        for record in sorted(records, key=lambda item: item.profile)
    )
    return f"""# NyssaBench {version}

Commit: `{commit_sha}`

## Public Positioning

{credibility["public_wording"]["wording"]}

The Phase 1 credibility state is `{credibility["highest_completed_gate"]}`.
`phase1_complete` is `{str(credibility["phase1_complete"]).lower()}`. Container
publication does not promote missing benchmark or hardware evidence.

## Capability Status

- Implemented: {counts.get("implemented", 0)}
- Integration only: {counts.get("integration_only", 0)}
- Experimental: {counts.get("experimental", 0)}
- Planned: {counts.get("planned", 0)}

## Immutable Images

{images}

Use image digests above for scientific results. Semantic-version tags are
immutable release aliases; `latest` is only a convenience tag.

## Evidence

Only result packs listed in the attached compatibility manifest are approved
headline evidence. An empty list means this release makes no benchmark-result
claim.
"""


def _load_container_record(path: str | Path) -> ContainerReleaseRecord:
    data = _load_json(Path(path))
    return ContainerReleaseRecord.model_validate(data)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return dict(value)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(_json_bytes(value))


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _version(value: str) -> str:
    if not re.fullmatch(
        r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:rc[1-9]\d*)?", value
    ):
        raise ValueError("release version must be semantic or a release candidate")
    return value


def _commit(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("release commit must be a full lowercase Git SHA")
    return value


def _datetime(value: str) -> datetime:
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("release build date must be RFC 3339") from exc
    if result.tzinfo is None:
        raise ValueError("release build date must include a timezone")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build NyssaBench release metadata.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    record = subparsers.add_parser("record")
    record.add_argument("--profile", choices=sorted(PROFILES), required=True)
    record.add_argument("--image", required=True)
    record.add_argument("--digest", required=True)
    record.add_argument("--tag", action="append", required=True)
    record.add_argument("--version", required=True)
    record.add_argument("--commit", required=True)
    record.add_argument("--build-date", required=True)
    record.add_argument("--out", required=True)
    bundle = subparsers.add_parser("bundle")
    bundle.add_argument("--repo-root", default=".")
    bundle.add_argument("--out", required=True)
    bundle.add_argument("--version", required=True)
    bundle.add_argument("--commit", required=True)
    bundle.add_argument("--build-date", required=True)
    bundle.add_argument("--container-record", action="append", required=True)
    bundle.add_argument("--distribution", action="append", required=True)
    args = parser.parse_args(argv)
    if args.command == "record":
        path = write_container_record(
            profile=args.profile,
            image=args.image,
            digest=args.digest,
            tags=args.tag,
            version=args.version,
            commit_sha=args.commit,
            build_date=args.build_date,
            out_path=args.out,
        )
        print(f"container_record: {path}")
        return 0
    paths = build_release_bundle(
        repo_root=args.repo_root,
        out_dir=args.out,
        version=args.version,
        commit_sha=args.commit,
        build_date=args.build_date,
        container_records=args.container_record,
        distributions=args.distribution,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
