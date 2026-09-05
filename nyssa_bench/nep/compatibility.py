from __future__ import annotations

from dataclasses import dataclass

from nyssa_bench.nep.protocol import NEP_VERSION


@dataclass(frozen=True)
class CompatibilityResult:
    compatible: bool
    reader_version: str
    artifact_version: str
    reason: str


def check_nep_compatibility(
    artifact_version: str, reader_version: str = NEP_VERSION
) -> CompatibilityResult:
    artifact = _parts(artifact_version)
    reader = _parts(reader_version)
    if artifact[0] == 0 or reader[0] == 0:
        compatible = artifact[:2] == reader[:2] and artifact[2] <= reader[2]
        reason = (
            "NEP 0.x accepts patch-level additions within the same minor line"
            if compatible
            else "NEP 0.x minor versions are compatibility boundaries"
        )
    else:
        compatible = artifact[0] == reader[0] and artifact[1] <= reader[1]
        reason = (
            "artifact major matches and its minor is not newer than the reader"
            if compatible
            else "artifact requires an incompatible major or newer reader"
        )
    return CompatibilityResult(
        compatible=compatible,
        reader_version=reader_version,
        artifact_version=artifact_version,
        reason=reason,
    )


def _parts(value: str) -> tuple[int, int, int]:
    core = value.split("-", 1)[0].split("+", 1)[0]
    pieces = core.split(".")
    if len(pieces) != 3 or any(not piece.isdigit() for piece in pieces):
        raise ValueError(f"invalid semantic version: {value}")
    return tuple(int(piece) for piece in pieces)  # type: ignore[return-value]
