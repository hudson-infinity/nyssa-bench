from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


@dataclass(frozen=True)
class ExperimentCell:
    policy: str
    seed: int
    run_dir: Path
    variant: str | None = None
    enable_verifier: bool = False
    enable_recovery: bool = False


class ExperimentRunner:
    """Expand and execute immutable policy/seed/variant run cells."""

    def __init__(self, runner_factory: Callable[[ExperimentCell], Any]) -> None:
        self.runner_factory = runner_factory

    def execute(self, suite: Any, cells: Iterable[ExperimentCell]) -> list[Path]:
        cells = tuple(cells)
        counts: dict[Path, int] = {}
        for cell in cells:
            counts[cell.run_dir] = counts.get(cell.run_dir, 0) + 1
        duplicates = sorted(path for path, count in counts.items() if count > 1)
        if duplicates:
            raise ValueError(
                "duplicate experiment run directory: "
                + ", ".join(path.as_posix() for path in duplicates)
            )
        run_dirs: list[Path] = []
        for cell in cells:
            runner = self.runner_factory(cell)
            runner.evaluate(suite)
            run_dirs.append(cell.run_dir)
        return run_dirs


def policy_seed_cells(
    *,
    policies: Sequence[str],
    seeds: Sequence[int],
    out_dir: str | Path,
    enable_verifier: bool,
    enable_recovery: bool,
) -> tuple[ExperimentCell, ...]:
    root = Path(out_dir)
    return tuple(
        ExperimentCell(
            policy=policy,
            seed=int(seed),
            run_dir=root / policy / f"seed_{seed}",
            enable_verifier=enable_verifier,
            enable_recovery=enable_recovery,
        )
        for policy in policies
        for seed in seeds
    )


def ablation_cells(
    *,
    policy: str,
    variants: Sequence[str],
    seeds: Sequence[int],
    out_dir: str | Path,
) -> tuple[ExperimentCell, ...]:
    root = Path(out_dir)
    cells = []
    for variant in variants:
        if variant not in {"base", "verifier", "recovery", "verifier_recovery"}:
            raise ValueError(f"unsupported ablation variant: {variant}")
        for seed in seeds:
            cells.append(
                ExperimentCell(
                    policy=policy,
                    seed=int(seed),
                    run_dir=root / variant / f"seed_{seed}",
                    variant=variant,
                    enable_verifier=variant in {"verifier", "verifier_recovery"},
                    enable_recovery=variant in {"recovery", "verifier_recovery"},
                )
            )
    return tuple(cells)
