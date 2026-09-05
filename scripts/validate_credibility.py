from __future__ import annotations

from pathlib import Path

from nyssa_bench.credibility import evaluate_credibility, load_credibility_spec


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "claims" / "phase1_credibility.json"


def main() -> int:
    try:
        report = evaluate_credibility(
            load_credibility_spec(SPEC), spec_root=ROOT, source_root=ROOT
        )
    except ValueError as exc:
        print(f"credibility validation failed: {exc}")
        return 1
    for gate_id, gate in report["gates"].items():
        print(f"Gate {gate_id}: {gate['status']}")
    print(f"highest_completed_gate: {report['highest_completed_gate']}")
    print(f"phase1_complete: {report['phase1_complete']}")
    return 0 if report["gates"]["A"]["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
