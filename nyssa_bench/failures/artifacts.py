from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


FAILURE_LEDGER_MANIFEST_FORMAT = "nyssa-failure-ledger-manifest-v1"


def summarize_failure_ledgers(episodes: list[Any]) -> dict[str, Any]:
    roles: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    provenance: Counter[str] = Counter()
    evidence_visibility: Counter[str] = Counter()
    event_count = 0
    episodes_with_events = 0
    causal_hypothesis_count = 0
    recovery_eligible_count = 0

    for episode in episodes:
        ledger = getattr(episode, "failure_ledger", None)
        if ledger is None:
            continue
        events = ledger.events if hasattr(ledger, "events") else []
        if events:
            episodes_with_events += 1
        for event in events:
            event_count += 1
            roles[event.role] += 1
            categories[event.category] += 1
            provenance[event.provenance.source] += 1
            causal_hypothesis_count += len(event.causal_hypotheses)
            recovery_eligible_count += event.recovery_eligibility == "eligible"
            for evidence in event.evidence:
                evidence_visibility[evidence.visibility] += 1

    return {
        "format": FAILURE_LEDGER_MANIFEST_FORMAT,
        "episodes": len(episodes),
        "episodes_with_events": episodes_with_events,
        "event_count": event_count,
        "role_counts": dict(sorted(roles.items())),
        "category_counts": dict(sorted(categories.items())),
        "provenance_counts": dict(sorted(provenance.items())),
        "evidence_visibility_counts": dict(sorted(evidence_visibility.items())),
        "causal_hypothesis_count": causal_hypothesis_count,
        "recovery_eligible_event_count": recovery_eligible_count,
    }


def write_failure_ledger_manifest(episodes: list[Any], out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    payload = {
        "format": FAILURE_LEDGER_MANIFEST_FORMAT,
        "summary": summarize_failure_ledgers(episodes),
        "episodes": [
            {
                "task_id": episode.task_id,
                "episode_index": episode.episode_index,
                "seed": episode.seed,
                "failure_label": episode.failure_label,
                "failure_ledger": episode.failure_ledger.to_dict()
                if getattr(episode, "failure_ledger", None) is not None
                else None,
            }
            for episode in episodes
        ],
    }
    path = out_dir / "failure_ledger.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path
