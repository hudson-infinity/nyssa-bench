from __future__ import annotations

from nyssa_bench.core.episode import EpisodeResult


def episode_timeline(episode: EpisodeResult) -> list[dict[str, object]]:
    events = (
        episode.failure_ledger.events if episode.failure_ledger is not None else ()
    )
    return [
        {
            "step": index,
            "reward": step.reward,
            "terminated": step.terminated,
            "truncated": step.truncated,
            "failure_label": step.info.get("failure_label"),
            "failure_events": [
                event.to_dict()
                for event in events
                if event.onset_step <= index <= _event_end_step(event)
            ],
        }
        for index, step in enumerate(episode.steps)
    ]


def _event_end_step(event: object) -> int:
    end_step = getattr(event, "end_step", None)
    return int(end_step) if end_step is not None else int(getattr(event, "onset_step"))
