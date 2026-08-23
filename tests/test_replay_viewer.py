from __future__ import annotations

from pathlib import Path

from nyssa_bench.replay.viewer import replay_viewer_placeholder, write_replay_viewer


def test_replay_viewer_file_is_emitted(tmp_path: Path) -> None:
    out = tmp_path / "run"
    out.mkdir()

    path = write_replay_viewer(out)

    assert path.exists()
    html = path.read_text(encoding="utf-8")
    assert html.startswith("<!doctype html>")
    assert "NyssaBench Replay" in html
    assert "replay_manifest.json" in html
    assert "status-filter" in html
    assert "stat-total" in html
    assert 'createElement("video")' in html


def test_replay_viewer_backward_compatibility_alias_exists(tmp_path: Path) -> None:
    out = tmp_path / "run"
    out.mkdir()

    assert replay_viewer_placeholder(out) == out / "replay.html"
