from __future__ import annotations

from pathlib import Path


def replay_viewer_placeholder(out_dir: str | Path) -> Path:
    path = Path(out_dir) / "replay.html"
    path.write_text(
        """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>NyssaBench Replay</title>
  <style>
    body { font-family: Inter, Arial, sans-serif; margin: 32px; color: #17202a; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border-bottom: 1px solid #d8dee4; padding: 8px; text-align: left; }
    video { max-width: 360px; width: 100%; }
  </style>
</head>
<body>
<h1>NyssaBench Replay</h1>
<p>This viewer reads <code>replay_manifest.json</code>. MP4 playback appears when the selected engine exports render frames.</p>
<table>
  <thead><tr><th>Task</th><th>Episode</th><th>Status</th><th>Failure</th><th>Failure timeline</th><th>Replay</th></tr></thead>
  <tbody id="rows"></tbody>
</table>
<script>
fetch("replay_manifest.json")
  .then((response) => response.json())
  .then((manifest) => {
    const rows = document.getElementById("rows");
    for (const episode of manifest.episodes) {
      const tr = document.createElement("tr");
      const status = episode.success ? "success" : "failure";
      for (const value of [episode.task_id, episode.episode_index, status, episode.failure_label || ""]) {
        const td = document.createElement("td");
        td.textContent = value;
        tr.appendChild(td);
      }
      const timelineCell = document.createElement("td");
      const events = episode.failure_ledger?.events || [];
      if (events.length) {
        const details = document.createElement("details");
        const summary = document.createElement("summary");
        summary.textContent = `${events.length} events`;
        const timeline = document.createElement("pre");
        timeline.textContent = JSON.stringify(events, null, 2);
        details.append(summary, timeline);
        timelineCell.appendChild(details);
      } else {
        timelineCell.textContent = "No events";
      }
      tr.appendChild(timelineCell);
      const replayCell = document.createElement("td");
      if (episode.replay_path) {
        const video = document.createElement("video");
        video.controls = true;
        video.src = episode.replay_path;
        replayCell.appendChild(video);
      } else {
        replayCell.textContent = "No video exported";
      }
      tr.appendChild(replayCell);
      rows.appendChild(tr);
    }
  });
</script>
</body>
</html>
""",
        encoding="utf-8",
    )
    return path
