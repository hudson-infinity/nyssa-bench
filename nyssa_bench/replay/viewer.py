from __future__ import annotations

from pathlib import Path


def write_replay_viewer(out_dir: str | Path) -> Path:
    path = Path(out_dir) / "replay.html"
    path.write_text(
        """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>NyssaBench Replay</title>
  <style>
    :root {
      --bg: #f4f7fa;
      --card: #f8fafc;
      --line: #d8dee4;
      --muted: #495867;
    }

    body {
      margin: 0;
      background: var(--bg);
      color: #17202a;
      font-family: Inter, Arial, sans-serif;
    }

    .page {
      max-width: 1200px;
      margin: 0 auto;
      padding: 24px 20px 36px;
    }

    h1 {
      margin-top: 0;
    }

    .toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin: 12px 0 18px;
      align-items: center;
    }

    .toolbar label {
      display: inline-flex;
      flex-direction: column;
      gap: 4px;
      font-size: 13px;
      color: var(--muted);
      min-width: 220px;
    }

    .toolbar input,
    .toolbar select {
      padding: 8px 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: white;
    }

    .stats {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 12px;
      margin-bottom: 14px;
    }

    .stat {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px 12px;
    }

    .stat .value {
      font-size: 26px;
      font-weight: 700;
      line-height: 1.2;
    }

    .stat .label {
      color: var(--muted);
      font-size: 13px;
    }

    table {
      border-collapse: collapse;
      width: 100%;
      background: white;
      border: 1px solid var(--line);
      border-radius: 12px;
      overflow: hidden;
      box-shadow: 0 1px 0 #dfe7f0;
      min-width: 840px;
    }

    th, td {
      border-bottom: 1px solid var(--line);
      padding: 10px;
      text-align: left;
      vertical-align: top;
    }

    th {
      background: #f1f5f9;
      position: sticky;
      top: 0;
      z-index: 1;
    }

    tr:hover td {
      background: #f8fbff;
    }

    .mono {
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    }

    .muted {
      color: var(--muted);
      font-size: 13px;
    }

    .status-success {
      color: #047857;
      font-weight: 700;
    }

    .status-failure {
      color: #b91c1c;
      font-weight: 700;
    }

    .status-skip {
      color: #7c3aed;
      font-weight: 700;
    }

    .timeline {
      max-width: 340px;
      white-space: pre-wrap;
      margin: 0;
    }

    video {
      max-width: 320px;
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #0b1020;
    }
  </style>
</head>
<body>
<div class="page">
  <h1>NyssaBench Replay</h1>
  <p>This dashboard renders <code>replay_manifest.json</code> and supports filtering by task, status, or text.</p>

  <div class="toolbar">
    <label>
      Task
      <input id="task-filter" type="text" placeholder="all tasks">
    </label>
    <label>
      Status
      <select id="status-filter">
        <option value="all">all</option>
        <option value="success">success</option>
        <option value="failure">failure</option>
        <option value="other">other</option>
      </select>
    </label>
    <label>
      Search
      <input id="search-filter" type="search" placeholder="failure label / episode / task">
    </label>
  </div>

  <div class="stats">
    <div class="stat"><div class="label">Total episodes</div><div class="value" id="stat-total">0</div></div>
    <div class="stat"><div class="label">Successes</div><div class="value" id="stat-success">0</div></div>
    <div class="stat"><div class="label">Failures</div><div class="value" id="stat-failure">0</div></div>
    <div class="stat"><div class="label">Missing replay</div><div class="value" id="stat-missing">0</div></div>
    <div class="stat"><div class="label">With failure events</div><div class="value" id="stat-events">0</div></div>
  </div>

  <div style="overflow-x:auto">
    <table>
      <thead>
        <tr>
          <th>Task</th>
          <th>Episode</th>
          <th>Seed</th>
          <th>Status</th>
          <th>Failure</th>
          <th>Failure timeline</th>
          <th>Replay</th>
        </tr>
      </thead>
      <tbody id="rows"></tbody>
    </table>
  </div>
</div>

<script>
fetch("replay_manifest.json")
  .then((response) => response.json())
  .then((manifest) => {
    const rows = document.getElementById("rows");
    const episodes = manifest.episodes || [];

    const taskFilter = document.getElementById("task-filter");
    const statusFilter = document.getElementById("status-filter");
    const searchFilter = document.getElementById("search-filter");

    const updateStats = (metrics) => {
      document.getElementById("stat-total").textContent = String(metrics.total);
      document.getElementById("stat-success").textContent = String(metrics.success);
      document.getElementById("stat-failure").textContent = String(metrics.failure);
      document.getElementById("stat-missing").textContent = String(metrics.missing);
      document.getElementById("stat-events").textContent = String(metrics.events);
    };

    const rowForEpisode = (episode) => {
      const tr = document.createElement("tr");
      const taskId = episode.task_id || "";
      const status = episode.success === true
        ? "success"
        : episode.success === false
          ? "failure"
          : "other";

      const cells = [taskId, String(episode.episode_index ?? ""), String(episode.seed ?? "")];
      for (const value of cells) {
        const td = document.createElement("td");
        td.textContent = value;
        td.className = "mono";
        tr.appendChild(td);
      }

      const statusTd = document.createElement("td");
      const statusSpan = document.createElement("span");
      statusSpan.textContent = status;
      statusSpan.className =
        status === "success" ? "status-success" : status === "failure" ? "status-failure" : "status-skip";
      statusTd.appendChild(statusSpan);
      tr.appendChild(statusTd);

      const failureTd = document.createElement("td");
      failureTd.textContent = episode.failure_label || "none";
      tr.appendChild(failureTd);

      const timelineCell = document.createElement("td");
      const timelineEvents = episode.failure_ledger?.events || [];
      if (timelineEvents.length) {
        const details = document.createElement("details");
        const summary = document.createElement("summary");
        summary.textContent = `${timelineEvents.length} events`;
        const timeline = document.createElement("pre");
        timeline.textContent = JSON.stringify(timelineEvents, null, 2);
        timeline.className = "timeline mono";
        details.append(summary, timeline);
        timelineCell.appendChild(details);
      } else {
        timelineCell.textContent = "No events";
        timelineCell.className = "muted";
      }
      tr.appendChild(timelineCell);

      const replayCell = document.createElement("td");
      const clipPath = episode.failure_clip_path || episode.replay_path || "";
      if (clipPath) {
        const video = document.createElement("video");
        video.controls = true;
        video.src = clipPath;
        replayCell.appendChild(video);
        if (episode.failure_clip_path) {
          const caption = document.createElement("div");
          caption.className = "muted";
          caption.textContent = "Using failure clip";
          replayCell.appendChild(caption);
        }
      } else {
        replayCell.textContent = "No video exported";
        replayCell.className = "muted";
      }
      tr.appendChild(replayCell);

      return tr;
    };

    const render = () => {
      rows.innerHTML = "";
      const selectedTask = taskFilter.value.trim().toLowerCase();
      const selectedStatus = statusFilter.value;
      const query = searchFilter.value.trim().toLowerCase();

      let total = 0;
      let success = 0;
      let failure = 0;
      let missing = 0;
      let events = 0;

      for (const episode of episodes) {
        const status = episode.success === true
          ? "success"
          : episode.success === false
            ? "failure"
            : "other";

        const searchable = [
          episode.task_id || "",
          episode.episode_index,
          episode.seed,
          episode.failure_label || "",
        ]
          .map((value) => String(value || "").toLowerCase())
          .join(" ");

        const hasEvents = (episode.failure_ledger?.events || []).length > 0;
        if (selectedTask && !String(episode.task_id || "").toLowerCase().includes(selectedTask)) {
          continue;
        }
        if (selectedStatus !== "all" && status !== selectedStatus) {
          continue;
        }
        if (query && !searchable.includes(query)) {
          continue;
        }

        total += 1;
        if (status === "success") {
          success += 1;
        } else if (status === "failure") {
          failure += 1;
        }
        if (!episode.replay_path) {
          missing += 1;
        }
        if (hasEvents) {
          events += 1;
        }
        rows.appendChild(rowForEpisode(episode));
      }

      updateStats({ total, success, failure, missing, events });
    };

    taskFilter.addEventListener("input", render);
    statusFilter.addEventListener("change", render);
    searchFilter.addEventListener("input", render);
    render();
  })
  .catch(() => {
    const rows = document.getElementById("rows");
    rows.innerHTML = "";
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 7;
    td.textContent = "Could not load replay_manifest.json";
    td.style.color = "#b91c1c";
    tr.appendChild(td);
    rows.appendChild(tr);
    document.getElementById("stat-total").textContent = "0";
    document.getElementById("stat-success").textContent = "0";
    document.getElementById("stat-failure").textContent = "0";
    document.getElementById("stat-missing").textContent = "0";
    document.getElementById("stat-events").textContent = "0";
  });
</script>
</body>
</html>
""",
        encoding="utf-8",
    )
    return path


def replay_viewer_placeholder(out_dir: str | Path) -> Path:
    """Compatibility wrapper retained for legacy imports."""

    return write_replay_viewer(out_dir)
