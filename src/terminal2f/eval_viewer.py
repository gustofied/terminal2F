"""HTML viewer for eval outputs. Reads metadata.json + results.jsonl and generates a browsable page.

Usage:
    t2f eval view outputs/evals/email-to-cc-bcc--openai--gpt-4.1/e50c41ac
    t2f eval view outputs/evals/.../e50c41ac outputs/evals/.../d02a5a4d
"""

from __future__ import annotations

import html
import json
import tempfile
import webbrowser
from pathlib import Path


def load_eval_dir(eval_dir: Path) -> dict:
    metadata = json.loads((eval_dir / "metadata.json").read_text())
    results = []
    for line in (eval_dir / "results.jsonl").read_text().splitlines():
        if line.strip():
            results.append(json.loads(line))
    return {"path": str(eval_dir), "metadata": metadata, "results": results}


def _fmt(val) -> str:
    if val is None:
        return "-"
    if isinstance(val, float):
        return f"{val:.3f}"
    return html.escape(str(val))


def _color_for_score(score: float) -> str:
    if score >= 0.8:
        return "#22c55e"
    if score >= 0.5:
        return "#eab308"
    return "#ef4444"


def _render_message(role: str, content: str) -> str:
    escaped = html.escape(content)
    if role == "system":
        cls = "msg-system"
        label = "System"
    elif role == "user":
        cls = "msg-user"
        label = "User"
    else:
        cls = "msg-assistant"
        label = "Assistant"
    return f'<div class="msg {cls}"><div class="msg-role">{label}</div><pre class="msg-content">{escaped}</pre></div>'


def _render_ground_truth(gt) -> str:
    text = json.dumps(gt, indent=2) if not isinstance(gt, str) else gt
    escaped = html.escape(text)
    return f'<div class="ground-truth"><div class="msg-role">Ground Truth</div><pre class="msg-content">{escaped}</pre></div>'


def _build_conversation_html(result: dict) -> str:
    parts = []
    prompt_msgs = result.get("prompt", [])
    completion_msgs = result.get("completion", [])
    ground_truths = result.get("info", {}).get("ground_truths", [])

    all_msgs = prompt_msgs + completion_msgs
    gt_idx = 0

    for msg in all_msgs:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        parts.append(_render_message(role, content))
        if role == "assistant" and gt_idx < len(ground_truths):
            parts.append(_render_ground_truth(ground_truths[gt_idx]))
            gt_idx += 1

    return "\n".join(parts)


def _render_metrics_badges(metrics: dict) -> str:
    badges = []
    for key, val in metrics.items():
        if isinstance(val, (int, float)):
            color = _color_for_score(val) if 0 <= val <= 1 else "#6b7280"
            badges.append(
                f'<span class="badge" style="background:{color}">'
                f"{html.escape(key)}: {_fmt(val)}</span>"
            )
    return " ".join(badges)


def _render_rollout_card(result: dict, idx: int) -> str:
    example_id = result.get("example_id", "?")
    reward = result.get("reward", 0)
    metrics = result.get("metrics", {})
    error = result.get("error")
    reward_color = _color_for_score(reward)

    header_parts = [
        f'<span class="badge" style="background:{reward_color}">reward: {_fmt(reward)}</span>',
        _render_metrics_badges(metrics),
    ]
    if error:
        header_parts.append(
            f'<span class="badge" style="background:#ef4444">ERROR: {html.escape(str(error)[:100])}</span>'
        )

    conversation = _build_conversation_html(result)

    return f"""
    <div class="rollout-card" data-reward="{reward}" data-example="{example_id}">
      <div class="rollout-header" onclick="this.parentElement.classList.toggle('collapsed')">
        <span class="rollout-title">Example {example_id} &mdash; Rollout #{idx}</span>
        <div class="rollout-badges">{" ".join(header_parts)}</div>
        <span class="collapse-icon">&#9660;</span>
      </div>
      <div class="rollout-body">
        {conversation}
      </div>
    </div>"""


def _render_summary_row(eval_data: dict) -> str:
    m = eval_data["metadata"]
    metrics_cells = ""
    for key, val in m.get("avg_metrics", {}).items():
        color = _color_for_score(val) if isinstance(val, (int, float)) and 0 <= val <= 1 else ""
        style = f' style="color:{color};font-weight:600"' if color else ""
        metrics_cells += f"<td{style}>{_fmt(val)}</td>"
    return f"""<tr>
      <td class="mono">{html.escape(str(Path(eval_data['path']).name))}</td>
      <td>{html.escape(m.get('model', '?'))}</td>
      <td style="color:{_color_for_score(m.get('avg_reward', 0))};font-weight:700">{_fmt(m.get('avg_reward'))}</td>
      {metrics_cells}
      <td>{m.get('num_examples', '?')}</td>
      <td>{_fmt(m.get('avg_error', 0))}</td>
      <td>{_fmt(m.get('time_ms', 0) / 1000)}s</td>
    </tr>"""


def _render_summary_headers(eval_data_list: list[dict]) -> str:
    all_metric_keys: list[str] = []
    for ed in eval_data_list:
        for key in ed["metadata"].get("avg_metrics", {}):
            if key not in all_metric_keys:
                all_metric_keys.append(key)
    return "".join(f"<th>{html.escape(k)}</th>" for k in all_metric_keys)


def generate_html(eval_data_list: list[dict]) -> str:
    metric_headers = _render_summary_headers(eval_data_list)
    summary_rows = "\n".join(_render_summary_row(ed) for ed in eval_data_list)

    rollout_cards = []
    for ed in eval_data_list:
        dir_name = Path(ed["path"]).name
        model = ed["metadata"].get("model", "?")
        rollout_cards.append(
            f'<h2 class="eval-section-title">{html.escape(model)} '
            f'<span class="eval-hash">({html.escape(dir_name)})</span></h2>'
        )
        for i, result in enumerate(ed["results"]):
            rollout_cards.append(_render_rollout_card(result, i))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Eval Viewer</title>
<style>
  :root {{
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --text-dim: #8b949e; --accent: #58a6ff;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    background: var(--bg); color: var(--text); padding: 24px; line-height: 1.5; }}
  h1 {{ margin-bottom: 16px; font-size: 1.5rem; }}
  h2.eval-section-title {{ margin: 32px 0 12px; font-size: 1.2rem; color: var(--accent); }}
  .eval-hash {{ color: var(--text-dim); font-weight: 400; font-size: 0.85rem; }}

  /* Controls */
  .controls {{ display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin-bottom: 20px; }}
  .controls label {{ color: var(--text-dim); font-size: 0.85rem; }}
  .controls select, .controls input {{
    background: var(--surface); color: var(--text); border: 1px solid var(--border);
    border-radius: 6px; padding: 4px 8px; font-size: 0.85rem; }}
  .controls button {{
    background: var(--accent); color: #000; border: none; border-radius: 6px;
    padding: 5px 14px; font-size: 0.85rem; cursor: pointer; font-weight: 600; }}

  /* Summary table */
  .summary-table {{ width: 100%; border-collapse: collapse; margin-bottom: 24px; font-size: 0.85rem; }}
  .summary-table th {{ background: var(--surface); text-align: left; padding: 8px 12px;
    border-bottom: 2px solid var(--border); color: var(--text-dim); white-space: nowrap; }}
  .summary-table td {{ padding: 8px 12px; border-bottom: 1px solid var(--border); white-space: nowrap; }}
  .mono {{ font-family: 'SF Mono', SFMono-Regular, Consolas, monospace; font-size: 0.8rem; }}

  /* Badges */
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 10px;
    font-size: 0.75rem; font-weight: 600; color: #000; margin-right: 4px; white-space: nowrap; }}

  /* Rollout cards */
  .rollout-card {{ background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; margin-bottom: 8px; overflow: hidden; }}
  .rollout-header {{ display: flex; align-items: center; gap: 12px; padding: 10px 16px;
    cursor: pointer; user-select: none; flex-wrap: wrap; }}
  .rollout-header:hover {{ background: rgba(255,255,255,0.03); }}
  .rollout-title {{ font-weight: 600; font-size: 0.9rem; white-space: nowrap; }}
  .rollout-badges {{ flex: 1; }}
  .collapse-icon {{ color: var(--text-dim); transition: transform 0.15s; font-size: 0.7rem; }}
  .rollout-card.collapsed .rollout-body {{ display: none; }}
  .rollout-card.collapsed .collapse-icon {{ transform: rotate(-90deg); }}
  .rollout-body {{ padding: 12px 16px; border-top: 1px solid var(--border); }}

  /* Messages */
  .msg {{ margin-bottom: 10px; border-radius: 6px; padding: 8px 12px; }}
  .msg-role {{ font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
    margin-bottom: 4px; letter-spacing: 0.5px; }}
  .msg-content {{ white-space: pre-wrap; word-break: break-word; font-size: 0.82rem;
    font-family: 'SF Mono', SFMono-Regular, Consolas, monospace; line-height: 1.4; }}
  .msg-system {{ background: rgba(139,148,158,0.1); }}
  .msg-system .msg-role {{ color: #8b949e; }}
  .msg-user {{ background: rgba(88,166,255,0.08); }}
  .msg-user .msg-role {{ color: #58a6ff; }}
  .msg-assistant {{ background: rgba(63,185,80,0.08); }}
  .msg-assistant .msg-role {{ color: #3fb950; }}
  .ground-truth {{ background: rgba(210,153,34,0.08); margin-bottom: 10px;
    border-radius: 6px; padding: 8px 12px; border-left: 3px solid #d2992280; }}
  .ground-truth .msg-role {{ color: #d29922; }}
  .ground-truth .msg-content {{ font-size: 0.82rem;
    font-family: 'SF Mono', SFMono-Regular, Consolas, monospace; }}
</style>
</head>
<body>
<h1>Eval Viewer</h1>

<table class="summary-table">
  <thead>
    <tr>
      <th>Run</th><th>Model</th><th>Reward</th>
      {metric_headers}
      <th>Examples</th><th>Errors</th><th>Time</th>
    </tr>
  </thead>
  <tbody>{summary_rows}</tbody>
</table>

<div class="controls">
  <label>Sort:</label>
  <select id="sortBy">
    <option value="example">Example ID</option>
    <option value="reward-asc">Reward (low&rarr;high)</option>
    <option value="reward-desc">Reward (high&rarr;low)</option>
  </select>
  <label>Min reward:</label>
  <input id="minReward" type="number" value="0" min="0" max="1" step="0.1" style="width:70px">
  <label>Max reward:</label>
  <input id="maxReward" type="number" value="1" min="0" max="1" step="0.1" style="width:70px">
  <button onclick="applyFilters()">Apply</button>
  <button onclick="toggleAll(true)">Expand All</button>
  <button onclick="toggleAll(false)">Collapse All</button>
</div>

<div id="rollouts">
{"".join(rollout_cards)}
</div>

<script>
function applyFilters() {{
  const sortBy = document.getElementById('sortBy').value;
  const minR = parseFloat(document.getElementById('minReward').value) || 0;
  const maxR = parseFloat(document.getElementById('maxReward').value) || 1;
  const container = document.getElementById('rollouts');
  const cards = Array.from(container.querySelectorAll('.rollout-card'));
  const sections = Array.from(container.querySelectorAll('.eval-section-title'));

  cards.forEach(c => {{
    const r = parseFloat(c.dataset.reward);
    c.style.display = (r >= minR && r <= maxR) ? '' : 'none';
  }});

  sections.forEach(sec => {{
    const group = [];
    let el = sec.nextElementSibling;
    while (el && !el.classList.contains('eval-section-title')) {{
      if (el.classList.contains('rollout-card')) group.push(el);
      el = el.nextElementSibling;
    }}
    group.sort((a, b) => {{
      if (sortBy === 'reward-asc') return parseFloat(a.dataset.reward) - parseFloat(b.dataset.reward);
      if (sortBy === 'reward-desc') return parseFloat(b.dataset.reward) - parseFloat(a.dataset.reward);
      return parseInt(a.dataset.example) - parseInt(b.dataset.example);
    }});
    group.forEach(c => c.parentElement.insertBefore(c, el));
  }});
}}

function toggleAll(expand) {{
  document.querySelectorAll('.rollout-card').forEach(c => {{
    c.classList.toggle('collapsed', !expand);
  }});
}}

toggleAll(false);
</script>
</body>
</html>"""


def find_eval_runs(root: Path) -> list[Path]:
    """Find all eval run dirs (containing metadata.json) under root."""
    return sorted(d.parent for d in root.rglob("metadata.json") if (d.parent / "results.jsonl").exists())


def view(eval_dirs: list[Path] | None = None) -> None:
    """Load eval dirs, generate HTML, and open in browser.

    If no dirs given, scans outputs/evals/ for all runs.
    """
    if not eval_dirs:
        eval_dirs = find_eval_runs(Path("outputs/evals"))
        if not eval_dirs:
            raise SystemExit("No eval runs found in outputs/evals/")
        print(f"Found {len(eval_dirs)} eval runs")

    eval_data_list = []
    for d in eval_dirs:
        if not (d / "metadata.json").exists():
            print(f"Warning: {d} has no metadata.json, skipping")
            continue
        if not (d / "results.jsonl").exists():
            print(f"Warning: {d} has no results.jsonl, skipping")
            continue
        eval_data_list.append(load_eval_dir(d))

    if not eval_data_list:
        raise SystemExit("No valid eval dirs found.")


    html_content = generate_html(eval_data_list)
    out = Path(tempfile.gettempdir()) / "eval_view.html"
    out.write_text(html_content)
    print(f"Wrote {out}")
    webbrowser.open(f"file://{out}")
