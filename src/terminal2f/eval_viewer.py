"""HTML viewer for eval outputs. Reads metadata.json + results.jsonl and generates a browsable page.

Starts a local server so browser refresh re-reads the files live.

Usage:
    t2f eval view outputs/evals/email-to-cc-bcc--openai--gpt-4.1/e50c41ac
    t2f eval view outputs/evals/.../e50c41ac outputs/evals/.../d02a5a4d
"""

from __future__ import annotations

import html
import json
import threading
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
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


def _score_cls(score: float) -> str:
    if score >= 0.8:
        return "score-high"
    if score >= 0.5:
        return "score-mid"
    return "score-low"


def _pretty_gt(gt) -> str:
    """Pretty-print a ground truth value."""
    if isinstance(gt, str):
        try:
            parsed = json.loads(gt)
            return json.dumps(parsed, indent=2)
        except (json.JSONDecodeError, TypeError):
            return gt
    return json.dumps(gt, indent=2)


def _render_message(role: str, content: str) -> str:
    escaped = html.escape(content)
    if role == "system":
        cls = "msg-system"
        label = "SYS"
    elif role == "user":
        cls = "msg-user"
        label = "USER"
    else:
        cls = "msg-assistant"
        label = "ASST"
    return f'<div class="msg {cls}"><span class="msg-role">{label}</span><pre class="msg-content">{escaped}</pre></div>'


def _render_turn_pair(assistant_content: str, gt) -> str:
    """Render assistant response side-by-side with ground truth."""
    escaped_asst = html.escape(assistant_content)
    escaped_gt = html.escape(_pretty_gt(gt))
    return f"""<div class="turn-pair">
  <div class="turn-col"><span class="msg-role">ASST</span><pre class="msg-content">{escaped_asst}</pre></div>
  <div class="turn-col turn-gt"><span class="msg-role">GT</span><pre class="msg-content">{escaped_gt}</pre></div>
</div>"""


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
        if role == "assistant" and gt_idx < len(ground_truths):
            parts.append(_render_turn_pair(content, ground_truths[gt_idx]))
            gt_idx += 1
        else:
            parts.append(_render_message(role, content))

    return "\n".join(parts)


def _render_metrics_badges(metrics: dict) -> str:
    badges = []
    for key, val in metrics.items():
        if isinstance(val, (int, float)):
            cls = _score_cls(val) if 0 <= val <= 1 else ""
            badges.append(
                f'<span class="badge {cls}">'
                f"{html.escape(key)}: {_fmt(val)}</span>"
            )
    return " ".join(badges)


def _render_rollout_card(result: dict, rollout_idx: int) -> str:
    reward = result.get("reward", 0)
    metrics = result.get("metrics", {})
    error = result.get("error")
    stop = result.get("stop_condition", "")

    header_parts = [
        f'<span class="badge {_score_cls(reward)}">reward: {_fmt(reward)}</span>',
        _render_metrics_badges(metrics),
    ]
    if error:
        header_parts.append(
            f'<span class="badge score-low">ERROR: {html.escape(str(error)[:80])}</span>'
        )

    conversation = _build_conversation_html(result)
    stop_label = f' <span class="stop-label">{html.escape(stop)}</span>' if stop else ""

    return f"""<div class="rollout-card collapsed" data-reward="{reward}">
  <div class="rollout-header" onclick="this.parentElement.classList.toggle('collapsed')">
    <span class="rollout-idx">#{rollout_idx}</span>
    <div class="rollout-badges">{" ".join(header_parts)}{stop_label}</div>
    <span class="collapse-icon"></span>
  </div>
  <div class="rollout-body">{conversation}</div>
</div>"""


def _render_example_group(example_id, rollouts: list[tuple[int, dict]]) -> str:
    """Render a collapsible group for all rollouts of one example."""
    rewards = [r.get("reward", 0) for _, r in rollouts]
    avg_reward = sum(rewards) / len(rewards) if rewards else 0
    n = len(rollouts)

    cards = "\n".join(_render_rollout_card(r, idx) for idx, r in rollouts)

    return f"""<div class="example-group" data-example="{example_id}" data-reward="{avg_reward}">
  <div class="example-header" onclick="this.parentElement.classList.toggle('collapsed')">
    <span class="example-title">Example {example_id}</span>
    <span class="badge {_score_cls(avg_reward)}">avg: {_fmt(avg_reward)}</span>
    <span class="example-count">{n} rollout{"s" if n != 1 else ""}</span>
    <span class="collapse-icon"></span>
  </div>
  <div class="example-body">{cards}</div>
</div>"""


def _render_summary_row(eval_data: dict) -> str:
    m = eval_data["metadata"]
    metrics_cells = ""
    for key, val in m.get("avg_metrics", {}).items():
        cls = _score_cls(val) if isinstance(val, (int, float)) and 0 <= val <= 1 else ""
        metrics_cells += f'<td class="{cls}">{_fmt(val)}</td>'
    avg_reward = m.get("avg_reward", 0)
    return f"""<tr>
  <td class="mono">{html.escape(str(Path(eval_data['path']).name))}</td>
  <td>{html.escape(m.get('model', '?'))}</td>
  <td class="{_score_cls(avg_reward)}">{_fmt(avg_reward)}</td>
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

    sections = []
    for ed in eval_data_list:
        dir_name = Path(ed["path"]).name
        model = ed["metadata"].get("model", "?")
        sections.append(
            f'<h2 class="eval-section">{html.escape(model)} '
            f'<span class="eval-hash">{html.escape(dir_name)}</span></h2>'
        )

        # Group by example_id
        by_example: dict[int | str, list[tuple[int, dict]]] = {}
        for i, result in enumerate(ed["results"]):
            eid = result.get("example_id", i)
            by_example.setdefault(eid, []).append((i, result))

        for eid in sorted(by_example):
            sections.append(_render_example_group(eid, by_example[eid]))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Eval Viewer</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500&display=swap');

  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif;
    font-size: 14.5px; line-height: 1.72;
    color: #2a2a2a; background: #fdfcfa;
    padding: 72px 1rem 120px;
    max-width: 1100px; margin: 0 auto;
    -webkit-font-smoothing: antialiased;
  }}

  .memo-head {{
    border-top: 2.5px solid #1a1a1a;
    border-bottom: 0.5px solid #1a1a1a;
    padding: 20px 0 16px;
    margin-bottom: 24px;
  }}
  .memo-head h1 {{
    font-size: 1.6rem; font-weight: normal;
    color: #333; line-height: 1.25; margin: 0;
  }}
  .memo-sub {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; color: #999;
    letter-spacing: 0.8px; margin-top: 6px;
  }}
  h2.eval-section {{
    font-size: 1.3rem; font-weight: normal;
    color: #333; margin: 56px 0 10px;
  }}
  .eval-hash {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; color: #bbb; font-weight: 400;
    letter-spacing: 0.5px;
  }}

  /* Summary table */
  .summary-table {{
    width: 100%; border-collapse: collapse;
    margin: 0 0 20px; font-size: 13.5px;
  }}
  .summary-table th {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; font-weight: 400;
    color: #aaa; letter-spacing: 1px;
    text-transform: uppercase; text-align: left;
    padding: 6px 16px;
    border-bottom: 1px solid #1a1a1a;
  }}
  .summary-table td {{
    padding: 10px 16px;
    border-bottom: 0.5px solid #eee;
    vertical-align: top;
  }}
  .summary-table tbody tr:hover {{
    background: #f5f4f0;
  }}
  .mono {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
  }}

  /* Score colors */
  .score-high {{ color: #16653a; font-weight: 700; }}
  .score-mid {{ color: #8a6d0b; font-weight: 600; }}
  .score-low {{ color: #b91c1c; font-weight: 600; }}

  /* Badges */
  .badge {{
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 9.5px; font-weight: 400;
    padding: 1px 6px; border-radius: 1px;
    margin-right: 4px; white-space: nowrap;
    background: #f5f4f0; color: #555;
    border-bottom: 1px solid #ddd;
  }}
  .badge.score-high {{ color: #16653a; font-weight: 500; border-bottom-color: #bbf7d0; }}
  .badge.score-mid {{ color: #8a6d0b; font-weight: 500; border-bottom-color: #fef08a; }}
  .badge.score-low {{ color: #b91c1c; font-weight: 500; border-bottom-color: #fecaca; }}
  .stop-label {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; color: #bbb;
  }}

  /* Controls */
  .controls {{
    position: sticky; top: 0; z-index: 10;
    display: flex; gap: 12px; align-items: center;
    flex-wrap: wrap;
    padding: 12px 0; margin: 0 0 20px;
    background: rgba(253, 252, 250, 0.95);
    backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px);
    border-bottom: 0.5px solid #e8e4de;
    font-family: 'JetBrains Mono', monospace; font-size: 9.5px;
    letter-spacing: 0.3px;
  }}
  .controls label {{ color: #aaa; text-transform: uppercase; font-size: 9px; letter-spacing: 1px; }}
  .controls select, .controls input {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; padding: 3px 6px;
    border: 0.5px solid #ddd; border-radius: 2px;
    background: #fff; color: #333;
    outline: none;
  }}
  .controls select:focus, .controls input:focus {{
    border-color: #aaa;
  }}
  .controls button {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; padding: 5px 16px;
    letter-spacing: 0.5px;
    color: #888; background: #fff;
    border: 0.5px solid #ddd; border-radius: 2px;
    cursor: pointer; transition: all 0.15s;
  }}
  .controls button:hover {{ color: #1a1a1a; border-color: #aaa; }}
  .controls .sep {{
    width: 0.5px; height: 14px; background: #ddd;
  }}

  /* Example groups */
  .example-group {{ margin-bottom: 2px; }}
  .example-header {{
    display: flex; align-items: center; gap: 10px;
    padding: 8px 0; cursor: pointer; user-select: none;
    border-bottom: 0.5px solid #eee;
  }}
  .example-header:hover {{ background: none; }}
  .example-header:hover .example-title {{ color: #000; }}
  .example-title {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; font-weight: 500; color: #333;
    transition: color 0.1s;
  }}
  .example-count {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; color: #bbb; letter-spacing: 0.5px;
  }}
  .collapse-icon {{
    margin-left: auto; color: #ddd; font-size: 8px;
    transition: transform 0.15s ease;
  }}
  .collapse-icon::after {{ content: '\\25BC'; }}
  .collapsed > .example-header .collapse-icon,
  .collapsed > .rollout-header .collapse-icon {{ transform: rotate(-90deg); }}
  .collapsed > .example-body,
  .collapsed > .rollout-body {{ display: none; }}
  .example-body {{ padding-left: 16px; }}

  /* Rollout cards */
  .rollout-card {{ margin: 0; }}
  .rollout-header {{
    display: flex; align-items: center; gap: 8px;
    padding: 5px 0; cursor: pointer; user-select: none;
  }}
  .rollout-header:hover .rollout-idx {{ color: #666; }}
  .rollout-idx {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; color: #ccc; min-width: 20px;
    transition: color 0.1s;
  }}
  .rollout-badges {{ flex: 1; }}
  .rollout-body {{
    padding: 16px 20px;
    border-left: 1.5px solid #e8e4de;
    margin-left: 6px;
    margin-bottom: 8px;
  }}

  /* Messages */
  .msg {{ margin-bottom: 10px; }}
  .msg-role {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 8.5px; font-weight: 400;
    color: #aaa; letter-spacing: 1px;
    text-transform: uppercase;
    display: block; margin-bottom: 3px;
  }}
  .msg-content {{
    white-space: pre-wrap; word-break: break-word;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px; line-height: 1.5;
    color: #333; margin: 0;
    padding: 16px 20px;
    background: #f5f4f0;
    border: 0.5px solid #e8e4de;
  }}
  .msg-system .msg-content {{
    font-size: 10.5px; opacity: 0.5;
    background: none; border: none; padding: 0;
  }}
  .msg-assistant .msg-content {{
    background: #fafaf8;
    border-color: #eee;
  }}
  .msg-user .msg-content {{
    background: #f5f4f0;
  }}

  /* Turn pair: assistant vs ground truth side by side */
  .turn-pair {{
    display: grid; grid-template-columns: 1fr 1fr; gap: 12px;
    margin-bottom: 10px;
  }}
  .turn-col .msg-content {{
    background: #fafaf8; border-color: #e8e4de;
  }}
  .turn-gt .msg-content {{
    background: #f5f4f0; border-color: #e8e4de;
  }}
  .turn-gt .msg-role {{ color: #999; }}

  @media (max-width: 700px) {{
    .turn-pair {{ grid-template-columns: 1fr; }}
    body {{ padding: 32px 0.75rem 80px; }}
  }}
</style>
</head>
<body>
<div class="memo-head">
  <h1>Eval Viewer</h1>
</div>

<table class="summary-table">
<thead><tr>
  <th>Run</th><th>Model</th><th>Reward</th>
  {metric_headers}
  <th>Examples</th><th>Errors</th><th>Time</th>
</tr></thead>
<tbody>{summary_rows}</tbody>
</table>

<div class="controls">
  <label>Sort:</label>
  <select id="sortBy">
    <option value="example">Example ID</option>
    <option value="reward-asc">Reward (low-high)</option>
    <option value="reward-desc">Reward (high-low)</option>
  </select>
  <label>Min:</label>
  <input id="minReward" type="number" value="0" min="0" max="1" step="0.1" style="width:55px">
  <label>Max:</label>
  <input id="maxReward" type="number" value="1" min="0" max="1" step="0.1" style="width:55px">
  <button onclick="applyFilters()">Apply</button>
  <div class="sep"></div>
  <button onclick="toggleAll(true)">Expand All</button>
  <button onclick="toggleAll(false)">Collapse All</button>
  <span id="countLabel" style="color:#bbb; margin-left:auto;"></span>
</div>

<div id="rollouts">
{"".join(sections)}
</div>

<script>
function applyFilters() {{
  const sortBy = document.getElementById('sortBy').value;
  const minR = parseFloat(document.getElementById('minReward').value) || 0;
  const maxR = parseFloat(document.getElementById('maxReward').value) || 1;
  const groups = document.querySelectorAll('.example-group');
  let visible = 0;

  groups.forEach(g => {{
    const r = parseFloat(g.dataset.reward);
    const show = r >= minR && r <= maxR;
    g.style.display = show ? '' : 'none';
    if (show) visible++;
  }});

  document.getElementById('countLabel').textContent =
    visible + '/' + groups.length + ' examples';

  // Sort within each eval section
  document.querySelectorAll('h2.eval-section').forEach(sec => {{
    const container = sec.parentElement;
    const sectionGroups = [];
    let el = sec.nextElementSibling;
    while (el && el.tagName !== 'H2') {{
      if (el.classList.contains('example-group')) sectionGroups.push(el);
      el = el.nextElementSibling;
    }}
    sectionGroups.sort((a, b) => {{
      if (sortBy === 'reward-asc') return parseFloat(a.dataset.reward) - parseFloat(b.dataset.reward);
      if (sortBy === 'reward-desc') return parseFloat(b.dataset.reward) - parseFloat(a.dataset.reward);
      return parseInt(a.dataset.example) - parseInt(b.dataset.example);
    }});
    sectionGroups.forEach(g => container.insertBefore(g, el));
  }});
}}

function toggleAll(expand) {{
  document.querySelectorAll('.example-group, .rollout-card').forEach(c => {{
    c.classList.toggle('collapsed', !expand);
  }});
}}

// Auto-apply on sort change
document.getElementById('sortBy').addEventListener('change', applyFilters);

// Start collapsed, show count
toggleAll(false);
applyFilters();
</script>
</body>
</html>"""


def find_eval_runs(root: Path) -> list[Path]:
    """Find all eval run dirs (containing metadata.json) under root."""
    return sorted(d.parent for d in root.rglob("metadata.json") if (d.parent / "results.jsonl").exists())


def _load_dirs(eval_dirs: list[Path]) -> list[dict]:
    """Load eval data from directories, skipping invalid ones."""
    data = []
    for d in eval_dirs:
        if not (d / "metadata.json").exists() or not (d / "results.jsonl").exists():
            continue
        data.append(load_eval_dir(d))
    return data


def view(eval_dirs: list[Path] | None = None, port: int = 8279) -> None:
    """Start a local server that re-reads eval data on every refresh.

    If no dirs given, scans outputs/evals/ for all runs.
    """
    if not eval_dirs:
        eval_dirs = find_eval_runs(Path("outputs/evals"))
        if not eval_dirs:
            raise SystemExit("No eval runs found in outputs/evals/")
        print(f"Found {len(eval_dirs)} eval runs")

    # verify at least one valid dir
    if not _load_dirs(eval_dirs):
        raise SystemExit("No valid eval dirs found.")

    dirs = eval_dirs  # captured by handler

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            data = _load_dirs(dirs)
            if not data:
                self.send_error(404, "No valid eval dirs")
                return
            content = generate_html(data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, fmt, *args):
            pass  # quiet

    server = HTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}"
    print(f"Serving at {url}  (refresh to reload data, ctrl-c to stop)")
    threading.Timer(0.3, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
