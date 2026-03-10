"""Rollout viewer for doltgres branches.

Reads branch structure, table data, dolt_log, and dolt_diff per rollout.
Groups by example, shows collapsible cards like the eval viewer.

Usage:
    uv run uvicorn rollout_viewer:app --port 8091
"""

import html
import re
import psycopg
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()

CONN_STR = "host=127.0.0.1 user=postgres password=password dbname=getting_started"

BRANCH_RE = re.compile(r"^e(\d+)_r(\d+)$")


def fresh_conn(dbname="getting_started"):
    conn = psycopg.connect(f"host=127.0.0.1 user=postgres password=password dbname={dbname}")
    conn.autocommit = True
    return conn


def get_branches():
    conn = fresh_conn()
    rows = conn.execute("SELECT name, hash FROM dolt_branches ORDER BY name").fetchall()
    conn.close()
    return rows


def get_table_data(branch):
    try:
        conn = fresh_conn(f"getting_started/{branch}")
        cur = conn.execute("SELECT * FROM persons ORDER BY 1")
        cols = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
        conn.close()
        return cols, rows
    except Exception:
        return [], []


def get_log(branch, limit=5):
    try:
        conn = fresh_conn(f"getting_started/{branch}")
        rows = conn.execute(
            f"SELECT commit_hash, committer, message FROM dolt_log LIMIT {limit}"
        ).fetchall()
        conn.close()
        return rows
    except Exception:
        return []


def get_seed_hash():
    """Get the earliest commit on main (the seed)."""
    try:
        conn = fresh_conn()
        rows = conn.execute(
            "SELECT commit_hash FROM dolt_log ORDER BY date ASC LIMIT 1"
        ).fetchall()
        conn.close()
        return rows[0][0] if rows else None
    except Exception:
        return None


def esc(val):
    return html.escape(str(val)) if val is not None else "-"


def render_table(cols, rows):
    if not rows:
        return '<span class="empty">(empty)</span>'
    hdr = "".join(f"<th>{esc(c)}</th>" for c in cols)
    body = ""
    for row in rows:
        body += "<tr>" + "".join(f"<td>{esc(v)}</td>" for v in row) + "</tr>"
    return f'<table class="data-table"><thead><tr>{hdr}</tr></thead><tbody>{body}</tbody></table>'


def render_log(log_rows):
    if not log_rows:
        return '<span class="empty">(no log)</span>'
    parts = []
    for hash_, committer, message in log_rows:
        parts.append(
            f'<div class="log-entry">'
            f'<span class="log-hash">{esc(hash_[:8])}</span>'
            f'<span class="log-committer">{esc(committer)}</span>'
            f'<span class="log-msg">{esc(message)}</span>'
            f'</div>'
        )
    return "\n".join(parts)


def render_rollout_card(branch, hash_, example_id, rollout_id):
    cols, rows = get_table_data(branch)
    log = get_log(branch)

    committer = log[0][1] if log else "?"
    row_count = len(rows)

    return f"""<div class="rollout-card collapsed">
  <div class="rollout-header" onclick="this.parentElement.classList.toggle('collapsed')">
    <span class="rollout-idx">r{rollout_id}</span>
    <span class="badge">{row_count} rows</span>
    <span class="badge badge-user">{esc(committer)}</span>
    <span class="branch-hash">{esc(hash_[:8])}</span>
    <span class="collapse-icon"></span>
  </div>
  <div class="rollout-body">
    <div class="section-label">data</div>
    {render_table(cols, rows)}
    <div class="section-label">log</div>
    {render_log(log)}
  </div>
</div>"""


def render_example_group(example_id, rollouts):
    row_counts = []
    for branch, hash_ in rollouts:
        try:
            conn = fresh_conn(f"getting_started/{branch}")
            count = conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
            conn.close()
            row_counts.append(count)
        except Exception:
            row_counts.append(0)

    avg_rows = sum(row_counts) / len(row_counts) if row_counts else 0
    n = len(rollouts)

    cards = ""
    for i, (branch, hash_) in enumerate(rollouts):
        m = BRANCH_RE.match(branch)
        rid = int(m.group(2)) if m else i
        cards += render_rollout_card(branch, hash_, example_id, rid)

    return f"""<div class="example-group">
  <div class="example-header" onclick="this.parentElement.classList.toggle('collapsed')">
    <span class="example-title">Example {example_id}</span>
    <span class="badge">{n} rollout{"s" if n != 1 else ""}</span>
    <span class="badge">avg {avg_rows:.0f} rows</span>
    <span class="collapse-icon"></span>
  </div>
  <div class="example-body">{cards}</div>
</div>"""


def render_other_branches(branches):
    if not branches:
        return ""
    cards = ""
    for branch, hash_ in branches:
        cols, rows = get_table_data(branch)
        log = get_log(branch, limit=3)
        cards += f"""<div class="rollout-card collapsed">
  <div class="rollout-header" onclick="this.parentElement.classList.toggle('collapsed')">
    <span class="rollout-idx">{esc(branch)}</span>
    <span class="badge">{len(rows)} rows</span>
    <span class="branch-hash">{esc(hash_[:8])}</span>
    <span class="collapse-icon"></span>
  </div>
  <div class="rollout-body">
    <div class="section-label">data</div>
    {render_table(cols, rows)}
    <div class="section-label">log</div>
    {render_log(log)}
  </div>
</div>"""
    return f"""<div class="example-group">
  <div class="example-header" onclick="this.parentElement.classList.toggle('collapsed')">
    <span class="example-title">Other branches</span>
    <span class="badge">{len(branches)}</span>
    <span class="collapse-icon"></span>
  </div>
  <div class="example-body">{cards}</div>
</div>"""


def generate_html():
    try:
        branches = get_branches()
    except Exception as e:
        return f"""<!DOCTYPE html><html><body style="font-family:monospace;background:#111;color:#a44;padding:40px">
        <h1>cannot connect to doltgres</h1><pre>{esc(str(e))}</pre></body></html>"""

    # group branches
    examples: dict[int, list[tuple[str, str]]] = {}
    other: list[tuple[str, str]] = []
    for name, hash_ in branches:
        m = BRANCH_RE.match(name)
        if m:
            ex_id = int(m.group(1))
            examples.setdefault(ex_id, []).append((name, hash_))
        else:
            other.append((name, hash_))

    total_branches = len(branches)
    total_examples = len(examples)
    total_rollouts = sum(len(v) for v in examples.values())

    sections = ""
    for ex_id in sorted(examples):
        sections += render_example_group(ex_id, examples[ex_id])
    sections += render_other_branches(other)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Rollout Viewer</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500&display=swap');

  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif;
    font-size: 14.5px; line-height: 1.72;
    color: #2a2a2a; background: #fdfcfa;
    padding: 72px 1rem 120px;
    max-width: 1100px; margin: 0 auto;
  }}

  .memo-head {{
    border-top: 2.5px solid #1a1a1a;
    border-bottom: 0.5px solid #1a1a1a;
    padding: 20px 0 16px; margin-bottom: 24px;
  }}
  .memo-head h1 {{ font-size: 1.6rem; font-weight: normal; color: #333; }}
  .memo-sub {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; color: #999; letter-spacing: 0.8px; margin-top: 6px;
  }}

  .controls {{
    position: sticky; top: 0; z-index: 10;
    display: flex; gap: 12px; align-items: center;
    padding: 12px 0; margin: 0 0 20px;
    background: rgba(253, 252, 250, 0.95);
    backdrop-filter: blur(8px);
    border-bottom: 0.5px solid #e8e4de;
    font-family: 'JetBrains Mono', monospace; font-size: 9.5px;
  }}
  .controls button {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; padding: 5px 16px;
    letter-spacing: 0.5px; color: #888; background: #fff;
    border: 0.5px solid #ddd; border-radius: 2px;
    cursor: pointer; transition: all 0.15s;
  }}
  .controls button:hover {{ color: #1a1a1a; border-color: #aaa; }}
  .controls .count {{ margin-left: auto; color: #bbb; font-size: 9px; }}

  .badge {{
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 9.5px; font-weight: 400;
    padding: 1px 6px; border-radius: 1px;
    margin-right: 4px; white-space: nowrap;
    background: #f5f4f0; color: #555;
    border-bottom: 1px solid #ddd;
  }}
  .badge-user {{ color: #16653a; border-bottom-color: #bbf7d0; }}

  .example-group {{ margin-bottom: 2px; }}
  .example-header {{
    display: flex; align-items: center; gap: 10px;
    padding: 10px 0; cursor: pointer; user-select: none;
    border-bottom: 0.5px solid #eee;
  }}
  .example-header:hover .example-title {{ color: #000; }}
  .example-title {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; font-weight: 500; color: #333;
  }}
  .collapse-icon {{ margin-left: auto; color: #ddd; font-size: 8px; }}
  .collapse-icon::after {{ content: '\\25BC'; }}
  .collapsed > .example-header .collapse-icon,
  .collapsed > .rollout-header .collapse-icon {{ transform: rotate(-90deg); }}
  .collapsed > .example-body,
  .collapsed > .rollout-body {{ display: none; }}
  .example-body {{ padding-left: 16px; }}

  .rollout-card {{ margin: 0; }}
  .rollout-header {{
    display: flex; align-items: center; gap: 8px;
    padding: 5px 0; cursor: pointer; user-select: none;
  }}
  .rollout-idx {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; color: #888; min-width: 24px;
  }}
  .branch-hash {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; color: #ccc;
  }}
  .rollout-body {{
    padding: 16px 20px;
    border-left: 1.5px solid #e8e4de;
    margin-left: 6px; margin-bottom: 8px;
  }}

  .section-label {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 8.5px; color: #aaa; letter-spacing: 1px;
    text-transform: uppercase; margin: 12px 0 4px;
  }}
  .section-label:first-child {{ margin-top: 0; }}

  .data-table {{
    width: 100%; border-collapse: collapse;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; margin: 4px 0 12px;
  }}
  .data-table th {{
    font-size: 9px; font-weight: 400; color: #aaa;
    letter-spacing: 0.5px; text-transform: uppercase;
    text-align: left; padding: 4px 12px;
    border-bottom: 1px solid #1a1a1a;
  }}
  .data-table td {{
    padding: 4px 12px;
    border-bottom: 0.5px solid #eee;
  }}
  .data-table tbody tr:hover {{ background: #f5f4f0; }}

  .log-entry {{
    display: flex; gap: 8px; padding: 2px 0;
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
  }}
  .log-hash {{ color: #c084fc; min-width: 64px; }}
  .log-committer {{ color: #16653a; min-width: 100px; }}
  .log-msg {{ color: #555; }}

  .empty {{ color: #ccc; font-style: italic; font-size: 12px; }}
</style>
</head>
<body>

<div class="memo-head">
  <h1>Rollout Viewer</h1>
  <div class="memo-sub">
    {total_branches} branches &middot; {total_examples} examples &middot; {total_rollouts} rollouts
  </div>
</div>

<div class="controls">
  <button onclick="toggleAll(true)">Expand All</button>
  <button onclick="toggleAll(false)">Collapse All</button>
  <button onclick="location.reload()">Refresh</button>
  <span class="count">{total_branches} branches</span>
</div>

{sections}

<script>
function toggleAll(expand) {{
  document.querySelectorAll('.example-group, .rollout-card').forEach(c => {{
    c.classList.toggle('collapsed', !expand);
  }});
}}
toggleAll(false);
</script>
</body>
</html>"""


@app.get("/")
async def index():
    return HTMLResponse(generate_html())
