import json
from pathlib import Path
import httpx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, PlainTextResponse, JSONResponse

app = FastAPI()

METRICS_URL = "http://localhost:11228/metrics"
TIMING_FILE = Path("epoch_timing.json")


@app.get("/")
async def index():
    return HTMLResponse(open("dashboard.html").read())


@app.get("/metrics")
async def metrics():
    async with httpx.AsyncClient() as client:
        r = await client.get(METRICS_URL)
        return PlainTextResponse(r.text)


@app.get("/timing")
async def timing():
    if TIMING_FILE.exists():
        return JSONResponse(json.loads(TIMING_FILE.read_text()))
    return JSONResponse([])


@app.get("/plots")
async def plots():
    return HTMLResponse(PLOTS_HTML)


PLOTS_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>epoch loop plots</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: monospace; background: #111; color: #ccc; padding: 24px; }
  h1 { font-size: 14px; color: #666; margin-bottom: 4px; }
  .status { font-size: 12px; color: #444; margin-bottom: 20px; }
  .status.live { color: #4a4; }
  .charts { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }
  .chart-box { background: #1a1a1a; border: 1px solid #222; padding: 16px; border-radius: 4px; }
  .chart-box h2 { font-size: 11px; color: #666; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px; }
  canvas { width: 100% !important; height: 200px !important; }
  .summary { font-size: 12px; color: #888; margin-top: 16px; }
  .summary span { color: #ccc; }
  @media (max-width: 700px) { .charts { grid-template-columns: 1fr; } }
</style>
</head>
<body>

<h1>epoch loop plots</h1>
<div class="status" id="status">waiting for data...</div>

<div class="charts">
  <div class="chart-box">
    <h2>step time (s)</h2>
    <canvas id="timeChart"></canvas>
  </div>
  <div class="chart-box">
    <h2>time breakdown (s)</h2>
    <canvas id="breakdownChart"></canvas>
  </div>
  <div class="chart-box">
    <h2>memory (MB)</h2>
    <canvas id="memChart"></canvas>
  </div>
  <div class="chart-box">
    <h2>disk (MB)</h2>
    <canvas id="diskChart"></canvas>
  </div>
  <div class="chart-box">
    <h2>create vs delete (s)</h2>
    <canvas id="createDeleteChart"></canvas>
  </div>
</div>

<div class="summary" id="summary"></div>

<script>
// minimal canvas chart — no dependencies
function drawChart(canvasId, datasets, labels) {
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);
  const W = rect.width, H = rect.height;
  const pad = { t: 10, r: 10, b: 24, l: 44 };
  const pW = W - pad.l - pad.r, pH = H - pad.t - pad.b;

  // find global min/max
  let allVals = datasets.flatMap(d => d.values.filter(v => v !== null));
  if (!allVals.length) return;
  let minV = Math.min(...allVals), maxV = Math.max(...allVals);
  if (minV === maxV) { minV -= 1; maxV += 1; }
  const rangeV = maxV - minV;
  const n = datasets[0].values.length;

  ctx.clearRect(0, 0, W, H);

  // grid
  ctx.strokeStyle = '#222';
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= 4; i++) {
    const y = pad.t + pH - (i / 4) * pH;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
    ctx.fillStyle = '#444';
    ctx.font = '9px monospace';
    ctx.textAlign = 'right';
    ctx.fillText((minV + (i / 4) * rangeV).toFixed(2), pad.l - 4, y + 3);
  }

  // x labels
  ctx.fillStyle = '#444';
  ctx.textAlign = 'center';
  const xStep = Math.max(1, Math.floor(n / 6));
  for (let i = 0; i < n; i += xStep) {
    const x = pad.l + (i / (n - 1 || 1)) * pW;
    ctx.fillText(labels[i], x, H - 4);
  }

  // lines
  const colors = ['#4a4', '#48f', '#f84', '#c84f', '#84f'];
  datasets.forEach((ds, di) => {
    ctx.strokeStyle = colors[di % colors.length];
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    let started = false;
    ds.values.forEach((v, i) => {
      if (v === null) return;
      const x = pad.l + (i / (n - 1 || 1)) * pW;
      const y = pad.t + pH - ((v - minV) / rangeV) * pH;
      if (!started) { ctx.moveTo(x, y); started = true; }
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });

  // legend
  if (datasets.length > 1) {
    ctx.font = '9px monospace';
    datasets.forEach((ds, di) => {
      const lx = pad.l + 8 + di * 72;
      ctx.fillStyle = colors[di % colors.length];
      ctx.fillRect(lx, pad.t + 2, 8, 2);
      ctx.fillText(ds.label, lx + 12, pad.t + 6);
    });
  }
}

async function poll() {
  try {
    const res = await fetch('/timing');
    const data = await res.json();
    if (!data.length) {
      document.getElementById('status').textContent = 'waiting for epoch_loop data...';
      return;
    }

    const el = document.getElementById('status');
    el.className = 'status live';
    el.textContent = data.length + ' steps recorded — polling every 5s';

    const labels = data.map(d => String(d.step));

    drawChart('timeChart',
      [{ label: 'total', values: data.map(d => d.total) }], labels);

    drawChart('breakdownChart', [
      { label: 'create', values: data.map(d => d.create) },
      { label: 'work', values: data.map(d => d.work) },
      { label: 'delete', values: data.map(d => d.delete) },
    ], labels);

    drawChart('memChart',
      [{ label: 'mem', values: data.map(d => d.mem_mb) }], labels);

    drawChart('diskChart',
      [{ label: 'disk', values: data.map(d => d.disk_mb) }], labels);

    drawChart('createDeleteChart', [
      { label: 'create', values: data.map(d => d.create) },
      { label: 'delete', values: data.map(d => d.delete) },
    ], labels);

    // summary
    const totals = data.map(d => d.total);
    const avg = totals.reduce((a, b) => a + b) / totals.length;
    const first5 = totals.slice(0, 5).reduce((a, b) => a + b) / Math.min(5, totals.length);
    const last5 = totals.slice(-5).reduce((a, b) => a + b) / Math.min(5, totals.length);
    const drift = last5 / first5;
    const mems = data.map(d => d.mem_mb).filter(m => m !== null);
    const memStr = mems.length ? `mem: <span>${mems[0].toFixed(0)} → ${mems[mems.length-1].toFixed(0)} MB</span>` : '';

    document.getElementById('summary').innerHTML =
      `steps: <span>${data.length}</span> &middot; ` +
      `avg: <span>${avg.toFixed(3)}s</span> &middot; ` +
      `drift: <span>${drift.toFixed(2)}x</span> &middot; ` +
      memStr;

  } catch (e) {
    document.getElementById('status').textContent = 'error: ' + e.message;
  }
}

poll();
setInterval(poll, 5000);
window.addEventListener('resize', poll);
</script>
</body>
</html>"""

# uv run uvicorn dashboard:app --port 8090
