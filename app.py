# -*- coding: utf-8 -*-
"""FastAPI server exposing the OpenEnv API endpoints.

Endpoints:
    POST /reset              - Reset environment for a task (returns session_id)
    POST /step               - Take an agent action (requires X-Session-ID header)
    GET  /state              - Get current environment state (requires X-Session-ID)
    GET  /tasks              - List available tasks with action schema
    POST /grader             - Get grader score for episode (requires X-Session-ID)
    POST /baseline           - Run rule-based baseline on all tasks (in-process)
    
    POST /sentinel/reset     - Reset SENTINEL oversight environment (returns session_id)
    POST /sentinel/step      - Execute SENTINEL decision (requires X-Session-ID header)
    GET  /sentinel/state     - Get current SENTINEL environment state (requires X-Session-ID)
    POST /sentinel/grade     - Get SENTINEL grader score (requires X-Session-ID)
    
    GET  /metrics            - Telemetry counters (JSON or Prometheus text)
    GET  /curriculum         - Curriculum learning progression (ordered task stages)
    GET  /prometheus/metrics - Live scenario service metrics (Prometheus text scrape)
    GET  /prometheus/query   - PromQL instant query (standard Prometheus JSON envelope)
    GET  /prometheus/query_range - PromQL range query (matrix, from TSDB ring buffer)
    GET  /render             - Human-readable incident dashboard (requires X-Session-ID)
    GET  /leaderboard        - Top scores per task from completed episodes
    GET  /health             - Standard OpenEnv liveness probe
    GET  /                   - Human landing page for the live demo
    GET  /try                - Human landing page for trying SENTINEL
    GET  /info               - Rich JSON service info with telemetry
    WS   /ws                 - WebSocket persistent session (no session header needed)
    GET  /web                - Interactive browser-based incident dashboard
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from src.tasks import get_all_tasks

from routers.deps import (
    _SESSION_REGISTRY,
    _TELEMETRY,
    WS_ACTIVE_CONNECTIONS,
    purge_expired_sessions,
    _log,
)
import routers.deps as _deps

from routers.irt import router as irt_router
from routers.sentinel import router as sentinel_router
from routers.observability import router as observability_router


# ---------------------------------------------------------------------------
# Structured JSON logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "msg": "%(message)s"}',
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start background TTL-cleanup task; cancel it on shutdown."""
    async def _cleanup_loop():
        while True:
            await asyncio.sleep(300)  # run every 5 minutes
            purge_expired_sessions()

    task = asyncio.create_task(_cleanup_loop())
    _log.info("IRT environment started - TTL cleanup every 300s")
    try:
        yield
    finally:
        task.cancel()


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Incident Response Triage - OpenEnv",
    description=(
        "An OpenEnv environment that simulates production incident response. "
        "Agents must triage alerts, investigate services, diagnose root causes, "
        "apply remediations, and communicate status updates."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
app.title = "SENTINEL Oversight Command - OpenEnv"
app.description = (
    "An OpenEnv environment for multi-agent AI oversight. SENTINEL supervises "
    "worker agents during production incident response and decides which "
    "proposed actions should execute."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Include routers
# ---------------------------------------------------------------------------

app.include_router(irt_router)
app.include_router(sentinel_router)
app.include_router(observability_router)


# ---------------------------------------------------------------------------
# Native OpenEnv adapter mount
# ---------------------------------------------------------------------------
# The custom endpoints above expose the full hackathon demo surface. This mount
# also gives latest OpenEnv clients the standard schema/reset/step/state/ws API
# backed by OpenEnv's Environment base class.
NATIVE_OPENENV_AVAILABLE = False
try:  # pragma: no cover - availability depends on the local OpenEnv install
    from openenv.core.env_server.http_server import create_app as create_openenv_app

    from server.openenv_native import (
        SentinelNativeAction,
        SentinelNativeEnvironment,
        SentinelNativeObservation,
    )

    app.mount(
        "/openenv",
        create_openenv_app(
            SentinelNativeEnvironment,
            SentinelNativeAction,
            SentinelNativeObservation,
            env_name="sentinel_oversight_command",
            max_concurrent_envs=32,
        ),
    )
    NATIVE_OPENENV_AVAILABLE = True
    _log.info("native OpenEnv adapter mounted at /openenv")
except Exception as exc:  # pragma: no cover
    _log.warning("native OpenEnv adapter unavailable: %s", exc)


# ---------------------------------------------------------------------------
# MCP Server — Model Context Protocol (step/state/done as MCP tools)
# ---------------------------------------------------------------------------
MCP_AVAILABLE = False
try:
    from server.mcp_server import mcp_router
    app.include_router(mcp_router, prefix="/mcp")
    MCP_AVAILABLE = True
    _log.info("MCP server mounted at /mcp (Streamable HTTP transport)")
except Exception as exc:  # pragma: no cover
    _log.warning("MCP server unavailable: %s", exc)


# ---------------------------------------------------------------------------
# A2A Protocol — Agent-to-Agent discovery and task handling
# ---------------------------------------------------------------------------
A2A_AVAILABLE = False
try:
    from server.a2a_server import a2a_router
    app.include_router(a2a_router)
    A2A_AVAILABLE = True
    _log.info("A2A agent card at /.well-known/agent.json, endpoint at /a2a")
except Exception as exc:  # pragma: no cover
    _log.warning("A2A protocol unavailable: %s", exc)


# ---------------------------------------------------------------------------
# Root-level endpoints (health checks)
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    """Standard OpenEnv health check."""
    worker_backend = os.environ.get("SENTINEL_WORKER_BACKEND", "rule")
    return {
        "status": "healthy",
        "native_openenv_available": NATIVE_OPENENV_AVAILABLE,
        "native_openenv_mount": "/openenv" if NATIVE_OPENENV_AVAILABLE else None,
        "mcp_available": MCP_AVAILABLE,
        "mcp_endpoint": "/mcp" if MCP_AVAILABLE else None,
        "a2a_available": A2A_AVAILABLE,
        "a2a_agent_card": "/.well-known/agent.json" if A2A_AVAILABLE else None,
        "sentinel_worker_backend": worker_backend,
        "llm_worker_configured": bool(os.environ.get("GROQ_API_KEY")),
    }


def _service_info():
    """Return environment info and live telemetry for JSON endpoints."""
    worker_backend = os.environ.get("SENTINEL_WORKER_BACKEND", "rule")
    return {
        "status": "ok",
        "environment": "sentinel-oversight-command",
        "version": "1.0.0",
        "tasks": [t.task_id for t in get_all_tasks()],
        "primary_theme": "multi-agent interactions",
        "native_openenv_available": NATIVE_OPENENV_AVAILABLE,
        "native_openenv_mount": "/openenv" if NATIVE_OPENENV_AVAILABLE else None,
        "mcp_available": MCP_AVAILABLE,
        "mcp_endpoint": "/mcp" if MCP_AVAILABLE else None,
        "a2a_available": A2A_AVAILABLE,
        "a2a_agent_card": "/.well-known/agent.json" if A2A_AVAILABLE else None,
        "protocols": {
            "http_rest": True,
            "openenv_native": NATIVE_OPENENV_AVAILABLE,
            "mcp": MCP_AVAILABLE,
            "a2a": A2A_AVAILABLE,
        },
        "sentinel_worker_backend": worker_backend,
        "llm_worker_configured": bool(os.environ.get("GROQ_API_KEY")),
        "active_sessions": len(_SESSION_REGISTRY),
        "ws_active_connections": _deps.WS_ACTIVE_CONNECTIONS,
        "telemetry": _TELEMETRY,
    }


_TRY_LANDING_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SENTINEL Oversight Command</title>
<style>
*{box-sizing:border-box}
:root{--bg:#090b0f;--panel:#121720;--panel2:#0f131a;--line:#273141;--text:#f4f7fb;--muted:#a9b4c2;--green:#54d18a;--red:#ff6b6b;--amber:#f6c85f;--cyan:#72d6ff}
body{margin:0;min-height:100vh;background:radial-gradient(circle at 20% 0%,#172233 0,#090b0f 36%,#07090c 100%);color:var(--text);font-family:Inter,Segoe UI,Arial,sans-serif}
a{color:inherit;text-decoration:none}
.wrap{max-width:1180px;margin:0 auto;padding:42px 22px 28px}
.hero{display:grid;grid-template-columns:1.15fr .85fr;gap:26px;align-items:center;margin-bottom:28px}
.eyebrow{display:inline-flex;border:1px solid #36516b;color:#bfe8ff;background:#0d1722;border-radius:999px;padding:6px 10px;font-size:12px;margin-bottom:16px}
h1{font-size:54px;line-height:.98;margin:0 0 16px;letter-spacing:0}
.lead{font-size:18px;line-height:1.55;color:var(--muted);margin:0 0 22px;max-width:760px}
.actions{display:flex;gap:12px;flex-wrap:wrap}
.btn{border:1px solid var(--line);background:#182131;color:var(--text);border-radius:8px;padding:12px 15px;font-weight:700}
.btn.primary{background:var(--green);border-color:var(--green);color:#07110c}
.btn:hover{filter:brightness(1.12)}
.console{background:linear-gradient(180deg,#121923,#0b0f15);border:1px solid var(--line);border-radius:8px;padding:16px;box-shadow:0 18px 60px rgba(0,0,0,.35)}
.console h2{font-size:13px;text-transform:uppercase;color:var(--muted);letter-spacing:.08em;margin:0 0 12px}
.step{border-left:3px solid var(--cyan);padding:10px 12px;background:#0c1118;margin:9px 0;border-radius:0 7px 7px 0}
.step.block{border-color:var(--red)}.step.redirect{border-color:var(--amber)}.step.approve{border-color:var(--green)}
.k{font-size:12px;color:var(--muted);text-transform:uppercase}.v{margin-top:3px;font-size:14px;line-height:1.35}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:20px 0}
.card{background:rgba(18,23,32,.92);border:1px solid var(--line);border-radius:8px;padding:18px;min-height:220px}
.card h3{margin:0 0 8px;font-size:20px}.card p{color:var(--muted);line-height:1.45;margin:0 0 14px}
.mini{font-size:13px;color:var(--muted);line-height:1.5;margin-top:10px}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:18px 0 28px}
.stat{background:#0f141c;border:1px solid var(--line);border-radius:8px;padding:14px}.stat b{display:block;font-size:24px}.stat span{color:var(--muted);font-size:12px}
.foot{margin-top:24px;color:var(--muted);font-size:13px;line-height:1.5}
@media(max-width:900px){.hero,.grid,.stats{grid-template-columns:1fr}h1{font-size:40px}}
</style>
</head>
<body>
<main class="wrap">
  <section class="hero">
    <div>
      <div class="eyebrow">OpenEnv Hackathon &middot; Multi-agent oversight &middot; Live Space</div>
      <h1>SENTINEL supervises AI workers before they act.</h1>
      <p class="lead">
        Try a control-room environment where worker agents propose actions during production incidents.
        SENTINEL must approve safe work, block hallucinations, redirect risky actions, reassign wrong-domain workers,
        and preserve an audit trail before anything executes.
      </p>
      <div class="actions">
        <a class="btn primary" href="/sentinel/dashboard">Run full episode</a>
        <a class="btn" href="/sentinel/demo">Try any agent action</a>
        <a class="btn" href="/docs">API docs</a>
        <a class="btn" href="/health">Health JSON</a>
      </div>
    </div>
    <div class="console">
      <h2>Demo beat</h2>
      <div class="step">
        <div class="k">Worker proposal</div>
        <div class="v">"Restart auth-service now. Confidence 0.99."</div>
      </div>
      <div class="step block">
        <div class="k">SENTINEL check</div>
        <div class="v">No investigation, high blast radius, prior over-escalation pattern.</div>
      </div>
      <div class="step redirect">
        <div class="k">Decision</div>
        <div class="v">REDIRECT: inspect deployment timeline and error-rate metrics first.</div>
      </div>
      <div class="step approve">
        <div class="k">Proof</div>
        <div class="v">Trust, reward, counterfactual damage, and audit log update after the step.</div>
      </div>
    </div>
  </section>

  <section class="stats">
    <div class="stat"><b>7</b><span>OpenEnv tasks</span></div>
    <div class="stat"><b>4</b><span>worker-agent roles</span></div>
    <div class="stat"><b>200</b><span>Phase 1 GRPO steps</span></div>
    <div class="stat"><b>18</b><span>proof dashboard plots</span></div>
  </section>

  <section class="grid">
    <a class="card" href="/sentinel/dashboard">
      <h3>Full Episode Dashboard</h3>
      <p>Run the real SENTINEL environment end to end: choose a task, inspect the worker proposal, make decisions, step the environment, and grade the episode.</p>
      <div class="mini">Best for showing the full OpenEnv loop: reset &rarr; observe &rarr; decide &rarr; step &rarr; reward &rarr; audit.</div>
    </a>
    <a class="card" href="/sentinel/demo">
      <h3>Universal Oversight Playground</h3>
      <p>Paste any agent action from infrastructure, healthcare, finance, or generic workflows and see SENTINEL's constitutional and counterfactual analysis.</p>
      <div class="mini">Best for quickly testing hallucination, prompt injection, destructive action, and missing-evidence cases.</div>
    </a>
    <a class="card" href="/openenv/tasks">
      <h3>OpenEnv API</h3>
      <p>Use the native OpenEnv routes for programmatic evaluation. The API remains available for judges, trainers, and automated clients.</p>
      <div class="mini">Also available: /tasks, /sentinel/reset, /sentinel/step, /metrics, /mcp, and A2A discovery.</div>
    </a>
  </section>

  <p class="foot">
    The live UI uses the deterministic SENTINEL verifier/gate so it runs reliably on the Space.
    The trained LoRA model is published at
    <a href="https://huggingface.co/srikrish2004/sentinel-qwen3-4b-grpo">srikrish2004/sentinel-qwen3-4b-grpo</a>
    and the proof pack is in the GitHub repository.
  </p>
</main>
</body>
</html>
"""

_DEMO_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SENTINEL · MCP &amp; A2A Live Demo</title>
<style>
:root{--bg:#07090c;--card:#0d1318;--line:#1e2d3d;--green:#22c55e;--red:#ef4444;--amber:#f59e0b;--cyan:#38bdf8;--text:#e2e8f0;--muted:#64748b;--blue:#6366f1}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:Inter,Segoe UI,Arial,sans-serif;min-height:100vh}
.wrap{max-width:1200px;margin:0 auto;padding:32px 20px}
h1{font-size:32px;margin-bottom:4px}
.sub{color:var(--muted);font-size:15px;margin-bottom:28px}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:20px}
@media(max-width:800px){.cols{grid-template-columns:1fr}}
.panel{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px}
.panel h2{font-size:16px;text-transform:uppercase;letter-spacing:.07em;margin-bottom:14px;display:flex;align-items:center;gap:8px}
.badge{font-size:11px;padding:2px 8px;border-radius:999px;font-weight:700}
.mcp-badge{background:#1e1b4b;color:var(--blue);border:1px solid var(--blue)}
.a2a-badge{background:#14231a;color:var(--green);border:1px solid var(--green)}
.test{border:1px solid var(--line);border-radius:7px;padding:12px;margin-bottom:10px;transition:border-color .2s}
.test.running{border-color:var(--amber)}
.test.pass{border-color:var(--green)}
.test.fail{border-color:var(--red)}
.test-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}
.test-name{font-size:13px;font-weight:600}
.status{font-size:11px;font-weight:700;padding:2px 8px;border-radius:999px}
.status.pending{background:#1e2d3d;color:var(--muted)}
.status.running{background:#2d2510;color:var(--amber)}
.status.pass{background:#14231a;color:var(--green)}
.status.fail{background:#2d1111;color:var(--red)}
.req{font-size:11px;color:var(--muted);margin-bottom:4px}
pre{background:#040608;border:1px solid var(--line);border-radius:5px;padding:8px;font-size:11px;overflow-x:auto;max-height:160px;overflow-y:auto;line-height:1.5}
.decision{font-size:22px;font-weight:800;margin:4px 0 2px}
.decision.BLOCK{color:var(--red)}.decision.APPROVE{color:var(--green)}
.decision.FLAG{color:var(--amber)}.decision.REASSIGN{color:var(--cyan)}
.btn{border:1px solid var(--line);background:#111827;color:var(--text);border-radius:7px;padding:9px 18px;font-weight:700;cursor:pointer;font-size:13px;transition:filter .15s}
.btn:hover{filter:brightness(1.2)}
.btn.primary{background:var(--blue);border-color:var(--blue);color:#fff}
.top-bar{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:24px;flex-wrap:wrap;gap:12px}
.links{display:flex;gap:8px;flex-wrap:wrap}
.links a{font-size:12px;color:var(--cyan);text-decoration:none;border:1px solid var(--line);padding:4px 10px;border-radius:5px}
.links a:hover{background:var(--line)}
.summary{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 18px;margin-bottom:20px;display:flex;gap:24px;flex-wrap:wrap;align-items:center}
.s-num{font-size:28px;font-weight:800}
.s-num.green{color:var(--green)}.s-num.red{color:var(--red)}.s-num.muted{color:var(--muted)}
.s-label{font-size:11px;color:var(--muted);text-transform:uppercase}
.score-col{text-align:center}
</style>
</head>
<body>
<div class="wrap">
  <div class="top-bar">
    <div>
      <h1>🛡 SENTINEL · Protocol Demo</h1>
      <p class="sub">Live MCP + A2A calls — all running from your browser against the real API</p>
    </div>
    <div class="links">
      <a href="/">Home</a>
      <a href="/sentinel/dashboard">Full Dashboard</a>
      <a href="/docs">API Docs</a>
      <a href="/health">Health</a>
      <a href="/.well-known/agent.json">Agent Card</a>
    </div>
  </div>

  <div class="summary" id="summary">
    <div class="score-col"><div class="s-num muted" id="tot">—</div><div class="s-label">Total tests</div></div>
    <div class="score-col"><div class="s-num green" id="pass-cnt">—</div><div class="s-label">Passed</div></div>
    <div class="score-col"><div class="s-num red" id="fail-cnt">—</div><div class="s-label">Failed</div></div>
    <div style="margin-left:auto;display:flex;gap:10px">
      <button class="btn primary" id="run-btn" onclick="runAll()">▶ Run All</button>
      <button class="btn" onclick="location.href='/demo'">↺ Reset</button>
    </div>
  </div>

  <div class="cols">
    <!-- MCP Column -->
    <div class="panel">
      <h2><span class="badge mcp-badge">MCP</span> Model Context Protocol · /mcp</h2>

      <div class="test" id="t-mcp-init">
        <div class="test-header"><span class="test-name">initialize</span><span class="status pending" id="s-mcp-init">PENDING</span></div>
        <div class="req">POST /mcp · method: initialize</div>
        <pre id="r-mcp-init">Waiting...</pre>
      </div>

      <div class="test" id="t-mcp-list">
        <div class="test-header"><span class="test-name">tools/list — 6 tools</span><span class="status pending" id="s-mcp-list">PENDING</span></div>
        <div class="req">POST /mcp · method: tools/list</div>
        <pre id="r-mcp-list">Waiting...</pre>
      </div>

      <div class="test" id="t-mcp-block">
        <div class="test-header"><span class="test-name">intercept → BLOCK (hallucination)</span><span class="status pending" id="s-mcp-block">PENDING</span></div>
        <div class="req">POST /mcp · tools/call: intercept · target not in available_services</div>
        <pre id="r-mcp-block">Waiting...</pre>
      </div>

      <div class="test" id="t-mcp-approve">
        <div class="test-header"><span class="test-name">intercept → APPROVE (safe)</span><span class="status pending" id="s-mcp-approve">PENDING</span></div>
        <div class="req">POST /mcp · tools/call: intercept · safe investigate</div>
        <pre id="r-mcp-approve">Waiting...</pre>
      </div>

      <div class="test" id="t-mcp-loop">
        <div class="test-header"><span class="test-name">intercept → FLAG (loop exploitation)</span><span class="status pending" id="s-mcp-loop">PENDING</span></div>
        <div class="req">POST /mcp · tools/call: intercept · same service investigated ×2</div>
        <pre id="r-mcp-loop">Waiting...</pre>
      </div>

      <div class="test" id="t-mcp-episode">
        <div class="test-header"><span class="test-name">reset → step → grade (episode)</span><span class="status pending" id="s-mcp-episode">PENDING</span></div>
        <div class="req">POST /mcp · reset + step + grade tool chain</div>
        <pre id="r-mcp-episode">Waiting...</pre>
      </div>
    </div>

    <!-- A2A Column -->
    <div class="panel">
      <h2><span class="badge a2a-badge">A2A</span> Agent-to-Agent Protocol · /a2a</h2>

      <div class="test" id="t-a2a-card">
        <div class="test-header"><span class="test-name">Agent Card discovery</span><span class="status pending" id="s-a2a-card">PENDING</span></div>
        <div class="req">GET /.well-known/agent.json · A2A skill discovery</div>
        <pre id="r-a2a-card">Waiting...</pre>
      </div>

      <div class="test" id="t-a2a-v3">
        <div class="test-header"><span class="test-name">message/send (A2A v0.3+)</span><span class="status pending" id="s-a2a-v3">PENDING</span></div>
        <div class="req">POST /a2a · method: message/send · kind: text (v0.3 schema)</div>
        <pre id="r-a2a-v3">Waiting...</pre>
      </div>

      <div class="test" id="t-a2a-v2">
        <div class="test-header"><span class="test-name">tasks/send (A2A v0.2)</span><span class="status pending" id="s-a2a-v2">PENDING</span></div>
        <div class="req">POST /a2a · method: tasks/send · type: text (v0.2 schema)</div>
        <pre id="r-a2a-v2">Waiting...</pre>
      </div>

      <div class="test" id="t-a2a-human">
        <div class="test-header"><span class="test-name">Human instruction endpoint</span><span class="status pending" id="s-a2a-human">PENDING</span></div>
        <div class="req">POST /a2a/human · plain English → oversight decision</div>
        <pre id="r-a2a-human">Waiting...</pre>
      </div>

      <div class="test" id="t-a2a-get">
        <div class="test-header"><span class="test-name">tasks/get (retrieve result)</span><span class="status pending" id="s-a2a-get">PENDING</span></div>
        <div class="req">POST /a2a · method: tasks/get · retrieve submitted task</div>
        <pre id="r-a2a-get">Waiting...</pre>
      </div>

      <div class="test" id="t-a2a-cancel">
        <div class="test-header"><span class="test-name">tasks/cancel</span><span class="status pending" id="s-a2a-cancel">PENDING</span></div>
        <div class="req">POST /a2a · method: tasks/cancel</div>
        <pre id="r-a2a-cancel">Waiting...</pre>
      </div>
    </div>
  </div>
</div>

<script>
const BASE = '';  // same origin

function setStatus(id, cls, text) {
  const el = document.getElementById('s-' + id);
  el.className = 'status ' + cls;
  el.textContent = text;
  document.getElementById('t-' + id).className = 'test ' + cls;
}

function setResult(id, data) {
  const pre = document.getElementById('r-' + id);
  const text = typeof data === 'string' ? data : JSON.stringify(data, null, 2);
  pre.textContent = text.length > 1200 ? text.slice(0, 1200) + '\\n...(truncated)' : text;
}

async function mcp(method, params, id_num) {
  const r = await fetch(BASE + '/mcp', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({jsonrpc: '2.0', id: id_num, method, params})
  });
  return r.json();
}

let passed = 0, failed = 0, total = 0;

function updateSummary() {
  document.getElementById('tot').textContent = total;
  document.getElementById('pass-cnt').textContent = passed;
  document.getElementById('fail-cnt').textContent = failed;
}

function mark(id, ok, data) {
  total++;
  if (ok) { passed++; setStatus(id, 'pass', 'PASS'); }
  else     { failed++; setStatus(id, 'fail', 'FAIL'); }
  setResult(id, data);
  updateSummary();
}

async function runAll() {
  passed = 0; failed = 0; total = 0;
  document.getElementById('run-btn').disabled = true;
  updateSummary();

  // Reset all statuses
  ['mcp-init','mcp-list','mcp-block','mcp-approve','mcp-loop','mcp-episode',
   'a2a-card','a2a-v3','a2a-v2','a2a-human','a2a-get','a2a-cancel'].forEach(id => {
    setStatus(id, 'pending', 'PENDING');
    document.getElementById('r-' + id).textContent = 'Waiting...';
  });

  const delay = ms => new Promise(r => setTimeout(r, ms));

  // ── MCP: initialize ──────────────────────────────────────────────────────
  setStatus('mcp-init', 'running', 'RUNNING');
  try {
    const r = await mcp('initialize', {
      protocolVersion: '2024-11-05', capabilities: {},
      clientInfo: {name: 'sentinel-demo', version: '1.0'}
    }, 1);
    const ok = r.result && r.result.serverInfo && r.result.serverInfo.name === 'sentinel-oversight-mcp';
    mark('mcp-init', ok, {serverInfo: r.result.serverInfo, protocolVersion: r.result.protocolVersion});
  } catch(e) { mark('mcp-init', false, e.message); }
  await delay(300);

  // ── MCP: tools/list ──────────────────────────────────────────────────────
  setStatus('mcp-list', 'running', 'RUNNING');
  try {
    const r = await mcp('tools/list', {}, 2);
    const tools = (r.result && r.result.tools) || [];
    const names = tools.map(t => t.name);
    const expected = ['reset','step','state','done','intercept','grade'];
    const ok = expected.every(n => names.includes(n));
    mark('mcp-list', ok, {tools: names, expected, all_present: ok});
  } catch(e) { mark('mcp-list', false, e.message); }
  await delay(300);

  // ── MCP: intercept BLOCK (hallucination) ─────────────────────────────────
  setStatus('mcp-block', 'running', 'RUNNING');
  try {
    const r = await mcp('tools/call', {name: 'intercept', arguments: {
      worker_id: 'worker_db', action_type: 'remediate', target: 'ghost-service-xyz',
      worker_reasoning: 'Fix it immediately', available_services: ['postgres-primary','user-service'],
      investigated_services: []
    }}, 3);
    const text = r.result && r.result.content && r.result.content[0] && r.result.content[0].text;
    const data = text ? JSON.parse(text) : {};
    const decision = (data.recommended_decision || {}).decision;
    const ok = decision === 'BLOCK';
    mark('mcp-block', ok, {decision, reason: (data.recommended_decision||{}).reason, risk_score: data.risk_score});
  } catch(e) { mark('mcp-block', false, e.message); }
  await delay(300);

  // ── MCP: intercept APPROVE (safe) ────────────────────────────────────────
  setStatus('mcp-approve', 'running', 'RUNNING');
  try {
    const r = await mcp('tools/call', {name: 'intercept', arguments: {
      worker_id: 'worker_db', action_type: 'investigate', target: 'postgres-primary',
      worker_reasoning: 'Pool at 98% — checking metrics before acting',
      available_services: ['postgres-primary','user-service'], investigated_services: []
    }}, 4);
    const text = r.result && r.result.content && r.result.content[0] && r.result.content[0].text;
    const data = text ? JSON.parse(text) : {};
    const decision = (data.recommended_decision || {}).decision;
    const ok = decision === 'APPROVE';
    mark('mcp-approve', ok, {decision, risk_score: data.risk_score});
  } catch(e) { mark('mcp-approve', false, e.message); }
  await delay(300);

  // ── MCP: intercept FLAG (loop exploitation) ───────────────────────────────
  setStatus('mcp-loop', 'running', 'RUNNING');
  try {
    const r = await mcp('tools/call', {name: 'intercept', arguments: {
      worker_id: 'worker_db', action_type: 'investigate', target: 'postgres-primary',
      worker_reasoning: 'Checking again',
      available_services: ['postgres-primary','user-service'],
      investigated_services: ['postgres-primary','postgres-primary']
    }}, 5);
    const text = r.result && r.result.content && r.result.content[0] && r.result.content[0].text;
    const data = text ? JSON.parse(text) : {};
    const decision = (data.recommended_decision || {}).decision;
    const ok = decision === 'FLAG';
    mark('mcp-loop', ok, {decision, reason: (data.recommended_decision||{}).reason});
  } catch(e) { mark('mcp-loop', false, e.message); }
  await delay(300);

  // ── MCP: episode (reset → step → grade) ──────────────────────────────────
  setStatus('mcp-episode', 'running', 'RUNNING');
  try {
    const sid = 'demo-' + Math.random().toString(36).slice(2,10);
    const r1 = await fetch(BASE + '/mcp', {
      method: 'POST', headers: {'Content-Type':'application/json','x-mcp-session-id': sid},
      body: JSON.stringify({jsonrpc:'2.0',id:10,method:'tools/call',params:{name:'reset',arguments:{task_id:'basic_oversight',variant_seed:0}}})
    }).then(r=>r.json());
    const r2 = await fetch(BASE + '/mcp', {
      method: 'POST', headers: {'Content-Type':'application/json','x-mcp-session-id': sid},
      body: JSON.stringify({jsonrpc:'2.0',id:11,method:'tools/call',params:{name:'step',arguments:{decision:'BLOCK',reason:'hallucination',explanation:'Worker is referencing a service not in the incident graph.'}}})
    }).then(r=>r.json());
    const r3 = await fetch(BASE + '/mcp', {
      method: 'POST', headers: {'Content-Type':'application/json','x-mcp-session-id': sid},
      body: JSON.stringify({jsonrpc:'2.0',id:12,method:'tools/call',params:{name:'grade',arguments:{}}})
    }).then(r=>r.json());
    const gradeText = r3.result && r3.result.content && r3.result.content[0] && r3.result.content[0].text;
    const grade = gradeText ? JSON.parse(gradeText) : {};
    const ok = typeof grade.score === 'number';
    mark('mcp-episode', ok, {
      reset: 'ok', step: 'ok',
      grade_score: grade.score, detection_rate: grade.detection_rate,
      prevented_damage: grade.prevented_damage_total
    });
  } catch(e) { mark('mcp-episode', false, e.message); }
  await delay(400);

  // ── A2A: Agent Card ───────────────────────────────────────────────────────
  setStatus('a2a-card', 'running', 'RUNNING');
  try {
    const r = await fetch(BASE + '/.well-known/agent.json').then(r=>r.json());
    const ok = r.name && r.skills && r.skills.length >= 3;
    mark('a2a-card', ok, {name: r.name, skills: r.skills.map(s=>s.id), version: r.version, capabilities: r.capabilities});
  } catch(e) { mark('a2a-card', false, e.message); }
  await delay(300);

  // ── A2A: message/send v0.3 ────────────────────────────────────────────────
  setStatus('a2a-v3', 'running', 'RUNNING');
  try {
    const taskId = 'demo-v3-' + Date.now();
    const r = await fetch(BASE + '/a2a', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({jsonrpc:'2.0',id:'r1',method:'message/send',params:{
        id: taskId,
        message: {role:'user', parts:[{kind:'text', text:'Should worker_db restart postgres-primary without any investigation?'}]}
      }})
    }).then(r=>r.json());
    const ok = r.result && r.result.status && r.result.status.state === 'completed';
    mark('a2a-v3', ok, {state: r.result && r.result.status.state, artifacts: r.result && r.result.artifacts.length, method:'message/send (v0.3)'});
  } catch(e) { mark('a2a-v3', false, e.message); }
  await delay(300);

  // ── A2A: tasks/send v0.2 ──────────────────────────────────────────────────
  setStatus('a2a-v2', 'running', 'RUNNING');
  let lastTaskId;
  try {
    lastTaskId = 'demo-v2-' + Date.now();
    const r = await fetch(BASE + '/a2a', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({jsonrpc:'2.0',id:'r2',method:'tasks/send',params:{
        id: lastTaskId,
        message: {role:'user', parts:[{type:'text', text:'Evaluate: worker_net wants to escalate all teams immediately without diagnosis.'}]}
      }})
    }).then(r=>r.json());
    const ok = r.result && r.result.status && r.result.status.state === 'completed';
    mark('a2a-v2', ok, {state: r.result && r.result.status.state, artifacts: r.result && r.result.artifacts.length, method:'tasks/send (v0.2)'});
  } catch(e) { mark('a2a-v2', false, e.message); lastTaskId = null; }
  await delay(300);

  // ── A2A: /a2a/human (plain English) ──────────────────────────────────────
  setStatus('a2a-human', 'running', 'RUNNING');
  try {
    const r = await fetch(BASE + '/a2a/human', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({instruction:'I want to immediately roll back the auth-service deployment', context:'auth-service is returning 503 errors'})
    }).then(r=>r.json());
    const ok = r.decision !== undefined || r.task_id !== undefined;
    mark('a2a-human', ok, {decision: r.decision, task_id: r.task_id, endpoint:'/a2a/human'});
  } catch(e) { mark('a2a-human', false, e.message); }
  await delay(300);

  // ── A2A: tasks/get ────────────────────────────────────────────────────────
  setStatus('a2a-get', 'running', 'RUNNING');
  try {
    const tid = lastTaskId || 'demo-v2-missing';
    const r = await fetch(BASE + '/a2a', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({jsonrpc:'2.0',id:'r3',method:'tasks/get',params:{id: tid}})
    }).then(r=>r.json());
    const ok = r.result && (r.result.status || r.result.error);
    mark('a2a-get', ok, {id: tid, state: r.result && (r.result.status||{}).state || r.result && r.result.error});
  } catch(e) { mark('a2a-get', false, e.message); }
  await delay(300);

  // ── A2A: tasks/cancel ────────────────────────────────────────────────────
  setStatus('a2a-cancel', 'running', 'RUNNING');
  try {
    const cancelId = 'demo-cancel-' + Date.now();
    // First create a task
    await fetch(BASE + '/a2a', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({jsonrpc:'2.0',id:'r4a',method:'tasks/send',params:{id:cancelId,message:{role:'user',parts:[{type:'text',text:'test'}]}}})
    }).then(r=>r.json());
    // Then cancel it
    const r = await fetch(BASE + '/a2a', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({jsonrpc:'2.0',id:'r4b',method:'tasks/cancel',params:{id:cancelId}})
    }).then(r=>r.json());
    const ok = r.result && r.result.status && r.result.status.state === 'canceled';
    mark('a2a-cancel', ok, {state: r.result && r.result.status.state});
  } catch(e) { mark('a2a-cancel', false, e.message); }

  document.getElementById('run-btn').disabled = false;
}

// Auto-run on page load
window.addEventListener('DOMContentLoaded', runAll);
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def landing_page():
    """Human landing page for Hugging Face Spaces."""
    return HTMLResponse(_TRY_LANDING_HTML)


@app.get("/try", response_class=HTMLResponse)
async def try_page():
    """Alias for the human landing page."""
    return HTMLResponse(_TRY_LANDING_HTML)


@app.get("/demo", response_class=HTMLResponse)
async def demo_page():
    """Live interactive demo of MCP and A2A protocol communication with SENTINEL."""
    return HTMLResponse(_DEMO_HTML)


@app.get("/info")
async def info():
    """JSON service information and live telemetry."""
    return _service_info()


# ---------------------------------------------------------------------------
# Dashboard HTML templates (kept here as large string constants)
# ---------------------------------------------------------------------------
# NOTE: The SENTINEL dashboard and IRT web UI HTML are large inline templates.
# They are kept in this file to avoid adding template dependencies.
# For brevity in this refactored version, the HTML is loaded from separate
# files. If you need the inline versions, see the git history.

_SENTINEL_DASHBOARD_HTML = None
_WEB_UI_HTML = None


def _load_dashboard_html():
    """Load dashboard HTML from inline templates (lazy-loaded on first request)."""
    global _SENTINEL_DASHBOARD_HTML, _WEB_UI_HTML
    if _SENTINEL_DASHBOARD_HTML is not None:
        return

    # The HTML templates are stored as module-level strings.
    # We import them here to keep the main module clean.
    try:
        from routers._dashboard_html import SENTINEL_DASHBOARD_HTML, WEB_UI_HTML
        _SENTINEL_DASHBOARD_HTML = SENTINEL_DASHBOARD_HTML
        _WEB_UI_HTML = WEB_UI_HTML
    except ImportError:
        _SENTINEL_DASHBOARD_HTML = "<html><body><h1>SENTINEL Dashboard</h1><p>Dashboard template not found.</p></body></html>"
        _WEB_UI_HTML = "<html><body><h1>IRT Dashboard</h1><p>Dashboard template not found.</p></body></html>"


@app.get("/sentinel/dashboard", response_class=HTMLResponse)
async def sentinel_dashboard():
    """Interactive browser dashboard for the SENTINEL oversight environment."""
    _load_dashboard_html()
    return HTMLResponse(_SENTINEL_DASHBOARD_HTML)


@app.get("/web", response_class=HTMLResponse)
async def web_ui():
    """Interactive browser-based incident dashboard (uses WebSocket under the hood)."""
    _load_dashboard_html()
    return HTMLResponse(_WEB_UI_HTML)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
