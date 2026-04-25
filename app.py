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
    return {
        "status": "healthy",
        "native_openenv_available": NATIVE_OPENENV_AVAILABLE,
        "native_openenv_mount": "/openenv" if NATIVE_OPENENV_AVAILABLE else None,
        "mcp_available": MCP_AVAILABLE,
        "mcp_endpoint": "/mcp" if MCP_AVAILABLE else None,
        "a2a_available": A2A_AVAILABLE,
        "a2a_agent_card": "/.well-known/agent.json" if A2A_AVAILABLE else None,
    }


def _service_info():
    """Return environment info and live telemetry for JSON endpoints."""
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


@app.get("/", response_class=HTMLResponse)
async def landing_page():
    """Human landing page for Hugging Face Spaces."""
    return HTMLResponse(_TRY_LANDING_HTML)


@app.get("/try", response_class=HTMLResponse)
async def try_page():
    """Alias for the human landing page."""
    return HTMLResponse(_TRY_LANDING_HTML)


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
