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
    GET  /                   - Rich health check with telemetry
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
    _log.info("IRT environment started — TTL cleanup every 300s")
    try:
        yield
    finally:
        task.cancel()


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Incident Response Triage – OpenEnv",
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
# Root-level endpoints (health checks)
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    """Standard OpenEnv health check."""
    return {
        "status": "healthy",
        "native_openenv_available": NATIVE_OPENENV_AVAILABLE,
        "native_openenv_mount": "/openenv" if NATIVE_OPENENV_AVAILABLE else None,
    }


@app.get("/")
async def health():
    """Health check - returns 200 with environment info and live telemetry."""
    return {
        "status": "ok",
        "environment": "sentinel-oversight-command",
        "version": "1.0.0",
        "tasks": [t.task_id for t in get_all_tasks()],
        "primary_theme": "multi-agent interactions",
        "native_openenv_available": NATIVE_OPENENV_AVAILABLE,
        "native_openenv_mount": "/openenv" if NATIVE_OPENENV_AVAILABLE else None,
        "active_sessions": len(_SESSION_REGISTRY),
        "ws_active_connections": _deps.WS_ACTIVE_CONNECTIONS,
        "telemetry": _TELEMETRY,
    }


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
