# -*- coding: utf-8 -*-
"""FastAPI server exposing the OpenEnv API endpoints.

Endpoints:
    POST /reset              – Reset environment for a task (returns session_id)
    POST /step               – Take an agent action (requires X-Session-ID header)
    GET  /state              – Get current environment state (requires X-Session-ID)
    GET  /tasks              – List available tasks with action schema
    POST /grader             – Get grader score for episode (requires X-Session-ID)
    POST /baseline           – Run rule-based baseline on all tasks (in-process)
    GET  /metrics            – Telemetry counters (JSON or Prometheus text)
    GET  /render             – Human-readable incident dashboard (requires X-Session-ID)
    GET  /leaderboard        – Top scores per task from completed episodes
    GET  /health             – Standard OpenEnv liveness probe
    GET  /                   – Rich health check with telemetry
    WS   /ws                 – WebSocket persistent session (no session header needed)
    GET  /web                – Interactive browser-based incident dashboard
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import secrets
import time
import traceback
from contextlib import asynccontextmanager
from typing import Any, Dict, List

from fastapi import Body, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel

from src.environment import IncidentResponseEnv
from src.models import Action, StepResult
from src.tasks import get_all_tasks


# ---------------------------------------------------------------------------
# Structured JSON logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "msg": "%(message)s"}',
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
_log = logging.getLogger("irt.api")

# ---------------------------------------------------------------------------
# Session-isolated environment registry
# ---------------------------------------------------------------------------
# Each session_id maps to its own IncidentResponseEnv instance.
# Sessions expire after SESSION_TTL seconds (default 1 hour) and are also
# evicted via FIFO when the cap is reached.

_SESSION_REGISTRY: Dict[str, IncidentResponseEnv] = {}
_SESSION_TIMESTAMPS: Dict[str, float] = {}  # session_id -> creation epoch
_MAX_SESSIONS = 256
_SESSION_TTL = int(os.environ.get("SESSION_TTL_SECONDS", 3600))

# ---------------------------------------------------------------------------
# Telemetry counters  (in-process; reset on restart)
# ---------------------------------------------------------------------------
_TELEMETRY: Dict[str, int] = {
    "sessions_created": 0,
    "sessions_evicted_fifo": 0,
    "sessions_expired_ttl": 0,
    "episodes_total": 0,
    "steps_total": 0,
    "grader_calls": 0,
    "baseline_runs": 0,
    "errors_total": 0,
    "ws_connections_total": 0,
}

# Active WebSocket connections (single-process; decremented on disconnect)
_WS_ACTIVE_CONNECTIONS: int = 0

# ---------------------------------------------------------------------------
# In-memory leaderboard  (top-10 scores per task)
# ---------------------------------------------------------------------------
_LEADERBOARD: Dict[str, List[Dict[str, Any]]] = {
    "severity_classification": [],
    "root_cause_analysis": [],
    "full_incident_management": [],
}
_LEADERBOARD_SIZE = 10


def _get_or_create_session(session_id: str | None) -> tuple[str, IncidentResponseEnv]:
    """Return (session_id, env). Creates a new session if id is None or unknown."""
    if session_id and session_id in _SESSION_REGISTRY:
        return session_id, _SESSION_REGISTRY[session_id]
    # New session — evict if at capacity
    if len(_SESSION_REGISTRY) >= _MAX_SESSIONS:
        oldest = next(iter(_SESSION_REGISTRY))
        del _SESSION_REGISTRY[oldest]
        _SESSION_TIMESTAMPS.pop(oldest, None)
        _TELEMETRY["sessions_evicted_fifo"] += 1
        _log.info("session evicted (FIFO): %s", oldest)
    new_id = session_id or secrets.token_hex(16)
    _SESSION_REGISTRY[new_id] = IncidentResponseEnv()
    _SESSION_TIMESTAMPS[new_id] = time.time()
    _TELEMETRY["sessions_created"] += 1
    return new_id, _SESSION_REGISTRY[new_id]


def _purge_expired_sessions() -> int:
    """Remove sessions older than SESSION_TTL. Returns number purged."""
    cutoff = time.time() - _SESSION_TTL
    stale = [sid for sid, ts in _SESSION_TIMESTAMPS.items() if ts < cutoff]
    for sid in stale:
        _SESSION_REGISTRY.pop(sid, None)
        _SESSION_TIMESTAMPS.pop(sid, None)
        _TELEMETRY["sessions_expired_ttl"] += 1
    if stale:
        _log.info("purged %d stale session(s)", len(stale))
    return len(stale)


def _record_leaderboard(task_id: str, score: float, steps: int) -> None:
    """Insert a completed episode score into the in-memory leaderboard."""
    board = _LEADERBOARD.get(task_id)
    if board is None:
        return
    board.append({"score": score, "steps": steps, "ts": round(time.time())})
    board.sort(key=lambda e: (-e["score"], e["steps"]))
    del board[_LEADERBOARD_SIZE:]  # keep top-N


# ---------------------------------------------------------------------------
# Prometheus metric helpers
# ---------------------------------------------------------------------------

# (prom_metric_name, ServiceMetrics field, HELP text)
_PROM_CORE_FIELDS: List[tuple] = [
    ("irt_cpu_percent",    "cpu_percent",    "CPU utilisation percent"),
    ("irt_memory_percent", "memory_percent", "Memory utilisation percent"),
    ("irt_request_rate",   "request_rate",   "Requests per second"),
    ("irt_error_rate",     "error_rate",     "HTTP error rate fraction 0.0-1.0"),
    ("irt_latency_p50_ms", "latency_p50_ms", "P50 response latency milliseconds"),
    ("irt_latency_p99_ms", "latency_p99_ms", "P99 response latency milliseconds"),
]


def _scenario_live_to_prom_text(
    live: Dict[str, Any],
    scenario_id: str,
    incident_id: str,
    step: int,
) -> str:
    """Serialize live scenario metrics to Prometheus text exposition format."""
    lines: List[str] = [
        f'# HELP irt_scenario_step Current episode step number',
        f'# TYPE irt_scenario_step gauge',
        f'irt_scenario_step{{scenario="{scenario_id}",incident="{incident_id}"}} {step}',
    ]
    for prom_name, field, help_text in _PROM_CORE_FIELDS:
        lines += [
            f"# HELP {prom_name} {help_text}",
            f"# TYPE {prom_name} gauge",
        ]
        for svc, m in live.items():
            val = getattr(m, field, 0.0)
            lines.append(
                f'{prom_name}{{service="{svc}",scenario="{scenario_id}",incident="{incident_id}"}} {val}'
            )
    # Custom metrics (e.g. connection_pool_used, heap_mb, ...)
    all_custom: Dict[str, str] = {}  # prom_name -> raw key
    for m in live.values():
        for raw_key in (m.custom or {}):
            prom_key = "irt_custom_" + re.sub(r"[^a-zA-Z0-9_]", "_", raw_key)
            all_custom[prom_key] = raw_key
    for prom_key in sorted(all_custom):
        raw_key = all_custom[prom_key]
        lines += [
            f"# HELP {prom_key} Custom scenario metric: {raw_key}",
            f"# TYPE {prom_key} gauge",
        ]
        for svc, m in live.items():
            val = (m.custom or {}).get(raw_key)
            if val is not None:
                lines.append(
                    f'{prom_key}{{service="{svc}",scenario="{scenario_id}",incident="{incident_id}"}} {val}'
                )
    return "\n".join(lines) + "\n"


_PROM_SELECTOR_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)?(?:\{(?P<labels>[^}]*)\})?$"
)
_PROM_LABEL_RE = re.compile(r'(\w+)\s*=\s*"([^"]*)"')


def _parse_prom_selector(query: str) -> tuple[str, Dict[str, str]]:
    """Parse a simple PromQL instant selector into (metric_name, label_filters)."""
    m = _PROM_SELECTOR_RE.match(query.strip())
    if not m:
        return query.strip(), {}
    name = m.group("name") or ""
    label_str = m.group("labels") or ""
    filters: Dict[str, str] = {
        lm.group(1): lm.group(2)
        for lm in _PROM_LABEL_RE.finditer(label_str)
    }
    return name, filters


def _build_prom_vector(
    live: Dict[str, Any],
    metric_name: str,
    label_filters: Dict[str, str],
    scenario_id: str,
    incident_id: str,
) -> List[Dict[str, Any]]:
    """Build a Prometheus instant-query vector result list."""
    ts = round(time.time(), 3)
    # Normalise: auto-prefix irt_ when caller omits it
    if metric_name and not metric_name.startswith("irt_"):
        metric_name = f"irt_{metric_name}"
    field_map = {pn: fn for pn, fn, _ in _PROM_CORE_FIELDS}
    candidates = [metric_name] if metric_name else [pn for pn, _, _ in _PROM_CORE_FIELDS]
    results: List[Dict[str, Any]] = []
    for prom_name in candidates:
        field = field_map.get(prom_name)
        for svc, m in live.items():
            if "service" in label_filters and label_filters["service"] != svc:
                continue
            if "scenario" in label_filters and label_filters["scenario"] != scenario_id:
                continue
            if field is not None:
                val = getattr(m, field, 0.0)
            elif prom_name.startswith("irt_custom_"):
                raw_key = prom_name[len("irt_custom_"):]
                val = (m.custom or {}).get(raw_key)
                if val is None:
                    continue
            else:
                continue
            results.append({
                "metric": {
                    "__name__": prom_name,
                    "service": svc,
                    "scenario": scenario_id,
                    "incident": incident_id,
                },
                "value": [ts, str(val)],
            })
    return results


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start background TTL-cleanup task; cancel it on shutdown."""
    async def _cleanup_loop():
        while True:
            await asyncio.sleep(300)  # run every 5 minutes
            _purge_expired_sessions()

    task = asyncio.create_task(_cleanup_loop())
    _log.info("IRT environment started — TTL cleanup every 300s")
    try:
        yield
    finally:
        task.cancel()


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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response helpers
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "severity_classification"
    session_id: str | None = None
    variant_seed: int | None = None


class BaselineResponse(BaseModel):
    results: list
    summary: Dict[str, Any]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    """Standard OpenEnv health check – returns {status: healthy}."""
    return {"status": "healthy"}


@app.get("/")
async def health():
    """Health check – returns 200 with environment info and live telemetry."""
    return {
        "status": "ok",
        "environment": "incident-response-triage",
        "version": "1.0.0",
        "tasks": [t.task_id for t in get_all_tasks()],
        "active_sessions": len(_SESSION_REGISTRY),
        "ws_active_connections": _WS_ACTIVE_CONNECTIONS,
        "telemetry": _TELEMETRY,
    }


@app.post("/reset")
async def reset(request: ResetRequest | None = Body(default=None)):
    """Reset the environment for a given task_id.

    Returns the initial observation plus a `session_id` that must be
    passed via the `X-Session-ID` header on all subsequent calls.
    """
    if request is None:
        request = ResetRequest()
    try:
        session_id, env = _get_or_create_session(request.session_id)
        # When no variant_seed is supplied randomise for anti-memorization;
        # explicit 0 keeps the primary (deterministic) scenario.
        seed = request.variant_seed if request.variant_seed is not None else secrets.randbelow(100)
        obs = env.reset(request.task_id, variant_seed=seed)
        _TELEMETRY["episodes_total"] += 1
        _log.info("episode reset task=%s session=%s variant=%d", request.task_id, session_id[:8], seed)
        data = obs.model_dump()
        data["session_id"] = session_id
        return data
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/step")
async def step(
    action: Action,
    x_session_id: str | None = Header(default=None, alias="X-Session-ID"),
):
    """Execute one action and return observation, reward, done, info."""
    if not x_session_id or x_session_id not in _SESSION_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail="Missing or unknown X-Session-ID header. Call /reset first.",
        )
    env = _SESSION_REGISTRY[x_session_id]
    try:
        result: StepResult = env.step(action)
        _TELEMETRY["steps_total"] += 1
        return result.model_dump()
    except RuntimeError as exc:
        _TELEMETRY["errors_total"] += 1
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        _TELEMETRY["errors_total"] += 1
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")


@app.get("/state")
async def state(
    x_session_id: str | None = Header(default=None, alias="X-Session-ID"),
):
    """Return full environment state."""
    if not x_session_id or x_session_id not in _SESSION_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail="Missing or unknown X-Session-ID header. Call /reset first.",
        )
    env = _SESSION_REGISTRY[x_session_id]
    try:
        return env.state().model_dump()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/tasks")
async def tasks():
    """List all tasks with descriptions and action schema."""
    return [t.model_dump() for t in get_all_tasks()]


@app.post("/grader")
async def grader(
    x_session_id: str | None = Header(default=None, alias="X-Session-ID"),
):
    """Return grader score for the current or most recent episode."""
    if not x_session_id or x_session_id not in _SESSION_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail="Missing or unknown X-Session-ID header. Call /reset first.",
        )
    env = _SESSION_REGISTRY[x_session_id]
    try:
        result = env.grade()
        _TELEMETRY["grader_calls"] += 1
        state = env.state()
        _record_leaderboard(state.task_id, result.score, state.total_steps_taken)
        _log.info("graded task=%s score=%.4f steps=%d", state.task_id, result.score, state.total_steps_taken)
        return result.model_dump()
    except RuntimeError as exc:
        _TELEMETRY["errors_total"] += 1
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/baseline")
async def baseline():
    """Run the rule-based baseline inference against all tasks (in-process).

    Creates a dedicated ephemeral env instance so it never interferes
    with any active session.
    """
    try:
        from baseline.inference import run_all_tasks
        dedicated_env = IncidentResponseEnv()
        results = run_all_tasks(base_url=None, env_instance=dedicated_env)
        _TELEMETRY["baseline_runs"] += 1
        summary = {
            "mean_score": round(
                sum(r["score"] for r in results) / len(results), 4
            ),
            "tasks_evaluated": len(results),
        }
        return BaselineResponse(results=results, summary=summary).model_dump()
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Baseline execution failed: {exc}",
        )


# ---------------------------------------------------------------------------
# Observability endpoints
# ---------------------------------------------------------------------------

@app.get("/metrics")
async def metrics(format: str = "json"):
    """Return telemetry counters.

    ?format=prometheus  → Prometheus text format
    ?format=json        → JSON (default)
    """
    if format == "prometheus":
        lines = ["# HELP irt_counter OpenEnv IRT telemetry", "# TYPE irt_counter gauge"]
        for key, value in _TELEMETRY.items():
            lines.append(f'irt_{key} {value}')
        lines.append(f'irt_active_sessions {len(_SESSION_REGISTRY)}')

        return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")
    return {
        **_TELEMETRY,
        "active_sessions": len(_SESSION_REGISTRY),
        "session_ttl_seconds": _SESSION_TTL,
        "max_sessions": _MAX_SESSIONS,
    }


@app.get("/render")
async def render(
    x_session_id: str | None = Header(default=None, alias="X-Session-ID"),
):
    """Return a human-readable incident dashboard for the current session.

    Useful for debugging agent behaviour or as a REPL-style interface.
    """
    if not x_session_id or x_session_id not in _SESSION_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail="Missing or unknown X-Session-ID header. Call /reset first.",
        )
    env = _SESSION_REGISTRY[x_session_id]
    try:
        s = env.state()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    sev = s.severity_classified.value if s.severity_classified else "(not classified)"
    status_icon = "✅" if s.done else "⏳"
    bar_filled = int((s.step_number / s.max_steps) * 20)
    progress_bar = "█" * bar_filled + "░" * (20 - bar_filled)

    lines = [
        f"## 🚨 INCIDENT DASHBOARD — {s.task_id.replace('_', ' ').upper()}",
        "",
        f"| Field          | Value |",
        f"|----------------|-------|",
        f"| **Incident ID**| `{s.task_id}` |",
        f"| **Status**     | {status_icon} `{s.incident_status.value}` |",
        f"| **Progress**   | `[{progress_bar}]` {s.step_number}/{s.max_steps} steps |",
        f"| **Severity**   | `{sev}` |",
        f"| **Diagnosis**  | `{s.diagnosis or '(none)'}` |",
        f"| **Reward**     | `{s.cumulative_reward:.4f}` |",
        "",
        "### 🚧 Actions Taken",
    ]
    if s.actions_history:
        for i, a in enumerate(s.actions_history, 1):
            lines.append(f"{i}. `{a['action_type'].value}` → `{a.get('target', '')}` | {a.get('reasoning', '')[:80]}")
    else:
        lines.append("_No actions yet._")

    lines += [
        "",
        f"### 🔍 Investigated Services",
        ", ".join(f"`{s}`" for s in s.investigated_services) or "_None_",
        "",
        f"### 🛠 Remediations Applied",
        ", ".join(f"`{r}`" for r in s.remediations_applied) or "_None_",
        "",
        f"### 📯 Escalations",
        ", ".join(f"`{e}`" for e in s.escalations_made) or "_None_",
    ]

    return {"markdown": "\n".join(lines), "state": s.model_dump()}


@app.get("/leaderboard")
async def leaderboard():
    """Return top scores per task from all completed episodes in this session.

    Scores are ranked by (score DESC, steps ASC) — accuracy first, then efficiency.
    """
    return {
        task_id: board
        for task_id, board in _LEADERBOARD.items()
    }


# ---------------------------------------------------------------------------
# Prometheus-compatible live metrics endpoints
# ---------------------------------------------------------------------------

@app.get("/prometheus/metrics")
async def prometheus_scenario_metrics(
    fmt: str = "text",
    x_session_id: str | None = Header(default=None, alias="X-Session-ID"),
):
    """Prometheus text-format scrape endpoint for the current scenario state.

    Returns all service metrics with blast-radius degradation applied at the
    current step — the system degrades the longer the agent waits, exactly as
    in production Prometheus. No action cost: purely passive observability.

    - ``?fmt=text`` (default) — Prometheus text exposition format (standard scrape)
    - ``?fmt=json``           — JSON dict keyed by service name
    """
    if not x_session_id or x_session_id not in _SESSION_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail="Missing or unknown X-Session-ID. Call /reset first.",
        )
    env = _SESSION_REGISTRY[x_session_id]
    live = env.live_metrics()
    if not live:
        raise HTTPException(status_code=400, detail="No active episode. Call /reset first.")
    s = env.state()
    if fmt == "json":
        return {svc: m.model_dump() for svc, m in live.items()}
    prom_text = _scenario_live_to_prom_text(live, s.scenario_id, s.task_id, s.step_number)
    return PlainTextResponse(prom_text, media_type="text/plain; version=0.0.4")


@app.get("/prometheus/query")
async def prometheus_instant_query(
    query: str,
    x_session_id: str | None = Header(default=None, alias="X-Session-ID"),
):
    """Simplified Prometheus instant-query API (subset of /api/v1/query).

    Returns a standard Prometheus JSON response envelope so agents can use
    ``prometheus-api-client`` or any PromQL helper directly.  No server-side
    evaluation of complex PromQL — selectors only.

    Supported selectors::

        irt_error_rate                            # all services
        irt_error_rate{service="auth-service"}    # specific service
        error_rate{service="payment-api"}         # irt_ prefix auto-added
        {service="payment-api"}                   # all metrics for one service
    """
    if not x_session_id or x_session_id not in _SESSION_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail="Missing or unknown X-Session-ID. Call /reset first.",
        )
    env = _SESSION_REGISTRY[x_session_id]
    live = env.live_metrics()
    if not live:
        raise HTTPException(status_code=400, detail="No active episode. Call /reset first.")
    s = env.state()
    metric_name, label_filters = _parse_prom_selector(query)
    vector = _build_prom_vector(live, metric_name, label_filters, s.scenario_id, s.task_id)
    return {
        "status": "success",
        "data": {
            "resultType": "vector",
            "result": vector,
        },
    }


# ---------------------------------------------------------------------------
# WebSocket endpoint — one env instance per connection, no session header
# ---------------------------------------------------------------------------

_WEB_UI_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>IRT \u2014 OpenEnv Interactive</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:monospace;background:#0d1117;color:#e6edf3;min-height:100vh;padding:16px}
h1{color:#f85149;margin-bottom:4px;font-size:19px}
.row{display:flex;gap:12px;flex-wrap:wrap;margin-top:12px}
.panel{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px;flex:1;min-width:260px;margin-bottom:12px}
h2{color:#58a6ff;font-size:11px;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px}
select,input,textarea{font-family:monospace;font-size:12px;background:#21262d;color:#e6edf3;border:1px solid #30363d;border-radius:4px;padding:5px 8px;width:100%;margin-bottom:8px}
button{font-family:monospace;font-size:12px;cursor:pointer;background:#238636;border:1px solid #2ea043;color:#fff;padding:7px 14px;border-radius:4px;width:100%;margin-top:4px}
button:hover{background:#2ea043}
.feed{max-height:260px;overflow-y:auto;font-size:11px}
.fi{padding:5px 8px;margin:3px 0;border-radius:3px;border-left:3px solid #30363d}
.fi.pos{border-left-color:#2ea043;background:#0f2618}
.fi.neg{border-left-color:#f85149;background:#260f0f}
.fi.inf{border-left-color:#58a6ff;background:#0a192a}
.alert{padding:5px 9px;border-radius:3px;margin:3px 0;font-size:11px}
.alert.CRITICAL{background:#2a0a0d;border-left:3px solid #f85149}
.alert.WARNING{background:#221a08;border-left:3px solid #d29922}
.alert.INFO{background:#091829;border-left:3px solid #58a6ff}
.tag{display:inline-block;background:#21262d;border:1px solid #30363d;border-radius:10px;padding:2px 8px;font-size:11px;margin:2px}
.tag.done{background:#0f2618;border-color:#2ea043;color:#2ea043}
.st{font-size:11px;color:#8b949e;padding:2px 0}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:#f85149;margin-right:6px;vertical-align:middle}
.dot.on{background:#2ea043}
.score{font-size:36px;font-weight:bold;text-align:center}
.bar{height:8px;background:#21262d;border-radius:4px;margin:8px 0}
.bar-fill{height:100%;border-radius:4px;transition:width .3s}
label{font-size:11px;color:#8b949e;display:block;margin-bottom:3px}
hr{border:none;border-top:1px solid #21262d;margin:10px 0}
#revealed{max-height:300px;overflow-y:auto;font-size:11px}
</style></head>
<body>
<h1>&#x1F6A8; Incident Response Triage <span style="font-size:13px;color:#8b949e">&mdash; OpenEnv Interactive</span></h1>
<p class="st"><span class="dot" id="dot"></span><span id="ctext">Connecting&hellip;</span></p>
<div class="row">
  <div class="panel" style="flex:0 0 228px;min-width:228px">
    <h2>Control</h2>
    <label>Task</label>
    <select id="task">
      <option value="severity_classification">Easy &mdash; Severity Classification</option>
      <option value="root_cause_analysis">Medium &mdash; Root Cause Analysis</option>
      <option value="full_incident_management">Hard &mdash; Full Incident Management</option>
    </select>
    <button onclick="doReset()">&#x25B6; New Episode</button>
    <hr>
    <div class="st">Step: <b id="snum">&mdash;</b> / <b id="smax">&mdash;</b></div>
    <div class="st">Reward: <b id="rew">&mdash;</b></div>
    <div class="st">Status: <b id="istatus">&mdash;</b></div>
    <div class="st">Severity: <b id="isev">&mdash;</b></div>
  </div>
  <div class="panel">
    <h2>Alerts</h2>
    <div id="alerts"><p class="st">Start an episode.</p></div>
    <h2 style="margin-top:10px">Services</h2>
    <div id="services"></div>
  </div>
</div>
<div class="row">
  <div class="panel" style="flex:0 0 310px;min-width:280px">
    <h2>Action</h2>
    <label>Type</label>
    <select id="atype" onchange="updateForm()">
      <option value="investigate">INVESTIGATE &mdash; reveal service data</option>
      <option value="classify">CLASSIFY &mdash; set incident severity</option>
      <option value="diagnose">DIAGNOSE &mdash; identify root cause</option>
      <option value="remediate">REMEDIATE &mdash; apply fix</option>
      <option value="escalate">ESCALATE &mdash; notify team</option>
      <option value="communicate">COMMUNICATE &mdash; status update</option>
    </select>
    <div id="aform"></div>
    <label>Reasoning</label>
    <textarea id="reasoning" rows="2" placeholder="Why this action?"></textarea>
    <button onclick="doStep()">&#x2192; Submit Action</button>
  </div>
  <div class="panel">
    <h2>Revealed Data (after INVESTIGATE)</h2>
    <div id="revealed"><p class="st">Investigate a service to see its logs &amp; metrics.</p></div>
  </div>
</div>
<div class="row">
  <div class="panel">
    <h2>Event Feed</h2>
    <div class="feed" id="feed"></div>
  </div>
  <div class="panel" style="flex:0 0 240px;min-width:200px">
    <h2>Grader Score</h2>
    <div id="grader"><p class="st">Complete an episode to see score.</p></div>
  </div>
</div>
<script>
const proto = location.protocol === 'https:' ? 'wss' : 'ws';
let ws, active = false;
function connect() {
  ws = new WebSocket(proto + '://' + location.host + '/ws');
  ws.onopen = function() {
    document.getElementById('dot').className = 'dot on';
    document.getElementById('ctext').textContent = 'Connected via WebSocket';
    updateForm();
  };
  ws.onmessage = function(e) { handle(JSON.parse(e.data)); };
  ws.onclose = function() {
    document.getElementById('dot').className = 'dot';
    document.getElementById('ctext').textContent = 'Reconnecting\u2026';
    active = false;
    setTimeout(connect, 2000);
  };
  ws.onerror = function() {};
}
function handle(m) {
  if (m.type === 'error') { feed('\u26a0\ufe0f ' + m.detail, 'neg'); return; }
  if (m.type === 'reset' || m.type === 'step') {
    var obs = m.type === 'reset' ? m : m.observation;
    active = true;
    updateObs(obs);
    if (m.type === 'step') {
      var r = m.reward, cls = r.value >= 0 ? 'pos' : 'neg';
      feed(r.message + '  [' + (r.value >= 0 ? '+' : '') + r.value.toFixed(4) + ']', cls);
      if (obs.logs && Object.keys(obs.logs).length) showRevealed(obs.logs, obs.metrics);
      if (m.done) { feed('\u2705 Episode done \u2014 fetching score\u2026', 'inf'); ws.send(JSON.stringify({type:'grade'})); }
    } else {
      feed('\u25b6 Started: ' + (obs.task_id || ''), 'inf');
    }
  }
  if (m.type === 'grade') showGrade(m);
}
function updateObs(obs) {
  document.getElementById('snum').textContent = obs.step_number || 0;
  document.getElementById('smax').textContent = obs.max_steps || '?';
  document.getElementById('rew').textContent = (obs.cumulative_reward || 0).toFixed(4);
  document.getElementById('istatus').textContent = obs.incident_status || '\u2014';
  document.getElementById('isev').textContent = obs.severity_classified || '(unclassified)';
  var al = (obs.alerts || []).map(function(a) {
    return '<div class="alert ' + a.severity + '">[' + a.severity + '] <b>' + a.service + '</b>: ' + a.message + '</div>';
  }).join('');
  document.getElementById('alerts').innerHTML = al || '<p class="st">No alerts.</p>';
  var inv = obs.investigated_services || [];
  var sv = (obs.available_services || []).map(function(s) {
    return '<span class="tag' + (inv.indexOf(s) >= 0 ? ' done' : '') + '">' + s + (inv.indexOf(s) >= 0 ? ' \u2713' : '') + '</span>';
  }).join('');
  document.getElementById('services').innerHTML = sv;
}
function showRevealed(logs, metrics) {
  var h = '';
  for (var s in logs) {
    h += '<b style="color:#58a6ff">' + s + '</b><br>';
    (logs[s] || []).forEach(function(e) {
      var c = e.level === 'ERROR' ? '#f85149' : e.level === 'WARN' ? '#d29922' : '#6e7681';
      h += '<span style="color:' + c + '">[' + e.level + ']</span> ' + e.message + '<br>';
    });
  }
  for (var svc in (metrics || {})) {
    var mm = metrics[svc];
    h += '<b style="color:#d29922">' + svc + '</b>: CPU ' + mm.cpu_percent + '% Mem ' + mm.memory_percent + '% Err ' + (mm.error_rate * 100).toFixed(1) + '%<br>';
  }
  document.getElementById('revealed').innerHTML = h || '<p class="st">No data.</p>';
}
function showGrade(m) {
  var sc = m.score || 0, pct = (sc * 100).toFixed(1);
  var col = sc >= 0.8 ? '#2ea043' : sc >= 0.5 ? '#d29922' : '#f85149';
  var h = '<div class="score" style="color:' + col + '">' + pct + '%</div>';
  h += '<div class="bar"><div class="bar-fill" style="width:' + pct + '%;background:' + col + '"></div></div>';
  for (var k in (m.breakdown || {})) {
    h += '<div class="st">' + k + ': <b>' + (m.breakdown[k] * 100).toFixed(1) + '%</b></div>';
  }
  if (m.feedback) h += '<p style="margin-top:8px;font-size:11px;color:#e6edf3">' + m.feedback + '</p>';
  document.getElementById('grader').innerHTML = h;
}
function feed(txt, cls) {
  var f = document.getElementById('feed'), d = document.createElement('div');
  d.className = 'fi ' + cls;
  d.textContent = new Date().toLocaleTimeString('en-US', {hour12:false}) + ' \u2014 ' + txt;
  f.insertBefore(d, f.firstChild);
}
function g(id) { var e = document.getElementById(id); return e ? e.value : ''; }
function updateForm() {
  var t = g('atype');
  var f = {
    investigate: '<label>Service to investigate</label><input id="p_target" placeholder="e.g. redis-session">',
    classify: '<label>Severity</label><select id="p_sev"><option>P1</option><option>P2</option><option>P3</option><option>P4</option></select>',
    diagnose: '<label>Service (root cause)</label><input id="p_target" placeholder="e.g. auth-service"><label>Root cause description</label><input id="p_rc" placeholder="Describe the root cause\u2026">',
    remediate: '<label>Service</label><input id="p_target" placeholder="e.g. auth-service"><label>Action</label><select id="p_ract"><option>restart</option><option>rollback</option><option>scale</option><option>config_change</option></select>',
    escalate: '<label>Team</label><input id="p_target" placeholder="e.g. platform-team"><label>Priority</label><select id="p_pri"><option>urgent</option><option>high</option><option>medium</option></select><label>Message</label><input id="p_emsg" placeholder="Escalation message\u2026">',
    communicate: '<label>Channel</label><select id="p_ch"><option>status_page</option><option>slack</option><option>email</option></select><label>Message</label><input id="p_cmsg" placeholder="Status update\u2026">'
  };
  document.getElementById('aform').innerHTML = f[t] || '';
}
function doReset() {
  if (!ws || ws.readyState !== 1) { alert('Not connected'); return; }
  document.getElementById('feed').innerHTML = '';
  document.getElementById('revealed').innerHTML = '<p class="st">Investigate a service to see data.</p>';
  document.getElementById('grader').innerHTML = '<p class="st">Complete an episode to see score.</p>';
  ws.send(JSON.stringify({type:'reset', task_id: g('task'), variant_seed: 0}));
}
function doStep() {
  if (!ws || ws.readyState !== 1) { alert('Not connected'); return; }
  if (!active) { alert('Start an episode first'); return; }
  var t = g('atype');
  var a = {action_type: t, reasoning: g('reasoning'), parameters: {}, target: ''};
  if (t === 'investigate') a.target = g('p_target');
  else if (t === 'classify') a.parameters = {severity: g('p_sev')};
  else if (t === 'diagnose') { a.target = g('p_target'); a.parameters = {root_cause: g('p_rc')}; }
  else if (t === 'remediate') { a.target = g('p_target'); a.parameters = {action: g('p_ract')}; }
  else if (t === 'escalate') { a.target = g('p_target'); a.parameters = {priority: g('p_pri'), message: g('p_emsg')}; }
  else if (t === 'communicate') { a.target = g('p_ch'); a.parameters = {message: g('p_cmsg')}; }
  ws.send(JSON.stringify({type:'step', action: a}));
}
connect();
updateForm();
</script>
</body></html>"""


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket persistent session — one isolated env instance per connection.

    Message protocol (JSON):
      Client sends: {"type": "reset", "task_id": "...", "variant_seed": 0}
      Client sends: {"type": "step",  "action": {action_type, target, parameters, reasoning}}
      Client sends: {"type": "state"}
      Client sends: {"type": "grade"}

    Server replies: {"type": "reset"|"step"|"state"|"grade"|"error", ...payload}

    No X-Session-ID header needed — the connection itself is the session.
    """
    global _WS_ACTIVE_CONNECTIONS
    await websocket.accept()
    env = IncidentResponseEnv()
    _WS_ACTIVE_CONNECTIONS += 1
    _TELEMETRY["ws_connections_total"] += 1
    _log.info("ws connected — active=%d", _WS_ACTIVE_CONNECTIONS)
    try:
        while True:
            raw = await websocket.receive_json()
            msg_type = raw.get("type", "")

            if msg_type == "reset":
                task_id = raw.get("task_id", "severity_classification")
                seed = raw.get("variant_seed")
                seed = seed if seed is not None else secrets.randbelow(100)
                try:
                    obs = env.reset(task_id, variant_seed=seed)
                    _TELEMETRY["episodes_total"] += 1
                    await websocket.send_json({"type": "reset", **obs.model_dump(mode="json")})
                except ValueError as exc:
                    await websocket.send_json({"type": "error", "detail": str(exc)})

            elif msg_type == "step":
                action_data = raw.get("action", {})
                try:
                    action = Action(**action_data)
                    result: StepResult = env.step(action)
                    _TELEMETRY["steps_total"] += 1
                    await websocket.send_json({"type": "step", **result.model_dump(mode="json")})
                except (RuntimeError, Exception) as exc:
                    _TELEMETRY["errors_total"] += 1
                    await websocket.send_json({"type": "error", "detail": str(exc)})

            elif msg_type == "state":
                try:
                    await websocket.send_json({"type": "state", **env.state().model_dump(mode="json")})
                except RuntimeError as exc:
                    await websocket.send_json({"type": "error", "detail": str(exc)})

            elif msg_type == "grade":
                try:
                    result = env.grade()
                    _TELEMETRY["grader_calls"] += 1
                    s = env.state()
                    _record_leaderboard(s.task_id, result.score, s.total_steps_taken)
                    await websocket.send_json({"type": "grade", **result.model_dump(mode="json")})
                except RuntimeError as exc:
                    await websocket.send_json({"type": "error", "detail": str(exc)})

            else:
                await websocket.send_json({
                    "type": "error",
                    "detail": f"Unknown type '{msg_type}'. Supported: reset, step, state, grade",
                })

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        _TELEMETRY["errors_total"] += 1
        try:
            await websocket.send_json({"type": "error", "detail": str(exc)})
        except Exception:
            pass
    finally:
        _WS_ACTIVE_CONNECTIONS -= 1
        _log.info("ws disconnected — active=%d", _WS_ACTIVE_CONNECTIONS)


@app.get("/web", response_class=HTMLResponse)
async def web_ui():
    """Interactive browser-based incident dashboard (uses WebSocket under the hood)."""
    return HTMLResponse(_WEB_UI_HTML)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
