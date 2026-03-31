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
    GET  /                   – Health check
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import time
import traceback
from contextlib import asynccontextmanager
from typing import Any, Dict, List

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
}

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
        "telemetry": _TELEMETRY,
    }


@app.post("/reset")
async def reset(request: ResetRequest):
    """Reset the environment for a given task_id.

    Returns the initial observation plus a `session_id` that must be
    passed via the `X-Session-ID` header on all subsequent calls.
    """
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
        from fastapi.responses import PlainTextResponse
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
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
