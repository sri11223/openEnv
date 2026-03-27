"""FastAPI server exposing the OpenEnv API endpoints.

Endpoints:
    POST /reset          – Reset environment for a task
    POST /step           – Take an agent action
    GET  /state          – Get current environment state
    GET  /tasks          – List available tasks with action schema
    POST /grader         – Get grader score for current/completed episode
    POST /baseline       – Run baseline inference on all tasks
    GET  /               – Health check
"""

from __future__ import annotations

import os
import secrets
import traceback
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.environment import IncidentResponseEnv
from src.models import Action, StepResult
from src.tasks import get_all_tasks


# ---------------------------------------------------------------------------
# Session-isolated environment registry
# ---------------------------------------------------------------------------
# Each session_id maps to its own IncidentResponseEnv instance.
# Clients receive a session_id on /reset and must pass it via the
# X-Session-ID header (or query param) on subsequent calls.
# This makes the server safe for concurrent multi-agent evaluation.

_SESSION_REGISTRY: Dict[str, IncidentResponseEnv] = {}
_MAX_SESSIONS = 256  # cap to avoid unbounded memory growth


def _get_or_create_session(session_id: str | None) -> tuple[str, IncidentResponseEnv]:
    """Return (session_id, env). Creates a new session if id is None or unknown."""
    if session_id and session_id in _SESSION_REGISTRY:
        return session_id, _SESSION_REGISTRY[session_id]
    # New session
    if len(_SESSION_REGISTRY) >= _MAX_SESSIONS:
        # Evict the oldest session (simple FIFO)
        oldest = next(iter(_SESSION_REGISTRY))
        del _SESSION_REGISTRY[oldest]
    new_id = session_id or secrets.token_hex(16)
    _SESSION_REGISTRY[new_id] = IncidentResponseEnv()
    return new_id, _SESSION_REGISTRY[new_id]


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


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
    task_id: str
    session_id: str | None = None
    variant_seed: int | None = None


class BaselineResponse(BaseModel):
    results: list
    summary: Dict[str, Any]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def health():
    """Health check – returns 200 with environment info."""
    return {
        "status": "ok",
        "environment": "incident-response-triage",
        "version": "1.0.0",
        "tasks": [t.task_id for t in get_all_tasks()],
        "active_sessions": len(_SESSION_REGISTRY),
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
        return result.model_dump()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
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
        return result.model_dump()
    except RuntimeError as exc:
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
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
