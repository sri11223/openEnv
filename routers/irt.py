# -*- coding: utf-8 -*-
"""IRT (Incident Response Triage) API endpoints.

Extracted from app.py - handles /reset, /step, /state, /tasks, /grader, /baseline.
"""

from __future__ import annotations

import secrets
import traceback
from typing import Any, Dict

from fastapi import APIRouter, Body, Header, HTTPException
from pydantic import BaseModel

from src.environment import IncidentResponseEnv
from src.models import Action, StepResult
from src.tasks import get_all_tasks

from routers.deps import (
    _SESSION_REGISTRY,
    _TELEMETRY,
    _log,
    get_or_create_session,
    record_leaderboard,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / response schemas
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

@router.post("/reset")
async def reset(request: ResetRequest | None = Body(default=None)):
    """Reset the environment for a given task_id.

    Returns the initial observation plus a `session_id` that must be
    passed via the `X-Session-ID` header on all subsequent calls.
    """
    if request is None:
        request = ResetRequest()
    try:
        session_id, env = get_or_create_session(request.session_id)
        # When no variant_seed is supplied randomise for anti-memorization;
        # explicit 0 keeps the primary (deterministic) scenario.
        seed = request.variant_seed if request.variant_seed is not None else secrets.randbelow(100)
        obs = env.reset(request.task_id, variant_seed=seed)
        _TELEMETRY["episodes_total"] += 1
        _log.info("episode reset task=%s session=%s variant=%d", request.task_id, session_id[:8], seed)
        data = obs.model_dump(mode="json")
        data["session_id"] = session_id
        return data
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/step")
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


@router.get("/state")
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


@router.get("/tasks")
async def tasks():
    """List all tasks with descriptions and action schema."""
    return [t.model_dump() for t in get_all_tasks()]


@router.post("/grader")
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
        record_leaderboard(state.task_id, result.score, state.total_steps_taken)
        _log.info("graded task=%s score=%.4f steps=%d", state.task_id, result.score, state.total_steps_taken)
        return result.model_dump()
    except RuntimeError as exc:
        _TELEMETRY["errors_total"] += 1
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/baseline")
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
