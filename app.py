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
import traceback
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.environment import IncidentResponseEnv
from src.models import Action, StepResult
from src.tasks import get_all_tasks


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

env = IncidentResponseEnv()


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
    }


@app.post("/reset")
async def reset(request: ResetRequest):
    """Reset the environment for a given task_id."""
    try:
        obs = env.reset(request.task_id)
        return obs.model_dump()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/step")
async def step(action: Action):
    """Execute one action and return observation, reward, done, info."""
    try:
        result: StepResult = env.step(action)
        return result.model_dump()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")


@app.get("/state")
async def state():
    """Return full environment state."""
    try:
        return env.state().model_dump()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/tasks")
async def tasks():
    """List all tasks with descriptions and action schema."""
    return [t.model_dump() for t in get_all_tasks()]


@app.post("/grader")
async def grader():
    """Return grader score for the current or most recent episode."""
    try:
        result = env.grade()
        return result.model_dump()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/baseline")
async def baseline():
    """Run the baseline inference script against all tasks.

    Requires OPENAI_API_KEY environment variable to be set.
    Falls back to a rule-based baseline if no API key is available.
    """
    try:
        from baseline.inference import run_all_tasks
        results = run_all_tasks(base_url=None, env_instance=env)
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
