# -*- coding: utf-8 -*-
"""Observability, metrics, dashboard, and WebSocket endpoints.

Extracted from app.py — handles /metrics, /render, /leaderboard, /curriculum,
/prometheus/*, /ws, /web, and /sentinel/dashboard.
"""

from __future__ import annotations

import secrets
import time
from typing import Any, Dict

from fastapi import APIRouter, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, PlainTextResponse

from src.environment import IncidentResponseEnv
from src.models import Action, StepResult

from routers.deps import (
    _LEADERBOARD,
    _SESSION_REGISTRY,
    _SENTINEL_REGISTRY,
    _SESSION_TTL,
    _TELEMETRY,
    _log,
    WS_ACTIVE_CONNECTIONS,
    record_leaderboard,
    scenario_live_to_prom_text,
    parse_prom_selector,
    build_prom_vector,
    build_prom_matrix,
)
import routers.deps as _deps

router = APIRouter()


# ---------------------------------------------------------------------------
# Metrics / telemetry
# ---------------------------------------------------------------------------

@router.get("/metrics")
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
        "max_sessions": 256,
    }


@router.get("/render")
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


@router.get("/leaderboard")
async def leaderboard():
    """Return top scores per task from all completed episodes in this session.

    Scores are ranked by (score DESC, steps ASC) — accuracy first, then efficiency.
    """
    return {
        task_id: board
        for task_id, board in _LEADERBOARD.items()
    }


@router.get("/curriculum")
async def curriculum():
    """Return the ordered curriculum learning progression for this environment.

    Tasks are listed from easiest to hardest so training agents can be
    scheduled to start from the first stage and progressively advance.
    Each stage carries the metadata needed to build a curriculum sampler:
    task_id, difficulty label, reward dimension count, step budget,
    temporal degradation rate, and number of distinct scenario variants.
    """
    return {
        "description": (
            "Curriculum from easy to hard: agents accumulate reward signal "
            "from the first episode and progressively face more complex scenarios."
        ),
        "stages": [
            {
                "stage": 1,
                "task_id": "severity_classification",
                "difficulty": "easy",
                "reward_components": 3,
                "max_steps": 10,
                "degradation_per_step": 0.005,
                "variants": 2,
                "graded_dimensions": ["severity_accuracy", "investigation_quality", "efficiency"],
                "rationale": (
                    "Introduces the action loop. Model must investigate then classify. "
                    "Guaranteed non-zero reward even with minimal exploration."
                ),
            },
            {
                "stage": 2,
                "task_id": "root_cause_analysis",
                "difficulty": "medium",
                "reward_components": 5,
                "max_steps": 15,
                "degradation_per_step": 0.010,
                "variants": 2,
                "graded_dimensions": [
                    "severity_accuracy", "investigated_root_cause",
                    "diagnosis_accuracy", "remediation_quality", "efficiency",
                ],
                "rationale": (
                    "Requires causal reasoning: distinguish root cause from downstream symptoms. "
                    "Adds diagnosis and remediation actions not present in stage 1."
                ),
            },
            {
                "stage": 3,
                "task_id": "full_incident_management",
                "difficulty": "hard",
                "reward_components": 8,
                "max_steps": 20,
                "degradation_per_step": 0.015,
                "variants": 3,
                "graded_dimensions": [
                    "severity_accuracy", "diagnosis_accuracy", "remediation_quality",
                    "escalation_quality", "communication", "investigation_thoroughness",
                    "investigation_precision", "efficiency",
                ],
                "rationale": (
                    "Full incident commander workflow requiring all 6 action types. "
                    "Includes red-herring services. Tests strategic investigation under "
                    "cascading blast-radius temporal pressure."
                ),
            },
        ],
    }


# ---------------------------------------------------------------------------
# Prometheus endpoints
# ---------------------------------------------------------------------------

@router.get("/prometheus/metrics")
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
    prom_text = scenario_live_to_prom_text(live, s.scenario_id, s.task_id, s.step_number)
    return PlainTextResponse(prom_text, media_type="text/plain; version=0.0.4")


@router.get("/prometheus/query")
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
    metric_name, label_filters = parse_prom_selector(query)
    vector = build_prom_vector(live, metric_name, label_filters, s.scenario_id, s.task_id)
    return {
        "status": "success",
        "data": {
            "resultType": "vector",
            "result": vector,
        },
    }


@router.get("/prometheus/query_range")
async def prometheus_range_query(
    query: str,
    start: float | None = None,
    end: float | None = None,
    step: float = 1.0,
    x_session_id: str | None = Header(default=None, alias="X-Session-ID"),
):
    """Prometheus range-query API (subset of /api/v1/query_range).

    Returns a standard Prometheus **matrix** response from the per-session
    TSDB ring buffer.  One sample is recorded per environment step, so the
    timeseries reflects real metric degradation over the episode lifetime.

    Parameters:
        query: PromQL selector (same syntax as /prometheus/query)
        start: Unix timestamp (inclusive). Defaults to episode start.
        end:   Unix timestamp (inclusive). Defaults to now.
        step:  Step duration seconds (accepted for API compatibility; ring buffer
               has one sample per episode step regardless).

    Example::

        GET /prometheus/query_range?query=irt_error_rate&start=1712500000&end=1712500060
    """
    if not x_session_id or x_session_id not in _SESSION_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail="Missing or unknown X-Session-ID. Call /reset first.",
        )
    env = _SESSION_REGISTRY[x_session_id]
    now = time.time()
    start_ts = start if start is not None else now - 3600
    end_ts = end if end is not None else now
    if start_ts > end_ts:
        raise HTTPException(status_code=400, detail="start must be <= end")
    history = env.metric_history(start_ts, end_ts, step_seconds=step)
    if history is None or (not history and env.live_metrics() == {}):
        raise HTTPException(status_code=400, detail="No active episode. Call /reset first.")
    s = env.state()
    metric_name, label_filters = parse_prom_selector(query)
    matrix = build_prom_matrix(history, metric_name, label_filters, s.scenario_id, s.task_id)
    return {
        "status": "success",
        "data": {
            "resultType": "matrix",
            "result": matrix,
        },
    }


# ---------------------------------------------------------------------------
# WebSocket endpoint — one env instance per connection, no session header
# ---------------------------------------------------------------------------

@router.websocket("/ws")
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
    await websocket.accept()
    env = IncidentResponseEnv()
    _deps.WS_ACTIVE_CONNECTIONS += 1
    _TELEMETRY["ws_connections_total"] += 1
    _log.info("ws connected — active=%d", _deps.WS_ACTIVE_CONNECTIONS)
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
                    record_leaderboard(s.task_id, result.score, s.total_steps_taken)
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
        _deps.WS_ACTIVE_CONNECTIONS -= 1
        _log.info("ws disconnected — active=%d", _deps.WS_ACTIVE_CONNECTIONS)
