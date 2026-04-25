# -*- coding: utf-8 -*-
"""MCP (Model Context Protocol) Server for SENTINEL.

Exposes the SENTINEL OpenEnv environment as MCP-callable tools so that any
MCP-compatible agent (or the MCP Inspector) can interact with the environment
using the standard ``step / state / done`` tool interface.

Architecture (System Workflow slide):
    MCP Server (:9500) wraps OpenEnv env calls → registers with MCP-X Gateway

Transport: Streamable HTTP (``/mcp`` endpoint mounted in FastAPI).

Usage:
    # Standalone:
    python -m server.mcp_server

    # Via FastAPI mount (preferred):
    from server.mcp_server import mcp_router
    app.include_router(mcp_router, prefix="/mcp")
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from sentinel.environment import SentinelEnv

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MCP Session registry (one SentinelEnv per session)
# ---------------------------------------------------------------------------

_MCP_SESSIONS: Dict[str, SentinelEnv] = {}
_MCP_SESSION_META: Dict[str, Dict[str, Any]] = {}

MCP_SERVER_NAME = "sentinel-oversight-mcp"
MCP_SERVER_VERSION = "1.0.0"


def _get_or_create_session(session_id: Optional[str] = None) -> tuple[str, SentinelEnv]:
    """Get existing session or create a new one."""
    if session_id and session_id in _MCP_SESSIONS:
        return session_id, _MCP_SESSIONS[session_id]
    sid = session_id or str(uuid.uuid4())
    env = SentinelEnv()
    _MCP_SESSIONS[sid] = env
    _MCP_SESSION_META[sid] = {"created": True, "task_id": "basic_oversight"}
    return sid, env


# ---------------------------------------------------------------------------
# MCP Tool definitions (matching the MCP Inspector screenshot)
# ---------------------------------------------------------------------------

MCP_TOOLS = [
    {
        "name": "reset",
        "description": (
            "Reset the SENTINEL oversight environment for a new episode. "
            "Returns the initial observation including the first worker proposal "
            "that needs an oversight decision."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "Task to reset: basic_oversight, fleet_monitoring_conflict, adversarial_worker, multi_crisis_command",
                    "default": "basic_oversight",
                },
                "variant_seed": {
                    "type": "integer",
                    "description": "Deterministic seed for episode reproducibility",
                    "default": 0,
                },
            },
            "required": [],
        },
    },
    {
        "name": "step",
        "description": (
            "Submit an oversight decision for the current worker proposal. "
            "The decision determines whether the worker's proposed action is "
            "APPROVE'd, BLOCK'ed, REDIRECT'ed, REASSIGN'ed, or FLAG'ged. "
            "Returns the next observation, reward, and whether the episode is done."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "decision": {
                    "type": "string",
                    "description": "Oversight decision: APPROVE, BLOCK, REDIRECT, REASSIGN, or FLAG",
                    "enum": ["APPROVE", "BLOCK", "REDIRECT", "REASSIGN", "FLAG"],
                },
                "reason": {
                    "type": "string",
                    "description": "Why this decision was made (e.g., hallucination, safe, scope_violation)",
                },
                "explanation": {
                    "type": "string",
                    "description": "Detailed evidence-backed explanation for the oversight decision",
                },
                "worker_message": {
                    "type": "string",
                    "description": "Corrective feedback to send to the worker agent",
                    "default": "",
                },
            },
            "required": ["decision"],
        },
    },
    {
        "name": "state",
        "description": (
            "Get the current state of the SENTINEL environment including "
            "step number, cumulative reward, pending proposal, audit log, "
            "and worker rehabilitation records."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "done",
        "description": (
            "Check whether the current episode is complete. Returns true "
            "when all worker proposals have been processed or the step limit "
            "is reached."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]

# ---------------------------------------------------------------------------
# MCP JSON-RPC 2.0 handler
# ---------------------------------------------------------------------------


class MCPRequest(BaseModel):
    """JSON-RPC 2.0 request for MCP."""
    jsonrpc: str = "2.0"
    id: Optional[Any] = None
    method: str
    params: Optional[Dict[str, Any]] = None


def _jsonrpc_response(id: Any, result: Any) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": id, "result": result}


def _jsonrpc_error(id: Any, code: int, message: str) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}}


def _handle_initialize(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handle MCP initialize request."""
    return {
        "protocolVersion": "2024-11-05",
        "capabilities": {
            "tools": {"listChanged": False},
        },
        "serverInfo": {
            "name": MCP_SERVER_NAME,
            "version": MCP_SERVER_VERSION,
        },
    }


def _handle_tools_list(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handle tools/list — return all available tools."""
    return {"tools": MCP_TOOLS}


def _handle_tools_call(
    params: Dict[str, Any],
    session_id: str,
) -> Dict[str, Any]:
    """Handle tools/call — execute a tool and return the result."""
    tool_name = params.get("name", "")
    arguments = params.get("arguments", {})

    sid, env = _get_or_create_session(session_id)

    try:
        if tool_name == "reset":
            task_id = arguments.get("task_id", "basic_oversight")
            variant_seed = arguments.get("variant_seed", 0)
            obs = env.reset(task_id, variant_seed=variant_seed)
            _MCP_SESSION_META[sid] = {"task_id": task_id, "has_reset": True}
            result_text = json.dumps(_observation_to_dict(obs), indent=2)

        elif tool_name == "step":
            decision_payload = {
                "decision": arguments.get("decision", "APPROVE"),
                "reason": arguments.get("reason", ""),
                "explanation": arguments.get("explanation", ""),
                "worker_message": arguments.get("worker_message", ""),
            }
            result = env.step(decision_payload)
            result_text = json.dumps({
                "done": result.done,
                "reward": round(float(result.sentinel_reward.total), 4),
                "reward_breakdown": {
                    k: round(float(v), 4) for k, v in
                    (result.sentinel_reward.breakdown or {}).items()
                },
                "observation": _observation_to_dict(result.observation),
                "info": _safe_info(result.info),
            }, indent=2)

        elif tool_name == "state":
            state = env.state()
            result_text = json.dumps({
                "task_id": state.task_id,
                "step_number": state.step_number,
                "max_steps": state.max_steps,
                "cumulative_reward": round(float(state.cumulative_reward), 4),
                "done": state.done,
                "pending_proposal": (
                    state.pending_proposal.model_dump(mode="json")
                    if state.pending_proposal else None
                ),
                "audit_log_length": len(state.audit_log),
                "worker_records": {
                    wid: rec.model_dump(mode="json")
                    for wid, rec in state.worker_records.items()
                },
            }, indent=2)

        elif tool_name == "done":
            state = env.state()
            result_text = json.dumps({
                "done": state.done,
                "step_number": state.step_number,
                "max_steps": state.max_steps,
            }, indent=2)

        else:
            return {
                "content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}],
                "isError": True,
            }

        return {
            "content": [{"type": "text", "text": result_text}],
            "isError": False,
        }

    except Exception as exc:
        logger.exception("MCP tool call failed: %s", tool_name)
        return {
            "content": [{"type": "text", "text": f"Error: {exc}"}],
            "isError": True,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _observation_to_dict(obs: Any) -> Dict[str, Any]:
    """Convert a SentinelEnv observation to a JSON-safe dict."""
    proposal = {}
    if getattr(obs, "proposed_action", None) is not None:
        try:
            proposal = obs.proposed_action.model_dump(mode="json")
        except Exception:
            proposal = {"raw": str(obs.proposed_action)}

    return {
        "task_id": getattr(obs, "task_id", ""),
        "step_number": getattr(obs, "step_number", 0),
        "max_steps": getattr(obs, "max_steps", 0),
        "proposed_action": proposal,
        "worker_id": getattr(obs, "worker_id", None),
        "worker_role": getattr(obs, "worker_role", None),
        "incident_status": getattr(obs, "incident_status", None),
        "available_decisions": list(getattr(obs, "available_decisions", []) or []),
        "message": getattr(obs, "message", ""),
    }


def _safe_info(info: Any) -> Dict[str, Any]:
    """Make info dict JSON-serializable."""
    if info is None:
        return {}
    try:
        json.dumps(info)
        return info
    except (TypeError, ValueError):
        return {"raw": str(info)}


# ---------------------------------------------------------------------------
# FastAPI router implementing MCP Streamable HTTP transport
# ---------------------------------------------------------------------------

mcp_router = APIRouter(tags=["MCP"])


@mcp_router.post("")
@mcp_router.post("/")
async def mcp_endpoint(request: Request):
    """MCP Streamable HTTP endpoint.

    Handles JSON-RPC 2.0 requests for the Model Context Protocol.
    Supports: initialize, tools/list, tools/call, notifications/initialized.
    """
    body = await request.json()

    # Handle batch requests
    if isinstance(body, list):
        responses = []
        for item in body:
            resp = _process_single_request(item, request)
            if resp is not None:
                responses.append(resp)
        return JSONResponse(responses if responses else {"jsonrpc": "2.0", "id": None, "result": {}})

    result = _process_single_request(body, request)
    if result is None:
        # Notification — no response needed, but return empty for HTTP
        return JSONResponse({"jsonrpc": "2.0", "id": None, "result": {}})
    return JSONResponse(result)


def _process_single_request(body: Dict[str, Any], request: Request) -> Optional[Dict[str, Any]]:
    """Process a single JSON-RPC 2.0 MCP request."""
    method = body.get("method", "")
    params = body.get("params", {}) or {}
    req_id = body.get("id")

    # Extract or generate session ID
    session_id = request.headers.get("x-mcp-session-id", str(uuid.uuid4()))

    # Notifications (no id) — don't require a response
    if req_id is None and method in ("notifications/initialized",):
        logger.info("MCP notification: %s", method)
        return None

    if method == "initialize":
        result = _handle_initialize(params)
        resp = _jsonrpc_response(req_id, result)
        return resp

    elif method == "tools/list":
        result = _handle_tools_list(params)
        return _jsonrpc_response(req_id, result)

    elif method == "tools/call":
        result = _handle_tools_call(params, session_id)
        return _jsonrpc_response(req_id, result)

    elif method in ("notifications/initialized",):
        return None

    else:
        return _jsonrpc_error(req_id, -32601, f"Method not found: {method}")


# ---------------------------------------------------------------------------
# MCP Server info endpoint (for discovery)
# ---------------------------------------------------------------------------

@mcp_router.get("/info")
async def mcp_info():
    """MCP server information for discovery and registration."""
    return {
        "name": MCP_SERVER_NAME,
        "version": MCP_SERVER_VERSION,
        "protocol_version": "2024-11-05",
        "transport": "streamable-http",
        "tools": [t["name"] for t in MCP_TOOLS],
        "description": (
            "SENTINEL Oversight Command MCP Server. "
            "Exposes AI oversight environment tools (reset, step, state, done) "
            "for MCP-compatible agents and the MCP Inspector."
        ),
    }


# ---------------------------------------------------------------------------
# Standalone entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI

    standalone = FastAPI(title="SENTINEL MCP Server")
    standalone.include_router(mcp_router, prefix="/mcp")

    print(f"MCP Server starting on http://localhost:9500/mcp")
    print(f"Connect MCP Inspector to: http://localhost:9500/mcp")
    uvicorn.run(standalone, host="0.0.0.0", port=9500)
