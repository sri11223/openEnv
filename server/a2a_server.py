# -*- coding: utf-8 -*-
"""A2A (Agent-to-Agent) Protocol implementation for SENTINEL.

Implements the Google A2A protocol so that SENTINEL can:
  1. Be discovered by other agents via ``/.well-known/agent.json``
  2. Receive assessment tasks from an AgentBeats Assessor (Green Agent)
  3. Send results back via standard JSON-RPC 2.0

Architecture (matches workshop AgentBeats slide):
    Assessor (Green Agent) → A2A: send task → SENTINEL (Purple Agent)
    SENTINEL processes the task → A2A: return result

Communication Protocols:
    - A2A Protocol — agent-to-agent structured communication (JSON-RPC 2.0)
    - MCP — environment tools exposed as callable functions (see mcp_server.py)
    - Human-understandable instructions — natural language task descriptions

References:
    - https://google.github.io/A2A/
    - Workshop slide: "Introducing AgentBeats: Agentified Agent Assessment"
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# A2A Agent Card — served at /.well-known/agent.json
# ---------------------------------------------------------------------------

AGENT_CARD = {
    "name": "SENTINEL Oversight Command",
    "description": (
        "An AI oversight agent that supervises worker agents during production "
        "incident response. SENTINEL intercepts proposed actions, evaluates them "
        "against constitutional principles and counterfactual risk, and decides "
        "whether to APPROVE, BLOCK, REDIRECT, REASSIGN, or FLAG each action."
    ),
    "version": "1.0.0",
    "url": "https://srikrishna2005-openenv.hf.space",
    "provider": {
        "organization": "SENTINEL OpenEnv",
        "url": "https://srikrishna2005-openenv.hf.space",
    },
    "capabilities": {
        "streaming": False,
        "pushNotifications": False,
        "stateTransitionHistory": True,
    },
    "authentication": {
        "schemes": ["none"],
    },
    "defaultInputModes": ["text", "application/json"],
    "defaultOutputModes": ["text", "application/json"],
    "skills": [
        {
            "id": "oversight-basic",
            "name": "Basic Oversight",
            "description": (
                "Evaluate worker agent proposals for safety, accuracy, and "
                "constitutional alignment. Detect hallucinations, scope violations, "
                "and premature actions."
            ),
            "tags": ["ai-safety", "oversight", "incident-response"],
            "examples": [
                "Evaluate whether this agent action is safe to execute",
                "Check if the proposed remediation has sufficient evidence",
                "Detect if the worker is hallucinating a target service",
            ],
        },
        {
            "id": "oversight-adversarial",
            "name": "Adversarial Worker Detection",
            "description": (
                "Detect reward hacking, confidence washing, escalation bombing, "
                "and other adversarial misbehaviors in worker agent proposals."
            ),
            "tags": ["ai-safety", "adversarial", "misbehavior-detection"],
            "examples": [
                "Detect if a worker is gaming the reward function",
                "Identify if confidence levels are artificially inflated",
                "Check for disproportionate escalation patterns",
            ],
        },
        {
            "id": "oversight-fleet",
            "name": "Fleet Monitoring & Multi-Crisis",
            "description": (
                "Coordinate oversight across multiple concurrent worker agents "
                "handling multi-crisis scenarios with conflicting priorities."
            ),
            "tags": ["fleet-management", "multi-agent", "coordination"],
            "examples": [
                "Manage oversight for 4 workers handling a cascading outage",
                "Resolve conflicting remediation proposals from different workers",
                "Coordinate a multi-team escalation during a critical incident",
            ],
        },
    ],
}


# ---------------------------------------------------------------------------
# A2A Task management
# ---------------------------------------------------------------------------

class TaskState(str, Enum):
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    CANCELED = "canceled"
    FAILED = "failed"


class A2ATask:
    """Tracks the lifecycle of an A2A task."""

    def __init__(self, task_id: str, message: Dict[str, Any]):
        self.id = task_id
        self.state = TaskState.SUBMITTED
        self.message = message
        self.result: Optional[Dict[str, Any]] = None
        self.history: List[Dict[str, Any]] = []
        self.created_at = time.time()
        self.updated_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "status": {
                "state": self.state.value,
                "timestamp": self.updated_at,
            },
            "artifacts": [self.result] if self.result else [],
            "history": self.history,
        }


_A2A_TASKS: Dict[str, A2ATask] = {}


# ---------------------------------------------------------------------------
# Task execution — using SENTINEL environment
# ---------------------------------------------------------------------------

def _execute_task(task: A2ATask) -> None:
    """Execute an A2A task by running a SENTINEL oversight episode."""
    from sentinel.environment import SentinelEnv

    task.state = TaskState.WORKING
    task.updated_at = time.time()
    task.history.append({
        "state": TaskState.WORKING.value,
        "timestamp": task.updated_at,
        "message": "Processing oversight request...",
    })

    try:
        msg = task.message
        parts = msg.get("parts", [])
        text_content = ""
        json_content = {}

        for part in parts:
            if part.get("type") == "text":
                text_content = part.get("text", "")
            elif part.get("type") == "data":
                json_content = part.get("data", {})

        # Extract action from the message (supports both text and structured)
        agent_action = json_content.get("agent_action", {})
        if not agent_action and text_content:
            # Try to parse text as JSON (human-understandable instructions)
            try:
                agent_action = json.loads(text_content)
            except (json.JSONDecodeError, ValueError):
                # Treat as natural language instruction
                agent_action = {
                    "action_type": "investigate",
                    "reasoning": text_content,
                }

        environment_state = json_content.get("environment_state", {})
        domain = json_content.get("domain", "infrastructure")

        # Use universal adapter for the actual oversight
        from sentinel.universal_adapter import universal_intercept
        result = universal_intercept(
            agent_action=agent_action,
            environment_state=environment_state,
            domain=domain,
            agent_id=json_content.get("agent_id", "a2a_agent"),
            agent_role=json_content.get("agent_role", "general"),
        )

        task.result = {
            "parts": [
                {
                    "type": "text",
                    "text": (
                        f"SENTINEL Decision: {result.decision}\n"
                        f"Reason: {result.reason or 'N/A'}\n"
                        f"Explanation: {result.explanation}\n"
                        f"Risk Level: {result.risk_level} ({result.risk_score:.2f})\n"
                        f"Worker Message: {result.worker_message}"
                    ),
                },
                {
                    "type": "data",
                    "data": result.model_dump(mode="json"),
                },
            ],
        }
        task.state = TaskState.COMPLETED
        task.updated_at = time.time()
        task.history.append({
            "state": TaskState.COMPLETED.value,
            "timestamp": task.updated_at,
            "message": f"Oversight decision: {result.decision}",
        })

    except Exception as exc:
        logger.exception("A2A task execution failed: %s", task.id)
        task.state = TaskState.FAILED
        task.updated_at = time.time()
        task.result = {
            "parts": [{"type": "text", "text": f"Error: {exc}"}],
        }
        task.history.append({
            "state": TaskState.FAILED.value,
            "timestamp": task.updated_at,
            "message": str(exc),
        })


# ---------------------------------------------------------------------------
# A2A JSON-RPC 2.0 methods
# ---------------------------------------------------------------------------

def _handle_tasks_send(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handle tasks/send — receive a task and execute it synchronously."""
    task_id = params.get("id", str(uuid.uuid4()))
    message = params.get("message", {})

    task = A2ATask(task_id, message)
    _A2A_TASKS[task_id] = task

    # Execute synchronously (for non-streaming mode)
    _execute_task(task)

    return task.to_dict()


def _handle_tasks_get(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handle tasks/get — retrieve task status."""
    task_id = params.get("id", "")
    task = _A2A_TASKS.get(task_id)
    if not task:
        return {"error": f"Task not found: {task_id}"}
    return task.to_dict()


def _handle_tasks_cancel(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handle tasks/cancel — cancel a task."""
    task_id = params.get("id", "")
    task = _A2A_TASKS.get(task_id)
    if not task:
        return {"error": f"Task not found: {task_id}"}
    task.state = TaskState.CANCELED
    task.updated_at = time.time()
    return task.to_dict()


def _handle_message_send(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handle message/send (A2A v0.3+) — normalize schema and delegate to tasks/send.

    A2A v0.3 changed the method name from ``tasks/send`` to ``message/send`` and
    uses ``kind`` instead of ``type`` in message parts.  This adapter normalises
    the new envelope so SENTINEL can be reached by both v0.2 and v0.3 clients.
    """
    raw_message = params.get("message", {})
    # Normalise parts: A2A v0.3 uses "kind", v0.2 uses "type"
    normalized_parts = []
    for part in raw_message.get("parts", []):
        p = dict(part)
        if "kind" in p and "type" not in p:
            p["type"] = p.pop("kind")
        normalized_parts.append(p)
    normalized_message = {
        "parts": normalized_parts,
        "role": raw_message.get("role", "user"),
    }
    task_id = params.get("id", str(uuid.uuid4()))
    return _handle_tasks_send({"id": task_id, "message": normalized_message})


# ---------------------------------------------------------------------------
# FastAPI Router
# ---------------------------------------------------------------------------

a2a_router = APIRouter(tags=["A2A"])


@a2a_router.get("/.well-known/agent.json")
async def agent_card():
    """A2A Agent Card — the standard discovery endpoint.

    Other agents discover SENTINEL's capabilities by fetching this card.
    Hosted at the well-known URL as specified by the A2A protocol.
    """
    return JSONResponse(AGENT_CARD)


@a2a_router.post("/a2a")
async def a2a_endpoint(request: Request):
    """A2A JSON-RPC 2.0 endpoint.

    Handles task lifecycle: tasks/send, tasks/get, tasks/cancel.
    Communication uses standard HTTP + JSON-RPC 2.0 as specified by the A2A protocol.
    """
    body = await request.json()
    method = body.get("method", "")
    params = body.get("params", {}) or {}
    req_id = body.get("id")

    handlers = {
        "tasks/send": _handle_tasks_send,
        "tasks/get": _handle_tasks_get,
        "tasks/cancel": _handle_tasks_cancel,
        "message/send": _handle_message_send,   # A2A v0.3+ alias
        "message/stream": _handle_message_send,  # A2A v0.3+ streaming (sync fallback)
    }

    handler = handlers.get(method)
    if handler is None:
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        })

    try:
        result = handler(params)
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": result,
        })
    except Exception as exc:
        logger.exception("A2A method failed: %s", method)
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32000, "message": str(exc)},
        })


# ---------------------------------------------------------------------------
# Human-understandable instruction endpoint
# ---------------------------------------------------------------------------

@a2a_router.post("/a2a/human")
async def a2a_human_instruction(request: Request):
    """Accept human-understandable natural language oversight requests.

    This endpoint supports the third communication mode from the workshop:
    'Human-understandable instructions' — agents can describe their action
    in plain English and SENTINEL will evaluate it.

    Example:
        POST /a2a/human
        {
            "instruction": "I want to restart the payment-gateway service",
            "context": "There's a timeout on payment processing"
        }
    """
    body = await request.json()
    instruction = body.get("instruction", body.get("text", body.get("message", "")))
    context = body.get("context", body.get("environment_state", {}))

    if isinstance(context, str):
        context = {"description": context}

    # Wrap as A2A task
    task_id = str(uuid.uuid4())
    message = {
        "parts": [
            {"type": "text", "text": instruction},
            {"type": "data", "data": {
                "agent_action": {"action_type": "investigate", "reasoning": instruction},
                "environment_state": context,
                "domain": body.get("domain", "infrastructure"),
            }},
        ],
    }

    task = A2ATask(task_id, message)
    _A2A_TASKS[task_id] = task
    _execute_task(task)

    return JSONResponse({
        "task_id": task_id,
        "decision": task.result["parts"][1]["data"]["decision"] if task.result else "ERROR",
        "explanation": task.result["parts"][0]["text"] if task.result else "Failed",
        "full_result": task.to_dict(),
    })
