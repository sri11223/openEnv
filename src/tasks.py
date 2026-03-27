"""Task definitions and action schema for the /tasks endpoint."""

from __future__ import annotations

from typing import Any, Dict, List

from src.models import ActionType, RemediationAction, TaskInfo
from src.scenarios import SCENARIOS


def _action_schema() -> Dict[str, Any]:
    """JSON Schema describing a valid Action payload."""
    return {
        "type": "object",
        "required": ["action_type"],
        "properties": {
            "action_type": {
                "type": "string",
                "enum": [a.value for a in ActionType],
                "description": "The type of action to take.",
            },
            "target": {
                "type": "string",
                "description": (
                    "For INVESTIGATE/DIAGNOSE/REMEDIATE: service name. "
                    "For ESCALATE: team name. "
                    "For COMMUNICATE: channel (status_page|slack|email)."
                ),
            },
            "parameters": {
                "type": "object",
                "description": "Action-specific parameters.",
                "properties": {
                    "severity": {
                        "type": "string",
                        "enum": ["P1", "P2", "P3", "P4"],
                        "description": "Required for CLASSIFY action.",
                    },
                    "root_cause": {
                        "type": "string",
                        "description": "Required for DIAGNOSE action. Free-text root cause description.",
                    },
                    "action": {
                        "type": "string",
                        "enum": [r.value for r in RemediationAction],
                        "description": "Required for REMEDIATE action.",
                    },
                    "priority": {
                        "type": "string",
                        "description": "Optional for ESCALATE. e.g. urgent, high, medium.",
                    },
                    "message": {
                        "type": "string",
                        "description": "Required for COMMUNICATE. Status update text.",
                    },
                },
            },
            "reasoning": {
                "type": "string",
                "description": "Free-text explanation of the agent's reasoning for this action.",
            },
        },
    }


_TASK_METADATA = {
    "severity_classification": {
        "name": "Severity Classification",
        "difficulty": "easy",
        "description": (
            "A production service is experiencing degradation. Review the "
            "alerts, investigate relevant services, and classify the incident "
            "severity (P1–P4). Score is based on classification accuracy, "
            "investigation quality, and efficiency."
        ),
    },
    "root_cause_analysis": {
        "name": "Root Cause Analysis",
        "difficulty": "medium",
        "description": (
            "Payment processing is failing. Multiple services show symptoms. "
            "Investigate to find the true root cause (not just symptoms), "
            "classify severity, diagnose the issue, and apply the correct "
            "remediation. Requires distinguishing root cause from downstream "
            "effects."
        ),
    },
    "full_incident_management": {
        "name": "Full Incident Management",
        "difficulty": "hard",
        "description": (
            "A cascading multi-service outage is in progress. Six alerts "
            "across eight services, including red herrings. You must: "
            "investigate strategically, classify severity, identify the root "
            "cause (a bad deployment), apply multiple remediations, escalate "
            "to the right teams, and communicate status updates. Tests the "
            "full incident commander workflow."
        ),
    },
}


def get_all_tasks() -> List[TaskInfo]:
    tasks = []
    schema = _action_schema()
    for task_id, meta in _TASK_METADATA.items():
        scenario = SCENARIOS[task_id]
        tasks.append(
            TaskInfo(
                task_id=task_id,
                name=meta["name"],
                difficulty=meta["difficulty"],
                description=meta["description"],
                max_steps=scenario.max_steps,
                action_schema=schema,
            )
        )
    return tasks
