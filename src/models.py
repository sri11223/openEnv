"""Typed Pydantic models for the Incident Response Triage environment.

Defines the full Observation, Action, and Reward schema used by
step() / reset() / state() and validated by openenv validate.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Primitive domain objects
# ---------------------------------------------------------------------------

class AlertSeverity(str, Enum):
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class Alert(BaseModel):
    alert_id: str
    service: str
    severity: AlertSeverity
    message: str
    timestamp: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LogEntry(BaseModel):
    timestamp: str
    service: str
    level: str  # ERROR, WARN, INFO, DEBUG
    message: str
    trace_id: Optional[str] = None


class ServiceMetrics(BaseModel):
    service: str
    cpu_percent: float
    memory_percent: float
    request_rate: float   # req/s
    error_rate: float     # fraction 0-1
    latency_p50_ms: float
    latency_p99_ms: float
    custom: Dict[str, float] = Field(default_factory=dict)


class IncidentSeverity(str, Enum):
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"
    P4 = "P4"


class IncidentStatus(str, Enum):
    OPEN = "open"
    INVESTIGATING = "investigating"
    MITIGATING = "mitigating"
    RESOLVED = "resolved"


# ---------------------------------------------------------------------------
# Action model
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    CLASSIFY = "classify"
    INVESTIGATE = "investigate"
    DIAGNOSE = "diagnose"
    REMEDIATE = "remediate"
    ESCALATE = "escalate"
    COMMUNICATE = "communicate"


class RemediationAction(str, Enum):
    RESTART = "restart"
    ROLLBACK = "rollback"
    SCALE = "scale"
    CONFIG_CHANGE = "config_change"


class Action(BaseModel):
    """Agent action submitted to step()."""
    action_type: ActionType
    target: Optional[str] = Field(
        None,
        description="Service name, team name, or channel depending on action_type.",
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Action-specific parameters (e.g. severity level, root_cause text).",
    )
    reasoning: str = Field(
        "",
        description="Free-text field for the agent to explain its reasoning.",
    )


# ---------------------------------------------------------------------------
# Observation model
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """Returned by reset() and step(). Represents what the agent can see."""
    incident_id: str
    timestamp: str
    step_number: int
    max_steps: int
    task_id: str
    task_description: str
    # Alert information (always visible)
    alerts: List[Alert]
    available_services: List[str]
    # Progressive disclosure – populated as agent investigates
    investigated_services: List[str] = Field(default_factory=list)
    logs: Dict[str, List[LogEntry]] = Field(default_factory=dict)
    metrics: Dict[str, ServiceMetrics] = Field(default_factory=dict)
    # Incident tracking
    incident_status: IncidentStatus = IncidentStatus.OPEN
    severity_classified: Optional[IncidentSeverity] = None
    diagnosis: Optional[str] = None
    # Action history
    actions_taken: List[str] = Field(default_factory=list)
    remediations_applied: List[str] = Field(default_factory=list)
    escalations_made: List[str] = Field(default_factory=list)
    communications_sent: List[str] = Field(default_factory=list)
    # Feedback
    message: str = "Incident opened. Review alerts and begin investigation."


# ---------------------------------------------------------------------------
# Reward model
# ---------------------------------------------------------------------------

class Reward(BaseModel):
    """Returned alongside each observation from step()."""
    value: float = Field(..., ge=-1.0, le=1.0)
    components: Dict[str, float] = Field(default_factory=dict)
    message: str = ""


# ---------------------------------------------------------------------------
# Composite return types
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class EnvironmentState(BaseModel):
    """Returned by state(). Full internal snapshot (for debugging / grading)."""
    task_id: str
    scenario_id: str
    step_number: int
    max_steps: int
    incident_status: IncidentStatus
    done: bool
    cumulative_reward: float
    total_steps_taken: int
    alerts: List[Alert] = Field(default_factory=list)
    actions_history: List[Dict[str, Any]] = Field(default_factory=list)
    severity_classified: Optional[IncidentSeverity] = None
    diagnosis: Optional[str] = None
    remediations_applied: List[str] = Field(default_factory=list)
    escalations_made: List[str] = Field(default_factory=list)
    communications_sent: List[str] = Field(default_factory=list)
    investigated_services: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Task / grader info (used by /tasks and /grader endpoints)
# ---------------------------------------------------------------------------

class TaskInfo(BaseModel):
    task_id: str
    name: str
    difficulty: str
    description: str
    max_steps: int
    action_schema: Dict[str, Any]


class GraderResult(BaseModel):
    task_id: str
    score: float = Field(..., ge=0.0, le=1.0)
    breakdown: Dict[str, float] = Field(default_factory=dict)
    feedback: str = ""


class BaselineResult(BaseModel):
    task_id: str
    score: float
    steps_taken: int
    grader_breakdown: Dict[str, float] = Field(default_factory=dict)
