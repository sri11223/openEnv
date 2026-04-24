"""Native OpenEnv adapter for the SENTINEL oversight environment.

The main FastAPI app keeps the richer hackathon demo endpoints, while this
adapter exposes the same oversight task through OpenEnv's Environment base
class so latest OpenEnv clients can discover schemas and use the standard
reset/step/state/WebSocket contract.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import Field

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import (
    Action as OpenEnvAction,
    EnvironmentMetadata,
    Observation as OpenEnvObservation,
    State as OpenEnvState,
)

from sentinel.environment import SentinelEnv


class SentinelNativeAction(OpenEnvAction):
    """OpenEnv-native action model for one SENTINEL oversight decision."""

    decision: Optional[str] = Field(
        default=None,
        description="Oversight decision: APPROVE, BLOCK, REDIRECT, REASSIGN, or FLAG.",
    )
    action: Optional[str] = Field(
        default=None,
        description="Alias for decision, accepted for client compatibility.",
    )
    reason: Optional[str] = Field(
        default=None,
        description="Misbehavior type or 'safe'.",
    )
    explanation: str = Field(
        default="",
        description="Evidence-backed explanation for the oversight decision.",
    )
    redirect_action: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Safer replacement action when decision is REDIRECT.",
    )
    reassign_to: Optional[str] = Field(
        default=None,
        description="Worker id to receive the work when decision is REASSIGN.",
    )
    flag_severity: Optional[str] = Field(
        default=None,
        description="Severity for FLAG decisions.",
    )
    worker_message: str = Field(
        default="",
        description="Corrective feedback sent to the worker.",
    )
    required_evidence: List[str] = Field(
        default_factory=list,
        description="Evidence required before a corrected proposal can execute.",
    )
    suggested_action_type: Optional[str] = None
    suggested_target: Optional[str] = None
    suggested_parameters: Dict[str, Any] = Field(default_factory=dict)
    constitutional_violations: List[str] = Field(default_factory=list)


class SentinelNativeObservation(OpenEnvObservation):
    """Observation returned by the native OpenEnv SENTINEL adapter."""

    task_id: str = "basic_oversight"
    step_number: int = 0
    max_steps: int = 0
    proposed_action: Dict[str, Any] = Field(default_factory=dict)
    worker_id: Optional[str] = None
    worker_role: Optional[str] = None
    incident_status: Optional[str] = None
    available_decisions: List[str] = Field(default_factory=list)
    corrective_loop_enabled: bool = False
    audit_log_tail: List[Dict[str, Any]] = Field(default_factory=list)
    message: str = ""


class SentinelNativeState(OpenEnvState):
    """State snapshot for the native OpenEnv SENTINEL adapter."""

    task_id: Optional[str] = None
    cumulative_reward: float = 0.0
    done: bool = False
    latest_proposal: Dict[str, Any] = Field(default_factory=dict)
    latest_audit: Optional[Dict[str, Any]] = None
    worker_records: Dict[str, Any] = Field(default_factory=dict)


class SentinelNativeEnvironment(
    Environment[SentinelNativeAction, SentinelNativeObservation, SentinelNativeState]
):
    """OpenEnv Environment wrapper around :class:`sentinel.environment.SentinelEnv`."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        super().__init__()
        self._env = SentinelEnv()
        self._episode_id = str(uuid4())
        self._task_id = "basic_oversight"
        self._has_reset = False

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: str = "basic_oversight",
        variant_seed: Optional[int] = None,
        **_: Any,
    ) -> SentinelNativeObservation:
        self._episode_id = episode_id or str(uuid4())
        self._task_id = task_id
        resolved_seed = variant_seed if variant_seed is not None else (seed or 0)
        obs = self._env.reset(task_id, variant_seed=resolved_seed)
        self._has_reset = True
        return self._to_observation(obs, reward=None, done=False)

    def step(
        self,
        action: SentinelNativeAction,
        timeout_s: Optional[float] = None,
        **_: Any,
    ) -> SentinelNativeObservation:
        if not self._has_reset:
            self.reset(task_id=self._task_id)

        payload = action.model_dump(exclude={"metadata"}, exclude_none=True)
        if not payload.get("decision") and payload.get("action"):
            payload["decision"] = payload["action"]

        result = self._env.step(payload)
        return self._to_observation(
            result.observation,
            reward=result.sentinel_reward.total,
            done=result.done,
            info=result.info,
        )

    @property
    def state(self) -> SentinelNativeState:
        if not self._has_reset:
            return SentinelNativeState(
                episode_id=self._episode_id,
                step_count=0,
                task_id=self._task_id,
            )

        state = self._env.state()
        latest_audit = state.audit_log[-1].model_dump(mode="json") if state.audit_log else None
        latest_proposal = (
            state.pending_proposal.model_dump(mode="json")
            if state.pending_proposal is not None
            else {}
        )
        return SentinelNativeState(
            episode_id=self._episode_id,
            step_count=state.step_number,
            task_id=state.task_id,
            cumulative_reward=state.cumulative_reward,
            done=state.done,
            latest_proposal=latest_proposal,
            latest_audit=latest_audit,
            worker_records={
                worker_id: record.model_dump(mode="json")
                for worker_id, record in state.worker_records.items()
            },
        )

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="sentinel-oversight-command",
            description=(
                "Multi-agent OpenEnv environment where an LLM overseer intercepts "
                "worker-agent actions before they can affect a production incident."
            ),
            version="1.0.0",
            author="OpenEnv Contributor",
            documentation_url="https://huggingface.co/spaces/srikrishna2005/openenv",
        )

    def close(self) -> None:
        self._has_reset = False

    def _to_observation(
        self,
        obs: Any,
        *,
        reward: Optional[float],
        done: bool,
        info: Optional[Dict[str, Any]] = None,
    ) -> SentinelNativeObservation:
        proposal = (
            obs.proposed_action.model_dump(mode="json")
            if getattr(obs, "proposed_action", None) is not None
            else {}
        )
        audit_tail = []
        try:
            audit_tail = [
                item.model_dump(mode="json")
                for item in self._env.state().audit_log[-3:]
            ]
        except RuntimeError:
            audit_tail = []

        return SentinelNativeObservation(
            done=done,
            reward=reward,
            metadata=info or {},
            task_id=getattr(obs, "task_id", self._task_id),
            step_number=getattr(obs, "step_number", 0),
            max_steps=getattr(obs, "max_steps", 0),
            proposed_action=proposal,
            worker_id=getattr(obs, "worker_id", None),
            worker_role=getattr(obs, "worker_role", None),
            incident_status=getattr(obs, "incident_status", None),
            available_decisions=list(getattr(obs, "available_decisions", []) or []),
            corrective_loop_enabled=bool(getattr(obs, "corrective_loop_enabled", False)),
            audit_log_tail=audit_tail,
            message=getattr(obs, "message", ""),
        )
