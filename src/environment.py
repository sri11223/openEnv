"""Core environment implementing the OpenEnv step() / reset() / state() API.

This module owns all mutable episode state.  It is deliberately a single-
episode, per-session environment — the FastAPI layer maintains one instance
per session ID, ensuring concurrent agents never share state.
"""

from __future__ import annotations

import copy
import random
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.graders import grade
from src.models import (
    Action,
    ActionType,
    EnvironmentState,
    GraderResult,
    IncidentSeverity,
    IncidentStatus,
    Observation,
    Reward,
    ServiceMetrics,
    StepResult,
)
from src.rewards import compute_step_reward
from src.scenarios import Scenario, apply_blast_radius, get_scenario


class IncidentResponseEnv:
    """Incident Response Triage environment.

    Lifecycle:
        env = IncidentResponseEnv()
        obs = env.reset("severity_classification")
        while not done:
            result = env.step(action)
            obs, reward, done, info = result.observation, result.reward, result.done, result.info
        grader_result = env.grade()
    """

    def __init__(self) -> None:
        self._scenario: Optional[Scenario] = None
        self._task_id: Optional[str] = None
        self._step: int = 0
        self._done: bool = True
        self._cumulative_reward: float = 0.0
        # Progressive state
        self._investigated: List[str] = []
        self._severity_classified: Optional[IncidentSeverity] = None
        self._diagnosis: Optional[str] = None
        self._remediations: List[str] = []
        self._escalations: List[str] = []
        self._communications: List[str] = []
        self._actions_history: List[Dict[str, Any]] = []
        self._incident_status: IncidentStatus = IncidentStatus.OPEN
        self._last_message: str = ""
        # Logs / metrics revealed so far
        self._revealed_logs: Dict[str, list] = {}
        self._revealed_metrics: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # reset()
    # ------------------------------------------------------------------

    def reset(self, task_id: str, variant_seed: int = 0) -> Observation:
        """Reset the environment for a new episode on the given task.

        Args:
            task_id:      Task to run.
            variant_seed: Scenario variant index (default 0 = primary scenario).
        """
        scenario = get_scenario(task_id, variant_seed=variant_seed)
        self._scenario = scenario
        self._task_id = task_id
        self._step = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._investigated = []
        self._severity_classified = None
        self._diagnosis = None
        self._remediations = []
        self._escalations = []
        self._communications = []
        self._actions_history = []
        self._incident_status = IncidentStatus.OPEN
        self._last_message = "Incident opened. Review the alerts and begin your investigation."
        self._revealed_logs = {}
        self._revealed_metrics = {}
        return self._build_observation()

    # ------------------------------------------------------------------
    # step()
    # ------------------------------------------------------------------

    def step(self, action: Action) -> StepResult:
        """Process one agent action and return the result."""
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")
        if self._scenario is None:
            raise RuntimeError("Environment not initialised. Call reset() first.")

        self._step += 1
        scenario = self._scenario

        # Record action
        self._actions_history.append(action.model_dump())

        # Process action effects
        self._process_action(action, scenario)

        # Compute reward
        reward = compute_step_reward(
            action=action,
            scenario=scenario,
            step_number=self._step,
            already_investigated=self._investigated,
            already_classified=self._severity_classified,
            already_diagnosed=self._diagnosis,
            already_remediated=self._remediations,
            already_escalated=self._escalations,
            already_communicated=self._communications,
            actions_history=self._actions_history,
        )
        self._cumulative_reward += reward.value

        # Apply action state changes (after reward so duplicates are penalised first)
        self._apply_state_changes(action, scenario)

        # Check episode termination
        done = self._check_done(scenario)
        self._done = done

        obs = self._build_observation()
        info: Dict[str, Any] = {
            "cumulative_reward": round(self._cumulative_reward, 4),
            "steps_remaining": max(0, scenario.max_steps - self._step),
        }
        if done:
            info["grader"] = self.grade().model_dump()

        return StepResult(observation=obs, reward=reward, done=done, info=info)

    # ------------------------------------------------------------------
    # state()
    # ------------------------------------------------------------------

    def state(self) -> EnvironmentState:
        """Return the full internal state snapshot."""
        return EnvironmentState(
            task_id=self._task_id or "",
            scenario_id=self._scenario.scenario_id if self._scenario else "",
            step_number=self._step,
            max_steps=self._scenario.max_steps if self._scenario else 0,
            incident_status=self._incident_status,
            done=self._done,
            cumulative_reward=round(self._cumulative_reward, 4),
            total_steps_taken=self._step,
            alerts=list(self._scenario.initial_alerts) if self._scenario else [],
            actions_history=copy.deepcopy(self._actions_history),
            severity_classified=self._severity_classified,
            diagnosis=self._diagnosis,
            remediations_applied=list(self._remediations),
            escalations_made=list(self._escalations),
            communications_sent=list(self._communications),
            investigated_services=list(self._investigated),
        )

    # ------------------------------------------------------------------
    # grade()
    # ------------------------------------------------------------------

    def grade(self) -> GraderResult:
        """Grade the current episode. Can be called mid-episode or after done."""
        if self._scenario is None or self._task_id is None:
            raise RuntimeError("No episode in progress.")
        return grade(self._task_id, self.state(), self._scenario)

    def live_metrics(self) -> Dict[str, ServiceMetrics]:
        """Return service metrics with blast-radius degradation at the current step.

        Safe to call at any point (including before any actions are taken).
        Returns an empty dict when no episode is in progress.

        This is the same numerical data the agent would eventually see via
        investigate actions, but served here without consuming an action slot —
        analogous to a Prometheus scrape that is always available passively.
        """
        if self._scenario is None:
            return {}
        return apply_blast_radius(self._scenario, self._step)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _process_action(self, action: Action, scenario: Scenario) -> None:
        """Process action effects (messages, state transitions)."""
        if action.action_type == ActionType.INVESTIGATE:
            target = (action.target or "").strip()
            if target not in scenario.available_services:
                self._last_message = (
                    f"Unknown service '{target}'. "
                    f"Available: {', '.join(scenario.available_services)}"
                )
            elif target in self._investigated:
                self._last_message = f"Already investigated {target}. Logs and metrics available."
            else:
                # Reveal logs (always static — logs are historical records)
                if target in scenario.service_logs:
                    self._revealed_logs[target] = [
                        entry.model_dump() for entry in scenario.service_logs[target]
                    ]
                # Reveal LIVE metrics with blast-radius degradation applied
                live_metrics = apply_blast_radius(scenario, self._step)
                if target in live_metrics:
                    self._revealed_metrics[target] = live_metrics[target].model_dump()
                self._last_message = (
                    f"Investigation of {target} complete. Logs and live metrics now available."
                )
                if self._incident_status == IncidentStatus.OPEN:
                    self._incident_status = IncidentStatus.INVESTIGATING

        elif action.action_type == ActionType.CLASSIFY:
            severity_str = action.parameters.get("severity", "")
            try:
                sev = IncidentSeverity(severity_str)
                self._last_message = f"Incident classified as {sev.value}."
            except ValueError:
                self._last_message = (
                    f"Invalid severity '{severity_str}'. Use P1, P2, P3, or P4."
                )

        elif action.action_type == ActionType.DIAGNOSE:
            root_cause = action.parameters.get("root_cause", "")
            target_svc = (action.target or "").strip()
            self._last_message = (
                f"Diagnosis recorded: root cause in {target_svc} — {root_cause[:120]}"
            )

        elif action.action_type == ActionType.REMEDIATE:
            rem_action = action.parameters.get("action", "")
            target_svc = (action.target or "").strip()
            if not rem_action or not target_svc:
                self._last_message = "Remediation requires 'action' parameter and 'target' service."
            else:
                self._last_message = (
                    f"Remediation '{rem_action}' applied to {target_svc}."
                )
                if self._incident_status in (IncidentStatus.OPEN, IncidentStatus.INVESTIGATING):
                    self._incident_status = IncidentStatus.MITIGATING

        elif action.action_type == ActionType.ESCALATE:
            team = (action.target or "").strip()
            priority = action.parameters.get("priority", "high")
            message = action.parameters.get("message", "")
            self._last_message = (
                f"Escalated to {team} (priority: {priority}). "
                f"Message: {message[:80]}"
            )

        elif action.action_type == ActionType.COMMUNICATE:
            channel = (action.target or "status_page").strip()
            message = action.parameters.get("message", "")
            self._last_message = (
                f"Status update posted to {channel}: {message[:100]}"
            )

    def _apply_state_changes(self, action: Action, scenario: Scenario) -> None:
        """Persist state changes after reward is computed."""
        if action.action_type == ActionType.INVESTIGATE:
            target = (action.target or "").strip()
            if target in scenario.available_services and target not in self._investigated:
                self._investigated.append(target)

        elif action.action_type == ActionType.CLASSIFY:
            severity_str = action.parameters.get("severity", "")
            try:
                self._severity_classified = IncidentSeverity(severity_str)
            except ValueError:
                pass

        elif action.action_type == ActionType.DIAGNOSE:
            if self._diagnosis is None:
                self._diagnosis = action.parameters.get("root_cause", "")

        elif action.action_type == ActionType.REMEDIATE:
            rem_action = action.parameters.get("action", "")
            target_svc = (action.target or "").strip()
            if rem_action and target_svc:
                key = f"{rem_action}:{target_svc}"
                if key not in self._remediations:
                    self._remediations.append(key)

        elif action.action_type == ActionType.ESCALATE:
            team = (action.target or "").strip()
            if team and team not in self._escalations:
                self._escalations.append(team)

        elif action.action_type == ActionType.COMMUNICATE:
            message = action.parameters.get("message", "")
            if message:
                self._communications.append(message[:200])

    def _check_done(self, scenario: Scenario) -> bool:
        """Episode ends when max steps reached or incident resolved."""
        if self._step >= scenario.max_steps:
            self._last_message += " [Episode ended: max steps reached.]"
            return True
        # For easy task: done once classified
        if scenario.task_id == "severity_classification" and self._severity_classified is not None:
            # Give agent a chance to investigate first, but if classified, we're done
            # Actually let them keep going for a few more steps if they want
            if self._step >= 2 or self._severity_classified is not None:
                # Check if the last action was classify
                if (self._actions_history and
                        self._actions_history[-1].get("action_type") == ActionType.CLASSIFY.value):
                    self._incident_status = IncidentStatus.RESOLVED
                    self._last_message += " [Episode complete: severity classified.]"
                    return True
        # For medium: done once diagnosed AND remediated
        if scenario.task_id == "root_cause_analysis":
            if self._diagnosis and len(self._remediations) > 0:
                self._incident_status = IncidentStatus.RESOLVED
                self._last_message += " [Episode complete: diagnosed and remediated.]"
                return True
        # For hard: done once all major actions taken
        if scenario.task_id == "full_incident_management":
            has_severity = self._severity_classified is not None
            has_diagnosis = self._diagnosis is not None
            has_remediation = len(self._remediations) > 0
            has_escalation = len(self._escalations) > 0
            has_communication = len(self._communications) > 0
            if all([has_severity, has_diagnosis, has_remediation, has_escalation, has_communication]):
                self._incident_status = IncidentStatus.RESOLVED
                self._last_message += " [Episode complete: full incident management workflow completed.]"
                return True
        return False

    def _build_observation(self) -> Observation:
        """Construct the current observation from internal state."""
        scenario = self._scenario
        if scenario is None:
            raise RuntimeError("No scenario loaded.")

        # Reconstruct typed logs from revealed data
        from src.models import LogEntry, ServiceMetrics
        typed_logs = {}
        for svc, entries in self._revealed_logs.items():
            typed_logs[svc] = [LogEntry(**e) for e in entries]
        typed_metrics = {}
        for svc, m in self._revealed_metrics.items():
            typed_metrics[svc] = ServiceMetrics(**m)

        return Observation(
            incident_id=scenario.incident_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            step_number=self._step,
            max_steps=scenario.max_steps,
            task_id=scenario.task_id,
            task_description=scenario.description,
            alerts=list(scenario.initial_alerts),
            available_services=list(scenario.available_services),
            investigated_services=list(self._investigated),
            logs=typed_logs,
            metrics=typed_metrics,
            incident_status=self._incident_status,
            severity_classified=self._severity_classified,
            diagnosis=self._diagnosis,
            actions_taken=[
                f"Step {i+1}: {a.get('action_type', '?')} → {a.get('target', 'N/A')}"
                for i, a in enumerate(self._actions_history)
            ],
            remediations_applied=list(self._remediations),
            escalations_made=list(self._escalations),
            communications_sent=list(self._communications),
            message=self._last_message,
        )
