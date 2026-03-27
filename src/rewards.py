"""Step-level reward computation for the IRT environment.

Provides dense reward signal over the full trajectory:
  - Positive for relevant investigations, correct classifications,
    accurate diagnoses, and appropriate remediations.
  - Negative for irrelevant actions, wrong classifications,
    destructive remediations, and wasted steps.
  - Temporal degradation penalty for delayed response.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.models import (
    Action,
    ActionType,
    IncidentSeverity,
    Reward,
)
from src.scenarios import Scenario


def _normalize(value: float) -> float:
    """Clamp reward to [-1.0, 1.0]."""
    return max(-1.0, min(1.0, value))


def compute_step_reward(
    action: Action,
    scenario: Scenario,
    step_number: int,
    already_investigated: List[str],
    already_classified: Optional[IncidentSeverity],
    already_diagnosed: Optional[str],
    already_remediated: List[str],
    already_escalated: List[str],
    already_communicated: List[str],
    actions_history: List[Dict[str, Any]],
) -> Reward:
    """Compute the reward for a single step."""

    components: Dict[str, float] = {}
    total = 0.0

    # -- Temporal degradation -----------------------------------------------
    degradation = -scenario.degradation_per_step * step_number
    components["temporal_degradation"] = degradation
    total += degradation

    # -- Action-specific rewards --------------------------------------------

    if action.action_type == ActionType.INVESTIGATE:
        target = (action.target or "").strip()
        if target in already_investigated:
            components["duplicate_investigation"] = -0.03
            total -= 0.03
        elif target in scenario.relevant_services:
            components["relevant_investigation"] = 0.06
            total += 0.06
        elif target in scenario.available_services:
            components["irrelevant_investigation"] = -0.02
            total -= 0.02
        else:
            components["invalid_target"] = -0.05
            total -= 0.05

    elif action.action_type == ActionType.CLASSIFY:
        severity_str = action.parameters.get("severity", "")
        if already_classified is not None:
            components["duplicate_classify"] = -0.03
            total -= 0.03
        else:
            try:
                given = IncidentSeverity(severity_str)
                if given == scenario.correct_severity:
                    components["correct_classification"] = 0.15
                    total += 0.15
                else:
                    diff = abs(
                        list(IncidentSeverity).index(given)
                        - list(IncidentSeverity).index(scenario.correct_severity)
                    )
                    penalty = -0.05 * diff
                    components["wrong_classification"] = penalty
                    total += penalty
            except ValueError:
                components["invalid_severity"] = -0.08
                total -= 0.08

    elif action.action_type == ActionType.DIAGNOSE:
        if already_diagnosed is not None:
            components["duplicate_diagnosis"] = -0.03
            total -= 0.03
        else:
            root_cause_text = action.parameters.get("root_cause", "").lower()
            target_svc = (action.target or "").lower()
            # Check service match
            if target_svc == scenario.correct_root_cause_service.lower():
                components["correct_service"] = 0.10
                total += 0.10
            elif target_svc:
                components["wrong_service"] = -0.05
                total -= 0.05
            # Check root cause keywords
            matched = any(
                kw.lower() in root_cause_text
                for kw in scenario.correct_root_cause_keywords
            )
            if matched:
                components["correct_root_cause"] = 0.15
                total += 0.15
            elif root_cause_text:
                components["wrong_root_cause"] = -0.05
                total -= 0.05

    elif action.action_type == ActionType.REMEDIATE:
        rem_action = action.parameters.get("action", "")
        rem_service = (action.target or "").strip()
        rem_key = f"{rem_action}:{rem_service}"
        if rem_key in already_remediated:
            components["duplicate_remediation"] = -0.03
            total -= 0.03
        else:
            valid = any(
                va.get("action") == rem_action and va.get("service") == rem_service
                for va in scenario.valid_remediation_actions
            )
            if valid:
                components["correct_remediation"] = 0.12
                total += 0.12
            else:
                components["wrong_remediation"] = -0.08
                total -= 0.08

    elif action.action_type == ActionType.ESCALATE:
        team = (action.target or "").strip().lower()
        if team in [t.lower() for t in already_escalated]:
            components["duplicate_escalation"] = -0.02
            total -= 0.02
        elif team in [t.lower() for t in scenario.expected_escalation_teams]:
            components["correct_escalation"] = 0.05
            total += 0.05
        else:
            components["unnecessary_escalation"] = -0.02
            total -= 0.02

    elif action.action_type == ActionType.COMMUNICATE:
        message = action.parameters.get("message", "")
        if len(message) < 10:
            components["low_quality_communication"] = -0.02
            total -= 0.02
        elif already_communicated and len(already_communicated) > 3:
            components["excessive_communication"] = -0.01
            total -= 0.01
        else:
            components["status_communication"] = 0.04
            total += 0.04

    # -- Reasoning bonus (small credit for providing reasoning) -------------
    if action.reasoning and len(action.reasoning) > 20:
        components["reasoning_provided"] = 0.01
        total += 0.01

    total = _normalize(total)
    message_parts = [f"{k}: {v:+.3f}" for k, v in components.items()]
    return Reward(
        value=round(total, 4),
        components={k: round(v, 4) for k, v in components.items()},
        message="; ".join(message_parts),
    )
