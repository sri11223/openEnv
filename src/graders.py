"""End-of-episode graders for each task.

Each grader evaluates the full trajectory and produces a score in [0.0, 1.0]
with a detailed breakdown.  Grading is deterministic given the same
trajectory.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.models import (
    EnvironmentState,
    GraderResult,
    IncidentSeverity,
)
from src.scenarios import Scenario


def _keyword_match(text: str, keywords: List[str]) -> bool:
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def _severity_distance(a: Optional[IncidentSeverity], b: IncidentSeverity) -> int:
    if a is None:
        return 4  # worst case
    order = list(IncidentSeverity)
    return abs(order.index(a) - order.index(b))


# --------------------------------------------------------------------------
# Task 1 – Easy: Severity Classification
# --------------------------------------------------------------------------

def grade_severity_classification(state: EnvironmentState, scenario: Scenario) -> GraderResult:
    breakdown: Dict[str, float] = {}

    # 1. Correct severity (0.50)
    sev_dist = _severity_distance(state.severity_classified, scenario.correct_severity)
    if sev_dist == 0:
        breakdown["severity_accuracy"] = 0.50
    elif sev_dist == 1:
        breakdown["severity_accuracy"] = 0.25
    else:
        breakdown["severity_accuracy"] = 0.0

    # 2. Investigated before classifying (0.25)
    inv_before_classify = len(state.investigated_services) > 0
    if inv_before_classify:
        relevant_inv = len(
            set(state.investigated_services) & set(scenario.relevant_services)
        )
        if relevant_inv > 0:
            breakdown["investigation_quality"] = 0.25
        else:
            breakdown["investigation_quality"] = 0.10
    else:
        breakdown["investigation_quality"] = 0.0

    # 3. Efficiency (0.25) – fewer steps is better; 0 steps = no credit
    max_s = scenario.max_steps
    used = state.total_steps_taken
    if used == 0:
        breakdown["efficiency"] = 0.0
    elif used <= 3:
        breakdown["efficiency"] = 0.25
    elif used <= 5:
        breakdown["efficiency"] = 0.20
    elif used <= max_s // 2:
        breakdown["efficiency"] = 0.10
    else:
        breakdown["efficiency"] = 0.0

    score = sum(breakdown.values())
    return GraderResult(
        task_id=scenario.task_id,
        score=round(min(1.0, score), 4),
        breakdown={k: round(v, 4) for k, v in breakdown.items()},
        feedback=_severity_feedback(breakdown),
    )


def _severity_feedback(bd: Dict[str, float]) -> str:
    parts = []
    if bd.get("severity_accuracy", 0) >= 0.50:
        parts.append("Severity correctly classified.")
    elif bd.get("severity_accuracy", 0) > 0:
        parts.append("Severity close but not exact.")
    else:
        parts.append("Severity classification incorrect or missing.")
    if bd.get("investigation_quality", 0) >= 0.25:
        parts.append("Good investigation of relevant services.")
    elif bd.get("investigation_quality", 0) > 0:
        parts.append("Investigated, but not the most relevant services.")
    else:
        parts.append("No investigation performed before classification.")
    return " ".join(parts)


# --------------------------------------------------------------------------
# Task 2 – Medium: Root Cause Analysis
# --------------------------------------------------------------------------

def grade_root_cause_analysis(state: EnvironmentState, scenario: Scenario) -> GraderResult:
    breakdown: Dict[str, float] = {}

    # 1. Correct severity (0.15)
    sev_dist = _severity_distance(state.severity_classified, scenario.correct_severity)
    breakdown["severity_accuracy"] = 0.15 if sev_dist == 0 else (0.08 if sev_dist == 1 else 0.0)

    # 2. Investigated root-cause service (0.15)
    if scenario.correct_root_cause_service in state.investigated_services:
        breakdown["investigated_root_cause_service"] = 0.15
    else:
        breakdown["investigated_root_cause_service"] = 0.0

    # 3. Correct diagnosis (0.30)
    if state.diagnosis and _keyword_match(state.diagnosis, scenario.correct_root_cause_keywords):
        breakdown["diagnosis_accuracy"] = 0.30
    elif state.diagnosis:
        breakdown["diagnosis_accuracy"] = 0.05
    else:
        breakdown["diagnosis_accuracy"] = 0.0

    # 4. Correct remediation (0.20)
    valid_keys = {
        f"{va['action']}:{va['service']}" for va in scenario.valid_remediation_actions
    }
    applied_valid = len(set(state.remediations_applied) & valid_keys)
    if applied_valid > 0:
        breakdown["remediation_quality"] = 0.20
    elif len(state.remediations_applied) > 0:
        breakdown["remediation_quality"] = 0.05
    else:
        breakdown["remediation_quality"] = 0.0

    # 5. Efficiency (0.20); 0 steps = no credit
    max_s = scenario.max_steps
    used = state.total_steps_taken
    ratio = used / max_s if used > 0 else 1.0
    if used == 0:
        breakdown["efficiency"] = 0.0
    elif ratio <= 0.4:
        breakdown["efficiency"] = 0.20
    elif ratio <= 0.6:
        breakdown["efficiency"] = 0.15
    elif ratio <= 0.8:
        breakdown["efficiency"] = 0.08
    else:
        breakdown["efficiency"] = 0.0

    score = sum(breakdown.values())
    return GraderResult(
        task_id=scenario.task_id,
        score=round(min(1.0, score), 4),
        breakdown={k: round(v, 4) for k, v in breakdown.items()},
        feedback=_rca_feedback(breakdown),
    )


def _rca_feedback(bd: Dict[str, float]) -> str:
    parts = []
    if bd.get("diagnosis_accuracy", 0) >= 0.30:
        parts.append("Root cause correctly identified.")
    elif bd.get("diagnosis_accuracy", 0) > 0:
        parts.append("Diagnosis attempted but inaccurate.")
    else:
        parts.append("No diagnosis provided.")
    if bd.get("remediation_quality", 0) >= 0.20:
        parts.append("Correct remediation applied.")
    elif bd.get("remediation_quality", 0) > 0:
        parts.append("Remediation attempted but not optimal.")
    else:
        parts.append("No remediation applied.")
    return " ".join(parts)


# --------------------------------------------------------------------------
# Task 3 – Hard: Full Incident Management
# --------------------------------------------------------------------------

def grade_full_incident_management(state: EnvironmentState, scenario: Scenario) -> GraderResult:
    breakdown: Dict[str, float] = {}

    # Pre-compute whether agent actually investigated root-cause service
    investigated_root_cause = scenario.correct_root_cause_service in state.investigated_services

    # 1. Correct severity (0.12)
    sev_dist = _severity_distance(state.severity_classified, scenario.correct_severity)
    breakdown["severity_accuracy"] = 0.12 if sev_dist == 0 else (0.06 if sev_dist == 1 else 0.0)

    # 2. Root cause identified (0.20)
    # Diagnosis without investigation gets at most 0.06 (prompted guess, not evidence-based)
    if state.diagnosis and _keyword_match(state.diagnosis, scenario.correct_root_cause_keywords):
        if investigated_root_cause:
            breakdown["diagnosis_accuracy"] = 0.20
        else:
            breakdown["diagnosis_accuracy"] = 0.06   # guessed correctly but didn't look
    elif state.diagnosis:
        breakdown["diagnosis_accuracy"] = 0.03
    else:
        breakdown["diagnosis_accuracy"] = 0.0

    # 3. Remediation (0.18)
    valid_keys = {
        f"{va['action']}:{va['service']}" for va in scenario.valid_remediation_actions
    }
    applied_valid = set(state.remediations_applied) & valid_keys
    if len(applied_valid) >= 2:
        breakdown["remediation_quality"] = 0.18
    elif len(applied_valid) == 1:
        breakdown["remediation_quality"] = 0.12
    elif len(state.remediations_applied) > 0:
        breakdown["remediation_quality"] = 0.03
    else:
        breakdown["remediation_quality"] = 0.0

    # 4. Escalation (0.15) — raised from 0.08/0.14 cap
    expected_lower = {t.lower() for t in scenario.expected_escalation_teams}
    escalated_lower = {e.lower() for e in state.escalations_made}
    matched = len(escalated_lower & expected_lower)
    if matched >= 2:
        breakdown["escalation_quality"] = 0.15
    elif matched == 1:
        breakdown["escalation_quality"] = 0.09
    else:
        breakdown["escalation_quality"] = 0.0

    # 5. Communication (0.10)
    if len(state.communications_sent) >= 2:
        breakdown["communication"] = 0.10
    elif len(state.communications_sent) == 1:
        breakdown["communication"] = 0.06
    else:
        breakdown["communication"] = 0.0

    # 6. Investigation thoroughness (0.15)
    relevant_investigated = len(
        set(state.investigated_services) & set(scenario.relevant_services)
    )
    total_relevant = len(scenario.relevant_services)
    if total_relevant > 0:
        inv_ratio = relevant_investigated / total_relevant
    else:
        inv_ratio = 0.0
    breakdown["investigation_thoroughness"] = round(0.15 * inv_ratio, 4)

    # 7. Efficiency (0.10); 0 steps = no credit
    max_s = scenario.max_steps
    used = state.total_steps_taken
    if used == 0:
        breakdown["efficiency"] = 0.0
    elif used <= int(max_s * 0.5):
        breakdown["efficiency"] = 0.10
    elif used <= int(max_s * 0.7):
        breakdown["efficiency"] = 0.07
    elif used <= int(max_s * 0.85):
        breakdown["efficiency"] = 0.04
    else:
        breakdown["efficiency"] = 0.0

    score = sum(breakdown.values())
    return GraderResult(
        task_id=scenario.task_id,
        score=round(min(1.0, score), 4),
        breakdown={k: round(v, 4) for k, v in breakdown.items()},
        feedback=_full_feedback(breakdown),
    )


def _full_feedback(bd: Dict[str, float]) -> str:
    parts = []
    if bd.get("diagnosis_accuracy", 0) >= 0.20:
        parts.append("Root cause correctly identified with evidence from investigation.")
    elif bd.get("diagnosis_accuracy", 0) >= 0.06:
        parts.append("Root cause guessed correctly but not confirmed by investigation.")
    else:
        parts.append("Root cause not identified or inaccurate.")
    if bd.get("remediation_quality", 0) >= 0.12:
        parts.append("Good remediation actions taken.")
    else:
        parts.append("Remediation insufficient.")
    if bd.get("escalation_quality", 0) >= 0.09:
        parts.append("Appropriate escalation made.")
    else:
        parts.append("Escalation missing or misdirected.")
    if bd.get("communication", 0) > 0:
        parts.append("Status communicated.")
    else:
        parts.append("No status communication.")
    return " ".join(parts)


# --------------------------------------------------------------------------
# Dispatcher
# --------------------------------------------------------------------------

_GRADERS = {
    "severity_classification": grade_severity_classification,
    "root_cause_analysis": grade_root_cause_analysis,
    "full_incident_management": grade_full_incident_management,
}


def grade(task_id: str, state: EnvironmentState, scenario: Scenario) -> GraderResult:
    grader_fn = _GRADERS.get(task_id)
    if grader_fn is None:
        raise ValueError(f"No grader for task_id '{task_id}'")
    return grader_fn(state, scenario)
