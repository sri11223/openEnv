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
        score=round(max(0.01, min(0.99, score)), 4),
        breakdown={k: round(v, 4) for k, v in breakdown.items()},
        feedback=_severity_feedback(breakdown),
    )


def _severity_feedback(bd: Dict[str, float]) -> str:
    parts = []
    sa = bd.get("severity_accuracy", 0)
    iq = bd.get("investigation_quality", 0)
    ef = bd.get("efficiency", 0)
    if sa >= 0.50:
        parts.append("✓ Severity classification correct. The connection pool saturation and partial error rate (~12%) indicate a degraded-but-not-down P2 incident.")
    elif sa > 0:
        parts.append("~ Severity off by one level. Review the alert signals: 98% connection pool utilisation and 12% error rate indicate degraded service (P2), not a full outage (P1) or minor issue (P3).")
    else:
        parts.append("✗ Severity classification missing or wrong. Examine alert severity levels and error rates before classifying. A P2 is correct: significant service degradation but not a full outage.")
    if iq >= 0.25:
        parts.append("✓ Good investigation — examined relevant services before classifying. Always investigate before classifying.")
    elif iq > 0:
        parts.append("~ Investigated services, but not the most relevant ones. postgres-primary (connection pool alert) and user-service (latency alert) are the critical paths.")
    else:
        parts.append("✗ No investigation performed before classification. Investigate postgres-primary and user-service first to confirm the root cause.")
    if ef >= 0.25:
        parts.append("✓ Efficient resolution — completed in 3 steps or fewer.")
    elif ef > 0:
        parts.append("~ Resolved but used more steps than optimal. Target: investigate 2 services → classify (3 steps total).")
    else:
        parts.append("✗ Too many steps or no actions taken. Optimal path: INVESTIGATE postgres-primary → INVESTIGATE user-service → CLASSIFY P2.")
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
        score=round(max(0.01, min(0.99, score)), 4),
        breakdown={k: round(v, 4) for k, v in breakdown.items()},
        feedback=_rca_feedback(breakdown),
    )


def _rca_feedback(bd: Dict[str, float]) -> str:
    parts = []
    da = bd.get("diagnosis_accuracy", 0)
    rq = bd.get("remediation_quality", 0)
    ir = bd.get("investigated_root_cause_service", 0)
    sa = bd.get("severity_accuracy", 0)
    ef = bd.get("efficiency", 0)
    if da >= 0.30:
        parts.append("✓ Root cause correctly identified: Redis session store hit maxmemory, causing active payment session tokens to be evicted before payment completion.")
    elif da > 0:
        parts.append("~ Diagnosis attempted but inaccurate. The root cause is Redis memory exhaustion (maxmemory reached → allkeys-lru eviction → payment tokens evicted mid-transaction). Not a payment-gateway issue.")
    else:
        parts.append("✗ No diagnosis provided. Root cause: redis-session hit its 4 GB maxmemory limit. Eviction policy evicted active payment session tokens faster than transactions completed.")
    if ir >= 0.15:
        parts.append("✓ Correctly investigated the root-cause service (redis-session).")
    else:
        parts.append("✗ Did not investigate redis-session. The eviction spike alert (ALT-012) is the key signal — always investigate the service generating eviction alerts.")
    if rq >= 0.20:
        parts.append("✓ Correct remediation: scaling redis-session memory clears the eviction pressure.")
    elif rq > 0:
        parts.append("~ Remediation attempted but suboptimal. Valid fixes: scale redis-session capacity, config_change to increase maxmemory, or restart to temporarily clear the eviction queue.")
    else:
        parts.append("✗ No remediation applied. Fix: REMEDIATE redis-session with action=scale or action=config_change to increase maxmemory.")
    if sa < 0.15:
        parts.append("✗ Severity wrong. Payment processing at 45% success rate is a P1 — revenue-impacting, customer-facing, requires immediate escalation.")
    if ef < 0.15:
        parts.append("~ Efficiency: optimal path is 5 steps (investigate ×2 → classify → diagnose → remediate).")
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

    # 6. Investigation thoroughness (0.12)
    relevant_investigated = len(
        set(state.investigated_services) & set(scenario.relevant_services)
    )
    total_relevant = len(scenario.relevant_services)
    if total_relevant > 0:
        inv_ratio = relevant_investigated / total_relevant
    else:
        inv_ratio = 0.0
    breakdown["investigation_thoroughness"] = round(0.12 * inv_ratio, 4)

    # 7. Investigation precision (0.03) — penalise unfocused investigation
    irrelevant_investigated = len(
        set(state.investigated_services) - set(scenario.relevant_services)
    )
    if irrelevant_investigated == 0:
        breakdown["investigation_precision"] = 0.03
    elif irrelevant_investigated <= 1:
        breakdown["investigation_precision"] = 0.01
    else:
        breakdown["investigation_precision"] = 0.0

    # 8. Efficiency (0.10); 0 steps = no credit
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
        score=round(max(0.01, min(0.99, score)), 4),
        breakdown={k: round(v, 4) for k, v in breakdown.items()},
        feedback=_full_feedback(breakdown),
    )


def _full_feedback(bd: Dict[str, float]) -> str:
    parts = []
    da = bd.get("diagnosis_accuracy", 0)
    rq = bd.get("remediation_quality", 0)
    eq = bd.get("escalation_quality", 0)
    comm = bd.get("communication", 0)
    it = bd.get("investigation_thoroughness", 0)
    ip = bd.get("investigation_precision", 0)
    sa = bd.get("severity_accuracy", 0)
    ef = bd.get("efficiency", 0)
    if da >= 0.20:
        parts.append("✓ Root cause correctly identified: auth-service v3.1.0 introduced an unbounded in-memory token cache causing OOMKill and cascading failures across all auth-dependent services.")
    elif da >= 0.06:
        parts.append("~ Root cause guessed but not confirmed via investigation. Investigate auth-service first — the deployment timestamp (v3.1.0 at 13:47) and memory climb logs are the definitive evidence.")
    else:
        parts.append("✗ Root cause not identified. Key signals: auth-service memory 45%→97% after v3.1.0 deploy at 13:47, changelog note 'Refactored token cache to in-memory store', 3 OOMKills in 5 min.")
    if rq >= 0.18:
        parts.append("✓ Comprehensive remediation: rolled back auth-service AND scaled order-service to drain the 15k+ message backlog.")
    elif rq >= 0.12:
        parts.append("~ Partial remediation. Also remediate order-service (scale) to drain the queue that built up during the auth outage.")
    elif rq > 0:
        parts.append("~ Some remediation applied but not optimal. Correct actions: rollback auth-service to v3.0.9 (primary fix) + scale order-service (queue drain).")
    else:
        parts.append("✗ No remediation applied. Critical: REMEDIATE auth-service action=rollback (roll back v3.1.0). Then REMEDIATE order-service action=scale to clear the queue backlog.")
    if eq >= 0.15:
        parts.append("✓ Correct teams escalated (platform-team + auth-team).")
    elif eq > 0:
        parts.append("~ Escalation partial. Escalate to both platform-team (infrastructure impact) and auth-team (owns the service with the bug).")
    else:
        parts.append("✗ No escalation. This is a P1 cascading outage — escalate to platform-team (urgent) and auth-team (owns the buggy deployment).")
    if comm > 0:
        parts.append("✓ Status communication sent.")
    else:
        parts.append("✗ No status communication. Send a COMMUNICATE action to status_page with root cause, impact, and ETA.")
    if it < 0.08:
        parts.append("~ Investigation incomplete. Key services to investigate: auth-service (root cause), api-gateway (circuit breaker), redis-auth-cache (cache bypass evidence), order-service (queue depth).")
    if ip < 0.03:
        parts.append("~ Investigation spread too wide. cdn-static and postgres-primary are red herrings — normal metrics, no alerts. Focus on auth-dependent services.")
    if sa < 0.12:
        parts.append("✗ Severity wrong. Multi-service cascading outage affecting auth, API gateway, orders, users = P1.")
    if ef < 0.04:
        parts.append("~ Efficiency: optimal path completes in 11 steps. Avoid re-investigating services or applying wrong remediations.")
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
