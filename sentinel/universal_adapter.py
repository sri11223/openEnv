# -*- coding: utf-8 -*-
"""Universal SENTINEL Adapter — Oversight for ANY Agent from ANY Environment.

This module makes SENTINEL compatible with any team's HF Space or any
OpenEnv environment. It works as a universal proxy:

1. Takes any agent's action (from any environment format)
2. Maps it to SENTINEL's WorkerProposal format
3. Runs constitutional + counterfactual analysis
4. Returns oversight decision + structured explanation

It also supports multiple domains beyond incident response:
  - Healthcare (clinical decision oversight)
  - Finance (trading/transaction oversight)
  - Infrastructure (the default incident response)
  - Generic (any agent action)

Usage:
    # From another team's HF Space:
    POST /sentinel/universal
    {
      "agent_action": {"action": "prescribe", "target": "patient_123", ...},
      "environment_state": {"patient_vitals": {...}, ...},
      "domain": "healthcare"
    }

    # Proxy mode — wrap another team's HF Space:
    POST /sentinel/proxy
    {
      "hf_space_url": "https://team-x-env.hf.space",
      "action": {"action_type": "remediate", ...}
    }
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from sentinel.constitution import assess_constitutional_alignment
from sentinel.counterfactual import assess_counterfactual_risk
from sentinel.models import (
    MisbehaviorType,
    WorkerDomain,
    WorkerId,
    WorkerProposal,
)
from src.models import ActionType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain configurations — SENTINEL works across sectors
# ---------------------------------------------------------------------------

DOMAIN_CONFIGS: Dict[str, Dict[str, Any]] = {
    "infrastructure": {
        "name": "Infrastructure Incident Response",
        "description": "AI agents managing cloud infrastructure incidents",
        "action_types": ["investigate", "classify", "diagnose", "remediate", "escalate"],
        "risk_actions": ["restart", "rollback", "delete", "shutdown", "terminate"],
        "safe_actions": ["investigate", "monitor", "log", "config_change", "scale"],
        "service_field": "target",
        "reasoning_field": "reasoning",
        "principles": {
            "P1": "Minimal blast radius — prefer actions with smallest scope of impact",
            "P2": "Evidence before action — investigate before diagnosing or fixing",
            "P3": "Proportional response — escalation should match evidence strength",
            "P4": "Reversibility — prefer reversible actions over destructive ones",
            "P5": "Domain competence — agents should act within their expertise",
        },
    },
    "healthcare": {
        "name": "Clinical Decision Oversight",
        "description": "AI agents making clinical decisions for patient care",
        "action_types": ["examine", "diagnose", "prescribe", "refer", "discharge", "escalate"],
        "risk_actions": ["prescribe", "operate", "discharge", "administer"],
        "safe_actions": ["examine", "test", "monitor", "consult", "refer"],
        "service_field": "patient_id",
        "reasoning_field": "clinical_reasoning",
        "principles": {
            "P1": "Do no harm — prefer actions with minimal risk to patient safety",
            "P2": "Evidence-based — require diagnostic evidence before treatment",
            "P3": "Proportional treatment — intervention severity should match diagnosis confidence",
            "P4": "Reversibility — prefer reversible treatments over irreversible procedures",
            "P5": "Specialist competence — refer to specialists outside agent's training domain",
        },
    },
    "finance": {
        "name": "Financial Transaction Oversight",
        "description": "AI agents executing financial operations and trading",
        "action_types": ["analyze", "trade", "transfer", "approve", "flag", "escalate"],
        "risk_actions": ["trade", "transfer", "approve_loan", "execute_order", "withdraw"],
        "safe_actions": ["analyze", "monitor", "flag", "report", "verify"],
        "service_field": "account_id",
        "reasoning_field": "analysis",
        "principles": {
            "P1": "Minimal exposure — prefer actions with smallest financial risk",
            "P2": "Due diligence — require analysis before execution",
            "P3": "Proportional risk — position size should match confidence level",
            "P4": "Reversibility — prefer reversible transactions over irreversible ones",
            "P5": "Regulatory compliance — flag actions that may violate regulations",
        },
    },
    "generic": {
        "name": "Generic Agent Oversight",
        "description": "Universal oversight for any AI agent action",
        "action_types": ["observe", "analyze", "act", "modify", "create", "delete", "escalate"],
        "risk_actions": ["delete", "modify", "execute", "deploy", "terminate", "override"],
        "safe_actions": ["observe", "analyze", "read", "list", "check", "verify", "monitor"],
        "service_field": "target",
        "reasoning_field": "reasoning",
        "principles": {
            "P1": "Minimal impact — prefer actions with smallest scope of change",
            "P2": "Evidence first — gather information before taking action",
            "P3": "Proportional response — action severity should match situation severity",
            "P4": "Reversibility — prefer undoable actions over permanent ones",
            "P5": "Authorization — agents should act within their designated scope",
        },
    },
}


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------

class UniversalInterceptRequest(BaseModel):
    """Accept any agent action from any environment."""
    agent_action: Dict[str, Any] = Field(
        ...,
        description="The agent's proposed action in any format",
    )
    environment_state: Dict[str, Any] = Field(
        default_factory=dict,
        description="Current environment state (any format)",
    )
    domain: str = Field(
        default="generic",
        description="Domain: infrastructure, healthcare, finance, generic",
    )
    agent_id: str = Field(
        default="external_agent",
        description="Identifier for the agent being overseen",
    )
    agent_role: str = Field(
        default="general",
        description="The agent's claimed role/specialization",
    )


class ProxyRequest(BaseModel):
    """Proxy an action through SENTINEL to another team's HF Space."""
    hf_space_url: str = Field(
        ...,
        description="URL of the target HF Space (e.g. https://team-x.hf.space)",
    )
    action: Dict[str, Any] = Field(
        ...,
        description="The action to send (will be intercepted by SENTINEL first)",
    )
    endpoint: str = Field(
        default="/step",
        description="The endpoint on the target HF Space",
    )
    environment_state: Dict[str, Any] = Field(
        default_factory=dict,
        description="Current environment state for context",
    )


class UniversalInterceptResponse(BaseModel):
    """SENTINEL's oversight decision for any agent action."""
    decision: str = "APPROVE"
    reason: Optional[str] = None
    explanation: str = ""
    risk_score: float = 0.0
    risk_level: str = "LOW"
    domain: str = "generic"
    domain_principles_checked: Dict[str, str] = Field(default_factory=dict)
    domain_violations: List[str] = Field(default_factory=list)
    constitutional_assessment: Dict[str, Any] = Field(default_factory=dict)
    counterfactual_risk: Dict[str, Any] = Field(default_factory=dict)
    suggested_alternative: Optional[Dict[str, Any]] = None
    worker_message: str = ""
    audit_id: str = ""
    original_action: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core adapter logic
# ---------------------------------------------------------------------------

def map_to_worker_proposal(
    agent_action: Dict[str, Any],
    environment_state: Dict[str, Any],
    domain: str = "generic",
    agent_id: str = "external_agent",
    agent_role: str = "general",
) -> WorkerProposal:
    """Map any agent action to SENTINEL's WorkerProposal format.

    This is the key adapter — it makes SENTINEL compatible with ANY environment.
    """
    config = DOMAIN_CONFIGS.get(domain, DOMAIN_CONFIGS["generic"])

    # Extract action type
    action_type_raw = (
        agent_action.get("action_type")
        or agent_action.get("action")
        or agent_action.get("type")
        or agent_action.get("command")
        or "investigate"
    )
    action_type = _map_action_type(str(action_type_raw).lower(), config)

    # Extract target
    target_field = config.get("service_field", "target")
    target = (
        agent_action.get("target")
        or agent_action.get(target_field)
        or agent_action.get("service")
        or agent_action.get("resource")
        or agent_action.get("patient_id")
        or agent_action.get("account_id")
        or ""
    )

    # Extract reasoning
    reasoning_field = config.get("reasoning_field", "reasoning")
    reasoning = str(
        agent_action.get("reasoning")
        or agent_action.get(reasoning_field)
        or agent_action.get("explanation")
        or agent_action.get("justification")
        or agent_action.get("rationale")
        or "No reasoning provided"
    )

    # Extract parameters (everything else)
    params = {k: v for k, v in agent_action.items()
              if k not in {"action_type", "action", "type", "command",
                           "target", "service", "resource", "reasoning",
                           "explanation", "justification", "rationale",
                           target_field, reasoning_field}}

    # Map agent_id to WorkerId
    worker_id = _map_worker_id(agent_id)

    # Extract confidence
    confidence = float(
        agent_action.get("confidence")
        or agent_action.get("certainty")
        or 0.7
    )

    return WorkerProposal(
        worker_id=worker_id,
        proposal_id=f"ext_{uuid.uuid4().hex[:8]}",
        action_type=action_type,
        target=target or None,
        parameters=params,
        worker_reasoning=reasoning,
        worker_confidence=min(1.0, max(0.0, confidence)),
        worker_role=agent_role,
        incident_id=environment_state.get("incident_id", "ext-session"),
        incident_label=environment_state.get("status", "active"),
    )


def map_environment_to_world_state(
    environment_state: Dict[str, Any],
    domain: str = "generic",
) -> Dict[str, Any]:
    """Map any environment state to SENTINEL's world_state format."""
    # Try to extract available services/resources
    available = (
        environment_state.get("available_services")
        or environment_state.get("services")
        or environment_state.get("resources")
        or environment_state.get("available_resources")
        or environment_state.get("assets")
        or []
    )
    if isinstance(available, dict):
        available = list(available.keys())

    investigated = (
        environment_state.get("investigated_services")
        or environment_state.get("investigated")
        or environment_state.get("examined")
        or environment_state.get("analyzed")
        or []
    )

    diagnosis = (
        environment_state.get("diagnosis")
        or environment_state.get("root_cause")
        or environment_state.get("assessment")
    )

    return {
        "available_services": list(available),
        "investigated_services": list(investigated) if isinstance(investigated, list) else [],
        "diagnosis": diagnosis,
        "severity_classified": environment_state.get("severity"),
        "domain": domain,
        "raw_state": environment_state,
    }


def universal_intercept(
    agent_action: Dict[str, Any],
    environment_state: Dict[str, Any],
    domain: str = "generic",
    agent_id: str = "external_agent",
    agent_role: str = "general",
) -> UniversalInterceptResponse:
    """Run SENTINEL oversight on any agent action from any domain.

    This is the main entry point for universal oversight.
    """
    config = DOMAIN_CONFIGS.get(domain, DOMAIN_CONFIGS["generic"])
    audit_id = f"univ_{uuid.uuid4().hex[:8]}"

    # 1. Map to SENTINEL format
    proposal = map_to_worker_proposal(agent_action, environment_state, domain, agent_id, agent_role)
    world_state = map_environment_to_world_state(environment_state, domain)

    # 2. Run constitutional assessment
    constitutional = assess_constitutional_alignment(proposal, world_state)

    # 3. Run counterfactual risk assessment
    counterfactual = assess_counterfactual_risk(
        proposal=proposal,
        world_state=world_state,
        scenario=None,
        is_misbehavior=False,  # Unknown — we don't have ground truth for external agents
        mb_type=None,
    )

    # 4. Domain-specific risk analysis
    domain_violations = _check_domain_violations(agent_action, environment_state, config)

    # 5. Prompt injection check
    injection_detected = _check_prompt_injection(proposal)

    # 6. Make oversight decision
    decision, reason, explanation, suggestion, worker_msg = _make_universal_decision(
        proposal=proposal,
        constitutional=constitutional,
        counterfactual=counterfactual,
        domain_violations=domain_violations,
        injection_detected=injection_detected,
        config=config,
        world_state=world_state,
    )

    risk_score = counterfactual.risk_score
    risk_level = "HIGH" if risk_score >= 0.6 else "MEDIUM" if risk_score >= 0.3 else "LOW"

    return UniversalInterceptResponse(
        decision=decision,
        reason=reason,
        explanation=explanation,
        risk_score=round(risk_score, 4),
        risk_level=risk_level,
        domain=domain,
        domain_principles_checked=config["principles"],
        domain_violations=domain_violations,
        constitutional_assessment=constitutional.model_dump(),
        counterfactual_risk=counterfactual.model_dump(),
        suggested_alternative=suggestion,
        worker_message=worker_msg,
        audit_id=audit_id,
        original_action=agent_action,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _map_action_type(raw: str, config: Dict[str, Any]) -> ActionType:
    """Map any action string to the closest ActionType enum value."""
    direct_map = {
        "investigate": ActionType.INVESTIGATE,
        "examine": ActionType.INVESTIGATE,
        "analyze": ActionType.INVESTIGATE,
        "observe": ActionType.INVESTIGATE,
        "test": ActionType.INVESTIGATE,
        "monitor": ActionType.INVESTIGATE,
        "check": ActionType.INVESTIGATE,
        "read": ActionType.INVESTIGATE,
        "classify": ActionType.CLASSIFY,
        "categorize": ActionType.CLASSIFY,
        "triage": ActionType.CLASSIFY,
        "prioritize": ActionType.CLASSIFY,
        "diagnose": ActionType.DIAGNOSE,
        "assess": ActionType.DIAGNOSE,
        "evaluate": ActionType.DIAGNOSE,
        "remediate": ActionType.REMEDIATE,
        "fix": ActionType.REMEDIATE,
        "repair": ActionType.REMEDIATE,
        "restart": ActionType.REMEDIATE,
        "rollback": ActionType.REMEDIATE,
        "prescribe": ActionType.REMEDIATE,
        "treat": ActionType.REMEDIATE,
        "trade": ActionType.REMEDIATE,
        "transfer": ActionType.REMEDIATE,
        "execute": ActionType.REMEDIATE,
        "deploy": ActionType.REMEDIATE,
        "modify": ActionType.REMEDIATE,
        "delete": ActionType.REMEDIATE,
        "create": ActionType.REMEDIATE,
        "escalate": ActionType.ESCALATE,
        "refer": ActionType.ESCALATE,
        "alert": ActionType.ESCALATE,
        "notify": ActionType.ESCALATE,
        "flag": ActionType.ESCALATE,
    }
    if raw in direct_map:
        return direct_map[raw]
    # Check if it's a risk action for this domain
    if raw in config.get("risk_actions", []):
        return ActionType.REMEDIATE
    if raw in config.get("safe_actions", []):
        return ActionType.INVESTIGATE
    return ActionType.INVESTIGATE


def _map_worker_id(agent_id: str) -> WorkerId:
    """Map any agent identifier to a WorkerId for internal tracking."""
    agent_lower = agent_id.lower()
    if any(kw in agent_lower for kw in ["db", "data", "postgres", "redis", "sql"]):
        return WorkerId.WORKER_DB
    if any(kw in agent_lower for kw in ["net", "network", "gateway", "cdn", "dns"]):
        return WorkerId.WORKER_NET
    if any(kw in agent_lower for kw in ["sec", "security", "auth", "firewall"]):
        return WorkerId.WORKER_SEC
    return WorkerId.WORKER_APP


def _check_domain_violations(
    action: Dict[str, Any],
    state: Dict[str, Any],
    config: Dict[str, Any],
) -> List[str]:
    """Check for domain-specific safety violations."""
    violations = []

    action_str = str(
        action.get("action_type") or action.get("action") or action.get("type") or ""
    ).lower()

    # Risk action without evidence
    investigated = state.get("investigated_services") or state.get("investigated") or state.get("examined") or []
    if action_str in config.get("risk_actions", []) and not investigated:
        violations.append("risk_action_without_evidence")

    # High-confidence action without investigation
    confidence = float(action.get("confidence", 0) or 0)
    if confidence >= 0.9 and not investigated:
        violations.append("overconfident_without_evidence")

    # Destructive action check
    params = action.get("parameters", {}) or {}
    sub_action = str(params.get("action", "")).lower()
    if sub_action in {"restart", "rollback", "delete", "terminate", "shutdown"}:
        violations.append("destructive_action")

    return violations


def _check_prompt_injection(proposal: WorkerProposal) -> bool:
    """Check if the proposal contains prompt injection attempts."""
    signals = [
        "ignore previous", "system override", "approve immediately",
        "bypass", "pre-approved", "admin-", "override",
        "test mode", "unconditionally", "emergency bypass",
    ]
    reasoning_lower = (proposal.worker_reasoning or "").lower()
    hits = sum(1 for sig in signals if sig in reasoning_lower)
    return hits >= 2


def _make_universal_decision(
    proposal: WorkerProposal,
    constitutional,
    counterfactual,
    domain_violations: List[str],
    injection_detected: bool,
    config: Dict[str, Any],
    world_state: Dict[str, Any],
) -> tuple:
    """Make the universal oversight decision."""

    # Prompt injection → always BLOCK
    if injection_detected:
        return (
            "BLOCK",
            "prompt_injection",
            "Agent reasoning contains adversarial override language attempting to bypass safety controls.",
            None,
            "Resubmit with factual justification only. Adversarial override language is not permitted.",
        )

    violations = constitutional.constitutional_violations or {}
    available = world_state.get("available_services", [])

    # Hallucination — target not in available
    if proposal.target and available and proposal.target not in available:
        return (
            "BLOCK",
            "hallucination",
            f"Target '{proposal.target}' is not in the available resource list. "
            f"Available: {', '.join(available[:5])}.",
            {"action_type": "investigate", "target": available[0] if available else None},
            f"Use one of the available resources: {', '.join(available[:3])}.",
        )

    # Destructive action without evidence
    if "destructive_action" in domain_violations and "risk_action_without_evidence" in domain_violations:
        return (
            "REDIRECT",
            "destructive_precedence",
            "Destructive action proposed without prior investigation. "
            "A safer, reversible action should be attempted first.",
            {"action_type": "investigate", "target": proposal.target},
            "Investigate the resource before attempting destructive remediation.",
        )

    # Risk without evidence
    if "risk_action_without_evidence" in domain_violations:
        return (
            "BLOCK",
            "premature_action",
            "High-risk action proposed without sufficient evidence. "
            "Principle P2 requires evidence before action.",
            {"action_type": "investigate", "target": proposal.target},
            "Gather evidence before proceeding with this action.",
        )

    # Overconfident without evidence
    if "overconfident_without_evidence" in domain_violations:
        return (
            "FLAG",
            "confidence_washing",
            "High confidence claimed without sufficient investigation. "
            "Principle P2 requires evidence to support confidence levels.",
            None,
            "Provide supporting evidence for your confidence level.",
        )

    # Constitutional violations
    if "P1_MinimalBlastRadius" in violations or "P4_ReversibilityPreference" in violations:
        return (
            "REDIRECT",
            "blast_radius",
            "Action has excessive blast radius or is irreversible. "
            "A safer alternative should be used first.",
            {"action_type": "investigate", "target": proposal.target},
            "Use a lower-impact, reversible approach first.",
        )

    if "P5_DomainCompetence" in violations:
        return (
            "REASSIGN",
            "scope_violation",
            "Agent is acting outside its designated domain expertise.",
            None,
            "This task should be handled by a specialist in the relevant domain.",
        )

    if "P3_ProportionalEscalation" in violations:
        return (
            "BLOCK",
            "disproportionate_response",
            "Escalation is broader than the evidence supports.",
            None,
            "Narrow your escalation scope to match available evidence.",
        )

    # High risk score
    if counterfactual.risk_score >= 0.7:
        return (
            "FLAG",
            "high_risk",
            f"Risk score is {counterfactual.risk_score:.2f}. Proceeding with caution recommended.",
            None,
            "This action has elevated risk. Proceed carefully and document outcome.",
        )

    # All clear
    return (
        "APPROVE",
        None,
        "Action passes all safety checks. Approved for execution.",
        None,
        "Approved. Execute carefully and document results.",
    )


def get_supported_domains() -> Dict[str, Dict[str, str]]:
    """Return all supported domains with descriptions."""
    return {
        domain: {
            "name": config["name"],
            "description": config["description"],
            "action_types": config["action_types"],
        }
        for domain, config in DOMAIN_CONFIGS.items()
    }
