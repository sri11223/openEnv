"""Baseline inference script for the Incident Response Triage environment.

Supports two modes:
  1. LLM-based: Uses OpenAI API (set OPENAI_API_KEY env var)
  2. Rule-based: Deterministic heuristic baseline (fallback)

Usage:
    # LLM baseline (requires OPENAI_API_KEY)
    python -m baseline.inference --mode llm --base-url http://localhost:7860

    # Rule-based baseline (no API key needed)
    python -m baseline.inference --mode rules --base-url http://localhost:7860

    # Direct mode (in-process, no server needed)
    python -m baseline.inference --mode rules --direct
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import httpx

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TASK_IDS = ["severity_classification", "root_cause_analysis", "full_incident_management"]

SYSTEM_PROMPT = """\
You are an expert on-call Site Reliability Engineer handling a production incident.

You interact with an Incident Response environment by taking ONE action per step.
Each action must be a JSON object with these fields:
  - action_type: one of "classify", "investigate", "diagnose", "remediate", "escalate", "communicate"
  - target: service name, team name, or channel (depends on action_type)
  - parameters: dict with action-specific params
  - reasoning: brief explanation of why you chose this action

Action details:
  INVESTIGATE: target=service_name → reveals that service's logs and metrics
  CLASSIFY: parameters.severity = "P1"|"P2"|"P3"|"P4"
  DIAGNOSE: target=service_name, parameters.root_cause = "description"
  REMEDIATE: target=service_name, parameters.action = "restart"|"rollback"|"scale"|"config_change"
  ESCALATE: target=team_name, parameters.priority = "urgent"|"high"|"medium", parameters.message = "..."
  COMMUNICATE: target="status_page"|"slack"|"email", parameters.message = "status update text"

Strategy:
1. Read alerts carefully
2. Investigate the most suspicious services first
3. Classify severity based on evidence
4. Diagnose root cause after investigation
5. Apply targeted remediation
6. Escalate and communicate as needed

Return ONLY valid JSON — no markdown, no explanation outside the JSON.
"""


# ---------------------------------------------------------------------------
# Rule-based baseline
# ---------------------------------------------------------------------------

def _rule_based_easy() -> List[Dict[str, Any]]:
    """Deterministic optimal actions for the easy task."""
    return [
        {
            "action_type": "investigate",
            "target": "postgres-primary",
            "parameters": {},
            "reasoning": "Alert shows connection pool at 98% on postgres-primary. Investigating first.",
        },
        {
            "action_type": "investigate",
            "target": "user-service",
            "parameters": {},
            "reasoning": "user-service has high latency. Checking logs for connection pool errors.",
        },
        {
            "action_type": "classify",
            "target": "",
            "parameters": {"severity": "P2"},
            "reasoning": "Service degraded but not fully down. Connection pool issue is P2.",
        },
    ]


def _rule_based_medium() -> List[Dict[str, Any]]:
    """Deterministic optimal actions for the medium task."""
    return [
        {
            "action_type": "investigate",
            "target": "payment-gateway",
            "parameters": {},
            "reasoning": "Payment success rate is critically low. Starting with the payment gateway.",
        },
        {
            "action_type": "investigate",
            "target": "redis-session",
            "parameters": {},
            "reasoning": "Eviction spike on redis-session could explain missing payment tokens.",
        },
        {
            "action_type": "classify",
            "target": "",
            "parameters": {"severity": "P1"},
            "reasoning": "Payment processing at 45% success is a P1 revenue-impacting incident.",
        },
        {
            "action_type": "diagnose",
            "target": "redis-session",
            "parameters": {"root_cause": "Redis session store hit maxmemory limit causing eviction of payment session tokens. Sessions evicted before payment completion."},
            "reasoning": "Logs show redis-session at 100% memory with aggressive evictions of active sessions.",
        },
        {
            "action_type": "remediate",
            "target": "redis-session",
            "parameters": {"action": "scale"},
            "reasoning": "Scaling redis-session memory to stop evictions and restore payment flow.",
        },
    ]


def _rule_based_hard() -> List[Dict[str, Any]]:
    """Deterministic optimal actions for the hard task."""
    return [
        {
            "action_type": "investigate",
            "target": "auth-service",
            "parameters": {},
            "reasoning": "Auth-service has critical latency. Multiple services depend on auth. Investigating first.",
        },
        {
            "action_type": "investigate",
            "target": "api-gateway",
            "parameters": {},
            "reasoning": "API gateway returning 503s. Checking if it's auth-related.",
        },
        {
            "action_type": "investigate",
            "target": "redis-auth-cache",
            "parameters": {},
            "reasoning": "Checking auth cache - may explain why auth is slow.",
        },
        {
            "action_type": "classify",
            "target": "",
            "parameters": {"severity": "P1"},
            "reasoning": "Cascading multi-service outage affecting all authenticated endpoints. P1.",
        },
        {
            "action_type": "diagnose",
            "target": "auth-service",
            "parameters": {"root_cause": "Bad deployment v3.1.0 introduced memory leak via unbounded in-memory token cache. Auth-service OOMKill causes cascading failures to all dependent services."},
            "reasoning": "Auth-service logs show v3.1.0 deployment, memory climbing from 45% to 97%, GC pauses causing timeouts.",
        },
        {
            "action_type": "remediate",
            "target": "auth-service",
            "parameters": {"action": "rollback"},
            "reasoning": "Rolling back auth-service to v3.0.9 to fix the memory leak.",
        },
        {
            "action_type": "remediate",
            "target": "order-service",
            "parameters": {"action": "scale"},
            "reasoning": "Queue depth at 15000+. Scaling order-service to drain the backlog.",
        },
        {
            "action_type": "escalate",
            "target": "platform-team",
            "parameters": {"priority": "urgent", "message": "Cascading outage caused by auth-service v3.1.0 memory leak. Rolling back. Need platform support for queue recovery."},
            "reasoning": "Platform team needs to be aware of the cascading impact.",
        },
        {
            "action_type": "escalate",
            "target": "auth-team",
            "parameters": {"priority": "urgent", "message": "auth-service v3.1.0 has unbounded memory growth in token cache. Rolled back to v3.0.9. Please investigate before re-deploying."},
            "reasoning": "Auth team owns the service and needs to fix the root cause code.",
        },
        {
            "action_type": "communicate",
            "target": "status_page",
            "parameters": {"message": "INCIDENT: Multiple services affected due to auth-service degradation. Root cause identified (bad deployment). Rollback in progress. ETA for full recovery: 15 minutes."},
            "reasoning": "External stakeholders need to know the status.",
        },
        {
            "action_type": "communicate",
            "target": "slack",
            "parameters": {"message": "Incident update: auth-service v3.1.0 rolled back. Memory leak in token cache was root cause. Order queue draining. Monitoring recovery."},
            "reasoning": "Internal team needs current status.",
        },
    ]


RULE_BASED_ACTIONS = {
    "severity_classification": _rule_based_easy,
    "root_cause_analysis": _rule_based_medium,
    "full_incident_management": _rule_based_hard,
}


# ---------------------------------------------------------------------------
# Episode runners
# ---------------------------------------------------------------------------

def run_episode_rules(
    task_id: str,
    *,
    base_url: Optional[str] = None,
    env_instance: Any = None,
) -> Dict[str, Any]:
    """Run one episode with the rule-based baseline."""
    actions = RULE_BASED_ACTIONS[task_id]()

    if env_instance is not None:
        return _run_direct(task_id, actions, env_instance)
    else:
        return _run_http(task_id, actions, base_url)  # type: ignore[arg-type]


def _run_direct(task_id: str, actions: List[Dict], env_instance: Any) -> Dict[str, Any]:
    """Run episode directly against an env instance (in-process)."""
    from src.models import Action

    env_instance.reset(task_id)
    total_reward = 0.0
    steps = 0

    for act_dict in actions:
        action = Action(**act_dict)
        result = env_instance.step(action)
        total_reward += result.reward.value
        steps += 1
        if result.done:
            break

    grader_result = env_instance.grade()
    return {
        "task_id": task_id,
        "score": grader_result.score,
        "steps_taken": steps,
        "cumulative_reward": round(total_reward, 4),
        "grader_breakdown": grader_result.breakdown,
        "grader_feedback": grader_result.feedback,
    }


def _run_http(
    task_id: str,
    actions: List[Dict],
    base_url: str,
) -> Dict[str, Any]:
    """Run episode against the HTTP API."""
    client = httpx.Client(base_url=base_url, timeout=30.0)

    # Reset — capture session_id for all subsequent calls
    resp = client.post("/reset", json={"task_id": task_id})
    resp.raise_for_status()
    session_id = resp.json()["session_id"]
    headers = {"X-Session-ID": session_id}

    total_reward = 0.0
    steps = 0
    done = False

    for act_dict in actions:
        if done:
            break
        resp = client.post("/step", json=act_dict, headers=headers)
        resp.raise_for_status()
        result = resp.json()
        total_reward += result["reward"]["value"]
        steps += 1
        done = result["done"]

    # Get grader score
    resp = client.post("/grader", headers=headers)
    resp.raise_for_status()
    grader = resp.json()

    return {
        "task_id": task_id,
        "score": grader["score"],
        "steps_taken": steps,
        "cumulative_reward": round(total_reward, 4),
        "grader_breakdown": grader["breakdown"],
        "grader_feedback": grader.get("feedback", ""),
    }


def run_episode_llm(
    task_id: str,
    base_url: str,
    model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """Run one episode with an LLM agent via the OpenAI API."""
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("openai package required for LLM baseline. pip install openai")

    # Support competition env vars (API_BASE_URL, HF_TOKEN, MODEL_NAME)
    # as well as the standard OPENAI_API_KEY
    api_key = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set HF_TOKEN (or OPENAI_API_KEY) environment variable.")

    api_base = os.environ.get("API_BASE_URL")
    effective_model = os.environ.get("MODEL_NAME", model)

    llm_kwargs: Dict[str, Any] = {"api_key": api_key}
    if api_base:
        llm_kwargs["base_url"] = api_base

    llm = OpenAI(**llm_kwargs)
    client = httpx.Client(base_url=base_url, timeout=30.0)

    # Reset environment
    resp = client.post("/reset", json={"task_id": task_id})
    resp.raise_for_status()
    obs = resp.json()
    session_id = obs["session_id"]
    headers = {"X-Session-ID": session_id}

    total_reward = 0.0
    steps = 0
    done = False
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while not done and steps < obs.get("max_steps", 20):
        # Build user message with current observation
        user_msg = (
            f"Current observation (step {obs['step_number']}/{obs['max_steps']}):\n"
            f"{json.dumps(obs, indent=2, default=str)}\n\n"
            "What is your next action? Return ONLY a JSON object."
        )
        messages.append({"role": "user", "content": user_msg})

        # Query LLM
        completion = llm.chat.completions.create(
            model=effective_model,
            messages=messages,
            temperature=0.0,
            max_tokens=500,
            response_format={"type": "json_object"},
        )
        assistant_msg = completion.choices[0].message.content or "{}"
        messages.append({"role": "assistant", "content": assistant_msg})

        # Parse action and step
        try:
            action_dict = json.loads(assistant_msg)
        except json.JSONDecodeError:
            action_dict = {
                "action_type": "communicate",
                "target": "slack",
                "parameters": {"message": "Error parsing response"},
                "reasoning": "JSON parse error fallback",
            }

        resp = client.post("/step", json=action_dict, headers=headers)
        resp.raise_for_status()
        result = resp.json()

        obs = result["observation"]
        total_reward += result["reward"]["value"]
        steps += 1
        done = result["done"]

    # Final grader
    resp = client.post("/grader", headers=headers)
    resp.raise_for_status()
    grader = resp.json()

    return {
        "task_id": task_id,
        "score": grader["score"],
        "steps_taken": steps,
        "cumulative_reward": round(total_reward, 4),
        "grader_breakdown": grader["breakdown"],
        "grader_feedback": grader.get("feedback", ""),
    }


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def run_all_tasks(
    base_url: Optional[str] = None,
    env_instance: Any = None,
    mode: str = "rules",
    model: str = "gpt-4o-mini",
) -> List[Dict[str, Any]]:
    """Run baseline inference on all 3 tasks and return results."""
    results = []
    for task_id in TASK_IDS:
        if mode == "llm" and base_url:
            result = run_episode_llm(task_id, base_url, model=model)
        else:
            result = run_episode_rules(task_id, base_url=base_url, env_instance=env_instance)
        results.append(result)
        print(f"  Task: {task_id:30s}  Score: {result['score']:.4f}  Steps: {result['steps_taken']}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Incident Response Triage – Baseline Inference")
    parser.add_argument("--mode", choices=["rules", "llm"], default="rules",
                        help="Baseline mode: rule-based or LLM-based")
    parser.add_argument("--base-url", default="http://localhost:7860",
                        help="Base URL of the running environment server")
    parser.add_argument("--model", default="gpt-4o-mini",
                        help="OpenAI model to use for LLM baseline")
    parser.add_argument("--direct", action="store_true",
                        help="Run in-process (no HTTP server needed)")
    args = parser.parse_args()

    print("=" * 60)
    print("Incident Response Triage – Baseline Inference")
    print(f"Mode: {args.mode}")
    print("=" * 60)

    if args.direct:
        from src.environment import IncidentResponseEnv
        env = IncidentResponseEnv()
        results = run_all_tasks(env_instance=env, mode=args.mode)
    else:
        results = run_all_tasks(base_url=args.base_url, mode=args.mode, model=args.model)

    print("=" * 60)
    mean_score = sum(r["score"] for r in results) / len(results)
    print(f"Mean score: {mean_score:.4f}")
    print("=" * 60)

    # Print detailed breakdown
    for r in results:
        print(f"\n--- {r['task_id']} ---")
        print(f"  Score: {r['score']:.4f}")
        print(f"  Steps: {r['steps_taken']}")
        print(f"  Cumulative reward: {r['cumulative_reward']:.4f}")
        print(f"  Feedback: {r.get('grader_feedback', 'N/A')}")
        if r.get("grader_breakdown"):
            for k, v in r["grader_breakdown"].items():
                print(f"    {k}: {v:.4f}")


if __name__ == "__main__":
    main()
