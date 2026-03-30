"""
Inference Script — Incident Response Triage (OpenEnv)
=====================================================
MANDATORY
- Before submitting, ensure the following variables are defined in your
  environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- The inference script must be named `inference.py` and placed in the root
  directory of the project.
- Participants must use OpenAI Client for all LLM calls using above variables.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

import httpx
from openai import OpenAI

# ---------------------------------------------------------------------------
# Required competition env vars
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")

# Environment endpoint (this OpenEnv server, not the LLM endpoint)
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

TASK_IDS = [
    "severity_classification",
    "root_cause_analysis",
    "full_incident_management",
]

MAX_STEPS_OVERRIDE = 20       # safety cap
TEMPERATURE = 0.0
MAX_TOKENS = 500

# ---------------------------------------------------------------------------
# System prompt for the LLM agent
# ---------------------------------------------------------------------------
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
# Rule-based baselines (deterministic, no LLM needed)
# ---------------------------------------------------------------------------

def _rule_based_easy() -> List[Dict[str, Any]]:
    return [
        {"action_type": "investigate", "target": "postgres-primary", "parameters": {},
         "reasoning": "Alert shows connection pool at 98% on postgres-primary."},
        {"action_type": "investigate", "target": "user-service", "parameters": {},
         "reasoning": "user-service has high latency. Checking logs for pool errors."},
        {"action_type": "classify", "target": "", "parameters": {"severity": "P2"},
         "reasoning": "Service degraded but not fully down. Connection pool issue is P2."},
    ]


def _rule_based_medium() -> List[Dict[str, Any]]:
    return [
        {"action_type": "investigate", "target": "payment-gateway", "parameters": {},
         "reasoning": "Payment success rate critically low. Starting here."},
        {"action_type": "investigate", "target": "redis-session", "parameters": {},
         "reasoning": "Eviction spike on redis-session could explain missing tokens."},
        {"action_type": "classify", "target": "", "parameters": {"severity": "P1"},
         "reasoning": "Payment processing at 45% success is P1 revenue-impacting."},
        {"action_type": "diagnose", "target": "redis-session",
         "parameters": {"root_cause": "Redis session store hit maxmemory limit causing eviction of payment session tokens. Sessions evicted before payment completion."},
         "reasoning": "Logs show redis-session at 100% memory with aggressive evictions."},
        {"action_type": "remediate", "target": "redis-session", "parameters": {"action": "scale"},
         "reasoning": "Scaling redis-session memory to stop evictions."},
    ]


def _rule_based_hard() -> List[Dict[str, Any]]:
    return [
        {"action_type": "investigate", "target": "auth-service", "parameters": {},
         "reasoning": "Auth-service has critical latency. Multiple services depend on auth."},
        {"action_type": "investigate", "target": "api-gateway", "parameters": {},
         "reasoning": "API gateway returning 503s. Checking if auth-related."},
        {"action_type": "investigate", "target": "redis-auth-cache", "parameters": {},
         "reasoning": "Checking auth cache — may explain why auth is slow."},
        {"action_type": "classify", "target": "", "parameters": {"severity": "P1"},
         "reasoning": "Cascading multi-service outage. P1."},
        {"action_type": "diagnose", "target": "auth-service",
         "parameters": {"root_cause": "Bad deployment v3.1.0 introduced memory leak via unbounded in-memory token cache. Auth-service OOMKill causes cascading failures."},
         "reasoning": "Auth-service logs show v3.1.0 deployment, memory climbing to 97%."},
        {"action_type": "remediate", "target": "auth-service", "parameters": {"action": "rollback"},
         "reasoning": "Rolling back auth-service to v3.0.9."},
        {"action_type": "remediate", "target": "order-service", "parameters": {"action": "scale"},
         "reasoning": "Queue depth at 15000+. Scaling to drain backlog."},
        {"action_type": "escalate", "target": "platform-team",
         "parameters": {"priority": "urgent", "message": "Cascading outage caused by auth-service v3.1.0 memory leak. Rolling back. Need platform support for queue recovery."},
         "reasoning": "Platform team needs to be aware."},
        {"action_type": "escalate", "target": "auth-team",
         "parameters": {"priority": "urgent", "message": "auth-service v3.1.0 has unbounded memory growth in token cache. Rolled back to v3.0.9. Please investigate before re-deploying."},
         "reasoning": "Auth team owns the service."},
        {"action_type": "communicate", "target": "status_page",
         "parameters": {"message": "INCIDENT: Multiple services affected due to auth-service degradation. Root cause identified. Rollback in progress. ETA 15 min."},
         "reasoning": "External stakeholders need status."},
        {"action_type": "communicate", "target": "slack",
         "parameters": {"message": "Incident update: auth-service v3.1.0 rolled back. Memory leak root cause. Order queue draining. Monitoring recovery."},
         "reasoning": "Internal team status update."},
    ]


RULE_BASED_ACTIONS = {
    "severity_classification": _rule_based_easy,
    "root_cause_analysis": _rule_based_medium,
    "full_incident_management": _rule_based_hard,
}


# ---------------------------------------------------------------------------
# Episode runners
# ---------------------------------------------------------------------------

def run_episode_rules(task_id: str, env_url: str) -> Dict[str, Any]:
    """Run one episode using the deterministic rule-based baseline."""
    actions = RULE_BASED_ACTIONS[task_id]()
    client = httpx.Client(base_url=env_url, timeout=30.0)

    resp = client.post("/reset", json={"task_id": task_id, "variant_seed": 0})
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


def run_episode_llm(task_id: str, env_url: str) -> Dict[str, Any]:
    """Run one episode with an LLM agent using the OpenAI Client."""
    if not API_KEY:
        raise RuntimeError(
            "HF_TOKEN (or API_KEY) environment variable not set. "
            "Required for LLM inference."
        )

    llm = OpenAI(
        api_key=API_KEY,
        base_url=API_BASE_URL,
    )
    client = httpx.Client(base_url=env_url, timeout=30.0)

    # Reset environment
    resp = client.post("/reset", json={"task_id": task_id})
    resp.raise_for_status()
    obs = resp.json()
    session_id = obs["session_id"]
    headers = {"X-Session-ID": session_id}

    total_reward = 0.0
    steps = 0
    done = False
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    max_steps = obs.get("max_steps", MAX_STEPS_OVERRIDE)

    while not done and steps < max_steps:
        user_msg = (
            f"Current observation (step {obs.get('step_number', steps)}/{max_steps}):\n"
            f"{json.dumps(obs, indent=2, default=str)}\n\n"
            "What is your next action? Return ONLY a JSON object."
        )
        messages.append({"role": "user", "content": user_msg})

        completion = llm.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        assistant_msg = completion.choices[0].message.content or "{}"
        messages.append({"role": "assistant", "content": assistant_msg})

        # Extract JSON from response (handle markdown fences)
        cleaned = assistant_msg.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines)

        try:
            action_dict = json.loads(cleaned)
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
# Main
# ---------------------------------------------------------------------------

def main():
    # Determine mode: if HF_TOKEN / API_KEY is set → try LLM, else rule-based
    use_llm = bool(API_KEY)
    mode = "llm" if use_llm else "rules"

    print("=" * 60)
    print("Incident Response Triage — Inference Script")
    print(f"Mode       : {mode}")
    print(f"ENV_BASE   : {ENV_BASE_URL}")
    if use_llm:
        print(f"API_BASE   : {API_BASE_URL}")
        print(f"MODEL      : {MODEL_NAME}")
    print("=" * 60)

    results: List[Dict[str, Any]] = []
    for task_id in TASK_IDS:
        if use_llm:
            result = run_episode_llm(task_id, ENV_BASE_URL)
        else:
            result = run_episode_rules(task_id, ENV_BASE_URL)
        results.append(result)
        print(f"  Task: {task_id:30s}  Score: {result['score']:.4f}  Steps: {result['steps_taken']}")

    print("=" * 60)
    mean_score = sum(r["score"] for r in results) / len(results)
    print(f"Mean score: {mean_score:.4f}")
    print("=" * 60)

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
