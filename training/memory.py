"""
Cross-episode agent memory system.

Stores observations, strategies, and lessons learned across training episodes.
Injected into the system prompt at the start of every new episode so the
agent builds on past experience.

Inspired by kube-sre-gym's episodic memory and the open-env-assistant
memory consolidation approach.

Usage:
    from training.memory import (
        load_agent_memory,
        save_agent_memory,
        record_episode,
        build_memory_context,
        maybe_consolidate_memory,
    )

    memory = load_agent_memory()
    context_str = build_memory_context(memory)
    # inject context_str into system prompt

    # after episode ends:
    memory = record_episode(memory, {
        "task_id": "root_cause_analysis",
        "score": 0.82,
        "steps": 7,
        "trajectory_summary": "Investigated auth-service first, found JWT expiry bug",
        "mistakes": ["Escalated too early before diagnosing"],
        "successes": ["Correctly identified root cause on step 3"],
    })
    save_agent_memory(memory)
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_PATH = os.path.join("outputs", "agent_memory.json")

# Max items stored per category before consolidation triggers
MAX_EPISODES_STORED = 100
MAX_RULES_PER_TASK  = 10
CONSOLIDATION_EVERY = 20   # consolidate after every N episodes


# ---------------------------------------------------------------------------
# Memory schema
# ---------------------------------------------------------------------------

def _empty_memory() -> Dict[str, Any]:
    return {
        "version": 1,
        "total_episodes": 0,
        "last_consolidated_at": None,
        "global_rules": [],        # list of str — apply to every task
        "task_rules": {},          # task_id → list of str
        "episode_log": [],         # last MAX_EPISODES_STORED episodes
        "score_history": {},       # task_id → list of float
        "mistakes": [],            # list of str — common mistakes to avoid
        "successes": [],           # list of str — things that worked well
    }


# ---------------------------------------------------------------------------
# Load / Save
# ---------------------------------------------------------------------------

def load_agent_memory(path: str = DEFAULT_PATH) -> Dict[str, Any]:
    """Load memory from disk. Returns empty memory if file doesn't exist."""
    if not os.path.exists(path):
        logger.info("No memory file found at %s, starting fresh", path)
        return _empty_memory()
    try:
        with open(path) as f:
            data = json.load(f)
        logger.info(
            "Loaded memory: %d episodes, %d global rules",
            data.get("total_episodes", 0),
            len(data.get("global_rules", [])),
        )
        return data
    except Exception as e:
        logger.warning("Failed to load memory from %s: %s — starting fresh", path, e)
        return _empty_memory()


def save_agent_memory(memory: Dict[str, Any], path: str = DEFAULT_PATH) -> None:
    """Save memory to disk."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # Trim episode log before saving
    memory["episode_log"] = memory["episode_log"][-MAX_EPISODES_STORED:]
    with open(path, "w") as f:
        json.dump(memory, f, indent=2)
    logger.debug("Saved memory to %s", path)


# ---------------------------------------------------------------------------
# Record an episode
# ---------------------------------------------------------------------------

def record_episode(
    memory: Dict[str, Any],
    episode_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Record a completed episode into memory.

    episode_data keys:
        task_id (str)            — which task was attempted
        score (float)            — 0.0–1.0 final score
        steps (int)              — number of steps taken
        trajectory_summary (str) — 1-2 sentence summary of what happened
        mistakes (list[str])     — things that went wrong (optional)
        successes (list[str])    — things that worked (optional)
    """
    task_id = episode_data.get("task_id", "unknown")
    score   = float(episode_data.get("score", 0.0))

    # Score history per task
    if task_id not in memory["score_history"]:
        memory["score_history"][task_id] = []
    memory["score_history"][task_id].append(score)

    # Episode log
    log_entry = {
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "task_id":      task_id,
        "score":        score,
        "steps":        episode_data.get("steps", 0),
        "summary":      episode_data.get("trajectory_summary", ""),
    }
    memory["episode_log"].append(log_entry)
    memory["total_episodes"] = memory.get("total_episodes", 0) + 1

    # Extract mistakes and successes
    for mistake in episode_data.get("mistakes", []):
        if mistake and mistake not in memory["mistakes"]:
            memory["mistakes"].append(mistake)

    for success in episode_data.get("successes", []):
        if success and success not in memory["successes"]:
            memory["successes"].append(success)

    # Auto-generate rules from patterns
    _update_rules_from_episode(memory, task_id, score, episode_data)

    return memory


def _update_rules_from_episode(
    memory: Dict[str, Any],
    task_id: str,
    score: float,
    episode_data: Dict[str, Any],
) -> None:
    """Derive rules from episode outcome and add to task_rules."""
    if task_id not in memory["task_rules"]:
        memory["task_rules"][task_id] = []

    task_rules = memory["task_rules"][task_id]

    # High-score episode: extract successes as rules
    if score >= 0.85 and episode_data.get("successes"):
        for s in episode_data["successes"]:
            rule = f"[WORKS] {s}"
            if rule not in task_rules:
                task_rules.append(rule)

    # Low-score episode: extract mistakes as rules
    if score < 0.50 and episode_data.get("mistakes"):
        for m in episode_data["mistakes"]:
            rule = f"[AVOID] {m}"
            if rule not in task_rules:
                task_rules.append(rule)

    # Trim to max
    memory["task_rules"][task_id] = task_rules[-MAX_RULES_PER_TASK:]


# ---------------------------------------------------------------------------
# Build context string for injection into system prompt
# ---------------------------------------------------------------------------

def build_memory_context(
    memory: Dict[str, Any],
    task_id: Optional[str] = None,
    max_rules: int = 5,
    max_recent: int = 3,
) -> str:
    """
    Build a concise memory context string for injection into the system prompt.

    Returns a string of ~200 tokens that summarizes key lessons learned.
    Inject this at the TOP of the system prompt before each episode.
    """
    lines: List[str] = ["## MEMORY FROM PAST EPISODES"]

    # Task-specific rules
    if task_id and task_id in memory.get("task_rules", {}):
        rules = memory["task_rules"][task_id][-max_rules:]
        if rules:
            lines.append(f"\nRules for {task_id}:")
            for rule in rules:
                lines.append(f"  - {rule}")

    # Global rules
    global_rules = memory.get("global_rules", [])[-max_rules:]
    if global_rules:
        lines.append("\nGeneral rules (all tasks):")
        for rule in global_rules:
            lines.append(f"  - {rule}")

    # Common mistakes
    mistakes = memory.get("mistakes", [])[-3:]
    if mistakes:
        lines.append("\nMistakes to avoid:")
        for m in mistakes:
            lines.append(f"  - AVOID: {m}")

    # Recent episode outcomes for this task
    if task_id:
        recent = [
            ep for ep in memory.get("episode_log", [])
            if ep.get("task_id") == task_id
        ][-max_recent:]
        if recent:
            lines.append(f"\nRecent {task_id} episodes:")
            for ep in recent:
                lines.append(
                    f"  - Score {ep['score']:.2f} in {ep['steps']} steps: {ep['summary'][:100]}"
                )

    # Mean score for this task (self-awareness)
    if task_id and task_id in memory.get("score_history", {}):
        scores = memory["score_history"][task_id]
        if scores:
            mean = sum(scores) / len(scores)
            lines.append(f"\nYour mean score on {task_id}: {mean:.2f} (over {len(scores)} episodes)")

    if len(lines) == 1:
        return ""   # No memory yet — return empty string

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM-based memory consolidation (optional, requires API key)
# ---------------------------------------------------------------------------

def maybe_consolidate_memory(
    memory: Dict[str, Any],
    api_key: Optional[str] = None,
    path: str = DEFAULT_PATH,
) -> Dict[str, Any]:
    """
    Every CONSOLIDATION_EVERY episodes, use an LLM to distill episode logs
    into concise, high-signal rules. Saves tokens in future prompts.

    If no API key is available, falls back to simple heuristic consolidation.
    """
    total = memory.get("total_episodes", 0)
    last  = memory.get("last_consolidated_at") or 0
    if isinstance(last, str):
        last = 0  # reset if it was stored as ISO string

    if total - last < CONSOLIDATION_EVERY:
        return memory   # not yet due

    if api_key or os.getenv("GROQ_API_KEY"):
        memory = _llm_consolidate(memory, api_key or os.getenv("GROQ_API_KEY"))
    else:
        memory = _heuristic_consolidate(memory)

    memory["last_consolidated_at"] = total
    save_agent_memory(memory, path)
    return memory


def _heuristic_consolidate(memory: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simple rule: look at episodes where score > 0.85, extract their summaries
    as global rules. Deduplicate. Trim old ones.
    """
    new_rules: List[str] = []
    for ep in memory.get("episode_log", []):
        if ep.get("score", 0.0) >= 0.85 and ep.get("summary"):
            rule = f"[HIGH SCORE {ep['score']:.2f}] {ep['summary'][:120]}"
            if rule not in new_rules:
                new_rules.append(rule)

    # Merge with existing global rules (keep most recent)
    combined = memory.get("global_rules", []) + new_rules
    memory["global_rules"] = list(dict.fromkeys(combined))[-MAX_RULES_PER_TASK * 2:]

    logger.info("Heuristic consolidation: %d global rules", len(memory["global_rules"]))
    return memory


def _llm_consolidate(memory: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    """Use LLM to distill episode logs into concise rules."""
    try:
        import httpx

        episode_summary = "\n".join(
            f"task={ep['task_id']} score={ep['score']:.2f}: {ep['summary']}"
            for ep in memory.get("episode_log", [])[-30:]  # last 30 episodes
        )

        prompt = f"""You are analyzing an AI agent's performance across multiple episodes.
Here are recent episode outcomes:

{episode_summary}

Extract 5 concise, actionable rules the agent should follow in future episodes.
Each rule should be 1 sentence. Focus on what WORKS and what to AVOID.

Return ONLY a JSON array of strings:
["Rule 1...", "Rule 2...", ...]
"""
        response = httpx.post(
            f"{os.getenv('API_BASE_URL', 'https://api.groq.com/openai/v1')}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 300,
            },
            timeout=30.0,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]

        start = content.find("[")
        end   = content.rfind("]") + 1
        if start != -1 and end > 0:
            new_rules: List[str] = json.loads(content[start:end])
            existing = memory.get("global_rules", [])
            combined = existing + [f"[CONSOLIDATED] {r}" for r in new_rules]
            memory["global_rules"] = list(dict.fromkeys(combined))[-MAX_RULES_PER_TASK * 2:]
            logger.info("LLM consolidation: extracted %d new rules", len(new_rules))

    except Exception as e:
        logger.warning("LLM consolidation failed: %s — falling back to heuristic", e)
        memory = _heuristic_consolidate(memory)

    return memory


# ---------------------------------------------------------------------------
# Utility: memory stats for logging
# ---------------------------------------------------------------------------

def memory_summary(memory: Dict[str, Any]) -> Dict[str, Any]:
    """Human-readable summary of current memory state."""
    return {
        "total_episodes":  memory.get("total_episodes", 0),
        "global_rules":    len(memory.get("global_rules", [])),
        "task_rules":      {k: len(v) for k, v in memory.get("task_rules", {}).items()},
        "mistakes_stored": len(memory.get("mistakes", [])),
        "scores_by_task":  {
            k: round(sum(v) / len(v), 3)
            for k, v in memory.get("score_history", {}).items()
            if v
        },
    }
