"""
Adversarial Designer — generates novel hard scenarios to break the agent.

Architecture (from kube-sre-gym/server/adversarial_designer.py, 1st place SF winner):
  - Triggered when curriculum reaches intermediate+ tier AND agent is doing well
  - Identifies weak spots (scenarios agent consistently fails)
  - Uses an LLM to generate new scenario variants that target those weak spots
  - Validates generated scenarios pass basic sanity checks
  - New scenarios are inserted into training rotation with difficulty = 0.85+

Key innovations added on top of kube-sre-gym:
  1. Scenario diversity check — new scenarios must be structurally different
  2. Adversarial warmup — first shows current weakest variants, then novel attacks
  3. Blinding antipattern detection — rejects scenarios with obvious tells
  4. Human-in-the-loop escape hatch — dump generated scenarios to JSON for manual review

Usage:
    from training.adversarial import AdversarialDesigner
    from training.curriculum import get_curriculum

    designer = AdversarialDesigner(api_key=os.environ["GROQ_API_KEY"])
    curriculum = get_curriculum()

    if curriculum.should_use_adversarial():
        weak_spots = curriculum.weak_spots(top_n=2)
        new_scenarios = designer.generate(weak_spots, n=3)
        # new_scenarios is a list of dicts compatible with Scenario dataclass
        # register them with your environment
        designer.save_generated("outputs/adversarial_scenarios.json")
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
DESIGNER_MODEL = os.getenv("ADVERSARIAL_MODEL", "llama-3.3-70b-versatile")


# ---------------------------------------------------------------------------
# Scenario template — what a generated adversarial scenario must provide
# ---------------------------------------------------------------------------

SCENARIO_SCHEMA_DESCRIPTION = """
A scenario is a JSON object with these required fields:
  scenario_id:              string — unique ID like "adv_001"
  task_id:                  string — one of: severity_classification, root_cause_analysis, full_incident_management
  incident_id:              string — like "INC-ADV-001"
  description:              string — brief description of the incident
  initial_alerts:           list of {alert_id, service, severity, message, timestamp}
  available_services:       list of service names (strings)
  service_logs:             dict mapping service_name → list of log strings
  service_metrics:          dict mapping service_name → {cpu_usage, memory_usage, error_rate, request_rate, latency_p99}
  correct_severity:         string — one of: P1, P2, P3, P4
  correct_root_cause_service: string — the service that is the root cause
  correct_root_cause_keywords: list of strings — keywords that must appear in diagnosis
  valid_remediation_actions: list of strings
  expected_escalation_teams: list of strings
  max_steps:                integer — how many actions the agent gets
  degradation_per_step:     float — how fast the situation degrades (0.02–0.10)
  relevant_services:        list of strings — subset of available_services to investigate
  blast_radius:             dict mapping service_name → {affected: bool, severity_delta: float}
"""

# Different adversarial attack strategies
ADVERSARIAL_STRATEGIES = [
    "red_herring",         # wrong service shows high error rates; real cause is elsewhere
    "delayed_causality",   # root cause happened N steps ago; not obvious from current metrics
    "cascading_failure",   # 3+ services failing; need to find the origin
    "misleading_severity", # metrics look like P3 but it's actually P1 due to user impact
    "noisy_logs",          # hundreds of benign log entries hiding the critical one
    "ambiguous_escalation", # two teams both responsible; escalating only one is wrong
    "multi_fault",         # two independent faults at same time; both need remediation
    "silent_degradation",  # metrics flat but latency p99 creeping up over time
]


# ---------------------------------------------------------------------------
# Generator prompt
# ---------------------------------------------------------------------------

GENERATOR_PROMPT = """\
You are designing adversarial test scenarios for an AI incident response agent.

The agent is currently WEAK at these scenario types:
{weak_spots}

Your goal: create a scenario that a well-trained agent should handle,
but which specifically targets the agent's current weaknesses.

Use attack strategy: {strategy}

Attack strategy description:
- red_herring: Make the most alarming service look like the root cause, but it's actually a downstream victim; the real root cause has subtle metrics
- delayed_causality: The root cause service had a spike 10 steps ago; current metrics are flat but errors are still propagating
- cascading_failure: auth → payments → order services all failing, need to trace back to auth-service
- misleading_severity: Low error rate but 95% of traffic is affected; proper severity = P1 despite low raw error count
- noisy_logs: 500+ log lines, most are routine INFO; needle is a single ERROR line with the root cause
- ambiguous_escalation: Incident spans two team boundaries; both TEAM_A and TEAM_B must be escalated
- multi_fault: Two independent issues at same time; each needs separate remediation
- silent_degradation: latency_p99 doubles over 8 steps but error_rate stays at 0.001; correct answer is P2

{schema}

Return ONLY a valid JSON object matching the schema above. No explanation, no markdown.
"""


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = [
    "scenario_id", "task_id", "incident_id", "description",
    "initial_alerts", "available_services", "service_logs", "service_metrics",
    "correct_severity", "correct_root_cause_service", "correct_root_cause_keywords",
    "valid_remediation_actions", "expected_escalation_teams",
    "max_steps", "degradation_per_step", "relevant_services", "blast_radius",
]

VALID_TASK_IDS = {
    "severity_classification",
    "root_cause_analysis",
    "full_incident_management",
}

VALID_SEVERITIES = {"P1", "P2", "P3", "P4"}


def _validate_scenario(d: Dict[str, Any]) -> Tuple[bool, str]:
    """Returns (is_valid, reason)."""
    for f in REQUIRED_FIELDS:
        if f not in d:
            return False, f"Missing field: {f}"

    if d["task_id"] not in VALID_TASK_IDS:
        return False, f"Invalid task_id: {d['task_id']}"

    if d["correct_severity"] not in VALID_SEVERITIES:
        return False, f"Invalid severity: {d['correct_severity']}"

    if not isinstance(d["initial_alerts"], list) or len(d["initial_alerts"]) == 0:
        return False, "initial_alerts must be a non-empty list"

    root_cause = d["correct_root_cause_service"]
    if root_cause not in d["available_services"]:
        return False, f"correct_root_cause_service '{root_cause}' not in available_services"

    if not isinstance(d["correct_root_cause_keywords"], list) or len(d["correct_root_cause_keywords"]) < 2:
        return False, "correct_root_cause_keywords must have at least 2 items"

    if d["correct_root_cause_service"] not in d["service_logs"]:
        return False, "root cause service must have logs"

    # Antipattern: don't give away the answer in the alert message
    alert_msgs = " ".join(
        a.get("message", "") for a in d["initial_alerts"]
    ).lower()
    keywords = [kw.lower() for kw in d["correct_root_cause_keywords"]]
    matching = [kw for kw in keywords if kw in alert_msgs]
    if len(matching) >= len(keywords) // 2:
        return False, "Scenario is too obvious (alert messages contain root cause keywords)"

    # Basic metric sanity: each service must have required keys
    metric_keys = {"cpu_usage", "memory_usage", "error_rate", "request_rate", "latency_p99"}
    for svc, metrics in d["service_metrics"].items():
        if not metric_keys.issubset(set(metrics.keys())):
            return False, f"Service '{svc}' metrics missing keys: {metric_keys - set(metrics.keys())}"

    return True, "ok"


# ---------------------------------------------------------------------------
# Diversity check — avoid generating the same scenario twice
# ---------------------------------------------------------------------------

def _is_diverse(
    candidate: Dict[str, Any],
    existing: List[Dict[str, Any]],
    threshold: float = 0.60,
) -> bool:
    """
    Returns True if candidate is sufficiently different from all existing scenarios.
    Compares description + service names via bag-of-words Jaccard similarity.
    """
    def _tokens(d: Dict[str, Any]) -> set:
        text = (d.get("description", "") + " " + " ".join(d.get("available_services", []))).lower()
        return set(re.findall(r"\w+", text))

    cand_tokens = _tokens(candidate)
    for ex in existing:
        ex_tokens = _tokens(ex)
        if not cand_tokens or not ex_tokens:
            continue
        jaccard = len(cand_tokens & ex_tokens) / len(cand_tokens | ex_tokens)
        if jaccard >= threshold:
            return False
    return True


# ---------------------------------------------------------------------------
# AdversarialDesigner
# ---------------------------------------------------------------------------

class AdversarialDesigner:
    """
    Generates novel adversarial scenarios to break the agent's current blind spots.

    Main interface:
        designer = AdversarialDesigner(api_key="...")
        scenarios = designer.generate(weak_spots=[("root_cause_analysis", 1)], n=3)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DESIGNER_MODEL,
        max_attempts_per_scenario: int = 3,
    ) -> None:
        self._api_key = api_key or API_BASE_URL
        self._model = model
        self._max_attempts = max_attempts_per_scenario
        self._generated: List[Dict[str, Any]] = []
        self._strategy_index = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        weak_spots: List[Tuple[str, int]],
        n: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Generate `n` adversarial scenarios targeting the given weak spots.

        Args:
            weak_spots: list of (task_id, variant_seed) the agent struggles with
            n:          number of new scenarios to generate

        Returns:
            list of scenario dicts, each compatible with the Scenario dataclass
        """
        new_scenarios: List[Dict[str, Any]] = []

        weak_descriptions = "\n".join(
            f"  - task_id={task_id}, variant={variant_seed}"
            for task_id, variant_seed in weak_spots
        )

        for i in range(n):
            strategy = ADVERSARIAL_STRATEGIES[self._strategy_index % len(ADVERSARIAL_STRATEGIES)]
            self._strategy_index += 1

            scenario = self._generate_one(weak_descriptions, strategy, i)
            if scenario:
                new_scenarios.append(scenario)
                self._generated.append(scenario)

        logger.info("Generated %d/%d adversarial scenarios", len(new_scenarios), n)
        return new_scenarios

    def save_generated(self, path: str) -> None:
        """Save all generated scenarios to a JSON file for human review."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._generated, f, indent=2)
        logger.info("Saved %d adversarial scenarios to %s", len(self._generated), path)

    def load_generated(self, path: str) -> List[Dict[str, Any]]:
        """Load previously generated scenarios."""
        if not os.path.exists(path):
            return []
        with open(path) as f:
            self._generated = json.load(f)
        logger.info("Loaded %d adversarial scenarios from %s", len(self._generated), path)
        return self._generated

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _generate_one(
        self,
        weak_descriptions: str,
        strategy: str,
        index: int,
    ) -> Optional[Dict[str, Any]]:
        """Generate one scenario, with retry logic and validation."""
        prompt = GENERATOR_PROMPT.format(
            weak_spots=weak_descriptions,
            strategy=strategy,
            schema=SCENARIO_SCHEMA_DESCRIPTION,
        )

        for attempt in range(self._max_attempts):
            try:
                raw = self._call_llm(prompt)
                scenario = self._parse_json(raw)
                if scenario is None:
                    continue

                # Assign unique ID
                scenario["scenario_id"] = f"adv_{int(time.time())}_{index:03d}"

                # Validate
                is_valid, reason = _validate_scenario(scenario)
                if not is_valid:
                    logger.debug("Attempt %d invalid: %s", attempt + 1, reason)
                    continue

                # Diversity check
                existing = self._generated + self._get_builtin_scenarios()
                if not _is_diverse(scenario, existing):
                    logger.debug("Attempt %d too similar to existing scenario", attempt + 1)
                    continue

                return scenario

            except Exception as e:
                logger.warning("Designer attempt %d/%d failed: %s", attempt + 1, self._max_attempts, e)
                if attempt < self._max_attempts - 1:
                    time.sleep(2 ** attempt)

        logger.warning("Failed to generate valid scenario for strategy=%s after %d attempts",
                        strategy, self._max_attempts)
        return None

    def _call_llm(self, prompt: str) -> str:
        """Synchronous LLM call for scenario generation."""
        api_key = self._api_key or os.getenv("GROQ_API_KEY") or os.getenv("API_KEY", "")
        if not api_key:
            raise ValueError("No API key set for AdversarialDesigner")

        with httpx.Client() as client:
            response = client.post(
                f"{API_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": self._model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.8,  # higher creativity for diversity
                    "max_tokens": 2000,
                },
                timeout=60.0,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

    def _parse_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from LLM response text."""
        # Try direct parse
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Try finding JSON block
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            logger.debug("No JSON found in response: %s", text[:100])
            return None
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError as e:
            logger.debug("JSON parse failed: %s", e)
            return None

    @staticmethod
    def _get_builtin_scenarios() -> List[Dict[str, Any]]:
        """Returns stubs of existing built-in scenarios for diversity comparison."""
        return [
            {"description": "auth service high error rate", "available_services": ["auth-service", "api-gateway"]},
            {"description": "database connection pool exhausted", "available_services": ["postgres", "auth-service"]},
            {"description": "memory leak in payment service", "available_services": ["payment-service", "order-service"]},
        ]


# ---------------------------------------------------------------------------
# Warmup adversarial schedule
# ---------------------------------------------------------------------------

def build_adversarial_schedule(
    curriculum_tier: int,
    weak_spots: List[Tuple[str, int]],
    n_generated: int,
    adversarial_scenarios: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Returns the subset of adversarial scenarios to use this training epoch.

    Strategy (from kube-sre-gym adversarial warmup):
      - Tier 2 (intermediate): 20% adversarial, 80% curriculum
      - Tier 3 (expert): 40% adversarial, 60% curriculum

    The adversarial scenarios targeting weak spots are shown first.
    """
    if not adversarial_scenarios:
        return []

    # Sort: weak spot scenarios first
    def _relevance(s: Dict[str, Any]) -> int:
        return -1 if s.get("task_id") in [ws[0] for ws in weak_spots] else 0

    sorted_scenarios = sorted(adversarial_scenarios, key=_relevance)

    # Proportion based on tier
    if curriculum_tier == 2:
        n = max(1, int(n_generated * 0.20))
    elif curriculum_tier >= 3:
        n = max(1, int(n_generated * 0.40))
    else:
        n = 0

    return sorted_scenarios[:n]
