"""
Curriculum Controller — drives progressive difficulty across training episodes.

Adapted from kube-sre-gym/server/curriculum.py (1st place SF hackathon winner).
Adapted for the IRT (Incident Response Triage) OpenEnv environment.

Training flow:
  1. Agent starts with easy single-fault scenarios (severity_classification easy variants)
  2. As it masters each scenario type (>= MASTERY_THRESHOLD over MASTERY_WINDOW),
     that type is "graduated" and harder variants unlock
  3. Difficulty tiers: warmup -> beginner -> intermediate -> expert
  4. In adversarial mode, curriculum feeds weak spots to the adversarial designer

Key design (from kube-sre-gym):
  - Clean state + ONE scenario per episode → clean reward signal for GRPO
  - Agent must sustain success rate over multiple episodes before advancing
  - Judge strictness scales with tier: lenient → normal → strict

Usage:
    from training.curriculum import CurriculumController

    curriculum = CurriculumController()
    task_id, variant_seed = curriculum.select_episode()
    # ... run episode ...
    curriculum.record_episode(task_id, variant_seed, score=0.82, steps=5)
    if curriculum.should_advance():
        curriculum.advance_tier()
"""

from __future__ import annotations

import logging
import os
import json
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scenario difficulty map
# Each (task_id, variant_seed) pair gets a difficulty score 0.0–1.0
# ---------------------------------------------------------------------------

SCENARIO_DIFFICULTY: Dict[Tuple[str, int], float] = {
    # Tier 1 — easy, obvious symptoms
    ("severity_classification", 0): 0.10,
    ("severity_classification", 1): 0.15,
    ("severity_classification", 2): 0.20,
    # Tier 2 — medium, requires investigation
    ("root_cause_analysis", 0): 0.35,
    ("root_cause_analysis", 1): 0.45,
    ("root_cause_analysis", 2): 0.50,
    # Tier 3 — hard, multi-step, escalation required
    ("full_incident_management", 0): 0.65,
    ("full_incident_management", 1): 0.75,
    ("full_incident_management", 2): 0.85,
}

# ---------------------------------------------------------------------------
# Difficulty tiers — agent must earn its way through each tier
# ---------------------------------------------------------------------------

DIFFICULTY_TIERS = [
    {
        "name": "warmup",
        "max_diff": 0.20,      # only easy severity scenarios
        "min_episodes": 3,
        "advance_rate": 0.60,  # 60% success rate to advance
    },
    {
        "name": "beginner",
        "max_diff": 0.50,      # severity + easy root cause
        "min_episodes": 5,
        "advance_rate": 0.65,
    },
    {
        "name": "intermediate",
        "max_diff": 0.75,      # all tasks, easy+medium variants
        "min_episodes": 8,
        "advance_rate": 0.68,
    },
    {
        "name": "expert",
        "max_diff": 1.00,      # all scenarios including hard
        "min_episodes": 0,     # no cap — stay here until training ends
        "advance_rate": 1.00,  # no advancing past expert
    },
]

MASTERY_THRESHOLD = 0.70   # 70% mean score = mastered
MASTERY_WINDOW = 10        # look at last N episodes per scenario type
MIN_EPISODES_FOR_MASTERY = 3  # need at least 3 attempts before graduating


@dataclass
class EpisodeRecord:
    task_id: str
    variant_seed: int
    score: float
    steps: int
    tier_name: str


@dataclass
class CurriculumState:
    tier_index: int = 0
    tier_episodes: int = 0  # episodes spent in current tier
    total_episodes: int = 0
    graduated: List[Tuple[str, int]] = field(default_factory=list)
    history: List[EpisodeRecord] = field(default_factory=list)


class CurriculumController:
    """
    Tracks agent skill across scenario types.
    Drives: scenario selection, difficulty progression, weak-spot targeting.

    Progression: warmup → beginner → intermediate → expert.
    Agent must sustain MASTERY_THRESHOLD success rate over MASTERY_WINDOW episodes
    before advancing to the next tier.

    Example:
        ctrl = CurriculumController()
        for episode in range(200):
            task_id, variant_seed = ctrl.select_episode()
            score = run_episode(task_id, variant_seed)
            ctrl.record_episode(task_id, variant_seed, score, steps)
    """

    def __init__(self, state_path: Optional[str] = None) -> None:
        self._state = CurriculumState()
        self._state_path = state_path or os.path.join(
            "outputs", "curriculum_state.json"
        )
        # Allow forcing min difficulty for eval (skips warmup)
        min_diff = float(os.environ.get("EVAL_MIN_DIFFICULTY", "0.0"))
        if min_diff > 0:
            for i, tier in enumerate(DIFFICULTY_TIERS):
                if tier["max_diff"] >= min_diff:
                    self._state.tier_index = i
                    break
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def tier_index(self) -> int:
        return self._state.tier_index

    @property
    def tier_name(self) -> str:
        return DIFFICULTY_TIERS[self._state.tier_index]["name"]

    @property
    def total_episodes(self) -> int:
        return self._state.total_episodes

    def select_episode(self, prefer_weak_spots: bool = True) -> Tuple[str, int]:
        """
        Select the next (task_id, variant_seed) to train on.

        Strategy:
        - Only expose scenarios within the current tier's max difficulty
        - If prefer_weak_spots=True, bias toward scenarios the agent scored lowest
        - Randomly sample among eligible scenarios with score-weighted probability
        """
        eligible = self._eligible_scenarios()
        if not eligible:
            # Fallback: use the easiest scenario
            return ("severity_classification", 0)

        if prefer_weak_spots and self._state.history:
            # Compute mean score per (task_id, variant_seed)
            scores: Dict[Tuple[str, int], List[float]] = defaultdict(list)
            for rec in self._state.history[-50:]:  # last 50 episodes
                key = (rec.task_id, rec.variant_seed)
                if key in eligible:
                    scores[key].append(rec.score)

            # Weight unseen / weak scenarios higher
            weights = []
            for key in eligible:
                if key not in scores:
                    weights.append(2.0)  # unseen → highest priority
                else:
                    mean = sum(scores[key]) / len(scores[key])
                    weights.append(max(0.1, 1.0 - mean))  # lower score → higher weight

            # Weighted random choice
            import random
            total = sum(weights)
            r = random.random() * total
            cumulative = 0.0
            for key, w in zip(eligible, weights):
                cumulative += w
                if r <= cumulative:
                    return key
            return eligible[-1]
        else:
            import random
            return random.choice(eligible)

    def record_episode(
        self,
        task_id: str,
        variant_seed: int,
        score: float,
        steps: int,
    ) -> None:
        """Record a completed episode and check for tier advancement."""
        rec = EpisodeRecord(
            task_id=task_id,
            variant_seed=variant_seed,
            score=score,
            steps=steps,
            tier_name=self.tier_name,
        )
        self._state.history.append(rec)
        self._state.tier_episodes += 1
        self._state.total_episodes += 1

        # Check if current scenario type is mastered (graduate it)
        key = (task_id, variant_seed)
        if key not in self._state.graduated:
            recent = [
                r.score for r in self._state.history
                if (r.task_id, r.variant_seed) == key
            ][-MASTERY_WINDOW:]
            if len(recent) >= MIN_EPISODES_FOR_MASTERY:
                mean = sum(recent) / len(recent)
                if mean >= MASTERY_THRESHOLD:
                    self._state.graduated.append(key)
                    logger.info(
                        "Graduated scenario %s variant %d (mean=%.2f)",
                        task_id, variant_seed, mean,
                    )

        # Attempt tier advancement
        self._maybe_advance_tier()
        self._save()

    def should_use_adversarial(self) -> bool:
        """True if agent is ready for adversarial designer to create new scenarios."""
        return (
            self._state.tier_index >= 2  # intermediate+
            and self._recent_mean_score() >= 0.70
        )

    def weak_spots(self, top_n: int = 3) -> List[Tuple[str, int]]:
        """Return the N scenarios with lowest recent mean score."""
        scores: Dict[Tuple[str, int], List[float]] = defaultdict(list)
        for rec in self._state.history[-30:]:
            scores[(rec.task_id, rec.variant_seed)].append(rec.score)
        ranked = sorted(scores.items(), key=lambda kv: sum(kv[1]) / len(kv[1]))
        return [k for k, _ in ranked[:top_n]]

    def summary(self) -> Dict:
        """Human-readable summary for logging."""
        return {
            "tier": self.tier_name,
            "tier_index": self._state.tier_index,
            "tier_episodes": self._state.tier_episodes,
            "total_episodes": self._state.total_episodes,
            "graduated": len(self._state.graduated),
            "recent_mean_score": round(self._recent_mean_score(), 3),
            "eligible_scenario_count": len(self._eligible_scenarios()),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _eligible_scenarios(self) -> List[Tuple[str, int]]:
        """All scenarios at or below the current tier's max difficulty."""
        max_diff = DIFFICULTY_TIERS[self._state.tier_index]["max_diff"]
        return [
            key for key, diff in SCENARIO_DIFFICULTY.items()
            if diff <= max_diff
        ]

    def _recent_mean_score(self, window: int = 20) -> float:
        """Mean score over the last `window` episodes."""
        recent = self._state.history[-window:]
        if not recent:
            return 0.0
        return sum(r.score for r in recent) / len(recent)

    def _maybe_advance_tier(self) -> None:
        tier = DIFFICULTY_TIERS[self._state.tier_index]
        if self._state.tier_index >= len(DIFFICULTY_TIERS) - 1:
            return  # already at expert

        min_episodes = tier["min_episodes"]
        advance_rate = tier["advance_rate"]

        if self._state.tier_episodes < min_episodes:
            return

        # Check recent mean score over the last `min_episodes` episodes in this tier
        tier_records = [
            r for r in self._state.history
            if r.tier_name == tier["name"]
        ][-MASTERY_WINDOW:]

        if len(tier_records) < min_episodes:
            return

        mean = sum(r.score for r in tier_records) / len(tier_records)
        if mean >= advance_rate:
            self._state.tier_index += 1
            self._state.tier_episodes = 0
            new_tier = DIFFICULTY_TIERS[self._state.tier_index]["name"]
            logger.info(
                "Advanced to tier '%s' (mean=%.2f >= %.2f)",
                new_tier, mean, advance_rate,
            )

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self._state_path) or ".", exist_ok=True)
        with open(self._state_path, "w") as f:
            json.dump(
                {
                    "tier_index": self._state.tier_index,
                    "tier_episodes": self._state.tier_episodes,
                    "total_episodes": self._state.total_episodes,
                    "graduated": self._state.graduated,
                    "history": [asdict(r) for r in self._state.history[-200:]],
                },
                f, indent=2,
            )

    def _load(self) -> None:
        if not os.path.exists(self._state_path):
            return
        try:
            with open(self._state_path) as f:
                data = json.load(f)
            self._state.tier_index = data.get("tier_index", 0)
            self._state.tier_episodes = data.get("tier_episodes", 0)
            self._state.total_episodes = data.get("total_episodes", 0)
            self._state.graduated = [tuple(g) for g in data.get("graduated", [])]
            self._state.history = [
                EpisodeRecord(**r) for r in data.get("history", [])
            ]
            logger.info("Loaded curriculum state: %s", self.summary())
        except Exception as e:
            logger.warning("Failed to load curriculum state: %s", e)


# ---------------------------------------------------------------------------
# Convenience singleton for use during training
# ---------------------------------------------------------------------------

_default_curriculum: Optional[CurriculumController] = None


def get_curriculum() -> CurriculumController:
    """Get or create the default curriculum controller."""
    global _default_curriculum
    if _default_curriculum is None:
        _default_curriculum = CurriculumController()
    return _default_curriculum
