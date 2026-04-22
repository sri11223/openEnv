"""
Curriculum Controller for progressive training difficulty.

This module supports both tracks in this repository:
  - IRT incident-response tasks
  - SENTINEL oversight tasks

The controller does three jobs:
  1. Filter scenarios to the currently unlocked difficulty tier
  2. Bias sampling toward weak spots and unseen scenarios
  3. Record outcomes and advance tiers once performance is sustained
"""

from __future__ import annotations

import json
import logging
import os
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


IRT_SCENARIO_DIFFICULTY: Dict[Tuple[str, int], float] = {
    ("severity_classification", 0): 0.10,
    ("severity_classification", 1): 0.15,
    ("severity_classification", 2): 0.20,
    ("root_cause_analysis", 0): 0.35,
    ("root_cause_analysis", 1): 0.45,
    ("root_cause_analysis", 2): 0.50,
    ("full_incident_management", 0): 0.65,
    ("full_incident_management", 1): 0.75,
    ("full_incident_management", 2): 0.85,
}

SENTINEL_SCENARIO_DIFFICULTY: Dict[Tuple[str, int], float] = {
    ("basic_oversight", 0): 0.10,
    ("basic_oversight", 1): 0.15,
    ("basic_oversight", 2): 0.20,
    ("fleet_monitoring_conflict", 0): 0.35,
    ("fleet_monitoring_conflict", 1): 0.45,
    ("fleet_monitoring_conflict", 2): 0.50,
    ("adversarial_worker", 0): 0.65,
    ("adversarial_worker", 1): 0.72,
    ("adversarial_worker", 2): 0.75,
    ("multi_crisis_command", 0): 0.82,
    ("multi_crisis_command", 1): 0.88,
    ("multi_crisis_command", 2): 0.92,
    ("multi_crisis_command", 3): 0.96,
    ("multi_crisis_command", 4): 1.00,
}

SCENARIO_DIFFICULTY: Dict[Tuple[str, int], float] = {
    **IRT_SCENARIO_DIFFICULTY,
    **SENTINEL_SCENARIO_DIFFICULTY,
}

_IRT_TASK_IDS = {task_id for task_id, _ in IRT_SCENARIO_DIFFICULTY}
_SENTINEL_TASK_IDS = {task_id for task_id, _ in SENTINEL_SCENARIO_DIFFICULTY}


DIFFICULTY_TIERS = [
    {"name": "warmup", "max_diff": 0.20, "min_episodes": 3, "advance_rate": 0.60},
    {"name": "beginner", "max_diff": 0.50, "min_episodes": 5, "advance_rate": 0.65},
    {"name": "intermediate", "max_diff": 0.75, "min_episodes": 8, "advance_rate": 0.68},
    {"name": "expert", "max_diff": 1.00, "min_episodes": 0, "advance_rate": 1.00},
]

MASTERY_THRESHOLD = 0.70
MASTERY_WINDOW = 10
MIN_EPISODES_FOR_MASTERY = 3


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
    tier_episodes: int = 0
    total_episodes: int = 0
    graduated: List[Tuple[str, int]] = field(default_factory=list)
    history: List[EpisodeRecord] = field(default_factory=list)


class CurriculumController:
    """Track progress and choose the next scenario from the active task set."""

    def __init__(
        self,
        state_path: Optional[str] = None,
        active_task_ids: Optional[List[str]] = None,
    ) -> None:
        self._state = CurriculumState()
        self._active_task_ids = tuple(
            active_task_ids or sorted({task_id for task_id, _ in SCENARIO_DIFFICULTY})
        )
        self._state_path = state_path or _default_state_path_for_tasks(self._active_task_ids)

        min_diff = float(os.environ.get("EVAL_MIN_DIFFICULTY", "0.0"))
        if min_diff > 0:
            for i, tier in enumerate(DIFFICULTY_TIERS):
                if tier["max_diff"] >= min_diff:
                    self._state.tier_index = i
                    break

        self._load()

    @property
    def tier_index(self) -> int:
        return self._state.tier_index

    @property
    def tier_name(self) -> str:
        return DIFFICULTY_TIERS[self._state.tier_index]["name"]

    @property
    def total_episodes(self) -> int:
        return self._state.total_episodes

    @property
    def active_task_ids(self) -> Tuple[str, ...]:
        return self._active_task_ids

    def select_episode(self, prefer_weak_spots: bool = True) -> Tuple[str, int]:
        eligible = self._eligible_scenarios()
        if not eligible:
            for task_id in self._active_task_ids:
                candidate = (task_id, 0)
                if candidate in SCENARIO_DIFFICULTY:
                    return candidate
            return ("severity_classification", 0)

        if not prefer_weak_spots or not self._state.history:
            return random.choice(eligible)

        scores: Dict[Tuple[str, int], List[float]] = defaultdict(list)
        for rec in self._state.history[-50:]:
            key = (rec.task_id, rec.variant_seed)
            if key in eligible:
                scores[key].append(rec.score)

        weights: List[float] = []
        for key in eligible:
            if key not in scores:
                weights.append(2.0)
                continue
            mean = sum(scores[key]) / len(scores[key])
            weights.append(max(0.1, 1.0 - mean))

        total = sum(weights)
        if total <= 0:
            return random.choice(eligible)

        draw = random.random() * total
        cumulative = 0.0
        for key, weight in zip(eligible, weights):
            cumulative += weight
            if draw <= cumulative:
                return key
        return eligible[-1]

    def record_episode(
        self,
        task_id: str,
        variant_seed: int,
        score: float,
        steps: int,
    ) -> None:
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
                        task_id,
                        variant_seed,
                        mean,
                    )

        self._maybe_advance_tier()
        self._save()

    def should_use_adversarial(self) -> bool:
        return self._state.tier_index >= 2 and self._recent_mean_score() >= 0.70

    def weak_spots(self, top_n: int = 3) -> List[Tuple[str, int]]:
        scores: Dict[Tuple[str, int], List[float]] = defaultdict(list)
        for rec in self._state.history[-30:]:
            if self._is_active_task(rec.task_id):
                scores[(rec.task_id, rec.variant_seed)].append(rec.score)
        ranked = sorted(scores.items(), key=lambda item: sum(item[1]) / len(item[1]))
        return [key for key, _ in ranked[:top_n]]

    def summary(self) -> Dict[str, object]:
        return {
            "tier": self.tier_name,
            "tier_index": self._state.tier_index,
            "tier_episodes": self._state.tier_episodes,
            "total_episodes": self._state.total_episodes,
            "graduated": len(self._state.graduated),
            "recent_mean_score": round(self._recent_mean_score(), 3),
            "eligible_scenario_count": len(self._eligible_scenarios()),
            "active_task_ids": list(self._active_task_ids),
        }

    def _eligible_scenarios(self) -> List[Tuple[str, int]]:
        max_diff = DIFFICULTY_TIERS[self._state.tier_index]["max_diff"]
        return [
            key for key, diff in SCENARIO_DIFFICULTY.items()
            if diff <= max_diff and self._is_active_task(key[0])
        ]

    def _recent_mean_score(self, window: int = 20) -> float:
        recent = [
            rec for rec in self._state.history[-window:]
            if self._is_active_task(rec.task_id)
        ]
        if not recent:
            return 0.0
        return sum(rec.score for rec in recent) / len(recent)

    def _is_active_task(self, task_id: str) -> bool:
        return not self._active_task_ids or task_id in self._active_task_ids

    def _maybe_advance_tier(self) -> None:
        tier = DIFFICULTY_TIERS[self._state.tier_index]
        if self._state.tier_index >= len(DIFFICULTY_TIERS) - 1:
            return
        if self._state.tier_episodes < tier["min_episodes"]:
            return

        tier_records = [
            rec for rec in self._state.history
            if rec.tier_name == tier["name"] and self._is_active_task(rec.task_id)
        ][-MASTERY_WINDOW:]
        if len(tier_records) < tier["min_episodes"]:
            return

        mean = sum(rec.score for rec in tier_records) / len(tier_records)
        if mean >= tier["advance_rate"]:
            self._state.tier_index += 1
            self._state.tier_episodes = 0
            logger.info(
                "Advanced to tier '%s' (mean=%.2f >= %.2f)",
                DIFFICULTY_TIERS[self._state.tier_index]["name"],
                mean,
                tier["advance_rate"],
            )

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self._state_path) or ".", exist_ok=True)
        payload = {
            "tier_index": self._state.tier_index,
            "tier_episodes": self._state.tier_episodes,
            "total_episodes": self._state.total_episodes,
            "graduated": self._state.graduated,
            "active_task_ids": list(self._active_task_ids),
            "history": [asdict(item) for item in self._state.history[-200:]],
        }
        with open(self._state_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def _load(self) -> None:
        if not os.path.exists(self._state_path):
            return
        try:
            with open(self._state_path, encoding="utf-8") as handle:
                data = json.load(handle)
            self._state.tier_index = data.get("tier_index", 0)
            self._state.tier_episodes = data.get("tier_episodes", 0)
            self._state.total_episodes = data.get("total_episodes", 0)
            self._state.graduated = [tuple(item) for item in data.get("graduated", [])]
            self._state.history = [EpisodeRecord(**item) for item in data.get("history", [])]
            logger.info("Loaded curriculum state: %s", self.summary())
        except Exception as exc:
            logger.warning("Failed to load curriculum state: %s", exc)


_default_curricula: Dict[Tuple[Tuple[str, ...], str], CurriculumController] = {}


def _default_state_path_for_tasks(active_task_ids: Tuple[str, ...]) -> str:
    if not active_task_ids:
        suffix = "all"
    else:
        task_set = set(active_task_ids)
        if task_set.issubset(_IRT_TASK_IDS):
            suffix = "irt"
        elif task_set.issubset(_SENTINEL_TASK_IDS):
            suffix = "sentinel"
        else:
            suffix = "mixed"
    return os.path.join("outputs", f"curriculum_state_{suffix}.json")


def get_curriculum(
    active_task_ids: Optional[List[str]] = None,
    state_path: Optional[str] = None,
) -> CurriculumController:
    task_key = tuple(active_task_ids or [])
    resolved_state_path = state_path or _default_state_path_for_tasks(task_key)
    cache_key = (task_key, resolved_state_path)
    if cache_key not in _default_curricula:
        _default_curricula[cache_key] = CurriculumController(
            state_path=resolved_state_path,
            active_task_ids=list(task_key) or None,
        )
    return _default_curricula[cache_key]
