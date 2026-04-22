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
CURRICULUM_DIFFICULTY_WINDOW = max(1, int(os.getenv("CURRICULUM_DIFFICULTY_WINDOW", "2")))
CURRICULUM_FRONTIER_MIN_ATTEMPTS = max(1, int(os.getenv("CURRICULUM_FRONTIER_MIN_ATTEMPTS", "3")))
CURRICULUM_FRONTIER_TARGET_RATE = float(os.getenv("CURRICULUM_FRONTIER_TARGET_RATE", "0.75"))
CURRICULUM_FRONTIER_FAILURE_RATE = float(os.getenv("CURRICULUM_FRONTIER_FAILURE_RATE", "0.10"))
ZERO_SIGNAL_REWARD_THRESHOLD = float(os.getenv("ZERO_SIGNAL_REWARD_THRESHOLD", "0.05"))
TRIVIAL_REWARD_THRESHOLD = float(os.getenv("TRIVIAL_REWARD_THRESHOLD", "0.95"))

TASK_SCENARIOS_BY_DIFFICULTY: Dict[str, List[Tuple[str, int]]] = {}
SCENARIO_RANK: Dict[Tuple[str, int], int] = {}
for _task_id in sorted({task_id for task_id, _ in SCENARIO_DIFFICULTY}):
    ordered = sorted(
        [key for key in SCENARIO_DIFFICULTY if key[0] == _task_id],
        key=lambda key: (SCENARIO_DIFFICULTY[key], key[1]),
    )
    TASK_SCENARIOS_BY_DIFFICULTY[_task_id] = ordered
    for rank, key in enumerate(ordered):
        SCENARIO_RANK[key] = rank


@dataclass
class EpisodeRecord:
    task_id: str
    variant_seed: int
    score: float
    steps: int
    tier_name: str
    difficulty_rank: int = 0
    difficulty_value: float = 0.0
    frontier_hit: bool = False


@dataclass
class CurriculumState:
    tier_index: int = 0
    tier_episodes: int = 0
    total_episodes: int = 0
    graduated: List[Tuple[str, int]] = field(default_factory=list)
    history: List[EpisodeRecord] = field(default_factory=list)
    difficulty_low: Dict[str, int] = field(default_factory=dict)
    difficulty_high: Dict[str, int] = field(default_factory=dict)
    mastery_attempts: Dict[str, int] = field(default_factory=dict)
    mastery_successes: Dict[str, int] = field(default_factory=dict)
    frontier_backoffs: Dict[str, int] = field(default_factory=dict)


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
        self._ensure_adaptive_state()

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
                fallback = self._fallback_scenario_for_task(task_id)
                if fallback:
                    return fallback
            return ("severity_classification", 0)

        if not prefer_weak_spots or not self._state.history:
            return random.choice(eligible)

        scores: Dict[Tuple[str, int], List[float]] = defaultdict(list)
        task_scores: Dict[str, List[float]] = defaultdict(list)
        for rec in self._state.history[-50:]:
            key = (rec.task_id, rec.variant_seed)
            if key in eligible:
                scores[key].append(rec.score)
                task_scores[rec.task_id].append(rec.score)

        eligible_by_task: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        for key in eligible:
            eligible_by_task[key[0]].append(key)

        task_weights: List[float] = []
        task_candidates = sorted(eligible_by_task)
        max_samples = max((len(task_scores.get(task_id, [])) for task_id in task_candidates), default=0)
        for task_id in task_candidates:
            values = task_scores.get(task_id, [])
            if not values:
                task_weights.append(2.5)
                continue
            mean = sum(values) / len(values)
            under_sampled = 1.0 - _safe_ratio(len(values), max_samples or 1)
            task_weights.append(max(0.2, 0.75 + (1.0 - mean) + 0.5 * under_sampled))

        chosen_task = self._weighted_choice(task_candidates, task_weights)
        task_eligible = eligible_by_task.get(chosen_task) or eligible

        weights: List[float] = []
        for key in task_eligible:
            if key not in scores:
                weights.append(2.0)
                continue
            mean = sum(scores[key]) / len(scores[key])
            weights.append(max(0.1, 1.0 - mean))

        return self._weighted_choice(task_eligible, weights)

    def record_episode(
        self,
        task_id: str,
        variant_seed: int,
        score: float,
        steps: int,
    ) -> None:
        scenario_key = (task_id, variant_seed)
        difficulty_rank = SCENARIO_RANK.get(scenario_key, 0)
        difficulty_value = float(SCENARIO_DIFFICULTY.get(scenario_key, 0.0))
        frontier_hit = difficulty_rank == self._state.difficulty_high.get(task_id, 0)
        rec = EpisodeRecord(
            task_id=task_id,
            variant_seed=variant_seed,
            score=score,
            steps=steps,
            tier_name=self.tier_name,
            difficulty_rank=difficulty_rank,
            difficulty_value=difficulty_value,
            frontier_hit=frontier_hit,
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

        self._update_adaptive_difficulty(task_id, variant_seed, score)
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
        eligible = self._eligible_scenarios()
        recent = [
            rec for rec in self._state.history[-MASTERY_WINDOW:]
            if self._is_active_task(rec.task_id)
        ]
        zero_signal = sum(1 for rec in recent if rec.score <= ZERO_SIGNAL_REWARD_THRESHOLD)
        trivial = sum(1 for rec in recent if rec.score >= TRIVIAL_REWARD_THRESHOLD)
        productive = max(0, len(recent) - zero_signal - trivial)
        frontier_hits = sum(1 for rec in recent if rec.frontier_hit)
        adaptive_by_task: Dict[str, object] = {}
        frontier_scenarios: List[Dict[str, object]] = []
        for task_id in self._active_task_ids:
            window = self._adaptive_window_for_task(task_id)
            frontier_key = self._frontier_scenario_for_task(task_id)
            frontier_variant_seed = frontier_key[1] if frontier_key else None
            if frontier_key:
                frontier_scenarios.append(
                    {
                        "task_id": task_id,
                        "variant_seed": frontier_variant_seed,
                        "difficulty": round(float(SCENARIO_DIFFICULTY.get(frontier_key, 0.0)), 4),
                    }
                )
            adaptive_by_task[task_id] = {
                **window,
                "available_variants": [key[1] for key in self._window_scenarios(task_id)],
                "frontier_variant_seed": frontier_variant_seed,
            }
        return {
            "tier": self.tier_name,
            "tier_index": self._state.tier_index,
            "tier_episodes": self._state.tier_episodes,
            "total_episodes": self._state.total_episodes,
            "graduated": len(self._state.graduated),
            "recent_mean_score": round(self._recent_mean_score(), 3),
            "eligible_scenario_count": len(eligible),
            "active_task_ids": list(self._active_task_ids),
            "zero_reward_fraction": round(_safe_ratio(zero_signal, len(recent)), 4),
            "trivially_solved_fraction": round(_safe_ratio(trivial, len(recent)), 4),
            "productive_fraction": round(_safe_ratio(productive, len(recent)), 4),
            "effective_prompt_ratio": round(_safe_ratio(productive, len(recent)), 4),
            "frontier_hit_rate": round(_safe_ratio(frontier_hits, len(recent)), 4),
            "adaptive_difficulty": {
                "window_size": CURRICULUM_DIFFICULTY_WINDOW,
                "frontier_min_attempts": CURRICULUM_FRONTIER_MIN_ATTEMPTS,
                "frontier_target_rate": round(CURRICULUM_FRONTIER_TARGET_RATE, 4),
                "frontier_failure_rate": round(CURRICULUM_FRONTIER_FAILURE_RATE, 4),
                "total_frontier_backoffs": sum(int(self._state.frontier_backoffs.get(task_id, 0)) for task_id in self._active_task_ids),
                "frontier_scenarios": frontier_scenarios,
                "per_task": adaptive_by_task,
            },
        }

    def _eligible_scenarios(self) -> List[Tuple[str, int]]:
        max_diff = DIFFICULTY_TIERS[self._state.tier_index]["max_diff"]
        eligible: List[Tuple[str, int]] = []
        for task_id in self._active_task_ids:
            windowed = [
                key
                for key in self._window_scenarios(task_id)
                if SCENARIO_DIFFICULTY.get(key, 1.0) <= max_diff
            ]
            if windowed:
                eligible.extend(windowed)
                continue
            fallback = self._fallback_scenario_for_task(task_id, max_diff=max_diff)
            if fallback is not None:
                eligible.append(fallback)
        return eligible

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

    def _ensure_adaptive_state(self) -> None:
        for task_id in self._active_task_ids:
            task_scenarios = TASK_SCENARIOS_BY_DIFFICULTY.get(task_id, [])
            max_rank = max(0, len(task_scenarios) - 1)
            low = int(self._state.difficulty_low.get(task_id, 0))
            high = int(self._state.difficulty_high.get(task_id, 0))
            low = max(0, min(low, max_rank))
            high = max(low, min(high, max_rank))
            self._state.difficulty_low[task_id] = low
            self._state.difficulty_high[task_id] = high
            self._state.mastery_attempts[task_id] = max(0, int(self._state.mastery_attempts.get(task_id, 0)))
            self._state.mastery_successes[task_id] = max(0, int(self._state.mastery_successes.get(task_id, 0)))
            self._state.frontier_backoffs[task_id] = max(0, int(self._state.frontier_backoffs.get(task_id, 0)))

    @staticmethod
    def _weighted_choice(candidates: List[Tuple[str, int]] | List[str], weights: List[float]):
        total = sum(weights)
        if total <= 0:
            return random.choice(candidates)
        draw = random.random() * total
        cumulative = 0.0
        for candidate, weight in zip(candidates, weights):
            cumulative += weight
            if draw <= cumulative:
                return candidate
        return candidates[-1]

    def _window_scenarios(self, task_id: str) -> List[Tuple[str, int]]:
        task_scenarios = TASK_SCENARIOS_BY_DIFFICULTY.get(task_id, [])
        if not task_scenarios:
            return []
        low = int(self._state.difficulty_low.get(task_id, 0))
        high = int(self._state.difficulty_high.get(task_id, 0))
        return [
            key for rank, key in enumerate(task_scenarios)
            if low <= rank <= high
        ]

    def _frontier_scenario_for_task(self, task_id: str) -> Optional[Tuple[str, int]]:
        task_scenarios = TASK_SCENARIOS_BY_DIFFICULTY.get(task_id, [])
        if not task_scenarios:
            return None
        high = int(self._state.difficulty_high.get(task_id, 0))
        if high < 0 or high >= len(task_scenarios):
            return None
        return task_scenarios[high]

    def _fallback_scenario_for_task(
        self,
        task_id: str,
        *,
        max_diff: Optional[float] = None,
    ) -> Optional[Tuple[str, int]]:
        task_scenarios = TASK_SCENARIOS_BY_DIFFICULTY.get(task_id, [])
        if not task_scenarios:
            return None
        allowed = [
            key for key in task_scenarios
            if max_diff is None or SCENARIO_DIFFICULTY.get(key, 1.0) <= max_diff
        ]
        if not allowed:
            return None
        return allowed[-1]

    def _adaptive_window_for_task(self, task_id: str) -> Dict[str, object]:
        frontier_key = self._frontier_scenario_for_task(task_id)
        attempts = int(self._state.mastery_attempts.get(task_id, 0))
        successes = int(self._state.mastery_successes.get(task_id, 0))
        return {
            "difficulty_low": int(self._state.difficulty_low.get(task_id, 0)),
            "difficulty_high": int(self._state.difficulty_high.get(task_id, 0)),
            "mastery_attempts": attempts,
            "mastery_successes": successes,
            "mastery_success_rate": round(_safe_ratio(successes, attempts), 4),
            "frontier_backoffs": int(self._state.frontier_backoffs.get(task_id, 0)),
            "frontier_difficulty": round(float(SCENARIO_DIFFICULTY.get(frontier_key, 0.0)), 4) if frontier_key else 0.0,
        }

    def _update_adaptive_difficulty(
        self,
        task_id: str,
        variant_seed: int,
        score: float,
    ) -> None:
        frontier_key = self._frontier_scenario_for_task(task_id)
        if frontier_key is None or frontier_key != (task_id, variant_seed):
            return

        attempts = self._state.mastery_attempts.get(task_id, 0) + 1
        successes = self._state.mastery_successes.get(task_id, 0)
        if score >= CURRICULUM_FRONTIER_TARGET_RATE:
            successes += 1

        self._state.mastery_attempts[task_id] = attempts
        self._state.mastery_successes[task_id] = successes

        if attempts < CURRICULUM_FRONTIER_MIN_ATTEMPTS:
            return

        current_high = int(self._state.difficulty_high.get(task_id, 0))
        success_rate = _safe_ratio(successes, attempts)
        if success_rate < CURRICULUM_FRONTIER_TARGET_RATE:
            if success_rate > CURRICULUM_FRONTIER_FAILURE_RATE:
                return

            current_low = int(self._state.difficulty_low.get(task_id, 0))
            if current_high <= 0 and current_low <= 0:
                return

            new_high = max(0, current_high - 1)
            new_low = max(0, min(current_low, new_high))
            if new_high - new_low + 1 < CURRICULUM_DIFFICULTY_WINDOW:
                new_low = max(0, new_high - CURRICULUM_DIFFICULTY_WINDOW + 1)

            self._state.difficulty_high[task_id] = new_high
            self._state.difficulty_low[task_id] = new_low
            self._state.mastery_attempts[task_id] = 0
            self._state.mastery_successes[task_id] = 0
            self._state.frontier_backoffs[task_id] = self._state.frontier_backoffs.get(task_id, 0) + 1
            logger.info(
                "Adaptive difficulty eased back for %s to window [%d, %d] after frontier success rate %.2f (%d/%d)",
                task_id,
                new_low,
                new_high,
                success_rate,
                successes,
                attempts,
            )
            return

        task_scenarios = TASK_SCENARIOS_BY_DIFFICULTY.get(task_id, [])
        max_rank = max(0, len(task_scenarios) - 1)
        if current_high >= max_rank:
            return

        new_high = current_high + 1
        self._state.difficulty_high[task_id] = new_high
        new_low = int(self._state.difficulty_low.get(task_id, 0))
        if new_high - new_low + 1 > CURRICULUM_DIFFICULTY_WINDOW:
            new_low = max(0, new_high - CURRICULUM_DIFFICULTY_WINDOW + 1)
        self._state.difficulty_low[task_id] = new_low
        self._state.mastery_attempts[task_id] = 0
        self._state.mastery_successes[task_id] = 0
        logger.info(
            "Advanced adaptive difficulty for %s to window [%d, %d] after frontier success rate %.2f (%d/%d)",
            task_id,
            new_low,
            new_high,
            success_rate,
            successes,
            attempts,
        )

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
            "difficulty_low": self._state.difficulty_low,
            "difficulty_high": self._state.difficulty_high,
            "mastery_attempts": self._state.mastery_attempts,
            "mastery_successes": self._state.mastery_successes,
            "frontier_backoffs": self._state.frontier_backoffs,
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
            self._state.difficulty_low = {
                str(key): int(value) for key, value in (data.get("difficulty_low") or {}).items()
            }
            self._state.difficulty_high = {
                str(key): int(value) for key, value in (data.get("difficulty_high") or {}).items()
            }
            self._state.mastery_attempts = {
                str(key): int(value) for key, value in (data.get("mastery_attempts") or {}).items()
            }
            self._state.mastery_successes = {
                str(key): int(value) for key, value in (data.get("mastery_successes") or {}).items()
            }
            self._state.frontier_backoffs = {
                str(key): int(value) for key, value in (data.get("frontier_backoffs") or {}).items()
            }
            self._state.history = [EpisodeRecord(**item) for item in data.get("history", [])]
            self._ensure_adaptive_state()
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


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


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
