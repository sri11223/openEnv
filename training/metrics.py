# -*- coding: utf-8 -*-
"""Training metrics: diversity, productive signal, coverage, and zero-gradient detection.

Extracted from train.py to keep the training pipeline modular.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Thresholds (mirrored from train.py config; imported at call sites)
# ---------------------------------------------------------------------------

ZERO_SIGNAL_REWARD_THRESHOLD = 0.05
TRIVIAL_REWARD_THRESHOLD = 0.95


def set_thresholds(zero: float, trivial: float) -> None:
    """Allow train.py to override the defaults at startup."""
    global ZERO_SIGNAL_REWARD_THRESHOLD, TRIVIAL_REWARD_THRESHOLD
    ZERO_SIGNAL_REWARD_THRESHOLD = zero
    TRIVIAL_REWARD_THRESHOLD = trivial


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _increment_counter(counter: Dict[str, int], key: Any) -> None:
    label = str(key or "unknown")
    counter[label] = counter.get(label, 0) + 1


def _normalize_completion_text(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def _extract_completion_choice(text: str) -> str:
    from training.episodes import parse_action
    payload = parse_action(str(text or "")) or {}
    choice = payload.get("decision") or payload.get("action") or payload.get("action_type") or ""
    return str(choice).upper()


def _shannon_entropy_from_labels(labels: List[str]) -> float:
    usable = [label for label in labels if label]
    if not usable:
        return 0.0
    total = float(len(usable))
    counts: Dict[str, int] = {}
    for label in usable:
        counts[label] = counts.get(label, 0) + 1
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log(p, 2)
    return float(entropy)


# ---------------------------------------------------------------------------
# Completion diversity
# ---------------------------------------------------------------------------

def completion_diversity_metrics(completions: Optional[List[str]]) -> Dict[str, Any]:
    if not completions:
        return {
            "unique_completion_ratio": 0.0,
            "decision_entropy": 0.0,
            "decision_variety": 0,
            "decision_distribution": {},
        }

    normalized = [_normalize_completion_text(text) for text in completions]
    unique_ratio = safe_ratio(len(set(normalized)), len(normalized))
    decisions = [_extract_completion_choice(text) for text in completions]
    decision_counts: Dict[str, int] = {}
    for choice in decisions:
        key = choice or "UNPARSED"
        decision_counts[key] = decision_counts.get(key, 0) + 1
    total = float(sum(decision_counts.values()) or 1.0)
    decision_distribution = {
        key: round(value / total, 4)
        for key, value in sorted(decision_counts.items(), key=lambda item: item[0])
    }
    return {
        "unique_completion_ratio": round(unique_ratio, 4),
        "decision_entropy": round(_shannon_entropy_from_labels(decisions), 4),
        "decision_variety": len(decision_counts),
        "decision_distribution": decision_distribution,
    }


# ---------------------------------------------------------------------------
# Frontier scenarios
# ---------------------------------------------------------------------------

def frontier_scenario_keys(curriculum_summary: Optional[Dict[str, Any]]) -> set[Tuple[str, int]]:
    if not curriculum_summary:
        return set()
    adaptive = curriculum_summary.get("adaptive_difficulty") or {}
    frontier_scenarios = adaptive.get("frontier_scenarios") or []
    resolved = set()
    for item in frontier_scenarios:
        try:
            resolved.add((str(item.get("task_id")), int(item.get("variant_seed", 0))))
        except (TypeError, ValueError):
            continue
    return resolved


# ---------------------------------------------------------------------------
# Productive signal metrics
# ---------------------------------------------------------------------------

def productive_signal_metrics(
    rewards: List[float],
    task_ids: List[str],
    variant_seeds: List[int],
    curriculum_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    reward_values = [float(value) for value in rewards]
    fkeys = frontier_scenario_keys(curriculum_summary)
    zero_signal = sum(1 for reward in reward_values if reward <= ZERO_SIGNAL_REWARD_THRESHOLD)
    trivial = sum(1 for reward in reward_values if reward >= TRIVIAL_REWARD_THRESHOLD)
    productive = max(0, len(reward_values) - zero_signal - trivial)
    frontier_hits = sum(
        1
        for task_id, variant_seed in zip(task_ids, variant_seeds)
        if (str(task_id), int(variant_seed)) in fkeys
    )
    active_task_ids = list((curriculum_summary or {}).get("active_task_ids") or [])
    task_diversity_ratio = safe_ratio(len(set(task_ids)), len(active_task_ids) or len(set(task_ids)) or 1)
    payload = {
        "zero_reward_fraction": round(safe_ratio(zero_signal, len(reward_values)), 4),
        "trivially_solved_fraction": round(safe_ratio(trivial, len(reward_values)), 4),
        "productive_fraction": round(safe_ratio(productive, len(reward_values)), 4),
        "effective_prompt_ratio": round(safe_ratio(productive, len(reward_values)), 4),
        "frontier_hit_rate": round(safe_ratio(frontier_hits, len(reward_values)), 4),
        "task_diversity_ratio": round(task_diversity_ratio, 4),
        "frontier_hit_count": frontier_hits,
    }
    if not fkeys and curriculum_summary and curriculum_summary.get("frontier_hit_rate") is not None:
        payload["frontier_hit_rate"] = float(curriculum_summary.get("frontier_hit_rate", 0.0))
    return payload


# ---------------------------------------------------------------------------
# Training coverage
# ---------------------------------------------------------------------------

def training_coverage_metrics(
    histories: List[List[Dict[str, Any]]],
    task_ids: List[str],
    variant_seeds: List[int],
    adversarial_cases: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Summarize what the batch actually exercised for judge-facing plots."""
    task_counts: Dict[str, int] = {}
    scenario_counts: Dict[str, int] = {}
    worker_counts: Dict[str, int] = {}
    worker_role_counts: Dict[str, int] = {}
    misbehavior_counts: Dict[str, int] = {}
    decision_counts: Dict[str, int] = {}
    corrective_counts: Dict[str, int] = {"attempted": 0, "approved": 0}

    for index, task_id in enumerate(task_ids):
        variant_seed = int(variant_seeds[index]) if index < len(variant_seeds) else 0
        _increment_counter(task_counts, task_id)
        _increment_counter(scenario_counts, f"{task_id}:seed{variant_seed}")

    for history in histories:
        for entry in history:
            audit = entry.get("audit") or {}
            info = entry.get("info") or {}
            decision = entry.get("decision") or {}
            revision = entry.get("worker_revision") or {}
            worker_id = audit.get("worker_id") or (entry.get("proposal") or {}).get("worker_id")
            if worker_id:
                _increment_counter(worker_counts, worker_id)
            worker_role = audit.get("worker_role") or info.get("worker_role")
            if worker_role:
                _increment_counter(worker_role_counts, worker_role)
            if audit.get("was_misbehavior") or info.get("is_misbehavior"):
                _increment_counter(misbehavior_counts, audit.get("reason") or info.get("mb_type") or "unknown")
            _increment_counter(
                decision_counts,
                audit.get("sentinel_decision") or decision.get("decision") or decision.get("action") or "unknown",
            )
            if revision.get("attempted"):
                corrective_counts["attempted"] += 1
            if revision.get("revision_approved"):
                corrective_counts["approved"] += 1

    adversarial_count = sum(1 for case in (adversarial_cases or []) if str(case or "").strip())
    total_cases = len(adversarial_cases or []) or len(task_ids) or 1
    return {
        "task_counts": dict(sorted(task_counts.items())),
        "scenario_counts": dict(sorted(scenario_counts.items())),
        "worker_counts": dict(sorted(worker_counts.items())),
        "worker_role_counts": dict(sorted(worker_role_counts.items())),
        "misbehavior_counts": dict(sorted(misbehavior_counts.items())),
        "oversight_decision_counts": dict(sorted(decision_counts.items())),
        "corrective_loop_counts": corrective_counts,
        "adversarial_case_count": adversarial_count,
        "adversarial_case_fraction": round(safe_ratio(adversarial_count, total_cases), 4),
    }


# ---------------------------------------------------------------------------
# Zero-gradient group detection
# ---------------------------------------------------------------------------

def zero_gradient_group_metrics(
    rewards: List[float],
    task_ids: List[str],
    variant_seeds: List[int],
    prompts: Optional[List[str]] = None,
    adversarial_cases: Optional[List[str]] = None,
    tolerance: float = 1e-9,
) -> Dict[str, Any]:
    """Detect GRPO groups where every sampled completion received the same reward."""
    groups: Dict[str, List[float]] = {}
    for index, reward in enumerate(rewards):
        if prompts and index < len(prompts):
            key = str(prompts[index])
        else:
            task_id = task_ids[index] if index < len(task_ids) else "unknown"
            variant_seed = int(variant_seeds[index]) if index < len(variant_seeds) else 0
            case = ""
            if adversarial_cases and index < len(adversarial_cases):
                case = str(adversarial_cases[index] or "")
            key = f"{task_id}:seed{variant_seed}:adv{bool(case.strip())}"
        groups.setdefault(key, []).append(float(reward))

    multi_sample_groups = [values for values in groups.values() if len(values) > 1]
    zero_gradient_groups = [
        values
        for values in multi_sample_groups
        if max(values) - min(values) <= tolerance
    ]
    group_std_values = [float(np.std(values)) for values in multi_sample_groups]
    return {
        "reward_group_count": len(multi_sample_groups),
        "zero_gradient_group_count": len(zero_gradient_groups),
        "zero_gradient_group_fraction": round(
            safe_ratio(len(zero_gradient_groups), len(multi_sample_groups)),
            4,
        ),
        "mean_reward_group_std": round(float(np.mean(group_std_values)), 4) if group_std_values else 0.0,
    }


# ---------------------------------------------------------------------------
# SENTINEL history summarization
# ---------------------------------------------------------------------------

def summarize_sentinel_history(history: List[Dict[str, Any]]) -> Dict[str, float]:
    audits = [entry.get("audit") or {} for entry in history if entry.get("audit")]
    misbehaviors = sum(1 for audit in audits if audit.get("was_misbehavior"))
    caught = sum(
        1
        for audit in audits
        if audit.get("was_misbehavior") and audit.get("sentinel_decision") != "APPROVE"
    )
    false_positives = sum(
        1
        for audit in audits
        if audit.get("sentinel_decision") != "APPROVE" and not audit.get("was_misbehavior")
    )
    false_negatives = sum(
        1
        for audit in audits
        if audit.get("was_misbehavior") and audit.get("sentinel_decision") == "APPROVE"
    )
    revision_attempts = sum(
        1
        for entry in history
        if (entry.get("worker_revision") or {}).get("attempted")
    )
    revision_successes = sum(
        1
        for entry in history
        if (entry.get("worker_revision") or {}).get("revision_approved")
    )
    coaching_values = [
        float((entry.get("reward_breakdown") or {}).get("coaching_quality"))
        for entry in history
        if (entry.get("reward_breakdown") or {}).get("coaching_quality") is not None
    ]
    prevented_damage = sum(float(audit.get("prevented_damage_score") or 0.0) for audit in audits)
    allowed_damage = sum(float(audit.get("allowed_damage_score") or 0.0) for audit in audits)
    twin_without_sentinel_damage = prevented_damage + allowed_damage
    safe_actions = max(0, len(audits) - misbehaviors)
    return {
        "steps": float(len(history)),
        "misbehaviors": float(misbehaviors),
        "caught": float(caught),
        "false_positives": float(false_positives),
        "false_negatives": float(false_negatives),
        "revision_attempts": float(revision_attempts),
        "revision_successes": float(revision_successes),
        "prevented_damage_total": round(prevented_damage, 4),
        "allowed_damage_total": round(allowed_damage, 4),
        "twin_without_sentinel_damage_total": round(twin_without_sentinel_damage, 4),
        "twin_with_sentinel_damage_total": round(allowed_damage, 4),
        "twin_prevented_damage_total": round(prevented_damage, 4),
        "twin_damage_reduction_rate": round(
            safe_ratio(prevented_damage, twin_without_sentinel_damage),
            4,
        ),
        "coaching_quality": round(float(np.mean(coaching_values)), 4) if coaching_values else 0.0,
        "detection_rate": round(safe_ratio(caught, misbehaviors), 4),
        "false_positive_rate": round(safe_ratio(false_positives, safe_actions), 4),
        "risk_reduction_rate": round(
            safe_ratio(prevented_damage, prevented_damage + allowed_damage),
            4,
        ),
        "worker_rehabilitation_rate": round(
            safe_ratio(revision_successes, revision_attempts),
            4,
        ),
    }


# ---------------------------------------------------------------------------
# Aggregate batch metrics
# ---------------------------------------------------------------------------

def aggregate_batch_metrics(
    rewards: List[float],
    histories: List[List[Dict[str, Any]]],
    task_ids: List[str],
    variant_seeds: List[int],
    sentinel_task_ids: Optional[List[str]] = None,
    completions: Optional[List[str]] = None,
    prompts: Optional[List[str]] = None,
    adversarial_cases: Optional[List[str]] = None,
    curriculum_summary: Optional[Dict[str, Any]] = None,
    prompt_refreshes: int = 0,
) -> Dict[str, Any]:
    if sentinel_task_ids is None:
        sentinel_task_ids = ["basic_oversight", "fleet_monitoring_conflict", "adversarial_worker", "multi_crisis_command"]
    is_sentinel_batch = any(task_id in sentinel_task_ids for task_id in task_ids)
    safe_rewards = [float(r) for r in rewards]
    prod_metrics = productive_signal_metrics(
        rewards=safe_rewards,
        task_ids=task_ids,
        variant_seeds=variant_seeds,
        curriculum_summary=curriculum_summary,
    )
    fkeys = frontier_scenario_keys(curriculum_summary)
    reward_mean = float(np.mean(safe_rewards)) if safe_rewards else 0.0
    reward_min = float(np.min(safe_rewards)) if safe_rewards else 0.0
    reward_max = float(np.max(safe_rewards)) if safe_rewards else 0.0
    reward_std = float(np.std(safe_rewards)) if safe_rewards else 0.0
    avg_steps = float(np.mean([len(history) for history in histories])) if histories else 0.0

    active_task_ids_for_fallback = sentinel_task_ids if is_sentinel_batch else task_ids

    per_task: Dict[str, Dict[str, Any]] = {}
    for idx, reward in enumerate(safe_rewards):
        task_id = task_ids[idx] if idx < len(task_ids) else active_task_ids_for_fallback[0]
        variant_seed = int(variant_seeds[idx]) if idx < len(variant_seeds) else 0
        history = histories[idx] if idx < len(histories) else []
        bucket = per_task.setdefault(
            task_id,
            {
                "count": 0,
                "reward_values": [],
                "step_values": [],
                "variant_seeds": set(),
                "misbehaviors": 0.0,
                "caught": 0.0,
                "false_positives": 0.0,
                "false_negatives": 0.0,
                "revision_attempts": 0.0,
                "revision_successes": 0.0,
                "prevented_damage_total": 0.0,
                "allowed_damage_total": 0.0,
                "twin_without_sentinel_damage_total": 0.0,
                "twin_with_sentinel_damage_total": 0.0,
                "twin_prevented_damage_total": 0.0,
                "coaching_quality_values": [],
                "zero_reward_count": 0,
                "trivial_reward_count": 0,
                "productive_count": 0,
                "frontier_hits": 0,
            },
        )
        bucket["count"] += 1
        bucket["reward_values"].append(float(reward))
        bucket["step_values"].append(len(history))
        bucket["variant_seeds"].add(variant_seed)
        if reward <= ZERO_SIGNAL_REWARD_THRESHOLD:
            bucket["zero_reward_count"] += 1
        elif reward >= TRIVIAL_REWARD_THRESHOLD:
            bucket["trivial_reward_count"] += 1
        else:
            bucket["productive_count"] += 1
        if (str(task_id), int(variant_seed)) in fkeys:
            bucket["frontier_hits"] += 1

        if is_sentinel_batch:
            rollup = summarize_sentinel_history(history)
            for key in (
                "misbehaviors",
                "caught",
                "false_positives",
                "false_negatives",
                "revision_attempts",
                "revision_successes",
                "prevented_damage_total",
                "allowed_damage_total",
                "twin_without_sentinel_damage_total",
                "twin_with_sentinel_damage_total",
                "twin_prevented_damage_total",
            ):
                bucket[key] += float(rollup[key])
            bucket["coaching_quality_values"].append(float(rollup.get("coaching_quality", 0.0)))

    for task_id, bucket in list(per_task.items()):
        task_summary: Dict[str, Any] = {
            "count": bucket["count"],
            "reward_mean": round(float(np.mean(bucket["reward_values"])), 4) if bucket["reward_values"] else 0.0,
            "avg_steps": round(float(np.mean(bucket["step_values"])), 4) if bucket["step_values"] else 0.0,
            "variant_seeds": sorted(bucket["variant_seeds"]),
            "zero_reward_fraction": round(safe_ratio(bucket["zero_reward_count"], bucket["count"]), 4),
            "trivially_solved_fraction": round(safe_ratio(bucket["trivial_reward_count"], bucket["count"]), 4),
            "productive_fraction": round(safe_ratio(bucket["productive_count"], bucket["count"]), 4),
            "frontier_hit_rate": round(safe_ratio(bucket["frontier_hits"], bucket["count"]), 4),
        }
        if is_sentinel_batch:
            task_summary.update(
                {
                    "misbehaviors": int(bucket["misbehaviors"]),
                    "caught": int(bucket["caught"]),
                    "false_positives": int(bucket["false_positives"]),
                    "false_negatives": int(bucket["false_negatives"]),
                    "revision_attempts": int(bucket["revision_attempts"]),
                    "revision_successes": int(bucket["revision_successes"]),
                    "prevented_damage_total": round(bucket["prevented_damage_total"], 4),
                    "allowed_damage_total": round(bucket["allowed_damage_total"], 4),
                    "twin_without_sentinel_damage_total": round(bucket["twin_without_sentinel_damage_total"], 4),
                    "twin_with_sentinel_damage_total": round(bucket["twin_with_sentinel_damage_total"], 4),
                    "twin_prevented_damage_total": round(bucket["twin_prevented_damage_total"], 4),
                    "twin_damage_reduction_rate": round(
                        safe_ratio(
                            bucket["twin_prevented_damage_total"],
                            bucket["twin_without_sentinel_damage_total"],
                        ),
                        4,
                    ),
                    "coaching_quality": round(
                        float(np.mean(bucket["coaching_quality_values"])),
                        4,
                    ) if bucket["coaching_quality_values"] else 0.0,
                    "detection_rate": round(
                        safe_ratio(bucket["caught"], bucket["misbehaviors"]),
                        4,
                    ),
                    "false_positive_rate": round(
                        safe_ratio(
                            bucket["false_positives"],
                            max(0.0, float(sum(bucket["step_values"])) - bucket["misbehaviors"]),
                        ),
                        4,
                    ),
                    "risk_reduction_rate": round(
                        safe_ratio(
                            bucket["prevented_damage_total"],
                            bucket["prevented_damage_total"] + bucket["allowed_damage_total"],
                        ),
                        4,
                    ),
                    "worker_rehabilitation_rate": round(
                        safe_ratio(bucket["revision_successes"], bucket["revision_attempts"]),
                        4,
                    ),
                }
            )
        per_task[task_id] = task_summary

    payload: Dict[str, Any] = {
        "reward_mean": round(reward_mean, 4),
        "reward_min": round(reward_min, 4),
        "reward_max": round(reward_max, 4),
        "reward_std": round(reward_std, 4),
        "avg_steps": round(avg_steps, 4),
        "batch_size": len(safe_rewards),
        "prompt_refreshes": prompt_refreshes,
        "per_task": per_task,
        "curriculum": curriculum_summary or {},
    }
    payload.update(completion_diversity_metrics(completions))
    payload.update(prod_metrics)
    payload.update(training_coverage_metrics(histories, task_ids, variant_seeds, adversarial_cases))
    payload.update(
        zero_gradient_group_metrics(
            rewards=safe_rewards,
            task_ids=task_ids,
            variant_seeds=variant_seeds,
            prompts=prompts,
            adversarial_cases=adversarial_cases,
        )
    )

    if is_sentinel_batch:
        overall = {
            "misbehaviors": 0.0,
            "caught": 0.0,
            "false_positives": 0.0,
            "false_negatives": 0.0,
            "revision_attempts": 0.0,
            "revision_successes": 0.0,
            "prevented_damage_total": 0.0,
            "allowed_damage_total": 0.0,
            "twin_without_sentinel_damage_total": 0.0,
            "twin_with_sentinel_damage_total": 0.0,
            "twin_prevented_damage_total": 0.0,
            "coaching_quality_sum": 0.0,
            "coaching_quality_count": 0.0,
        }
        for history in histories:
            rollup = summarize_sentinel_history(history)
            for key in (
                "misbehaviors",
                "caught",
                "false_positives",
                "false_negatives",
                "revision_attempts",
                "revision_successes",
                "prevented_damage_total",
                "allowed_damage_total",
                "twin_without_sentinel_damage_total",
                "twin_with_sentinel_damage_total",
                "twin_prevented_damage_total",
            ):
                overall[key] += float(rollup[key])
            overall["coaching_quality_sum"] += float(rollup.get("coaching_quality", 0.0))
            overall["coaching_quality_count"] += 1.0

        safe_actions = max(0.0, float(sum(len(history) for history in histories)) - overall["misbehaviors"])
        payload.update(
            {
                "misbehaviors": int(overall["misbehaviors"]),
                "caught": int(overall["caught"]),
                "false_positives": int(overall["false_positives"]),
                "false_negatives": int(overall["false_negatives"]),
                "revision_attempts": int(overall["revision_attempts"]),
                "revision_successes": int(overall["revision_successes"]),
                "prevented_damage_total": round(overall["prevented_damage_total"], 4),
                "allowed_damage_total": round(overall["allowed_damage_total"], 4),
                "twin_without_sentinel_damage_total": round(overall["twin_without_sentinel_damage_total"], 4),
                "twin_with_sentinel_damage_total": round(overall["twin_with_sentinel_damage_total"], 4),
                "twin_prevented_damage_total": round(overall["twin_prevented_damage_total"], 4),
                "twin_damage_reduction_rate": round(
                    safe_ratio(
                        overall["twin_prevented_damage_total"],
                        overall["twin_without_sentinel_damage_total"],
                    ),
                    4,
                ),
                "coaching_quality": round(
                    safe_ratio(overall["coaching_quality_sum"], overall["coaching_quality_count"]),
                    4,
                ),
                "detection_rate": round(safe_ratio(overall["caught"], overall["misbehaviors"]), 4),
                "false_positive_rate": round(safe_ratio(overall["false_positives"], safe_actions), 4),
                "risk_reduction_rate": round(
                    safe_ratio(
                        overall["prevented_damage_total"],
                        overall["prevented_damage_total"] + overall["allowed_damage_total"],
                    ),
                    4,
                ),
                "worker_rehabilitation_rate": round(
                    safe_ratio(overall["revision_successes"], overall["revision_attempts"]),
                    4,
                ),
            }
        )

    return payload
