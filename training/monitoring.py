# -*- coding: utf-8 -*-
"""Training monitoring: TrainingMonitor, GRPOStabilityCallback, RolloutAuditSampler.

Extracted from train.py to keep the training pipeline modular.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from transformers import TrainerCallback

from training.metrics import safe_ratio, aggregate_batch_metrics, summarize_sentinel_history


# ---------------------------------------------------------------------------
# TrainingMonitor
# ---------------------------------------------------------------------------

class TrainingMonitor:
    """Write structured per-batch training metrics for proof-pack and judge review."""

    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / "training_metrics.jsonl"
        self.stability_path = self.output_dir / "training_stability.jsonl"
        self.summary_path = self.output_dir / "latest_summary.json"
        self.stack_path = self.output_dir / "training_stack_versions.json"
        self.batch_index = 0
        self.running_reward_total = 0.0
        self.running_batch_count = 0
        self.best_reward_mean = float("-inf")
        self.latest_batch_metrics: Dict[str, Any] = {}
        self.latest_trainer_metrics: Dict[str, Any] = {}
        self.latest_guardrail: Dict[str, Any] = {}

    def write_stack_versions(self, stack_versions: Dict[str, Any]) -> None:
        self.stack_path.write_text(
            json.dumps(stack_versions, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _write_latest_summary(self) -> None:
        payload = dict(self.latest_batch_metrics)
        if self.latest_trainer_metrics:
            payload.update(self.latest_trainer_metrics)
            payload["trainer_metrics"] = dict(self.latest_trainer_metrics)
        if self.latest_guardrail:
            payload["kl_guardrail"] = dict(self.latest_guardrail)
            payload["adaptive_beta"] = self.latest_guardrail.get("current_beta")
        if not payload:
            return
        self.summary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def log_batch(
        self,
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
        reward_schedule: Optional[Dict[str, Any]] = None,
        memory_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.batch_index += 1
        if sentinel_task_ids is None:
            sentinel_task_ids = ["basic_oversight", "fleet_monitoring_conflict", "adversarial_worker", "multi_crisis_command"]
        metrics = aggregate_batch_metrics(
            rewards=rewards,
            histories=histories,
            task_ids=task_ids,
            variant_seeds=variant_seeds,
            sentinel_task_ids=sentinel_task_ids,
            completions=completions,
            prompts=prompts,
            adversarial_cases=adversarial_cases,
            curriculum_summary=curriculum_summary,
            prompt_refreshes=prompt_refreshes,
        )
        metrics["batch_index"] = self.batch_index
        metrics["monitoring_mode"] = (
            "sentinel"
            if any(task_id in sentinel_task_ids for task_id in task_ids)
            else "irt"
        )
        if reward_schedule:
            metrics["reward_schedule"] = reward_schedule
        if memory_summary:
            metrics["memory"] = memory_summary

        self.running_batch_count += 1
        self.running_reward_total += metrics["reward_mean"]
        self.best_reward_mean = max(self.best_reward_mean, metrics["reward_mean"])
        metrics["running_reward_mean"] = round(
            safe_ratio(self.running_reward_total, self.running_batch_count),
            4,
        )
        metrics["best_reward_mean"] = round(self.best_reward_mean, 4)

        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, sort_keys=True))
            handle.write("\n")

        self.latest_batch_metrics = dict(metrics)
        self._write_latest_summary()
        return metrics

    def log_twin_replay(
        self,
        histories: List[List[Dict[str, Any]]],
        task_ids: List[str],
        variant_seeds: List[int],
        rewards: List[float],
    ) -> Optional[Dict[str, Any]]:
        """Run digital twin counterfactual replay and log metrics."""
        try:
            from sentinel.twin_replay import compute_batch_twin_metrics
            twin_metrics = compute_batch_twin_metrics(histories, task_ids, variant_seeds, rewards)
            if twin_metrics.get("twin_replays", 0) > 0:
                twin_path = self.output_dir / "twin_replay_metrics.jsonl"
                twin_metrics["batch_index"] = self.batch_index
                with twin_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(twin_metrics, sort_keys=True))
                    handle.write("\n")
                # Merge into latest batch metrics for dashboard
                self.latest_batch_metrics.update(twin_metrics)
                self._write_latest_summary()
            return twin_metrics
        except Exception as exc:
            import logging
            logging.getLogger(__name__).debug("Twin replay skipped: %s", exc)
            return None

    def log_reputation_update(
        self,
        histories: List[List[Dict[str, Any]]],
    ) -> Optional[Dict[str, Dict[str, Any]]]:
        """Update cross-episode worker reputation profiles."""
        try:
            from sentinel.reputation import WorkerReputationTracker
            tracker = WorkerReputationTracker(str(self.output_dir / "worker_reputation.json"))
            all_updated = {}
            for history in histories:
                if history:
                    updated = tracker.update_from_episode(history)
                    all_updated.update(updated)
            return all_updated
        except Exception as exc:
            import logging
            logging.getLogger(__name__).debug("Reputation update skipped: %s", exc)
            return None

    def log_trainer_metrics(
        self,
        *,
        global_step: int,
        trainer_metrics: Dict[str, Any],
        guardrail: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "global_step": int(global_step),
            **trainer_metrics,
        }
        if guardrail:
            payload["kl_guardrail"] = guardrail
        with self.stability_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True))
            handle.write("\n")

        self.latest_trainer_metrics = dict(trainer_metrics)
        if guardrail:
            self.latest_guardrail = dict(guardrail)
        self._write_latest_summary()
        return payload


# ---------------------------------------------------------------------------
# GRPOStabilityCallback
# ---------------------------------------------------------------------------

class GRPOStabilityCallback(TrainerCallback):
    """Hook TRL trainer logs to persist KL/entropy metrics and adapt beta conservatively."""

    def __init__(
        self,
        training_monitor: TrainingMonitor,
        *,
        initial_beta: float,
        target_kl: float,
        adaptive: bool,
        low_factor: float,
        high_factor: float,
        beta_up_mult: float,
        beta_down_mult: float,
        min_beta: float,
        max_beta: float,
        hard_stop_enabled: bool,
        hard_stop_mult: float,
    ) -> None:
        self.training_monitor = training_monitor
        self.current_beta = float(initial_beta)
        self.target_kl = float(target_kl)
        self.adaptive = bool(adaptive)
        self.low_factor = float(low_factor)
        self.high_factor = float(high_factor)
        self.beta_up_mult = float(beta_up_mult)
        self.beta_down_mult = float(beta_down_mult)
        self.min_beta = float(min_beta)
        self.max_beta = float(max_beta)
        self.hard_stop_enabled = bool(hard_stop_enabled)
        self.hard_stop_mult = float(hard_stop_mult)
        self.trainer = None

    def bind_trainer(self, trainer) -> None:
        self.trainer = trainer
        self.current_beta = float(getattr(trainer, "beta", self.current_beta) or self.current_beta)

    @staticmethod
    def _first_float(logs: Dict[str, Any], keys: List[str]) -> Optional[float]:
        for key in keys:
            value = logs.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    def _apply_beta(self, value: float) -> None:
        if self.trainer is None:
            self.current_beta = float(value)
            return
        self.current_beta = float(value)
        setattr(self.trainer, "beta", self.current_beta)
        if hasattr(self.trainer, "args"):
            if hasattr(self.trainer.args, "beta"):
                setattr(self.trainer.args, "beta", self.current_beta)
            if hasattr(self.trainer.args, "kl_coef"):
                setattr(self.trainer.args, "kl_coef", self.current_beta)

    def _guardrail_update(self, approx_kl: Optional[float]):
        low_threshold = self.target_kl / max(self.low_factor, 1.0)
        high_threshold = self.target_kl * max(self.high_factor, 1.0)
        guardrail = {
            "enabled": self.adaptive,
            "target_kl": round(self.target_kl, 4),
            "low_threshold": round(low_threshold, 4),
            "high_threshold": round(high_threshold, 4),
            "previous_beta": round(self.current_beta, 6),
            "current_beta": round(self.current_beta, 6),
            "action": "hold",
            "hard_stop_triggered": False,
        }
        if approx_kl is None:
            return guardrail

        new_beta = self.current_beta
        if self.adaptive and approx_kl > high_threshold:
            new_beta = min(self.max_beta, self.current_beta * self.beta_up_mult)
            guardrail["action"] = "increase_beta"
        elif self.adaptive and approx_kl < low_threshold:
            new_beta = max(self.min_beta, self.current_beta * self.beta_down_mult)
            guardrail["action"] = "decrease_beta"

        if abs(new_beta - self.current_beta) > 1e-12:
            self._apply_beta(new_beta)
            guardrail["current_beta"] = round(self.current_beta, 6)

        if self.hard_stop_enabled and approx_kl > self.target_kl * max(self.hard_stop_mult, 1.0):
            guardrail["hard_stop_triggered"] = True
            guardrail["action"] = "hard_stop"
        return guardrail

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        if any(str(key).startswith("eval_") for key in logs):
            return control

        approx_kl = self._first_float(logs, ["kl", "objective/kl"])
        policy_entropy = self._first_float(logs, ["entropy", "policy/entropy"])
        clip_ratio = self._first_float(logs, ["clip_ratio/region_mean", "clip_ratio", "objective/clip_ratio"])
        if approx_kl is None and policy_entropy is None and clip_ratio is None:
            return control

        guardrail = self._guardrail_update(approx_kl)
        trainer_metrics = {
            "approx_kl": round(float(approx_kl), 6) if approx_kl is not None else None,
            "policy_entropy": round(float(policy_entropy), 6) if policy_entropy is not None else None,
            "clip_ratio": round(float(clip_ratio), 6) if clip_ratio is not None else None,
        }
        self.training_monitor.log_trainer_metrics(
            global_step=int(getattr(state, "global_step", 0) or 0),
            trainer_metrics={key: value for key, value in trainer_metrics.items() if value is not None},
            guardrail=guardrail,
        )
        if guardrail.get("hard_stop_triggered"):
            control.should_training_stop = True
        return control


# ---------------------------------------------------------------------------
# RolloutAuditSampler
# ---------------------------------------------------------------------------

def _truncate_text(text: str, limit: int = 700) -> str:
    clean = (text or "").strip()
    if len(clean) <= limit:
        return clean
    return clean[: max(0, limit - 3)].rstrip() + "..."


def _audit_priority(task_id: str, reward: float, history: List[Dict[str, Any]], sentinel_task_ids: List[str]) -> float:
    priority = max(0.0, 1.0 - float(reward))
    if task_id in sentinel_task_ids:
        rollup = summarize_sentinel_history(history)
        priority += rollup["false_negatives"] * 2.0
        priority += rollup["false_positives"] * 1.5
        priority += (1.0 - rollup["risk_reduction_rate"]) * 0.8
        priority += rollup["revision_attempts"] * 0.25
    else:
        priority += len(history) * 0.05
    return round(priority, 4)


class RolloutAuditSampler:
    """Persist a periodic sample of rollout traces for human audit during training."""

    def __init__(self, output_dir: str, every: int, sample_limit: int) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.every = max(0, every)
        self.sample_limit = max(0, sample_limit)
        self.latest_markdown_path = self.output_dir / "latest.md"

    def record_batch(
        self,
        *,
        batch_index: int,
        prompts: List[str],
        completions: List[str],
        rewards: List[float],
        histories: List[List[Dict[str, Any]]],
        task_ids: List[str],
        variant_seeds: List[int],
        sentinel_task_ids: Optional[List[str]] = None,
        active_task_ids: Optional[List[str]] = None,
        monitor_summary: Dict[str, Any],
        reward_schedule: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        if sentinel_task_ids is None:
            sentinel_task_ids = ["basic_oversight", "fleet_monitoring_conflict", "adversarial_worker", "multi_crisis_command"]
        if active_task_ids is None:
            active_task_ids = task_ids
        if self.every <= 0 or self.sample_limit <= 0:
            return None
        if batch_index % self.every != 0:
            return None

        candidates: List[Dict[str, Any]] = []
        for index, reward in enumerate(rewards):
            task_id = str(task_ids[index]) if index < len(task_ids) else active_task_ids[0]
            variant_seed = int(variant_seeds[index]) if index < len(variant_seeds) else 0
            history = histories[index] if index < len(histories) else []
            history_summary = (
                summarize_sentinel_history(history)
                if task_id in sentinel_task_ids
                else {"steps": float(len(history))}
            )
            candidates.append(
                {
                    "task_id": task_id,
                    "variant_seed": variant_seed,
                    "reward": round(float(reward), 4),
                    "priority": _audit_priority(task_id, reward, history, sentinel_task_ids),
                    "prompt": prompts[index] if index < len(prompts) else "",
                    "completion": completions[index] if index < len(completions) else "",
                    "history_summary": history_summary,
                    "history": history,
                }
            )

        top_samples = sorted(
            candidates,
            key=lambda item: (item["priority"], item["reward"]),
            reverse=True,
        )[: self.sample_limit]

        payload = {
            "batch_index": batch_index,
            "reward_schedule": reward_schedule or {},
            "monitor_summary": monitor_summary,
            "samples": top_samples,
        }
        json_path = self.output_dir / f"batch_{batch_index:04d}.json"
        json_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )

        lines = [
            f"# Rollout Audit Batch {batch_index}",
            "",
            f"- Samples: {len(top_samples)}",
            f"- Reward mean: {monitor_summary.get('reward_mean', 0.0):.4f}",
            f"- Running reward mean: {monitor_summary.get('running_reward_mean', 0.0):.4f}",
        ]
        if "approx_kl" in monitor_summary:
            lines.append(f"- Approx KL: {monitor_summary.get('approx_kl', 0.0):.6f}")
        if "adaptive_beta" in monitor_summary:
            lines.append(f"- Adaptive beta: {monitor_summary.get('adaptive_beta', 0.0):.6f}")
        if "policy_entropy" in monitor_summary:
            lines.append(f"- Policy entropy: {monitor_summary.get('policy_entropy', 0.0):.6f}")
        if "decision_entropy" in monitor_summary:
            lines.append(f"- Decision entropy: {monitor_summary.get('decision_entropy', 0.0):.4f}")
        if "unique_completion_ratio" in monitor_summary:
            lines.append(f"- Unique completion ratio: {monitor_summary.get('unique_completion_ratio', 0.0):.4f}")
        if "effective_prompt_ratio" in monitor_summary:
            lines.append(f"- Effective prompt ratio: {monitor_summary.get('effective_prompt_ratio', 0.0):.4f}")
        if "frontier_hit_rate" in monitor_summary:
            lines.append(f"- Frontier hit rate: {monitor_summary.get('frontier_hit_rate', 0.0):.4f}")
        if "task_diversity_ratio" in monitor_summary:
            lines.append(f"- Task diversity ratio: {monitor_summary.get('task_diversity_ratio', 0.0):.4f}")
        if "zero_gradient_group_fraction" in monitor_summary:
            lines.append(f"- Zero-gradient group fraction: {monitor_summary.get('zero_gradient_group_fraction', 0.0):.4f}")
        if "adversarial_case_fraction" in monitor_summary:
            lines.append(f"- Adversarial case fraction: {monitor_summary.get('adversarial_case_fraction', 0.0):.4f}")
        if "twin_damage_reduction_rate" in monitor_summary:
            lines.append(f"- Twin damage reduction rate: {monitor_summary.get('twin_damage_reduction_rate', 0.0):.4f}")
        if "coaching_quality" in monitor_summary:
            lines.append(f"- Coaching quality: {monitor_summary.get('coaching_quality', 0.0):.4f}")
        if monitor_summary.get("memory"):
            mem = monitor_summary["memory"]
            lines.append(
                f"- Memory: enabled={mem.get('agent_memory_enabled')} cards={mem.get('mistake_cards_stored', 0)}"
            )
        if reward_schedule:
            lines.append(
                f"- Reward schedule: {reward_schedule.get('stage', 'unknown')} ({reward_schedule.get('mode', 'unknown')})"
            )
        lines.append("")

        for sample_index, sample in enumerate(top_samples, start=1):
            history_summary = sample.get("history_summary") or {}
            lines.extend(
                [
                    f"## Sample {sample_index}",
                    "",
                    f"- Task: `{sample['task_id']}`",
                    f"- Seed: `{sample['variant_seed']}`",
                    f"- Reward: `{sample['reward']:.4f}`",
                    f"- Audit priority: `{sample['priority']:.4f}`",
                ]
            )
            if "detection_rate" in history_summary:
                lines.extend(
                    [
                        f"- Detection rate: `{history_summary.get('detection_rate', 0.0):.4f}`",
                        f"- False positive rate: `{history_summary.get('false_positive_rate', 0.0):.4f}`",
                        f"- Risk reduction rate: `{history_summary.get('risk_reduction_rate', 0.0):.4f}`",
                        f"- Twin without SENTINEL damage: `{history_summary.get('twin_without_sentinel_damage_total', 0.0):.4f}`",
                        f"- Twin with SENTINEL damage: `{history_summary.get('twin_with_sentinel_damage_total', 0.0):.4f}`",
                        f"- Rehabilitation rate: `{history_summary.get('worker_rehabilitation_rate', 0.0):.4f}`",
                        f"- Coaching quality: `{history_summary.get('coaching_quality', 0.0):.4f}`",
                    ]
                )
            lines.extend(
                [
                    "",
                    "### Prompt",
                    "",
                    "```text",
                    _truncate_text(str(sample.get("prompt", ""))),
                    "```",
                    "",
                    "### Completion",
                    "",
                    "```json",
                    _truncate_text(str(sample.get("completion", ""))),
                    "```",
                    "",
                ]
            )

        self.latest_markdown_path.write_text("\n".join(lines), encoding="utf-8")
        return str(json_path)
