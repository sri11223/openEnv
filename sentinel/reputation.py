# -*- coding: utf-8 -*-
"""Cross-Episode Worker Reputation Learning.

Builds persistent reputation profiles for each worker that carry across
training episodes. SENTINEL uses these profiles to make better-informed
oversight decisions — implementing genuine theory-of-mind reasoning.

Usage:
    from sentinel.reputation import WorkerReputationTracker

    tracker = WorkerReputationTracker("outputs/reputation.json")
    tracker.record_episode("worker_db", episode_stats)
    profile = tracker.get_profile("worker_db")
    context = tracker.build_reputation_context()  # inject into prompts
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default reputation for a new worker
_DEFAULT_PROFILE = {
    "episodes_seen": 0,
    "total_proposals": 0,
    "misbehaviors_total": 0,
    "misbehaviors_caught": 0,
    "false_positives_caused": 0,
    "trust_trajectory": [],
    "misbehavior_type_counts": {},
    "domains_reliable": [],
    "domains_unreliable": [],
    "rehabilitation_attempts": 0,
    "rehabilitation_successes": 0,
    "current_trust_score": 0.70,
    "trend": "stable",
}


class WorkerReputationTracker:
    """Persistent cross-episode reputation tracker for worker agents."""

    def __init__(self, path: str = "outputs/worker_reputation.json", max_trajectory: int = 50):
        self.path = Path(path)
        self.max_trajectory = max_trajectory
        self.profiles: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                self.profiles = json.loads(self.path.read_text(encoding="utf-8"))
                logger.info("Loaded reputation profiles for %d workers", len(self.profiles))
            except Exception as exc:
                logger.warning("Failed to load reputation: %s", exc)
                self.profiles = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.profiles, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )

    def _ensure_profile(self, worker_id: str) -> Dict[str, Any]:
        if worker_id not in self.profiles:
            self.profiles[worker_id] = dict(_DEFAULT_PROFILE)
            self.profiles[worker_id]["trust_trajectory"] = []
            self.profiles[worker_id]["misbehavior_type_counts"] = {}
            self.profiles[worker_id]["domains_reliable"] = []
            self.profiles[worker_id]["domains_unreliable"] = []
        return self.profiles[worker_id]

    def record_episode(
        self,
        worker_id: str,
        episode_stats: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Record one episode's stats for a worker.

        episode_stats should contain:
            proposals: int — total proposals made
            misbehaviors: int — number of misbehavior proposals
            caught: int — misbehaviors SENTINEL caught
            false_positives: int — valid proposals SENTINEL incorrectly blocked
            misbehavior_types: list[str] — types of misbehavior in this episode
            domain: str — worker's domain
            rehabilitation_attempted: bool
            rehabilitation_success: bool
        """
        profile = self._ensure_profile(worker_id)
        profile["episodes_seen"] += 1
        profile["total_proposals"] += int(episode_stats.get("proposals", 0))
        profile["misbehaviors_total"] += int(episode_stats.get("misbehaviors", 0))
        profile["misbehaviors_caught"] += int(episode_stats.get("caught", 0))
        profile["false_positives_caused"] += int(episode_stats.get("false_positives", 0))

        # Track misbehavior type distribution
        for mb_type in episode_stats.get("misbehavior_types", []):
            key = str(mb_type)
            profile["misbehavior_type_counts"][key] = profile["misbehavior_type_counts"].get(key, 0) + 1

        # Rehabilitation tracking
        if episode_stats.get("rehabilitation_attempted"):
            profile["rehabilitation_attempts"] += 1
        if episode_stats.get("rehabilitation_success"):
            profile["rehabilitation_successes"] += 1

        # Compute trust score
        total = max(1, profile["total_proposals"])
        misbehavior_rate = profile["misbehaviors_total"] / total
        trust = max(0.0, min(1.0, 1.0 - misbehavior_rate * 1.5))
        profile["current_trust_score"] = round(trust, 4)

        # Track trajectory
        profile["trust_trajectory"].append(round(trust, 4))
        if len(profile["trust_trajectory"]) > self.max_trajectory:
            profile["trust_trajectory"] = profile["trust_trajectory"][-self.max_trajectory:]

        # Compute trend
        traj = profile["trust_trajectory"]
        if len(traj) >= 5:
            recent = sum(traj[-5:]) / 5
            older = sum(traj[-10:-5]) / 5 if len(traj) >= 10 else sum(traj[:5]) / max(1, min(5, len(traj)))
            if recent > older + 0.05:
                profile["trend"] = "improving"
            elif recent < older - 0.05:
                profile["trend"] = "declining"
            else:
                profile["trend"] = "stable"

        # Domain reliability
        domain = episode_stats.get("domain", "")
        if domain:
            if misbehavior_rate < 0.15 and domain not in profile["domains_reliable"]:
                profile["domains_reliable"].append(domain)
            elif misbehavior_rate > 0.30 and domain not in profile["domains_unreliable"]:
                profile["domains_unreliable"].append(domain)

        # Find most common misbehavior
        if profile["misbehavior_type_counts"]:
            most_common = max(profile["misbehavior_type_counts"], key=profile["misbehavior_type_counts"].get)
            profile["most_common_misbehavior"] = most_common
        else:
            profile["most_common_misbehavior"] = None

        # Compute rehabilitation rate
        if profile["rehabilitation_attempts"] > 0:
            profile["rehabilitation_rate"] = round(
                profile["rehabilitation_successes"] / profile["rehabilitation_attempts"], 4
            )
        else:
            profile["rehabilitation_rate"] = 0.0

        # Compute misbehavior frequency
        profile["misbehavior_frequency"] = round(misbehavior_rate, 4)

        self._save()
        return profile

    def get_profile(self, worker_id: str) -> Dict[str, Any]:
        return self._ensure_profile(worker_id)

    def get_all_profiles(self) -> Dict[str, Dict[str, Any]]:
        return dict(self.profiles)

    def build_reputation_context(self, max_chars: int = 600) -> str:
        """Build a text context block for injection into SENTINEL prompts.

        Returns a concise summary of each worker's reputation that helps
        SENTINEL make better-informed oversight decisions.
        """
        if not self.profiles:
            return ""

        lines = ["WORKER REPUTATION PROFILES (cross-episode):"]
        for worker_id, profile in sorted(self.profiles.items()):
            trust = profile.get("current_trust_score", 0.7)
            trend = profile.get("trend", "stable")
            freq = profile.get("misbehavior_frequency", 0.0)
            most_common = profile.get("most_common_misbehavior", "none")
            episodes = profile.get("episodes_seen", 0)
            rehab_rate = profile.get("rehabilitation_rate", 0.0)

            trust_label = "HIGH" if trust >= 0.75 else "MEDIUM" if trust >= 0.50 else "LOW"
            trend_icon = "↑" if trend == "improving" else "↓" if trend == "declining" else "→"

            line = (
                f"  {worker_id}: trust={trust_label}({trust:.2f}{trend_icon}) "
                f"misbehavior_rate={freq:.0%} "
                f"primary_risk={most_common or 'none'} "
                f"episodes={episodes} "
                f"rehab={rehab_rate:.0%}"
            )
            lines.append(line)
            if len("\n".join(lines)) > max_chars:
                break

        return "\n".join(lines)

    def extract_from_episode_history(
        self,
        history: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Extract per-worker stats from a SENTINEL episode history.

        Returns a dict keyed by worker_id with episode_stats suitable
        for record_episode().
        """
        worker_stats: Dict[str, Dict[str, Any]] = {}

        for entry in history:
            audit = entry.get("audit", {}) or {}
            proposal = entry.get("proposal", {}) or {}
            revision = entry.get("worker_revision", {}) or {}
            info = entry.get("info", {}) or {}

            worker_id = str(audit.get("worker_id") or proposal.get("worker_id") or "unknown")
            if worker_id not in worker_stats:
                worker_stats[worker_id] = {
                    "proposals": 0,
                    "misbehaviors": 0,
                    "caught": 0,
                    "false_positives": 0,
                    "misbehavior_types": [],
                    "domain": "",
                    "rehabilitation_attempted": False,
                    "rehabilitation_success": False,
                }

            stats = worker_stats[worker_id]
            stats["proposals"] += 1
            stats["domain"] = str(audit.get("worker_role") or info.get("worker_role") or "")

            was_mb = bool(audit.get("was_misbehavior") or info.get("is_misbehavior"))
            decision = audit.get("sentinel_decision") or ""

            if was_mb:
                stats["misbehaviors"] += 1
                mb_type = str(audit.get("reason") or info.get("mb_type") or "")
                if mb_type:
                    stats["misbehavior_types"].append(mb_type)
                if decision and decision != "APPROVE":
                    stats["caught"] += 1
            elif decision and decision != "APPROVE":
                stats["false_positives"] += 1

            if revision.get("attempted"):
                stats["rehabilitation_attempted"] = True
            if revision.get("revision_approved"):
                stats["rehabilitation_success"] = True

        return worker_stats

    def update_from_episode(self, history: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Convenience: extract stats from history and record all workers."""
        per_worker = self.extract_from_episode_history(history)
        updated = {}
        for worker_id, stats in per_worker.items():
            updated[worker_id] = self.record_episode(worker_id, stats)
        return updated
