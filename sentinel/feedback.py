"""SENTINEL feedback memory.

Tracks two aligned feedback loops:
  1. Global oversight lessons (what SENTINEL keeps learning across episodes)
  2. Per-worker mistake memory (what each worker repeatedly gets wrong)

This memory is used in two places:
  - Runtime: better explanations, reassignment hints, and worker-pattern summaries
  - Training: prompt context so the overseer sees recurring mistakes and corrections
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional

from sentinel.models import WorkerId

DEFAULT_FEEDBACK_PATH = os.path.join("outputs", "sentinel_feedback_memory.json")
MAX_EVENTS = 200
MAX_ITEMS_PER_LIST = 20


def empty_feedback_memory() -> Dict[str, Any]:
    return {
        "version": 1,
        "total_events": 0,
        "global_mistakes": [],
        "global_corrections": [],
        "global_rehabilitations": [],
        "task_notes": {},
        "worker_profiles": {},
        "events": [],
    }


def load_feedback_memory(path: str = DEFAULT_FEEDBACK_PATH) -> Dict[str, Any]:
    if not os.path.exists(path):
        return empty_feedback_memory()
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return empty_feedback_memory()
    return _normalize_memory(data)


def save_feedback_memory(memory: Dict[str, Any], path: str = DEFAULT_FEEDBACK_PATH) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    trimmed = _normalize_memory(memory)
    trimmed["events"] = trimmed.get("events", [])[-MAX_EVENTS:]
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(trimmed, handle, indent=2)


def record_feedback_event(
    memory: Dict[str, Any],
    event: Dict[str, Any],
) -> Dict[str, Any]:
    memory = _normalize_memory(memory)
    worker_id = str(event.get("worker_id") or "unknown")
    task_id = str(event.get("task_id") or "unknown")
    incident_label = str(event.get("incident_label") or event.get("incident_id") or "incident")
    decision = str(event.get("decision") or "")
    reason = str(event.get("reason") or "")
    action_type = str(event.get("action_type") or "")
    target = str(event.get("target") or "N/A")
    signature = f"{action_type}:{target}"
    profile = _worker_profile(memory, worker_id)
    task_notes = _task_notes(memory, task_id)

    mistake_line = (
        f"{reason or 'unsafe_pattern'} via {signature} on {incident_label}"
        if event.get("is_misbehavior")
        else ""
    )
    correction_line = _correction_line(event)
    rehabilitation_line = _rehabilitation_line(event)
    safe_line = (
        f"safe {signature} approved on {incident_label}"
        if not event.get("is_misbehavior") and decision == "APPROVE"
        else ""
    )

    if mistake_line:
        _append_unique(profile["mistakes"], mistake_line)
        _append_unique(memory["global_mistakes"], mistake_line)
        _append_unique(task_notes["mistakes"], mistake_line)
        profile["violation_counts"][reason or "unknown"] = (
            profile["violation_counts"].get(reason or "unknown", 0) + 1
        )

    if correction_line:
        _append_unique(profile["corrections"], correction_line)
        _append_unique(memory["global_corrections"], correction_line)
        _append_unique(task_notes["corrections"], correction_line)

    if rehabilitation_line:
        _append_unique(profile["rehabilitations"], rehabilitation_line)
        _append_unique(memory["global_rehabilitations"], rehabilitation_line)
        _append_unique(task_notes["rehabilitations"], rehabilitation_line)

    if safe_line:
        _append_unique(profile["successes"], safe_line)

    reassign_to = event.get("reassign_to")
    if reassign_to:
        key = str(reassign_to)
        profile["preferred_reassignments"][key] = (
            profile["preferred_reassignments"].get(key, 0) + 1
        )

    if event.get("suggested_reassign_to"):
        suggested = str(event["suggested_reassign_to"])
        profile["suggested_targets"][suggested] = (
            profile["suggested_targets"].get(suggested, 0) + 1
        )

    profile["last_feedback"] = rehabilitation_line or correction_line or mistake_line or safe_line
    profile["last_task_id"] = task_id
    profile["last_incident_id"] = str(event.get("incident_id") or "")
    _append_unique(
        profile["recent_incidents"],
        f"{incident_label}:{decision or 'unknown'}:{signature}",
    )

    memory["events"].append(
        {
            "task_id": task_id,
            "incident_id": event.get("incident_id"),
            "incident_label": event.get("incident_label"),
            "worker_id": worker_id,
            "decision": decision,
            "reason": reason,
            "action_type": action_type,
            "target": target,
            "is_misbehavior": bool(event.get("is_misbehavior")),
            "revision_attempted": bool(event.get("revision_attempted")),
            "revision_approved": bool(event.get("revision_approved")),
            "revised_by": event.get("revised_by"),
            "revised_action_type": event.get("revised_action_type"),
            "revised_target": event.get("revised_target"),
            "executed_action_source": event.get("executed_action_source"),
        }
    )
    memory["events"] = memory["events"][-MAX_EVENTS:]
    memory["total_events"] = int(memory.get("total_events", 0)) + 1
    return memory


def record_episode_feedback(
    memory: Dict[str, Any],
    task_id: str,
    history: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    updated = _normalize_memory(memory)
    for entry in history:
        audit = entry.get("audit") or {}
        if not audit:
            continue
        info = entry.get("info") or {}
        decision = entry.get("decision") or {}
        updated = record_feedback_event(
            updated,
            {
                "task_id": task_id,
                "incident_id": audit.get("incident_id"),
                "incident_label": audit.get("incident_label"),
                "worker_id": audit.get("worker_id"),
                "decision": audit.get("sentinel_decision") or decision.get("action") or decision.get("decision"),
                "reason": audit.get("reason") or decision.get("reason"),
                "action_type": audit.get("proposed_action_type"),
                "target": audit.get("proposed_target"),
                "is_misbehavior": audit.get("was_misbehavior"),
                "reassign_to": audit.get("reassign_to") or decision.get("reassign_to"),
                "suggested_reassign_to": info.get("feedback_memory", {}).get("suggested_reassign_to"),
                "constitutional_violations": audit.get("constitutional_violations", []),
                "revision_attempted": (entry.get("worker_revision") or {}).get("attempted"),
                "revision_approved": (entry.get("worker_revision") or {}).get("revision_approved"),
                "revised_by": (entry.get("worker_revision") or {}).get("revised_by"),
                "revised_action_type": ((entry.get("worker_revision") or {}).get("revised_proposal") or {}).get("action_type"),
                "revised_target": ((entry.get("worker_revision") or {}).get("revised_proposal") or {}).get("target"),
                "executed_action_source": (entry.get("executed_action") or {}).get("source"),
            },
        )
    return updated


def build_feedback_summary(
    memory: Dict[str, Any],
    worker_id: Optional[str] = None,
    task_id: Optional[str] = None,
    available_workers: Optional[Iterable[Any]] = None,
) -> Dict[str, Any]:
    memory = _normalize_memory(memory)
    profile = _worker_profile(memory, worker_id) if worker_id else None
    task_notes = _task_notes(memory, task_id) if task_id else {"mistakes": [], "corrections": []}
    summary = {
        "global_mistakes": list(memory.get("global_mistakes", [])[-3:]),
        "global_corrections": list(memory.get("global_corrections", [])[-3:]),
        "global_rehabilitations": list(memory.get("global_rehabilitations", [])[-2:]),
        "task_mistakes": list(task_notes.get("mistakes", [])[-2:]),
        "task_corrections": list(task_notes.get("corrections", [])[-2:]),
        "task_rehabilitations": list(task_notes.get("rehabilitations", [])[-2:]),
        "worker_mistakes": list(profile.get("mistakes", [])[-3:]) if profile else [],
        "worker_successes": list(profile.get("successes", [])[-2:]) if profile else [],
        "worker_rehabilitations": list(profile.get("rehabilitations", [])[-2:]) if profile else [],
        "last_feedback": profile.get("last_feedback", "") if profile else "",
    }
    suggested = recommended_reassign_to(memory, worker_id, available_workers=available_workers)
    if suggested:
        summary["suggested_reassign_to"] = suggested
    if profile and profile.get("violation_counts"):
        top_violation = max(
            profile["violation_counts"].items(),
            key=lambda item: item[1],
        )[0]
        summary["top_violation"] = top_violation
    return summary


def build_feedback_context(
    memory: Dict[str, Any],
    task_id: Optional[str] = None,
    worker_ids: Optional[Iterable[Any]] = None,
) -> str:
    memory = _normalize_memory(memory)
    lines: List[str] = ["## FEEDBACK LOOP MEMORY"]
    if memory.get("global_mistakes"):
        lines.append("Global mistakes to avoid:")
        for item in memory["global_mistakes"][-3:]:
            lines.append(f"  - {item}")
    if memory.get("global_corrections"):
        lines.append("Global corrections that worked:")
        for item in memory["global_corrections"][-3:]:
            lines.append(f"  - {item}")
    if memory.get("global_rehabilitations"):
        lines.append("Rehabilitations that worked after supervisor feedback:")
        for item in memory["global_rehabilitations"][-2:]:
            lines.append(f"  - {item}")
    if task_id:
        task_notes = _task_notes(memory, task_id)
        if task_notes["mistakes"] or task_notes["corrections"] or task_notes["rehabilitations"]:
            lines.append(f"Task memory for {task_id}:")
            for item in task_notes["mistakes"][-2:]:
                lines.append(f"  - Avoid: {item}")
            for item in task_notes["corrections"][-2:]:
                lines.append(f"  - Prefer: {item}")
            for item in task_notes["rehabilitations"][-2:]:
                lines.append(f"  - Rehabilitation: {item}")
    for worker in list(worker_ids or [])[:4]:
        worker_key = str(worker.value if isinstance(worker, WorkerId) else worker)
        profile = _worker_profile(memory, worker_key)
        if not profile["mistakes"] and not profile["successes"]:
            continue
        lines.append(f"Worker profile {worker_key}:")
        for item in profile["mistakes"][-2:]:
            lines.append(f"  - Repeated mistake: {item}")
        for item in profile["successes"][-1:]:
            lines.append(f"  - Reliable pattern: {item}")
        for item in profile["rehabilitations"][-1:]:
            lines.append(f"  - Rehab pattern: {item}")
        suggested = recommended_reassign_to(memory, worker_key)
        if suggested:
            lines.append(f"  - Best reassignment target so far: {suggested}")
    return "" if len(lines) == 1 else "\n".join(lines)


def recommended_reassign_to(
    memory: Dict[str, Any],
    worker_id: Optional[str],
    available_workers: Optional[Iterable[Any]] = None,
) -> Optional[str]:
    if not worker_id:
        return None
    memory = _normalize_memory(memory)
    profile = _worker_profile(memory, worker_id)
    candidates = {
        **profile.get("preferred_reassignments", {}),
        **{
            key: profile.get("suggested_targets", {}).get(key, 0)
            + profile.get("preferred_reassignments", {}).get(key, 0)
            for key in set(profile.get("suggested_targets", {})) | set(profile.get("preferred_reassignments", {}))
        },
    }
    allowed = {
        str(item.value if isinstance(item, WorkerId) else item)
        for item in (available_workers or [])
    }
    best: Optional[str] = None
    best_score = -1
    for candidate, score in candidates.items():
        if candidate == worker_id:
            continue
        if allowed and candidate not in allowed:
            continue
        if score > best_score:
            best = candidate
            best_score = score
    return best


def _normalize_memory(memory: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    data = dict(empty_feedback_memory())
    if memory:
        data.update(memory)
    data.setdefault("task_notes", {})
    data.setdefault("worker_profiles", {})
    data.setdefault("events", [])
    data.setdefault("global_mistakes", [])
    data.setdefault("global_corrections", [])
    data.setdefault("global_rehabilitations", [])
    data.setdefault("total_events", 0)
    return data


def _task_notes(memory: Dict[str, Any], task_id: Optional[str]) -> Dict[str, Any]:
    key = task_id or "unknown"
    notes = memory["task_notes"].setdefault(
        key,
        {"mistakes": [], "corrections": [], "rehabilitations": []},
    )
    notes.setdefault("mistakes", [])
    notes.setdefault("corrections", [])
    notes.setdefault("rehabilitations", [])
    return notes


def _worker_profile(memory: Dict[str, Any], worker_id: Optional[str]) -> Dict[str, Any]:
    key = worker_id or "unknown"
    profile = memory["worker_profiles"].setdefault(
        key,
        {
            "mistakes": [],
            "successes": [],
            "corrections": [],
            "rehabilitations": [],
            "violation_counts": {},
            "preferred_reassignments": {},
            "suggested_targets": {},
            "recent_incidents": [],
            "last_feedback": "",
            "last_task_id": "",
            "last_incident_id": "",
        },
    )
    profile.setdefault("mistakes", [])
    profile.setdefault("successes", [])
    profile.setdefault("corrections", [])
    profile.setdefault("rehabilitations", [])
    profile.setdefault("violation_counts", {})
    profile.setdefault("preferred_reassignments", {})
    profile.setdefault("suggested_targets", {})
    profile.setdefault("recent_incidents", [])
    profile.setdefault("last_feedback", "")
    return profile


def _append_unique(items: List[str], value: str) -> None:
    value = value.strip()
    if not value:
        return
    if value in items:
        items.remove(value)
    items.append(value)
    del items[:-MAX_ITEMS_PER_LIST]


def _correction_line(event: Dict[str, Any]) -> str:
    decision = str(event.get("decision") or "")
    reason = str(event.get("reason") or "unsafe_pattern")
    target = str(event.get("target") or "N/A")
    if decision == "BLOCK":
        return f"BLOCK {reason} on {target} until evidence is present."
    if decision == "REDIRECT":
        return f"REDIRECT {reason} on {target} to a lower-blast-radius action."
    if decision == "REASSIGN":
        assignee = event.get("reassign_to") or event.get("suggested_reassign_to")
        if assignee:
            return f"REASSIGN {reason} on {target} to {assignee}."
        return f"REASSIGN {reason} on {target} to the domain owner."
    if decision == "FLAG":
        return f"FLAG suspicious {reason} pattern on {target} for follow-up."
    return ""


def _rehabilitation_line(event: Dict[str, Any]) -> str:
    if not event.get("revision_attempted") or not event.get("revision_approved"):
        return ""
    revised_by = str(event.get("revised_by") or event.get("worker_id") or "worker")
    revised_action = str(event.get("revised_action_type") or event.get("action_type") or "action")
    revised_target = str(event.get("revised_target") or event.get("target") or "N/A")
    source = str(event.get("executed_action_source") or "revised")
    return (
        f"{revised_by} recovered safely with {revised_action}:{revised_target} "
        f"(executed via {source}) after supervisor feedback."
    )
