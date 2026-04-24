# -*- coding: utf-8 -*-
"""Shared state and helpers used by all routers.

Centralizes session registries, telemetry counters, leaderboard,
and Prometheus metric helpers so that routers can import them
without circular dependencies back to app.py.
"""

from __future__ import annotations

import logging
import os
import re
import secrets
import time
from typing import Any, Dict, List

_log = logging.getLogger("irt.api")

# ---------------------------------------------------------------------------
# Session-isolated environment registry
# ---------------------------------------------------------------------------

_SESSION_REGISTRY: Dict[str, Any] = {}
_SESSION_TIMESTAMPS: Dict[str, float] = {}
_MAX_SESSIONS = 256
_SESSION_TTL = int(os.environ.get("SESSION_TTL_SECONDS", 3600))

# SENTINEL session registry (separate from IRT)
_SENTINEL_REGISTRY: Dict[str, Any] = {}
_SENTINEL_TIMESTAMPS: Dict[str, float] = {}

# ---------------------------------------------------------------------------
# Telemetry counters  (in-process; reset on restart)
# ---------------------------------------------------------------------------

_TELEMETRY: Dict[str, int] = {
    "sessions_created": 0,
    "sessions_evicted_fifo": 0,
    "sessions_expired_ttl": 0,
    "episodes_total": 0,
    "steps_total": 0,
    "grader_calls": 0,
    "baseline_runs": 0,
    "errors_total": 0,
    "ws_connections_total": 0,
    "sentinel_sessions_created": 0,
    "sentinel_episodes_total": 0,
    "sentinel_steps_total": 0,
    "sentinel_grader_calls": 0,
}

# Active WebSocket connections (single-process; decremented on disconnect)
WS_ACTIVE_CONNECTIONS: int = 0

# ---------------------------------------------------------------------------
# In-memory leaderboard  (top-10 scores per task)
# ---------------------------------------------------------------------------

_LEADERBOARD: Dict[str, List[Dict[str, Any]]] = {
    "severity_classification": [],
    "root_cause_analysis": [],
    "full_incident_management": [],
    "basic_oversight": [],
    "fleet_monitoring_conflict": [],
    "adversarial_worker": [],
    "multi_crisis_command": [],
}
_LEADERBOARD_SIZE = 10


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def get_or_create_session(session_id: str | None):
    """Return (session_id, env). Creates a new session if id is None or unknown."""
    from src.environment import IncidentResponseEnv

    if session_id and session_id in _SESSION_REGISTRY:
        return session_id, _SESSION_REGISTRY[session_id]
    # New session - evict if at capacity
    if len(_SESSION_REGISTRY) >= _MAX_SESSIONS:
        oldest = next(iter(_SESSION_REGISTRY))
        del _SESSION_REGISTRY[oldest]
        _SESSION_TIMESTAMPS.pop(oldest, None)
        _TELEMETRY["sessions_evicted_fifo"] += 1
        _log.info("session evicted (FIFO): %s", oldest)
    new_id = session_id or secrets.token_hex(16)
    _SESSION_REGISTRY[new_id] = IncidentResponseEnv()
    _SESSION_TIMESTAMPS[new_id] = time.time()
    _TELEMETRY["sessions_created"] += 1
    return new_id, _SESSION_REGISTRY[new_id]


def get_or_create_sentinel_session(session_id: str | None):
    """Return (session_id, sentinel_env). Creates a new SENTINEL session if id is None or unknown."""
    from sentinel.environment import SentinelEnv

    if session_id and session_id in _SENTINEL_REGISTRY:
        return session_id, _SENTINEL_REGISTRY[session_id]
    # New session - evict if at capacity
    if len(_SENTINEL_REGISTRY) >= _MAX_SESSIONS:
        oldest = next(iter(_SENTINEL_REGISTRY))
        del _SENTINEL_REGISTRY[oldest]
        _SENTINEL_TIMESTAMPS.pop(oldest, None)
        _TELEMETRY["sessions_evicted_fifo"] += 1
        _log.info("sentinel session evicted (FIFO): %s", oldest)
    new_id = session_id or secrets.token_hex(16)
    _SENTINEL_REGISTRY[new_id] = SentinelEnv()
    _SENTINEL_TIMESTAMPS[new_id] = time.time()
    _TELEMETRY["sentinel_sessions_created"] += 1
    return new_id, _SENTINEL_REGISTRY[new_id]


def purge_expired_sessions() -> int:
    """Remove sessions older than SESSION_TTL. Returns number purged."""
    cutoff = time.time() - _SESSION_TTL
    stale = [sid for sid, ts in _SESSION_TIMESTAMPS.items() if ts < cutoff]
    stale_sentinel = [sid for sid, ts in _SENTINEL_TIMESTAMPS.items() if ts < cutoff]

    for sid in stale:
        _SESSION_REGISTRY.pop(sid, None)
        _SESSION_TIMESTAMPS.pop(sid, None)
        _TELEMETRY["sessions_expired_ttl"] += 1

    for sid in stale_sentinel:
        _SENTINEL_REGISTRY.pop(sid, None)
        _SENTINEL_TIMESTAMPS.pop(sid, None)
        _TELEMETRY["sessions_expired_ttl"] += 1

    total_purged = len(stale) + len(stale_sentinel)
    if total_purged:
        _log.info("purged %d stale session(s) (%d IRT, %d SENTINEL)", total_purged, len(stale), len(stale_sentinel))
    return total_purged


def record_leaderboard(task_id: str, score: float, steps: int) -> None:
    """Insert a completed episode score into the in-memory leaderboard."""
    board = _LEADERBOARD.get(task_id)
    if board is None:
        return
    board.append({"score": score, "steps": steps, "ts": round(time.time())})
    board.sort(key=lambda e: (-e["score"], e["steps"]))
    del board[_LEADERBOARD_SIZE:]  # keep top-N


# ---------------------------------------------------------------------------
# Prometheus metric helpers
# ---------------------------------------------------------------------------

# (prom_metric_name, ServiceMetrics field, HELP text)
_PROM_CORE_FIELDS: List[tuple] = [
    ("irt_cpu_percent",    "cpu_percent",    "CPU utilisation percent"),
    ("irt_memory_percent", "memory_percent", "Memory utilisation percent"),
    ("irt_request_rate",   "request_rate",   "Requests per second"),
    ("irt_error_rate",     "error_rate",     "HTTP error rate fraction 0.0-1.0"),
    ("irt_latency_p50_ms", "latency_p50_ms", "P50 response latency milliseconds"),
    ("irt_latency_p99_ms", "latency_p99_ms", "P99 response latency milliseconds"),
]


def scenario_live_to_prom_text(
    live: Dict[str, Any],
    scenario_id: str,
    incident_id: str,
    step: int,
) -> str:
    """Serialize live scenario metrics to Prometheus text exposition format."""
    lines: List[str] = [
        f'# HELP irt_scenario_step Current episode step number',
        f'# TYPE irt_scenario_step gauge',
        f'irt_scenario_step{{scenario="{scenario_id}",incident="{incident_id}"}} {step}',
    ]
    for prom_name, field, help_text in _PROM_CORE_FIELDS:
        lines += [
            f"# HELP {prom_name} {help_text}",
            f"# TYPE {prom_name} gauge",
        ]
        for svc, m in live.items():
            val = getattr(m, field, 0.0)
            lines.append(
                f'{prom_name}{{service="{svc}",scenario="{scenario_id}",incident="{incident_id}"}} {val}'
            )
    # Custom metrics (e.g. connection_pool_used, heap_mb, ...)
    all_custom: Dict[str, str] = {}  # prom_name -> raw key
    for m in live.values():
        for raw_key in (m.custom or {}):
            prom_key = "irt_custom_" + re.sub(r"[^a-zA-Z0-9_]", "_", raw_key)
            all_custom[prom_key] = raw_key
    for prom_key in sorted(all_custom):
        raw_key = all_custom[prom_key]
        lines += [
            f"# HELP {prom_key} Custom scenario metric: {raw_key}",
            f"# TYPE {prom_key} gauge",
        ]
        for svc, m in live.items():
            val = (m.custom or {}).get(raw_key)
            if val is not None:
                lines.append(
                    f'{prom_key}{{service="{svc}",scenario="{scenario_id}",incident="{incident_id}"}} {val}'
                )
    return "\n".join(lines) + "\n"


_PROM_SELECTOR_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)?(?:\{(?P<labels>[^}]*)\})?$"
)
_PROM_LABEL_RE = re.compile(r'(\w+)\s*=\s*"([^"]*)"')


def parse_prom_selector(query: str) -> tuple[str, Dict[str, str]]:
    """Parse a simple PromQL instant selector into (metric_name, label_filters)."""
    m = _PROM_SELECTOR_RE.match(query.strip())
    if not m:
        return query.strip(), {}
    name = m.group("name") or ""
    label_str = m.group("labels") or ""
    filters: Dict[str, str] = {
        lm.group(1): lm.group(2)
        for lm in _PROM_LABEL_RE.finditer(label_str)
    }
    return name, filters


def build_prom_vector(
    live: Dict[str, Any],
    metric_name: str,
    label_filters: Dict[str, str],
    scenario_id: str,
    incident_id: str,
) -> List[Dict[str, Any]]:
    """Build a Prometheus instant-query vector result list."""
    ts = round(time.time(), 3)
    # Normalise: auto-prefix irt_ when caller omits it
    if metric_name and not metric_name.startswith("irt_"):
        metric_name = f"irt_{metric_name}"
    field_map = {pn: fn for pn, fn, _ in _PROM_CORE_FIELDS}
    candidates = [metric_name] if metric_name else [pn for pn, _, _ in _PROM_CORE_FIELDS]
    results: List[Dict[str, Any]] = []
    for prom_name in candidates:
        field = field_map.get(prom_name)
        for svc, m in live.items():
            if "service" in label_filters and label_filters["service"] != svc:
                continue
            if "scenario" in label_filters and label_filters["scenario"] != scenario_id:
                continue
            if field is not None:
                val = getattr(m, field, 0.0)
            elif prom_name.startswith("irt_custom_"):
                raw_key = prom_name[len("irt_custom_"):]
                val = (m.custom or {}).get(raw_key)
                if val is None:
                    continue
            else:
                continue
            results.append({
                "metric": {
                    "__name__": prom_name,
                    "service": svc,
                    "scenario": scenario_id,
                    "incident": incident_id,
                },
                "value": [ts, str(val)],
            })
    return results


def build_prom_matrix(
    history: Dict[str, Any],
    metric_name: str,
    label_filters: Dict[str, str],
    scenario_id: str,
    incident_id: str,
) -> List[Dict[str, Any]]:
    """Build a Prometheus range-query matrix result from ring-buffer history.

    ``history`` is the dict returned by ``env.metric_history(start, end)``:
        {service_name: [(ts, ServiceMetrics), ...], ...}

    Returns the standard Prometheus matrix result shape:
        [{"metric": {...labels}, "values": [[ts, "value"], ...]}, ...]
    """
    if metric_name and not metric_name.startswith("irt_"):
        metric_name = f"irt_{metric_name}"
    field_map = {pn: fn for pn, fn, _ in _PROM_CORE_FIELDS}
    candidates = [metric_name] if metric_name else [pn for pn, _, _ in _PROM_CORE_FIELDS]
    # Build one result stream per (prom_name, service)
    streams: Dict[tuple, List] = {}  # (prom_name, svc) -> [[ts, "val"],...]
    for svc, samples in history.items():
        if "service" in label_filters and label_filters["service"] != svc:
            continue
        if "scenario" in label_filters and label_filters["scenario"] != scenario_id:
            continue
        for prom_name in candidates:
            field = field_map.get(prom_name)
            for ts, m in samples:
                if field is not None:
                    val = getattr(m, field, 0.0)
                elif prom_name.startswith("irt_custom_"):
                    raw_key = prom_name[len("irt_custom_"):]
                    val = (m.custom or {}).get(raw_key)
                    if val is None:
                        continue
                else:
                    continue
                key = (prom_name, svc)
                if key not in streams:
                    streams[key] = []
                streams[key].append([round(ts, 3), str(val)])
    results: List[Dict[str, Any]] = []
    for (prom_name, svc), values in streams.items():
        results.append({
            "metric": {
                "__name__": prom_name,
                "service": svc,
                "scenario": scenario_id,
                "incident": incident_id,
            },
            "values": sorted(values, key=lambda x: x[0]),
        })
    return results
