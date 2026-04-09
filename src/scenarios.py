"""Deterministic incident scenarios for the IRT environment.

Each scenario is a self-contained data definition:
  - Initial alerts visible to the agent
  - Hidden logs and metrics per service (revealed on INVESTIGATE)
  - Ground truth for grading (severity, root cause, valid remediations)

Scenarios are keyed by task_id for 1-to-1 task↔scenario mapping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from src.models import (
    Alert,
    AlertSeverity,
    IncidentSeverity,
    LogEntry,
    ServiceMetrics,
)


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    task_id: str
    incident_id: str
    description: str
    # Initial state
    initial_alerts: List[Alert]
    available_services: List[str]
    # Hidden – revealed on INVESTIGATE
    service_logs: Dict[str, List[LogEntry]]
    service_metrics: Dict[str, ServiceMetrics]
    # Ground truth
    correct_severity: IncidentSeverity
    correct_root_cause_service: str
    correct_root_cause_keywords: List[str]  # any of these in diagnosis → credit
    valid_remediation_actions: List[Dict[str, Any]]
    expected_escalation_teams: List[str]
    # Params
    max_steps: int
    degradation_per_step: float = 0.0  # additional penalty per idle step
    relevant_services: List[str] = field(default_factory=list)
    # Blast radius: maps service → metric key → (rate_per_step, cap) that
    # worsens dynamically as the agent delays.  Applied before metrics are
    # revealed so the agent observes a live, worsening picture.
    # Format: {"service": {"metric_key": (delta_per_step, max_value)}}
    blast_radius: Dict[str, Dict[str, tuple]] = field(default_factory=dict)


def apply_blast_radius(scenario: Scenario, step: int) -> Dict[str, ServiceMetrics]:
    """Return a copy of service_metrics with blast-radius degradation applied.

    Each entry in scenario.blast_radius defines a (delta_per_step, cap) tuple
    per metric key.  The returned dict can be used as revealed metrics so each
    INVESTIGATE at a higher step number sees a more degraded system.
    """
    if not scenario.blast_radius:
        return dict(scenario.service_metrics)

    result: Dict[str, ServiceMetrics] = {}
    for svc, base_metrics in scenario.service_metrics.items():
        blast = scenario.blast_radius.get(svc)
        if blast is None:
            result[svc] = base_metrics
            continue
        # Build an updated custom dict
        d = base_metrics.model_dump()
        custom: Dict[str, float] = dict(d.get("custom") or {})
        # Core fields we also allow to degrade
        degradable_core = {
            "error_rate", "latency_p50_ms", "latency_p99_ms",
            "cpu_percent", "memory_percent", "request_rate",
        }
        for metric_key, (delta, cap) in blast.items():
            if metric_key in degradable_core:
                old_val = d.get(metric_key, 0.0)
                new_val = min(cap, old_val + delta * step) if delta > 0 else max(cap, old_val + delta * step)
                d[metric_key] = round(new_val, 3)
            else:
                # Custom metric field
                old_val = custom.get(metric_key, 0.0)
                new_val = min(cap, old_val + delta * step) if delta > 0 else max(cap, old_val + delta * step)
                custom[metric_key] = round(new_val, 3)
        d["custom"] = custom
        result[svc] = ServiceMetrics(**d)
    return result


# ---- helpers ----------------------------------------------------------------

def _alert(aid: str, svc: str, sev: AlertSeverity, msg: str, ts: str, **meta: Any) -> Alert:
    return Alert(alert_id=aid, service=svc, severity=sev, message=msg, timestamp=ts, metadata=meta)


def _log(ts: str, svc: str, lvl: str, msg: str, tid: str | None = None) -> LogEntry:
    return LogEntry(timestamp=ts, service=svc, level=lvl, message=msg, trace_id=tid)


def _metrics(svc: str, cpu: float, mem: float, rr: float, er: float, p50: float, p99: float, **custom: float) -> ServiceMetrics:
    return ServiceMetrics(service=svc, cpu_percent=cpu, memory_percent=mem, request_rate=rr, error_rate=er, latency_p50_ms=p50, latency_p99_ms=p99, custom=custom)


# ==========================================================================
# SCENARIO 1 – Easy: Database Connection Pool Exhaustion
# ==========================================================================

_SCENARIO_EASY = Scenario(
    scenario_id="db-conn-pool-001",
    task_id="severity_classification",
    incident_id="INC-20260327-001",
    description=(
        "The user-service API is experiencing elevated latency and errors. "
        "Alerts indicate the PostgreSQL primary database connection pool is "
        "nearly saturated. Classify the incident severity."
    ),
    initial_alerts=[
        _alert("ALT-001", "user-service", AlertSeverity.WARNING,
               "p99 latency exceeded 3000 ms threshold (current: 4200 ms)", "2026-03-27T02:15:00Z"),
        _alert("ALT-002", "postgres-primary", AlertSeverity.CRITICAL,
               "Connection pool utilization at 98% (max 200 connections)", "2026-03-27T02:14:30Z"),
        _alert("ALT-003", "user-service", AlertSeverity.WARNING,
               "Error rate at 12% over the last 5 minutes", "2026-03-27T02:15:30Z"),
    ],
    available_services=["user-service", "postgres-primary", "redis-cache", "api-gateway"],
    service_logs={
        "user-service": [
            _log("2026-03-27T02:10:00Z", "user-service", "INFO", "Deployment v2.3.1 completed successfully"),
            _log("2026-03-27T02:12:00Z", "user-service", "WARN", "DB query took 2800 ms for /api/users/profile"),
            _log("2026-03-27T02:13:00Z", "user-service", "ERROR", "Connection acquisition timeout after 5000 ms", "trace-a1b2"),
            _log("2026-03-27T02:13:30Z", "user-service", "ERROR", "Connection acquisition timeout after 5000 ms", "trace-c3d4"),
            _log("2026-03-27T02:14:00Z", "user-service", "ERROR", "Failed to acquire connection from pool: pool exhausted", "trace-e5f6"),
            _log("2026-03-27T02:14:30Z", "user-service", "WARN", "Retry #3 for DB connection – backing off 500 ms"),
            _log("2026-03-27T02:15:00Z", "user-service", "ERROR", "HTTP 503 returned for GET /api/users/profile", "trace-g7h8"),
        ],
        "postgres-primary": [
            _log("2026-03-27T02:08:00Z", "postgres-primary", "INFO", "Active connections: 120/200"),
            _log("2026-03-27T02:10:30Z", "postgres-primary", "WARN", "Active connections: 175/200"),
            _log("2026-03-27T02:12:00Z", "postgres-primary", "WARN", "Active connections: 190/200 – approaching limit"),
            _log("2026-03-27T02:13:00Z", "postgres-primary", "ERROR", "Active connections: 196/200 – new connection rejected"),
            _log("2026-03-27T02:14:00Z", "postgres-primary", "ERROR", "Connection count 198/200. Longest idle: 1800 s. Possible connection leak detected."),
            _log("2026-03-27T02:15:00Z", "postgres-primary", "ERROR", "Active connections: 200/200 – pool fully saturated"),
        ],
        "redis-cache": [
            _log("2026-03-27T02:15:00Z", "redis-cache", "INFO", "Memory usage: 45%. Operations normal."),
            _log("2026-03-27T02:15:00Z", "redis-cache", "INFO", "Hit rate: 94%. No evictions."),
        ],
        "api-gateway": [
            _log("2026-03-27T02:14:00Z", "api-gateway", "WARN", "Upstream user-service returning 503 for 8% of requests"),
            _log("2026-03-27T02:15:00Z", "api-gateway", "INFO", "All other upstream services healthy"),
        ],
    },
    service_metrics={
        "user-service": _metrics("user-service", 65.0, 58.0, 450.0, 0.12, 320.0, 4200.0),
        "postgres-primary": _metrics("postgres-primary", 78.0, 72.0, 450.0, 0.05, 45.0, 890.0, connection_pool_pct=98.0),
        "redis-cache": _metrics("redis-cache", 15.0, 45.0, 1200.0, 0.001, 1.2, 3.5),
        "api-gateway": _metrics("api-gateway", 22.0, 30.0, 2200.0, 0.08, 85.0, 4500.0),
    },
    correct_severity=IncidentSeverity.P2,
    correct_root_cause_service="postgres-primary",
    correct_root_cause_keywords=["connection pool", "connection leak", "pool exhaustion", "pool saturated", "connection exhaustion"],
    valid_remediation_actions=[
        {"action": "restart", "service": "user-service"},
        {"action": "config_change", "service": "postgres-primary", "detail": "increase pool size"},
    ],
    expected_escalation_teams=["database-team"],
    max_steps=10,
    degradation_per_step=0.005,
    relevant_services=["user-service", "postgres-primary"],
    # Blast radius: connection pool fully saturates, user-service error rate climbs
    blast_radius={
        "postgres-primary": {
            "connection_pool_pct": (0.5, 100.0),   # +0.5%/step → caps at 100%
        },
        "user-service": {
            "error_rate": (0.02, 0.60),             # +2pp/step → caps at 60%
            "latency_p99_ms": (200.0, 10000.0),     # +200ms/step → caps at 10s
        },
    },
)


# ==========================================================================
# SCENARIO 2 – Medium: Payment Processing Failure
# ==========================================================================

_SCENARIO_MEDIUM = Scenario(
    scenario_id="payment-failure-001",
    task_id="root_cause_analysis",
    incident_id="INC-20260327-002",
    description=(
        "Payment success rate has dropped sharply. Multiple services show "
        "degradation. Investigate the services, identify the root cause, "
        "classify severity, and apply the correct remediation."
    ),
    initial_alerts=[
        _alert("ALT-010", "payment-gateway", AlertSeverity.CRITICAL,
               "Payment success rate dropped to 45% (threshold: 95%)", "2026-03-27T09:30:00Z"),
        _alert("ALT-011", "payment-processor", AlertSeverity.WARNING,
               "Timeout errors increased 10x in last 10 minutes", "2026-03-27T09:30:30Z"),
        _alert("ALT-012", "redis-session", AlertSeverity.WARNING,
               "Key eviction rate spike: 1500 evictions/min (normal: <10)", "2026-03-27T09:29:00Z"),
        _alert("ALT-013", "order-service", AlertSeverity.WARNING,
               "Error rate elevated to 8%", "2026-03-27T09:31:00Z"),
    ],
    available_services=["payment-gateway", "payment-processor", "redis-session", "order-service", "user-service", "postgres-primary"],
    service_logs={
        "payment-gateway": [
            _log("2026-03-27T09:25:00Z", "payment-gateway", "INFO", "Processing 320 payments/min"),
            _log("2026-03-27T09:28:00Z", "payment-gateway", "WARN", "Payment token validation failed: token not found in session store", "trace-pay-01"),
            _log("2026-03-27T09:28:30Z", "payment-gateway", "ERROR", "Payment failed: session token expired or missing for txn TXN-8842", "trace-pay-02"),
            _log("2026-03-27T09:29:00Z", "payment-gateway", "ERROR", "Batch failure: 55% of payment attempts failing with SESSION_TOKEN_MISSING"),
            _log("2026-03-27T09:30:00Z", "payment-gateway", "ERROR", "Success rate critical: 45%. All failures correlate with session token lookup errors."),
        ],
        "payment-processor": [
            _log("2026-03-27T09:28:00Z", "payment-processor", "WARN", "Upstream payment-gateway sending incomplete requests"),
            _log("2026-03-27T09:29:00Z", "payment-processor", "ERROR", "Timeout waiting for payment-gateway response: 12 s", "trace-pp-01"),
            _log("2026-03-27T09:30:00Z", "payment-processor", "WARN", "Retry queue depth: 450 (normal: <20)"),
        ],
        "redis-session": [
            _log("2026-03-27T09:20:00Z", "redis-session", "INFO", "Memory usage: 95%. Approaching maxmemory limit (4 GB)."),
            _log("2026-03-27T09:22:00Z", "redis-session", "WARN", "maxmemory reached. Eviction policy: allkeys-lru. Beginning evictions."),
            _log("2026-03-27T09:25:00Z", "redis-session", "WARN", "Evicted 800 keys in last 3 minutes. Active sessions being evicted."),
            _log("2026-03-27T09:28:00Z", "redis-session", "ERROR", "Eviction rate critical: 1500 keys/min. Payment session tokens are being evicted before use."),
            _log("2026-03-27T09:30:00Z", "redis-session", "ERROR", "Memory at 100%. Continuous eviction. Session TTL effectively reduced from 30 min to ~45 s."),
        ],
        "order-service": [
            _log("2026-03-27T09:30:00Z", "order-service", "WARN", "Downstream payment-gateway returning errors for order confirmations"),
            _log("2026-03-27T09:31:00Z", "order-service", "ERROR", "8% of orders failing at payment step – propagated from payment-gateway"),
        ],
        "user-service": [
            _log("2026-03-27T09:30:00Z", "user-service", "INFO", "All endpoints healthy. Latency normal."),
        ],
        "postgres-primary": [
            _log("2026-03-27T09:30:00Z", "postgres-primary", "INFO", "Active connections: 85/200. Operations normal."),
        ],
    },
    service_metrics={
        "payment-gateway": _metrics("payment-gateway", 45.0, 52.0, 320.0, 0.55, 250.0, 12000.0, payment_success_rate=0.45),
        "payment-processor": _metrics("payment-processor", 35.0, 40.0, 150.0, 0.30, 180.0, 8000.0),
        "redis-session": _metrics("redis-session", 30.0, 99.5, 5000.0, 0.02, 0.8, 2.5, memory_used_gb=3.98, evictions_per_min=1500.0),
        "order-service": _metrics("order-service", 28.0, 35.0, 200.0, 0.08, 120.0, 950.0),
        "user-service": _metrics("user-service", 20.0, 32.0, 400.0, 0.002, 45.0, 120.0),
        "postgres-primary": _metrics("postgres-primary", 40.0, 55.0, 300.0, 0.001, 12.0, 45.0),
    },
    correct_severity=IncidentSeverity.P1,
    correct_root_cause_service="redis-session",
    correct_root_cause_keywords=[
        "redis", "memory", "eviction", "session token", "maxmemory",
        "session eviction", "cache eviction", "redis memory",
    ],
    valid_remediation_actions=[
        {"action": "scale", "service": "redis-session"},
        {"action": "config_change", "service": "redis-session", "detail": "increase maxmemory"},
        {"action": "restart", "service": "redis-session"},
    ],
    expected_escalation_teams=["payments-team", "platform-team"],
    max_steps=15,
    degradation_per_step=0.01,
    relevant_services=["payment-gateway", "redis-session", "payment-processor"],
    # Blast radius: Redis keeps evicting, payment success rate collapses
    blast_radius={
        "redis-session": {
            "evictions_per_min": (150.0, 5000.0),  # +150 evictions/min/step
            "memory_used_gb": (0.005, 4.0),         # creeps toward hard limit
        },
        "payment-gateway": {
            "payment_success_rate": (-0.04, 0.05),  # drops 4pp/step → 5% floor
            "error_rate": (0.03, 0.90),
        },
        "order-service": {
            "error_rate": (0.02, 0.50),
        },
    },
)


# ==========================================================================
# SCENARIO 3 – Hard: Cascading Multi-Service Outage
# ==========================================================================

_SCENARIO_HARD = Scenario(
    scenario_id="cascading-outage-001",
    task_id="full_incident_management",
    incident_id="INC-20260327-003",
    description=(
        "Multiple services are degraded simultaneously. The API gateway is "
        "returning 503s, the auth service has extreme latency, and downstream "
        "services are failing. This is a cascading outage. You must triage, "
        "investigate, identify the root cause, remediate, escalate, and "
        "communicate status updates."
    ),
    initial_alerts=[
        _alert("ALT-100", "api-gateway", AlertSeverity.CRITICAL,
               "503 error rate at 35% across all endpoints", "2026-03-27T14:00:00Z"),
        _alert("ALT-101", "auth-service", AlertSeverity.CRITICAL,
               "p99 latency > 5000 ms (threshold: 200 ms)", "2026-03-27T14:00:30Z"),
        _alert("ALT-102", "order-service", AlertSeverity.WARNING,
               "Message queue depth growing: 15000 (normal: <500)", "2026-03-27T14:01:00Z"),
        _alert("ALT-103", "notification-service", AlertSeverity.WARNING,
               "Connection timeout to auth-service: 100% failure rate", "2026-03-27T14:01:30Z"),
        _alert("ALT-104", "cdn-static", AlertSeverity.INFO,
               "Cache miss rate elevated to 15% (normal: 2%)", "2026-03-27T14:02:00Z"),
        _alert("ALT-105", "user-service", AlertSeverity.WARNING,
               "Intermittent HTTP 401 responses (token validation failing)", "2026-03-27T14:01:00Z"),
        _alert("ALT-106", "deployment-tracker", AlertSeverity.CRITICAL,
               "auth-service v3.1.0 deployed at 13:47 — memory climb started immediately. Escalate to auth-team and platform-team.", "2026-03-27T14:02:00Z"),
    ],
    available_services=[
        "api-gateway", "auth-service", "user-service",
        "order-service", "notification-service", "cdn-static",
        "postgres-primary", "redis-auth-cache",
    ],
    service_logs={
        "api-gateway": [
            _log("2026-03-27T13:58:00Z", "api-gateway", "INFO", "All upstreams healthy. Traffic: 5500 req/s."),
            _log("2026-03-27T14:00:00Z", "api-gateway", "ERROR", "Upstream auth-service: 503 for 35% of auth checks"),
            _log("2026-03-27T14:00:30Z", "api-gateway", "ERROR", "Circuit breaker OPEN for auth-service after 50 consecutive failures"),
            _log("2026-03-27T14:01:00Z", "api-gateway", "ERROR", "Cascading: requests requiring auth are failing. Public endpoints OK."),
        ],
        "auth-service": [
            _log("2026-03-27T13:45:00Z", "auth-service", "INFO", "Deployment v3.1.0 started (canary 10%)"),
            _log("2026-03-27T13:47:00Z", "auth-service", "INFO", "Deployment v3.1.0 promoted to 100%"),
            _log("2026-03-27T13:50:00Z", "auth-service", "WARN", "Memory usage climbing: 72% (was 45% before deploy)"),
            _log("2026-03-27T13:55:00Z", "auth-service", "WARN", "Memory usage: 88%. GC pauses increasing: avg 350 ms"),
            _log("2026-03-27T13:58:00Z", "auth-service", "ERROR", "Memory usage: 95%. GC pause: 2100 ms. Requests timing out."),
            _log("2026-03-27T14:00:00Z", "auth-service", "ERROR", "OOMKill risk. Memory: 97%. Token validation taking 4800 ms avg."),
            _log("2026-03-27T14:00:30Z", "auth-service", "ERROR", "v3.1.0 changelog: 'Refactored token cache to in-memory store' – possible unbounded cache growth"),
            _log("2026-03-27T14:01:00Z", "auth-service", "ERROR", "Pod restarts: 3 in last 5 min due to OOMKill. Service effectively down."),
        ],
        "user-service": [
            _log("2026-03-27T14:00:00Z", "user-service", "WARN", "Auth token validation calls timing out"),
            _log("2026-03-27T14:01:00Z", "user-service", "ERROR", "Returning 401 for 40% of requests – cannot validate tokens with auth-service"),
        ],
        "order-service": [
            _log("2026-03-27T14:00:00Z", "order-service", "WARN", "Order processing slowing – auth dependency failing"),
            _log("2026-03-27T14:01:00Z", "order-service", "ERROR", "Queue depth: 15000. Orders stuck awaiting auth validation."),
            _log("2026-03-27T14:02:00Z", "order-service", "ERROR", "Queue depth: 25000. Risk of message broker disk overflow."),
        ],
        "notification-service": [
            _log("2026-03-27T14:01:00Z", "notification-service", "ERROR", "Cannot reach auth-service. All notification deliveries paused."),
            _log("2026-03-27T14:02:00Z", "notification-service", "WARN", "Buffered 8000 pending notifications."),
        ],
        "cdn-static": [
            _log("2026-03-27T14:00:00Z", "cdn-static", "INFO", "Cache miss rate elevated. Likely due to increased full page reloads from client-side auth failures."),
            _log("2026-03-27T14:02:00Z", "cdn-static", "INFO", "No CDN-side issues detected. Origin healthy."),
        ],
        "postgres-primary": [
            _log("2026-03-27T14:00:00Z", "postgres-primary", "INFO", "Connections: 90/200. Query performance normal."),
        ],
        "redis-auth-cache": [
            _log("2026-03-27T14:00:00Z", "redis-auth-cache", "INFO", "Memory: 30%. Operations normal."),
            _log("2026-03-27T14:00:30Z", "redis-auth-cache", "WARN", "Cache hit rate dropped from 92% to 15%. auth-service v3.1.0 appears to bypass cache."),
        ],
    },
    service_metrics={
        "api-gateway": _metrics("api-gateway", 55.0, 40.0, 5500.0, 0.35, 150.0, 8500.0),
        "auth-service": _metrics("auth-service", 95.0, 97.0, 800.0, 0.65, 2500.0, 5200.0, gc_pause_ms=2100.0, pod_restarts=3.0),
        "user-service": _metrics("user-service", 30.0, 35.0, 400.0, 0.40, 80.0, 4800.0),
        "order-service": _metrics("order-service", 40.0, 45.0, 200.0, 0.25, 300.0, 3500.0, queue_depth=15000.0),
        "notification-service": _metrics("notification-service", 10.0, 20.0, 0.0, 1.0, 0.0, 0.0),
        "cdn-static": _metrics("cdn-static", 12.0, 18.0, 8000.0, 0.001, 8.0, 25.0, cache_miss_rate=0.15),
        "postgres-primary": _metrics("postgres-primary", 38.0, 52.0, 250.0, 0.001, 10.0, 40.0),
        "redis-auth-cache": _metrics("redis-auth-cache", 12.0, 30.0, 2000.0, 0.005, 0.5, 1.8, cache_hit_rate=0.15),
    },
    correct_severity=IncidentSeverity.P1,
    correct_root_cause_service="auth-service",
    correct_root_cause_keywords=[
        "memory leak", "v3.1.0", "deployment", "oom", "unbounded cache",
        "in-memory", "bad deployment", "auth-service deployment",
        "token cache", "gc pause", "out of memory",
    ],
    valid_remediation_actions=[
        {"action": "rollback", "service": "auth-service"},
        {"action": "restart", "service": "auth-service"},
        {"action": "scale", "service": "order-service"},
        {"action": "restart", "service": "order-service"},
    ],
    expected_escalation_teams=["platform-team", "auth-team"],
    max_steps=20,
    degradation_per_step=0.015,
    relevant_services=["auth-service", "api-gateway", "redis-auth-cache", "order-service"],
    # Blast radius: auth-service OOMKills more often, order queue grows unbounded
    blast_radius={
        "auth-service": {
            "memory_percent": (0.5, 100.0),        # +0.5%/step → OOM at 100%
            "error_rate": (0.02, 0.95),             # cascades toward full outage
            "latency_p99_ms": (100.0, 15000.0),
            "pod_restarts": (0.3, 15.0),            # accumulating restarts
        },
        "order-service": {
            "queue_depth": (1500.0, 100000.0),      # queue grows 1500/step
            "error_rate": (0.02, 0.80),
        },
        "api-gateway": {
            "error_rate": (0.015, 0.70),            # more requests fail over time
        },
        "user-service": {
            "error_rate": (0.02, 0.80),
        },
    },
)


# ==========================================================================
# SCENARIO 1-B – Easy variant: Disk space exhaustion on log volume
# ==========================================================================

_SCENARIO_EASY_B = Scenario(
    scenario_id="disk-full-001",
    task_id="severity_classification",
    incident_id="INC-20260327-101",
    description=(
        "The search-service and its underlying Elasticsearch cluster are "
        "experiencing errors. Alerts indicate disk usage is critically high. "
        "Classify the incident severity."
    ),
    initial_alerts=[
        _alert("ALT-201", "elasticsearch", AlertSeverity.CRITICAL,
               "Disk usage at 95% on data node es-node-01", "2026-03-27T06:10:00Z"),
        _alert("ALT-202", "search-service", AlertSeverity.WARNING,
               "Bulk indexing failures: 400% increase", "2026-03-27T06:10:30Z"),
        _alert("ALT-203", "elasticsearch", AlertSeverity.WARNING,
               "write.low_watermark crossed – shard allocation blocked", "2026-03-27T06:09:00Z"),
    ],
    available_services=["search-service", "elasticsearch", "kibana", "log-aggregator"],
    service_logs={
        "search-service": [
            _log("2026-03-27T06:08:00Z", "search-service", "WARN", "Indexing queue backing up: 12000 documents pending"),
            _log("2026-03-27T06:09:00Z", "search-service", "ERROR", "BulkIndexException: ClusterBlockException[blocked: FORBIDDEN/12/index]"),
            _log("2026-03-27T06:10:00Z", "search-service", "ERROR", "Search degraded – last index refresh 8 min ago. Serving stale results."),
        ],
        "elasticsearch": [
            _log("2026-03-27T06:05:00Z", "elasticsearch", "WARN", "Disk usage: 90% on es-node-01. Threshold: 85%."),
            _log("2026-03-27T06:07:00Z", "elasticsearch", "WARN", "Disk: 93%. flood_stage watermark approaching."),
            _log("2026-03-27T06:09:00Z", "elasticsearch", "ERROR", "Disk: 95%. flood_stage reached. All indices set to read-only."),
            _log("2026-03-27T06:10:00Z", "elasticsearch", "ERROR", "Shard allocation disabled. Cluster status: YELLOW. Write ops blocked."),
        ],
        "kibana": [
            _log("2026-03-27T06:10:00Z", "kibana", "INFO", "Dashboard loading normally. Read-only ops unaffected."),
        ],
        "log-aggregator": [
            _log("2026-03-27T06:09:00Z", "log-aggregator", "WARN", "Log shipping to elasticsearch failing. Retrying. Buffer: 50000 lines."),
        ],
    },
    service_metrics={
        "search-service": _metrics("search-service", 42.0, 50.0, 200.0, 0.35, 180.0, 2200.0),
        "elasticsearch": _metrics("elasticsearch", 60.0, 80.0, 50.0, 0.40, 200.0, 5000.0, disk_pct=95.0),
        "kibana": _metrics("kibana", 15.0, 25.0, 30.0, 0.0, 90.0, 350.0),
        "log-aggregator": _metrics("log-aggregator", 25.0, 35.0, 300.0, 0.15, 50.0, 400.0),
    },
    correct_severity=IncidentSeverity.P2,
    correct_root_cause_service="elasticsearch",
    correct_root_cause_keywords=["disk", "disk full", "disk space", "flood_stage", "watermark", "read-only", "disk usage"],
    valid_remediation_actions=[
        {"action": "config_change", "service": "elasticsearch", "detail": "clear read-only flag"},
        {"action": "scale", "service": "elasticsearch"},
    ],
    expected_escalation_teams=["infrastructure-team"],
    max_steps=10,
    degradation_per_step=0.005,
    relevant_services=["search-service", "elasticsearch"],
)


# ==========================================================================
# SCENARIO 2-B – Medium variant: Slow memory leak in background worker
# ==========================================================================

_SCENARIO_MEDIUM_B = Scenario(
    scenario_id="worker-memleak-001",
    task_id="root_cause_analysis",
    incident_id="INC-20260327-102",
    description=(
        "The report-generation service is timing out and users cannot export "
        "data. Multiple related services show elevated errors. Find the true "
        "root cause, classify severity, diagnose, and remediate."
    ),
    initial_alerts=[
        _alert("ALT-210", "report-service", AlertSeverity.CRITICAL,
               "Request timeout rate 60% for /api/export", "2026-03-27T11:20:00Z"),
        _alert("ALT-211", "worker-pool", AlertSeverity.WARNING,
               "Worker memory usage: 94% (4 of 5 workers OOMKilling)", "2026-03-27T11:19:00Z"),
        _alert("ALT-212", "s3-upload", AlertSeverity.WARNING,
               "Upload failures – 503s from report-service", "2026-03-27T11:20:30Z"),
        _alert("ALT-213", "postgres-reports", AlertSeverity.INFO,
               "Long-running queries detected: 5 queries > 10 s", "2026-03-27T11:18:00Z"),
        _alert("ALT-214", "health-monitor", AlertSeverity.INFO,
               "Core services healthy: payment, auth, user-api all nominal. Issue isolated to report-export subsystem.", "2026-03-27T11:20:00Z"),
    ],
    available_services=["report-service", "worker-pool", "s3-upload", "postgres-reports", "redis-cache", "api-gateway"],
    service_logs={
        "report-service": [
            _log("2026-03-27T11:15:00Z", "report-service", "INFO", "Report job queued: RPT-9981, format: xlsx, rows: 1M"),
            _log("2026-03-27T11:16:00Z", "report-service", "WARN", "Worker RPT-9981 memory: 2.1 GB (limit 2 GB). Nearing OOM."),
            _log("2026-03-27T11:18:00Z", "report-service", "ERROR", "Worker OOMKilled during xlsx serialization. Job failed."),
            _log("2026-03-27T11:19:00Z", "report-service", "ERROR", "3 concurrent OOMKills. Export endpoint returning 503."),
        ],
        "worker-pool": [
            _log("2026-03-27T11:10:00Z", "worker-pool", "INFO", "Workers: 5 active, 0 idle. Load: nominal."),
            _log("2026-03-27T11:14:00Z", "worker-pool", "WARN", "Worker memory climbing. Suspected unbounded row accumulation in xlsx writer."),
            _log("2026-03-27T11:17:00Z", "worker-pool", "ERROR", "Worker #3 OOMKilled. Memory at 100%."),
            _log("2026-03-27T11:19:00Z", "worker-pool", "ERROR", "4/5 workers OOMKilled. Effective worker capacity: 1. Queue depth: 45."),
            _log("2026-03-27T11:19:30Z", "worker-pool", "ERROR", "Root cause: xlsx writer buffers all rows in memory before flushing. No streaming."),
        ],
        "s3-upload": [
            _log("2026-03-27T11:20:00Z", "s3-upload", "WARN", "Upstream report-service returning 503. S3 uploads queued."),
        ],
        "postgres-reports": [
            _log("2026-03-27T11:17:00Z", "postgres-reports", "INFO", "Large sequential scan for 1M row export. Query time: 12 s. This is normal for large exports."),
        ],
        "redis-cache": [_log("2026-03-27T11:20:00Z", "redis-cache", "INFO", "Operations normal.")],
        "api-gateway": [_log("2026-03-27T11:20:00Z", "api-gateway", "WARN", "report-service upstream: 60% 503 errors.")],
    },
    service_metrics={
        "report-service": _metrics("report-service", 55.0, 75.0, 10.0, 0.60, 8000.0, 30000.0),
        "worker-pool": _metrics("worker-pool", 90.0, 94.0, 5.0, 0.80, 15000.0, 60000.0, oom_kills=4.0),
        "s3-upload": _metrics("s3-upload", 10.0, 15.0, 2.0, 0.60, 500.0, 3000.0),
        "postgres-reports": _metrics("postgres-reports", 55.0, 60.0, 15.0, 0.0, 200.0, 12000.0),
        "redis-cache": _metrics("redis-cache", 12.0, 30.0, 500.0, 0.0, 1.0, 3.0),
        "api-gateway": _metrics("api-gateway", 20.0, 28.0, 800.0, 0.08, 80.0, 2000.0),
    },
    correct_severity=IncidentSeverity.P2,
    correct_root_cause_service="worker-pool",
    correct_root_cause_keywords=["memory", "oom", "out of memory", "xlsx", "buffering", "unbounded", "memory leak", "worker memory", "worker", "oomkill", "streaming", "row accumulation"],
    # Note: P2 not P1 — only the report-export subsystem is affected, core services healthy.
    valid_remediation_actions=[
        {"action": "restart", "service": "worker-pool"},
        {"action": "scale", "service": "worker-pool"},
        {"action": "config_change", "service": "worker-pool", "detail": "enable streaming"},
    ],
    expected_escalation_teams=["backend-team", "platform-team"],
    max_steps=15,
    degradation_per_step=0.008,
    relevant_services=["report-service", "worker-pool"],
)


# ==========================================================================
# SCENARIO 3-B – Hard variant: Kubernetes node pressure / pod eviction cascade
# ==========================================================================

_SCENARIO_HARD_B = Scenario(
    scenario_id="k8s-node-pressure-001",
    task_id="full_incident_management",
    incident_id="INC-20260327-004",
    description=(
        "Multiple pods are being evicted across the cluster. The checkout "
        "service is returning 502s, node-exporter reports memory pressure on "
        "three nodes, and the HPA has been scaling aggressively. This is a "
        "node-level resource exhaustion event triggered by an HPA/resource-limit "
        "misconfiguration. Full incident management required."
    ),
    initial_alerts=[
        _alert("ALT-200", "checkout-service", AlertSeverity.CRITICAL,
               "502 error rate 28% across checkout endpoints", "2026-03-27T16:00:00Z"),
        _alert("ALT-201", "k8s-node-01", AlertSeverity.CRITICAL,
               "MemoryPressure=True — 3/8 pods evicted in last 5 min", "2026-03-27T16:00:30Z"),
        _alert("ALT-202", "k8s-node-02", AlertSeverity.WARNING,
               "MemoryPressure=True — node at 92% memory", "2026-03-27T16:01:00Z"),
        _alert("ALT-203", "hpa-controller", AlertSeverity.WARNING,
               "HPA for recommendation-service scaled to maxReplicas=20 (was 4)", "2026-03-27T15:55:00Z"),
        _alert("ALT-204", "cart-service", AlertSeverity.WARNING,
               "Downstream checkout-service returning 502s for 35% of cart completions", "2026-03-27T16:01:30Z"),
        _alert("ALT-205", "cdn-static", AlertSeverity.INFO,
               "Slight latency increase: p99 68ms (normal: 20ms)", "2026-03-27T16:02:00Z"),
    ],
    available_services=[
        "checkout-service", "k8s-node-01", "k8s-node-02",
        "recommendation-service", "cart-service", "hpa-controller",
        "cdn-static", "postgres-checkout",
    ],
    service_logs={
        "checkout-service": [
            _log("2026-03-27T15:58:00Z", "checkout-service", "INFO", "Processing normally. 180 req/s."),
            _log("2026-03-27T15:59:30Z", "checkout-service", "WARN", "3 pods restarting. Connections dropped."),
            _log("2026-03-27T16:00:00Z", "checkout-service", "ERROR", "502 Bad Gateway — upstream recommendation-service pods unavailable"),
            _log("2026-03-27T16:01:00Z", "checkout-service", "ERROR", "Circuit breaker half-open. 28% of requests failing."),
        ],
        "k8s-node-01": [
            _log("2026-03-27T15:50:00Z", "k8s-node-01", "INFO", "Memory: 78%."),
            _log("2026-03-27T15:53:00Z", "k8s-node-01", "WARN", "Memory: 88%. kubelet setting eviction threshold."),
            _log("2026-03-27T15:56:00Z", "k8s-node-01", "ERROR", "Memory: 95%. OOM eviction beginning. Evicting low-priority pods."),
            _log("2026-03-27T15:58:00Z", "k8s-node-01", "ERROR", "Evicted: recommendation-service-7d8f (2 GB). Memory: 91%."),
            _log("2026-03-27T16:00:00Z", "k8s-node-01", "ERROR", "Memory back to 95%. HPA-spawned recommendation-service pods consuming all available memory."),
        ],
        "k8s-node-02": [
            _log("2026-03-27T15:58:00Z", "k8s-node-02", "WARN", "Memory: 90%. recommendation-service HPA placed 6 new pods here."),
            _log("2026-03-27T16:00:30Z", "k8s-node-02", "ERROR", "Memory: 92%. Approaching eviction threshold."),
        ],
        "recommendation-service": [
            _log("2026-03-27T15:45:00Z", "recommendation-service", "INFO", "Memory usage tracking: v2.4.0 deployed. ML model loaded."),
            _log("2026-03-27T15:50:00Z", "recommendation-service", "WARN", "Each pod consuming 2.1 GB (limit: 2.0 GB) — requests.memory too low."),
            _log("2026-03-27T15:53:00Z", "recommendation-service", "WARN", "HPA triggered: latency spike caused scale-out. 8→12 pods"),
            _log("2026-03-27T15:57:00Z", "recommendation-service", "ERROR", "HPA at maxReplicas=20. 20 pods × 2.1 GB = 42 GB on nodes with 32 GB capacity."),
            _log("2026-03-27T16:00:00Z", "recommendation-service", "ERROR", "Pod eviction loop: evicted pods restart, consume memory, trigger eviction again."),
        ],
        "hpa-controller": [
            _log("2026-03-27T15:52:00Z", "hpa-controller", "INFO", "recommendation-service: scaling 4→8 due to latency"),
            _log("2026-03-27T15:55:00Z", "hpa-controller", "WARN", "recommendation-service: scaling 8→20 (maxReplicas). Memory requests underspecified."),
            _log("2026-03-27T16:00:00Z", "hpa-controller", "ERROR", "Eviction loop detected. Scaling is worsening node pressure."),
        ],
        "cart-service": [
            _log("2026-03-27T16:01:00Z", "cart-service", "WARN", "Checkout dependency failing. 35% cart completions blocked."),
        ],
        "cdn-static": [
            _log("2026-03-27T16:02:00Z", "cdn-static", "INFO", "Slight latency increase correlates with client retries. No CDN-side issue."),
        ],
        "postgres-checkout": [
            _log("2026-03-27T16:00:00Z", "postgres-checkout", "INFO", "All queries normal. Connections: 45/200."),
        ],
    },
    service_metrics={
        "checkout-service": _metrics("checkout-service", 55.0, 60.0, 180.0, 0.28, 200.0, 5500.0),
        "k8s-node-01":       _metrics("k8s-node-01",       70.0, 95.0,   0.0, 0.0,   0.0,    0.0, evicted_pods=3.0),
        "k8s-node-02":       _metrics("k8s-node-02",       65.0, 92.0,   0.0, 0.0,   0.0,    0.0),
        "recommendation-service": _metrics("recommendation-service", 85.0, 105.0, 80.0, 0.60, 800.0, 12000.0, memory_per_pod_gb=2.1, pod_count=20.0),
        "cart-service":      _metrics("cart-service",      30.0, 35.0, 250.0, 0.15,  90.0, 2200.0),
        "hpa-controller":    _metrics("hpa-controller",    10.0, 15.0,   0.0, 0.0,   0.0,    0.0, current_replicas=20.0),
        "cdn-static":        _metrics("cdn-static",        10.0, 12.0, 9000.0, 0.001, 12.0,  68.0),
        "postgres-checkout": _metrics("postgres-checkout", 35.0, 48.0, 200.0, 0.001, 12.0,   38.0),
    },
    correct_severity=IncidentSeverity.P1,
    correct_root_cause_service="recommendation-service",
    correct_root_cause_keywords=[
        "memory request", "resource limit", "hpa", "eviction loop", "pod eviction",
        "memory limit", "recommendation-service memory", "node pressure",
        "oom eviction", "hpa scale", "memory requests underspecified",
    ],
    valid_remediation_actions=[
        {"action": "config_change", "service": "recommendation-service"},
        {"action": "scale",         "service": "recommendation-service"},
        {"action": "restart",       "service": "recommendation-service"},
        {"action": "config_change", "service": "hpa-controller"},
    ],
    expected_escalation_teams=["platform-team", "sre-team"],
    max_steps=20,
    degradation_per_step=0.015,
    relevant_services=["recommendation-service", "k8s-node-01", "hpa-controller", "checkout-service"],
    blast_radius={
        "recommendation-service": {
            "error_rate": (0.03, 0.95),
            "pod_count":  (0.5,  20.0),
        },
        "k8s-node-01": {
            "memory_percent": (0.4, 100.0),
            "evicted_pods":   (0.4,  20.0),
        },
        "k8s-node-02": {
            "memory_percent": (0.5, 100.0),
        },
        "checkout-service": {
            "error_rate": (0.025, 0.85),
        },
    },
)


# ==========================================================================
# SCENARIO 3-C – Hard variant: Database failover split-brain
# ==========================================================================

_SCENARIO_HARD_C = Scenario(
    scenario_id="db-failover-race-001",
    task_id="full_incident_management",
    incident_id="INC-20260327-005",
    description=(
        "The primary PostgreSQL instance failed over to the replica 18 minutes "
        "ago but several services still route writes to the old primary (now "
        "read-only) because pgbouncer's connection string was never updated. "
        "A split-brain scenario is actively corrupting order state. Full "
        "incident commander workflow required: triage, diagnose, remediate, "
        "escalate, communicate."
    ),
    initial_alerts=[
        _alert("ALT-300", "order-service", AlertSeverity.CRITICAL,
               "Write failures: 65% of order commits failing with ReadOnlyError", "2026-03-27T18:10:00Z"),
        _alert("ALT-301", "postgres-primary-old", AlertSeverity.CRITICAL,
               "Instance is READ-ONLY (promoted replica took writes 18 min ago)", "2026-03-27T18:10:30Z"),
        _alert("ALT-302", "postgres-replica-new", AlertSeverity.WARNING,
               "Becoming primary: only 30% of expected write traffic received", "2026-03-27T18:11:00Z"),
        _alert("ALT-303", "payment-service", AlertSeverity.WARNING,
               "Double-charge risk: orders appearing in both DB instances for 8% of txns", "2026-03-27T18:11:30Z"),
        _alert("ALT-304", "inventory-service", AlertSeverity.WARNING,
               "Stock deduction failing silently: items over-sold", "2026-03-27T18:12:00Z"),
        _alert("ALT-305", "monitoring-dashboard", AlertSeverity.INFO,
               "DB failover event recorded at 2026-03-27T17:52:00Z", "2026-03-27T18:12:30Z"),
        _alert("ALT-306", "pgbouncer", AlertSeverity.CRITICAL,
               "pgbouncer still routing ALL writes to postgres-primary-old (read-only). Connection string not updated after failover.", "2026-03-27T18:13:00Z"),
    ],
    available_services=[
        "order-service", "postgres-primary-old", "postgres-replica-new",
        "payment-service", "inventory-service", "config-service",
        "monitoring-dashboard", "pgbouncer",
    ],
    service_logs={
        "order-service": [
            _log("2026-03-27T17:52:00Z", "order-service", "WARN", "DB failover detected. Using cached connection string."),
            _log("2026-03-27T17:55:00Z", "order-service", "ERROR", "INSERT failed: ERROR: cannot execute INSERT in a read-only transaction"),
            _log("2026-03-27T18:00:00Z", "order-service", "ERROR", "65% of order writes failing. Service still pointing to old primary."),
            _log("2026-03-27T18:10:00Z", "order-service", "ERROR", "Connection pool: all connections to postgres-primary-old. Failover not propagated."),
        ],
        "postgres-primary-old": [
            _log("2026-03-27T17:52:00Z", "postgres-primary-old", "WARN", "Promotion event: replica assumed primary role. This instance now read-only."),
            _log("2026-03-27T18:05:00Z", "postgres-primary-old", "ERROR", "Receiving 1800 write attempts/min from services — all rejected (read-only)."),
            _log("2026-03-27T18:10:00Z", "postgres-primary-old", "ERROR", "Active connections: 198/200. Service retry loops filling pool."),
        ],
        "postgres-replica-new": [
            _log("2026-03-27T17:52:00Z", "postgres-replica-new", "INFO", "Promoted to primary. Accepting writes."),
            _log("2026-03-27T18:05:00Z", "postgres-replica-new", "WARN", "Only 30% of expected write traffic received. Split-brain suspected."),
            _log("2026-03-27T18:10:00Z", "postgres-replica-new", "WARN", "Diverging from old primary: 1240 transactions only in new primary."),
        ],
        "payment-service": [
            _log("2026-03-27T18:05:00Z", "payment-service", "ERROR", "Idempotency check failing: order state inconsistent between DB instances"),
            _log("2026-03-27T18:10:00Z", "payment-service", "ERROR", "8% txn double-charge risk. Halting charge processing for affected orders."),
        ],
        "inventory-service": [
            _log("2026-03-27T18:05:00Z", "inventory-service", "ERROR", "Stock deduction writes going to old primary (read-only) — silently lost."),
            _log("2026-03-27T18:10:00Z", "inventory-service", "ERROR", "Oversold items: 340 SKUs with negative virtual stock. Revenue impact growing."),
        ],
        "config-service": [
            _log("2026-03-27T17:52:00Z", "config-service", "INFO", "DB failover event received. Updated DB_PRIMARY_HOST in config store."),
            _log("2026-03-27T17:52:30Z", "config-service", "WARN", "Config propagation: order-service and payment-service did NOT acknowledge new config."),
            _log("2026-03-27T17:55:00Z", "config-service", "ERROR", "Config ack missing for 4/8 services. Manual pgbouncer reload required."),
        ],
        "pgbouncer": [
            _log("2026-03-27T17:52:00Z", "pgbouncer", "WARN", "Failover detected. pgbouncer config NOT auto-updated (static connection string)."),
            _log("2026-03-27T18:10:00Z", "pgbouncer", "ERROR", "Routing 100% of writes to postgres-primary-old (read-only). Update target_db required immediately."),
        ],
        "monitoring-dashboard": [
            _log("2026-03-27T17:52:00Z", "monitoring-dashboard", "INFO", "Auto-failover triggered at 17:52:00Z by health check failure on primary."),
            _log("2026-03-27T18:12:00Z", "monitoring-dashboard", "INFO", "Split-brain duration: 18 min. Financial impact estimate: $42,000 in at-risk transactions."),
        ],
    },
    service_metrics={
        "order-service":         _metrics("order-service",         55.0, 60.0,  800.0, 0.65, 300.0,  8000.0, write_failure_rate=0.65),
        "postgres-primary-old":  _metrics("postgres-primary-old",  80.0, 70.0, 1800.0, 1.0,    5.0,    50.0, is_read_only=1.0, connection_pct=99.0),
        "postgres-replica-new":  _metrics("postgres-replica-new",  30.0, 45.0,  600.0, 0.0,    8.0,    30.0, write_pct_expected=0.30),
        "payment-service":       _metrics("payment-service",       40.0, 45.0,  200.0, 0.25, 180.0,  3500.0, double_charge_risk_pct=0.08),
        "inventory-service":     _metrics("inventory-service",     35.0, 40.0,  300.0, 0.30, 120.0,  2500.0, oversold_skus=340.0),
        "config-service":        _metrics("config-service",        15.0, 20.0,   50.0, 0.10,  30.0,   200.0),
        "monitoring-dashboard":  _metrics("monitoring-dashboard",  10.0, 15.0,  100.0, 0.0,   50.0,   150.0),
        "pgbouncer":             _metrics("pgbouncer",             25.0, 30.0, 2000.0, 0.65,   2.0,     8.0, routing_to_old_primary=1.0),
    },
    correct_severity=IncidentSeverity.P1,
    correct_root_cause_service="pgbouncer",
    correct_root_cause_keywords=[
        "pgbouncer", "connection string", "split-brain", "failover", "read-only",
        "config not propagated", "stale connection", "db routing", "pgbouncer config",
        "connection pool routing", "failover not propagated",
    ],
    valid_remediation_actions=[
        {"action": "config_change", "service": "pgbouncer"},
        {"action": "restart",       "service": "order-service"},
        {"action": "config_change", "service": "order-service"},
        {"action": "restart",       "service": "payment-service"},
    ],
    expected_escalation_teams=["database-team", "platform-team"],
    max_steps=20,
    degradation_per_step=0.02,
    relevant_services=["pgbouncer", "postgres-primary-old", "postgres-replica-new", "order-service"],
    blast_radius={
        "order-service": {
            "write_failure_rate": (0.02, 1.0),
            "error_rate":         (0.02, 0.95),
        },
        "inventory-service": {
            "oversold_skus":      (25.0, 5000.0),
            "error_rate":         (0.02, 0.80),
        },
        "payment-service": {
            "double_charge_risk_pct": (0.005, 0.30),
            "error_rate":             (0.02,  0.60),
        },
        "postgres-primary-old": {
            "connection_pct": (0.2, 100.0),
        },
    },
)


# ==========================================================================
# SCENARIO 1-C – Easy variant: DNS resolution failure
# ==========================================================================

_SCENARIO_EASY_C = Scenario(
    scenario_id="dns-fail-001",
    task_id="severity_classification",
    incident_id="INC-20260327-201",
    description=(
        "Multiple microservices are reporting connection timeouts to downstream "
        "dependencies. Alerts indicate DNS resolution failures across the "
        "internal service mesh. Classify the incident severity."
    ),
    initial_alerts=[
        _alert("ALT-301", "api-gateway", AlertSeverity.CRITICAL,
               "Upstream connection timeout rate 40% to backend services", "2026-03-27T14:00:00Z"),
        _alert("ALT-302", "coredns", AlertSeverity.CRITICAL,
               "DNS query failure rate 65% — SERVFAIL responses", "2026-03-27T13:58:00Z"),
        _alert("ALT-303", "notification-service", AlertSeverity.WARNING,
               "Failed to resolve smtp-relay.internal: NXDOMAIN", "2026-03-27T14:01:00Z"),
    ],
    available_services=["api-gateway", "coredns", "notification-service", "istio-proxy"],
    service_logs={
        "api-gateway": [
            _log("2026-03-27T13:58:00Z", "api-gateway", "ERROR", "upstream connect error: dns_resolution_failure for user-service.default.svc.cluster.local"),
            _log("2026-03-27T13:59:00Z", "api-gateway", "ERROR", "circuit breaker tripped: 5/10 upstream failures in 30s. Returning 503."),
            _log("2026-03-27T14:00:00Z", "api-gateway", "WARN", "Retry budget exhausted for payment-service. DNS not resolving."),
        ],
        "coredns": [
            _log("2026-03-27T13:55:00Z", "coredns", "WARN", "Cache miss rate increasing: 80%. Upstream forwarder slow."),
            _log("2026-03-27T13:57:00Z", "coredns", "ERROR", "OOMKilled: coredns-7d8f9b pod restarted. Memory limit 128Mi exceeded."),
            _log("2026-03-27T13:58:00Z", "coredns", "ERROR", "SERVFAIL for *.default.svc.cluster.local — upstream timeout after 5s"),
            _log("2026-03-27T14:00:00Z", "coredns", "ERROR", "Pod restart count: 4 in last 10 minutes. CrashLoopBackOff."),
        ],
        "notification-service": [
            _log("2026-03-27T14:00:00Z", "notification-service", "WARN", "Email delivery failing: cannot resolve smtp-relay.internal"),
        ],
        "istio-proxy": [
            _log("2026-03-27T14:00:00Z", "istio-proxy", "INFO", "Sidecar healthy. mTLS handshake OK. Issue is upstream DNS, not mesh."),
        ],
    },
    service_metrics={
        "api-gateway": _metrics("api-gateway", 25.0, 40.0, 1200.0, 0.40, 800.0, 5000.0),
        "coredns": _metrics("coredns", 95.0, 98.0, 5000.0, 0.65, 50.0, 5000.0, restart_count=4.0, cache_miss_pct=80.0),
        "notification-service": _metrics("notification-service", 10.0, 20.0, 50.0, 0.80, 200.0, 3000.0),
        "istio-proxy": _metrics("istio-proxy", 5.0, 10.0, 1200.0, 0.01, 2.0, 10.0),
    },
    correct_severity=IncidentSeverity.P1,
    correct_root_cause_service="coredns",
    correct_root_cause_keywords=["dns", "coredns", "OOM", "memory", "resolution", "SERVFAIL", "CrashLoop"],
    valid_remediation_actions=[
        {"action": "restart", "service": "coredns"},
        {"action": "scale", "service": "coredns"},
        {"action": "config_change", "service": "coredns", "detail": "increase memory limit"},
    ],
    expected_escalation_teams=["platform-team"],
    max_steps=10,
    degradation_per_step=0.008,
    relevant_services=["api-gateway", "coredns"],
    blast_radius={
        "coredns": {
            "error_rate": (0.03, 0.95),
            "restart_count": (1.0, 15.0),
        },
        "api-gateway": {
            "error_rate": (0.03, 0.80),
            "latency_p99_ms": (500.0, 15000.0),
        },
    },
)


# ==========================================================================
# SCENARIO 2-C – Medium variant: TLS certificate expiry
# ==========================================================================

_SCENARIO_MEDIUM_C = Scenario(
    scenario_id="tls-expiry-001",
    task_id="root_cause_analysis",
    incident_id="INC-20260327-301",
    description=(
        "The checkout-service is returning 502 errors for all HTTPS calls to "
        "the payment provider API. Internal health checks pass but external "
        "payment calls fail. Diagnose the root cause and remediate."
    ),
    initial_alerts=[
        _alert("ALT-401", "checkout-service", AlertSeverity.CRITICAL,
               "Payment API calls failing: 502 rate 95%", "2026-03-27T09:00:00Z"),
        _alert("ALT-402", "cert-manager", AlertSeverity.WARNING,
               "Certificate renewal failed for payments.example.com — ACME challenge timeout", "2026-03-27T08:00:00Z"),
        _alert("ALT-403", "nginx-ingress", AlertSeverity.WARNING,
               "TLS handshake failures: 200/min on payments upstream", "2026-03-27T09:01:00Z"),
    ],
    available_services=["checkout-service", "cert-manager", "nginx-ingress", "payment-provider-stub"],
    service_logs={
        "checkout-service": [
            _log("2026-03-27T08:55:00Z", "checkout-service", "ERROR", "PaymentGatewayError: SSL certificate has expired (payments.example.com)"),
            _log("2026-03-27T08:58:00Z", "checkout-service", "ERROR", "javax.net.ssl.SSLHandshakeException: PKIX path validation failed: certificate expired at 2026-03-27T00:00:00Z"),
            _log("2026-03-27T09:00:00Z", "checkout-service", "ERROR", "Circuit breaker OPEN for payment-provider. 48/50 calls failed in 60s."),
        ],
        "cert-manager": [
            _log("2026-03-27T02:00:00Z", "cert-manager", "INFO", "Certificate renewal triggered for payments.example.com (expires in 24h)"),
            _log("2026-03-27T02:01:00Z", "cert-manager", "ERROR", "ACME HTTP-01 challenge failed: upstream DNS not resolving challenge token"),
            _log("2026-03-27T02:05:00Z", "cert-manager", "ERROR", "Retry 3/3 failed. Certificate NOT renewed. Expiry: 2026-03-27T00:00:00Z"),
            _log("2026-03-27T08:00:00Z", "cert-manager", "CRITICAL", "Certificate EXPIRED: payments.example.com. Last valid: 2026-03-26T23:59:59Z"),
        ],
        "nginx-ingress": [
            _log("2026-03-27T09:00:00Z", "nginx-ingress", "ERROR", "SSL_do_handshake() failed: certificate verify failed (expired)"),
            _log("2026-03-27T09:01:00Z", "nginx-ingress", "WARN", "Upstream payments backend: 200 TLS errors/min. Peer certificate expired."),
        ],
        "payment-provider-stub": [
            _log("2026-03-27T09:00:00Z", "payment-provider-stub", "INFO", "Healthy. Accepting connections on port 443 with valid certificate."),
        ],
    },
    service_metrics={
        "checkout-service": _metrics("checkout-service", 15.0, 30.0, 300.0, 0.95, 50.0, 200.0, payment_success_pct=5.0, revenue_loss_per_min=8500.0),
        "cert-manager": _metrics("cert-manager", 5.0, 10.0, 1.0, 0.0, 10.0, 50.0, certs_expired=1.0, renewal_failures=3.0),
        "nginx-ingress": _metrics("nginx-ingress", 10.0, 20.0, 500.0, 0.40, 5.0, 30.0, tls_handshake_failures_per_min=200.0),
        "payment-provider-stub": _metrics("payment-provider-stub", 5.0, 15.0, 50.0, 0.0, 20.0, 80.0),
    },
    correct_severity=IncidentSeverity.P1,
    correct_root_cause_service="cert-manager",
    correct_root_cause_keywords=["certificate", "TLS", "SSL", "expired", "cert-manager", "renewal", "ACME", "expiry"],
    valid_remediation_actions=[
        {"action": "restart", "service": "cert-manager"},
        {"action": "config_change", "service": "cert-manager", "detail": "force renewal"},
        {"action": "config_change", "service": "nginx-ingress", "detail": "update certificate"},
    ],
    expected_escalation_teams=["security-team", "platform-team"],
    max_steps=15,
    degradation_per_step=0.010,
    relevant_services=["checkout-service", "cert-manager", "nginx-ingress"],
    blast_radius={
        "checkout-service": {
            "error_rate": (0.005, 1.0),
            "payment_success_pct": (-0.5, 0.0),
            "revenue_loss_per_min": (500.0, 50000.0),
        },
        "nginx-ingress": {
            "tls_handshake_failures_per_min": (20.0, 1000.0),
        },
    },
)


# ---- registry ---------------------------------------------------------------

# Multiple variants per task — environment randomly selects one per reset()
SCENARIO_VARIANTS: Dict[str, List[Scenario]] = {
    "severity_classification": [_SCENARIO_EASY, _SCENARIO_EASY_B, _SCENARIO_EASY_C],
    "root_cause_analysis": [_SCENARIO_MEDIUM, _SCENARIO_MEDIUM_B, _SCENARIO_MEDIUM_C],
    "full_incident_management": [_SCENARIO_HARD, _SCENARIO_HARD_B, _SCENARIO_HARD_C],
}

# Always maps task_id → primary (deterministic) scenario for testing/baseline
SCENARIOS: Dict[str, Scenario] = {
    "severity_classification": _SCENARIO_EASY,
    "root_cause_analysis": _SCENARIO_MEDIUM,
    "full_incident_management": _SCENARIO_HARD,
}


def get_scenario(task_id: str, variant_seed: int = 0) -> Scenario:
    """Return a scenario for the given task_id.

    Args:
        task_id:      One of the three registered task IDs.
        variant_seed: Index into SCENARIO_VARIANTS[task_id]. Wraps around.
                      Pass 0 for the primary/deterministic scenario.
    """
    if task_id not in SCENARIO_VARIANTS:
        raise ValueError(f"Unknown task_id '{task_id}'. Valid: {list(SCENARIO_VARIANTS.keys())}")
    variants = SCENARIO_VARIANTS[task_id]
    return variants[variant_seed % len(variants)]

