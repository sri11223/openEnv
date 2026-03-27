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
    expected_escalation_teams=["payments-team", "infrastructure-team", "platform-team"],
    max_steps=15,
    degradation_per_step=0.01,
    relevant_services=["payment-gateway", "redis-session", "payment-processor"],
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
    expected_escalation_teams=["platform-team", "auth-team", "security-team", "on-call-lead"],
    max_steps=20,
    degradation_per_step=0.015,
    relevant_services=["auth-service", "api-gateway", "redis-auth-cache", "order-service"],
)


# ---- registry ---------------------------------------------------------------

SCENARIOS: Dict[str, Scenario] = {
    "severity_classification": _SCENARIO_EASY,
    "root_cause_analysis": _SCENARIO_MEDIUM,
    "full_incident_management": _SCENARIO_HARD,
}


def get_scenario(task_id: str) -> Scenario:
    if task_id not in SCENARIOS:
        raise ValueError(f"Unknown task_id '{task_id}'. Valid: {list(SCENARIOS.keys())}")
    return SCENARIOS[task_id]
