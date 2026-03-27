# Incident Response Triage — OpenEnv Environment

> **Domain:** Site Reliability Engineering / On-Call Incident Management  
> **Difficulty:** Easy → Medium → Hard  
> **Framework:** FastAPI + Pydantic v2  
> **Tags:** `openenv` `incident-response` `site-reliability` `operations` `triage`

---

## What Is This?

A **production incident response simulator** where an AI agent acts as an on-call engineer. The agent receives alerts from a monitoring system and must:

1. **Investigate** — query service logs and metrics (progressive disclosure)
2. **Classify** — assign incident severity (P1–P4)
3. **Diagnose** — identify the root cause service and failure mode
4. **Remediate** — apply the correct fix (restart, rollback, scale, config change)
5. **Escalate** — notify the right teams with context
6. **Communicate** — post status updates for stakeholders

This models a task that tens of thousands of engineers perform daily, with real consequences for delays and misdiagnosis. The environment features **progressive information disclosure** — logs and metrics are hidden until the agent actively investigates a service — creating a genuine information-gathering challenge under temporal pressure.

---

## Why This Matters

| Problem | How This Env Addresses It |
|---------|--------------------------|
| Incident response training is expensive | Agents can practice on realistic scenarios at zero cost |
| On-call fatigue leads to errors | AI agents can assist with triage, reducing human toil |
| Root cause analysis requires connecting symptoms across services | Multi-service scenarios with red herrings test reasoning |
| Incident management is a workflow, not a single decision | Hard task requires 6 different action types in sequence |

---

## Tasks

### Task 1: Severity Classification (Easy)
**Scenario:** Database connection pool exhaustion on PostgreSQL primary.  
**Objective:** Review 3 alerts, investigate 2 services, classify as correct severity.  
**Max steps:** 10  
**Graded on:** Classification accuracy (50%), investigation quality (25%), efficiency (25%).

### Task 2: Root Cause Analysis (Medium)  
**Scenario:** Payment processing failures caused by Redis session eviction.  
**Objective:** Distinguish root cause (Redis memory) from symptoms (payment gateway errors), classify severity, diagnose, and remediate.  
**Max steps:** 15  
**Graded on:** Severity (15%), root cause investigation (15%), diagnosis (30%), remediation (20%), efficiency (20%).

### Task 3: Full Incident Management (Hard)
**Scenario:** Cascading multi-service outage from a bad deployment (auth-service v3.1.0 memory leak).  
**Objective:** 6 alerts across 8 services. Must investigate strategically, classify, diagnose, apply multiple remediations, escalate to correct teams, and communicate status.  
**Max steps:** 20  
**Graded on:** Severity (12%), diagnosis (18%), remediations (18%), escalation (14%), communication (13%), investigation thoroughness (12%), efficiency (13%).

---

## Action Space

```json
{
  "action_type": "classify | investigate | diagnose | remediate | escalate | communicate",
  "target": "service-name | team-name | channel",
  "parameters": {
    "severity": "P1 | P2 | P3 | P4",
    "root_cause": "free text description",
    "action": "restart | rollback | scale | config_change",
    "priority": "urgent | high | medium",
    "message": "status update text"
  },
  "reasoning": "explanation of agent's decision"
}
```

| Action Type | Target | Required Parameters |
|-------------|--------|-------------------|
| `investigate` | service name | — |
| `classify` | — | `parameters.severity` |
| `diagnose` | service name | `parameters.root_cause` |
| `remediate` | service name | `parameters.action` |
| `escalate` | team name | `parameters.priority`, `parameters.message` |
| `communicate` | channel | `parameters.message` |

---

## Observation Space

Each observation includes:

| Field | Type | Description |
|-------|------|-------------|
| `incident_id` | string | Unique incident identifier |
| `step_number` | int | Current step (0-indexed) |
| `max_steps` | int | Maximum steps before episode ends |
| `alerts` | Alert[] | Active monitoring alerts (always visible) |
| `available_services` | string[] | Services that can be investigated |
| `investigated_services` | string[] | Already investigated services |
| `logs` | Dict[str, LogEntry[]] | Service logs (**revealed on `investigate`**) |
| `metrics` | Dict[str, ServiceMetrics] | Service metrics (**revealed on `investigate`**) |
| `incident_status` | enum | `open` → `investigating` → `mitigating` → `resolved` |
| `severity_classified` | P1-P4 or null | Set after `classify` action |
| `diagnosis` | string or null | Set after `diagnose` action |
| `message` | string | Feedback from last action |

---

## Reward Function

**Dense per-step rewards** (not just binary end-of-episode):

| Component | Range | Trigger |
|-----------|-------|---------|
| Relevant investigation | +0.06 | Investigating a service related to the root cause |
| Irrelevant investigation | −0.02 | Investigating an unrelated service |
| Correct classification | +0.15 | Exact severity match |
| Wrong classification | −0.05 per level | Off by 1 → −0.05, off by 2 → −0.10 |
| Correct diagnosis | +0.15 | Root cause keywords matched |
| Correct remediation | +0.12 | Valid fix for the scenario |
| Wrong remediation | −0.08 | Applying a fix to the wrong service |
| Temporal degradation | −0.005 to −0.015/step | Penalty increases with task difficulty |
| Duplicate action | −0.03 | Repeating an already-completed action |
| Reasoning bonus | +0.01 | Providing non-trivial reasoning text |

---

## Setup & Usage

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python app.py
# → http://localhost:7860

# Run tests
pip install pytest
pytest tests/ -v

# Run validation
python validate.py

# Run baseline (in-process, no server needed)
python -m baseline.inference --mode rules --direct

# Run baseline (against running server)
python -m baseline.inference --mode rules --base-url http://localhost:7860

# Run LLM baseline (requires OPENAI_API_KEY)
export OPENAI_API_KEY=sk-...
python -m baseline.inference --mode llm --base-url http://localhost:7860
```

### Docker

```bash
docker build -t incident-response-triage .
docker run -p 7860:7860 incident-response-triage
```

### Hugging Face Spaces

Deploy as a Docker Space tagged with `openenv`. The Dockerfile is configured to expose port 7860.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check |
| POST | `/reset` | Reset environment for a task (`{"task_id": "..."}`) |
| POST | `/step` | Take an action (Action JSON body) |
| GET | `/state` | Full environment state snapshot |
| GET | `/tasks` | List tasks with action schema |
| POST | `/grader` | Get grader score for current episode |
| POST | `/baseline` | Run baseline inference on all tasks |

---

## Baseline Scores (Rule-Based)

| Task | Score | Steps |
|------|-------|-------|
| severity_classification | ~0.95 | 3 |
| root_cause_analysis | ~0.85 | 5 |
| full_incident_management | ~0.85 | 11 |
| **Mean** | **~0.88** | — |

These scores represent a near-optimal deterministic policy. An LLM agent typically scores lower due to reasoning errors and inefficiency.

---

## Architecture

```
openEnv/
├── app.py                    # FastAPI server with all endpoints
├── openenv.yaml              # OpenEnv metadata spec
├── Dockerfile                # Container definition
├── requirements.txt          # Python dependencies
├── validate.py               # Pre-submission validation
├── src/
│   ├── __init__.py
│   ├── models.py             # Pydantic models (Observation, Action, Reward, etc.)
│   ├── scenarios.py          # Deterministic incident scenario data
│   ├── environment.py        # Core env: reset(), step(), state(), grade()
│   ├── rewards.py            # Dense per-step reward computation
│   ├── graders.py            # End-of-episode grading (0.0–1.0)
│   └── tasks.py              # Task metadata for /tasks endpoint
├── baseline/
│   ├── __init__.py
│   └── inference.py          # Rule-based + LLM baseline scripts
└── tests/
    ├── __init__.py
    └── test_env.py           # Comprehensive test suite
```

---

## Design Decisions

1. **Progressive Disclosure**: Logs and metrics are hidden until investigated. This prevents the agent from seeing the answer immediately and forces genuine investigation strategy.

2. **Temporal Degradation**: Real incidents get worse over time. The environment penalizes slow response with per-step degradation that scales with task difficulty.

3. **Multi-dimensional Grading**: Each task grades multiple aspects (accuracy, process quality, efficiency) rather than just binary success/fail.

4. **Red Herrings**: The hard scenario includes 8 services but only 4 are relevant. CDN and postgres show normal behavior, testing the agent's ability to focus.

5. **Deterministic Scenarios**: Scenarios are fixed data, not randomly generated. This ensures reproducibility and fair comparison between agents.

---

## License

MIT
