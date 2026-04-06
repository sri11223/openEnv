---
title: Incident Response Triage
emoji: 🚨
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - sre
  - incident-response
---

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

# Run inference script (root-level, competition-compliant)
python inference.py
# → Uses rules baseline by default; set HF_TOKEN + API_BASE_URL + MODEL_NAME for LLM mode

# Run baseline module (alternative)
python -m baseline.inference --mode rules --direct

# Run baseline (against running server)
python -m baseline.inference --mode rules --base-url http://localhost:7860

# Run LLM baseline
export HF_TOKEN=hf_...            # or OPENAI_API_KEY=sk-...
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct
python inference.py
```

### Docker

```bash
docker build -t incident-response-triage .
docker run -p 7860:7860 incident-response-triage
```

### Hugging Face Spaces

This environment is deployed as a Docker Space. To deploy your own copy:

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Choose **Docker** as the SDK
3. Set the Space name and add tag `openenv`
4. In the Space settings, link this repository (or push directly)
5. HF Spaces will run `docker build` then `docker run` automatically

The container exposes port **7860** as required by HF Spaces.

```bash
# Test the deployed space
curl https://<your-username>-incident-response-triage.hf.space/
```

---

## API Endpoints

| Method | Path | Headers | Description |
|--------|------|---------|-------------|
| `GET` | `/` | — | Health check + telemetry summary |
| `POST` | `/reset` | — | Start new episode; returns initial observation **and** `session_id` |
| `POST` | `/step` | `X-Session-ID` | Take an action; returns observation, reward, done, info |
| `GET` | `/state` | `X-Session-ID` | Full environment state snapshot |
| `GET` | `/tasks` | — | List tasks with action schema |
| `POST` | `/grader` | `X-Session-ID` | Grader score for current/completed episode |
| `POST` | `/baseline` | — | Run rule-based baseline on all tasks (in-process) |
| `GET` | `/metrics` | — | Telemetry counters (JSON or `?format=prometheus`) |
| `GET` | `/render` | `X-Session-ID` | Human-readable Markdown incident dashboard |
| `GET` | `/leaderboard` | — | Top-10 scores per task from completed episodes |

> **Session flow:** `/reset` returns a `session_id`. Pass it as the `X-Session-ID` HTTP header on all subsequent `/step`, `/state`, `/grader`, and `/render` calls. This enables safe concurrent multi-agent evaluation.

```bash
# 1. Start episode
SESSION=$(curl -s -X POST http://localhost:7860/reset \
  -H 'Content-Type: application/json' \
  -d '{"task_id": "severity_classification"}' | python -c "import sys,json; print(json.load(sys.stdin)['session_id'])")

# 2. Take a step
curl -X POST http://localhost:7860/step \
  -H 'Content-Type: application/json' \
  -H "X-Session-ID: $SESSION" \
  -d '{"action_type": "investigate", "target": "postgres-primary", "parameters": {}, "reasoning": "check connection pool"}'

# 3. Grade the episode
curl -X POST http://localhost:7860/grader -H "X-Session-ID: $SESSION"
```

---

## Baseline Scores (Rule-Based)

| Task | Score | Steps |
|------|-------|-------|
| severity_classification | **1.00** | 3 |
| root_cause_analysis | **1.00** | 5 |
| full_incident_management | **0.95** | 12 |
| **Mean** | **0.98** | — |

These scores represent a near-optimal deterministic policy. An LLM agent typically scores lower due to reasoning errors and sub-optimal action ordering.

---

## Architecture

```
openEnv/
├── inference.py                  # Root inference script (competition-compliant, [START]/[STEP]/[END] logs)
├── app.py                        # FastAPI server with all endpoints + WebSocket + web UI
├── openenv.yaml                  # OpenEnv metadata spec
├── Dockerfile                    # Multi-stage container definition (non-root, health check)
├── requirements.txt              # Python dependencies
├── validate.py                   # Pre-submission validation
├── src/
│   ├── __init__.py
│   ├── models.py                 # Pydantic v2 models (Observation, Action, Reward, StepResult, etc.)
│   ├── scenarios.py              # Deterministic incident scenario data (7 scenario variants)
│   ├── environment.py            # Core env: reset(), step(), state(), grade()
│   ├── rewards.py                # Dense per-step reward computation (12 reward components)
│   ├── graders.py                # End-of-episode grading (0.0–1.0, multi-dimensional breakdown)
│   └── tasks.py                  # Task metadata for /tasks endpoint
├── baseline/
│   ├── __init__.py
│   └── inference.py              # Rule-based + LLM baseline scripts (module)
└── tests/
    ├── __init__.py
    └── test_env.py               # Comprehensive test suite (concurrency, variants, boundaries)
```

---

## Design Decisions

1. **Progressive Disclosure**: Logs and metrics are hidden until investigated. This prevents the agent from seeing the answer immediately and forces genuine investigation strategy.

2. **Temporal Degradation**: Real incidents get worse over time. The environment penalizes slow response with per-step degradation that scales with task difficulty (0.005/step easy → 0.015/step hard).

3. **Dynamic Blast Radius**: Metrics degrade in real-time as the agent delays. Connection pools fill, error rates climb, queue depths grow — revealed via INVESTIGATE so the agent sees a progressively worsening system.

4. **Multi-dimensional Grading**: Each task grades multiple orthogonal aspects (accuracy, investigation quality, efficiency, precision) rather than just binary success/fail. The hard task has 8 scoring dimensions.

5. **Investigation Precision**: Agents that investigate irrelevant services (red herrings like CDN, postgres-primary in the hard scenario) receive a lower precision score. This rewards strategic, focused investigation.

6. **Red Herrings**: The hard scenario includes 8 services but only 4 are relevant. CDN and postgres show normal behavior, testing the agent's ability to focus.

7. **Deterministic Scenarios**: Scenarios are fixed data with multiple variants per task (2+2+3 = 7 total). Variant seed 0 always returns the primary scenario for reproducibility.

8. **Content-Aware Reasoning Bonus**: The reward function introspects the agent's `reasoning` field and gives higher bonuses when it references relevant services or root-cause keywords — encouraging agents to show their work.

---

## Pre-Submission Checklist

- [x] HF Space deploys and responds to `GET /` with 200
- [x] `openenv.yaml` valid with 3 tasks, action/observation spaces, reward spec
- [x] Typed Pydantic v2 models for Observation, Action, Reward, StepResult
- [x] `step()` / `reset()` / `state()` endpoints functional
- [x] 3 tasks with difficulty progression: easy → medium → hard
- [x] Graders produce deterministic scores in [0.0, 1.0] range
- [x] Dense per-step rewards (not just binary end-of-episode)
- [x] Baseline inference script (`inference.py`) with reproducible scores
- [x] Structured stdout logs: `[START]`, `[STEP]`, `[END]` format
- [x] `Dockerfile` builds and runs cleanly
- [x] Environment variables: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`
- [x] OpenAI Client used for all LLM calls
- [x] Runtime < 20 minutes on vcpu=2, 8GB RAM
- [x] `python validate.py` passes all checks

---

## License

MIT

