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
> **Phase 1 & 2:** ✅ PASSED  
> **Tags:** `openenv` `incident-response` `site-reliability` `operations` `triage`

<table>
<tr>
<td><strong>Baseline Score</strong></td><td>0.99 / 0.99 / 0.93 (mean 0.97)</td>
</tr>
<tr>
<td><strong>Tasks</strong></td><td>3 (Easy, Medium, Hard)</td>
</tr>
<tr>
<td><strong>Scenarios</strong></td><td>7 variants across 3 difficulty levels</td>
</tr>
<tr>
<td><strong>Reward Components</strong></td><td>12 (dense, per-step)</td>
</tr>
<tr>
<td><strong>Unit Tests</strong></td><td>145 / 145 passing</td>
</tr>
<tr>
<td><strong>Validation Checks</strong></td><td>7 / 7 passing</td>
</tr>
<tr>
<td><strong>CI</strong></td><td>GitHub Actions — tests + Docker build + lint</td>
</tr>
</table>

---

## Requirements Compliance

### Functional Requirements

| Requirement | Status | Evidence |
|-------------|--------|----------|
| **Real-world task simulation** | ✅ | On-call SRE incident response — a real workflow performed by thousands of engineers daily. Not a game or toy. |
| **OpenEnv spec: typed Observation/Action/Reward Pydantic models** | ✅ | `src/models.py` — `Observation`, `Action`, `Reward`, `StepResult` all Pydantic v2 |
| **`step(action)` → observation, reward, done, info** | ✅ | `src/environment.py::IncidentResponseEnv.step()` |
| **`reset()` → initial observation** | ✅ | `src/environment.py::IncidentResponseEnv.reset()` |
| **`state()` → current state** | ✅ | `src/environment.py::IncidentResponseEnv.state()` |
| **`openenv.yaml` with metadata** | ✅ | Root-level `openenv.yaml` — 3 tasks, full action/obs/reward spec |
| **Tested via `openenv validate`** | ✅ | 7/7 checks pass — see [Validation Output](#validation-output) |
| **Minimum 3 tasks with graders (0–1, easy→medium→hard)** | ✅ | `severity_classification` (easy), `root_cause_analysis` (medium), `full_incident_management` (hard) |
| **Graders: deterministic success/failure criteria** | ✅ | `src/graders.py` — multi-dimensional, keyword-matched, no randomness in grading |
| **Meaningful reward: dense, partial progress, penalizes bad behavior** | ✅ | 12 components, per-step rewards, `−0.005` to `−0.015/step` temporal degradation, `−0.08` wrong remediation |
| **Baseline inference script using OpenAI API** | ✅ | `inference.py` — uses `openai.OpenAI` client; rules mode for default, LLM mode via env vars |
| **Reads credentials from environment variables** | ✅ | `HF_TOKEN` / `OPENAI_API_KEY` and `API_BASE_URL`, `MODEL_NAME` |
| **Reproducible baseline score** | ✅ | Rule-based: 0.99, 0.99, 0.93 every run — see [Validation Output](#validation-output) |

### Non-Functional Requirements

| Requirement | Status | Evidence |
|-------------|--------|----------|
| **Deploys to Hugging Face Space** | ✅ | [srikrishna2005/openenv](https://huggingface.co/spaces/srikrishna2005/openenv) — Docker Space, tagged `openenv` |
| **Working Dockerfile** | ✅ | Single-stage `python:3.12-slim`, non-root user, health check, port 7860 |
| **`docker build && docker run` works cleanly** | ✅ | Phase 2 Docker build passed |
| **README: environment description + motivation** | ✅ | [What Is This?](#what-is-this) and [Why This Matters](#why-this-matters) |
| **README: action and observation space definitions** | ✅ | [Action Space](#action-space) and [Observation Space](#observation-space) |
| **README: task descriptions with expected difficulty** | ✅ | [Tasks](#tasks) |
| **README: setup and usage instructions** | ✅ | [Setup & Usage](#setup--usage) |
| **README: baseline scores** | ✅ | [Baseline Scores](#baseline-scores-rule-based) |

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
| `GET` | `/curriculum` | — | Ordered curriculum learning stages with difficulty metadata |
| `GET` | `/prometheus/metrics` | opt. `X-Session-ID` | Prometheus text scrape of live scenario service metrics |
| `GET` | `/prometheus/query` | opt. `X-Session-ID` | PromQL instant query (standard Prometheus JSON envelope) |
| `GET` | `/prometheus/query_range` | opt. `X-Session-ID` | PromQL range query — matrix result from TSDB ring buffer |
| `GET` | `/web` | — | Interactive browser-based incident dashboard (WebSocket-backed) |

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

| Task | Score | Steps | Cumulative Reward |
|------|-------|-------|-------------------|
| `severity_classification` | **0.99** | 3 | 0.29 |
| `root_cause_analysis` | **0.99** | 5 | 0.56 |
| `full_incident_management` | **0.93** | 11 | 0.13 |
| **Mean** | **0.97** | — | — |

Scores are deterministic and reproducible. Clamped to (0.01, 0.99) per spec requirement.

---

## Example Inference Transcript

Running `python inference.py` (rules mode, no credentials needed):

```
[START] task=severity_classification env=incident_response_triage model=meta-llama/Meta-Llama-3-8B-Instruct
[STEP] step=1 action={"action_type":"investigate","target":"postgres-primary","parameters":{},"reasoning":"Alert shows connection pool at 98% on postgres-primary."} reward=0.07 done=false error=null
[STEP] step=2 action={"action_type":"investigate","target":"user-service","parameters":{},"reasoning":"user-service has high latency. Checking logs for pool errors."} reward=0.07 done=false error=null
[STEP] step=3 action={"action_type":"classify","target":"","parameters":{"severity":"P2"},"reasoning":"Service degraded but not fully down. Connection pool issue is P2."} reward=0.15 done=true error=null
[END] success=true steps=3 score=0.99 rewards=0.07,0.07,0.15

[START] task=root_cause_analysis env=incident_response_triage model=meta-llama/Meta-Llama-3-8B-Instruct
[STEP] step=1 action={"action_type":"investigate","target":"payment-gateway","parameters":{},"reasoning":"Payment success rate critically low. Starting here."} reward=0.06 done=false error=null
[STEP] step=2 action={"action_type":"investigate","target":"redis-session","parameters":{},"reasoning":"Eviction spike on redis-session could explain missing tokens."} reward=0.06 done=false error=null
[STEP] step=3 action={"action_type":"classify","target":"","parameters":{"severity":"P1"},"reasoning":"Payment processing at 45% success is P1 revenue-impacting."} reward=0.12 done=false error=null
[STEP] step=4 action={"action_type":"diagnose","target":"redis-session","parameters":{"root_cause":"Redis session store hit maxmemory limit causing eviction of payment session tokens."},"reasoning":"Logs show redis-session at 100% memory with aggressive evictions."} reward=0.23 done=false error=null
[STEP] step=5 action={"action_type":"remediate","target":"redis-session","parameters":{"action":"scale"},"reasoning":"Scaling redis-session memory to stop evictions."} reward=0.09 done=true error=null
[END] success=true steps=5 score=0.99 rewards=0.06,0.06,0.12,0.23,0.09

[START] task=full_incident_management env=incident_response_triage model=meta-llama/Meta-Llama-3-8B-Instruct
[STEP] step=1 action={"action_type":"investigate","target":"auth-service","parameters":{},"reasoning":"Auth-service has critical latency. Multiple services depend on auth."} reward=0.07 done=false error=null
[STEP] step=2 action={"action_type":"investigate","target":"api-gateway","parameters":{},"reasoning":"API gateway returning 503s. Checking if auth-related."} reward=0.04 done=false error=null
[STEP] step=3 action={"action_type":"investigate","target":"redis-auth-cache","parameters":{},"reasoning":"Checking auth cache — may explain why auth is slow."} reward=0.02 done=false error=null
[STEP] step=4 action={"action_type":"investigate","target":"order-service","parameters":{},"reasoning":"Order queue depth at 15000+."} reward=0.01 done=false error=null
[STEP] step=5 action={"action_type":"classify","target":"","parameters":{"severity":"P1"},"reasoning":"Cascading multi-service outage. P1."} reward=0.08 done=false error=null
[STEP] step=6 action={"action_type":"diagnose","target":"auth-service","parameters":{"root_cause":"v3.1.0 deploy introduced memory leak via unbounded in-memory token cache. OOMKill cascades."},"reasoning":"Auth logs show v3.1.0 at 13:47, memory climbing to 97%."} reward=0.18 done=false error=null
[STEP] step=7 action={"action_type":"remediate","target":"auth-service","parameters":{"action":"rollback"},"reasoning":"Rolling back to v3.0.9."} reward=0.04 done=false error=null
[STEP] step=8 action={"action_type":"remediate","target":"order-service","parameters":{"action":"scale"},"reasoning":"Queue depth 15000+. Scaling to drain backlog."} reward=0.01 done=false error=null
[STEP] step=9 action={"action_type":"escalate","target":"platform-team","parameters":{"priority":"urgent","message":"Cascading outage from auth v3.1.0 memory leak. Rolling back. Need queue recovery support."},"reasoning":"Platform team needs infra visibility."} reward=-0.08 done=false error=null
[STEP] step=10 action={"action_type":"escalate","target":"auth-team","parameters":{"priority":"urgent","message":"v3.1.0 has unbounded memory growth in token cache. Rolled back to v3.0.9."},"reasoning":"Auth team owns the service."} reward=-0.10 done=false error=null
[STEP] step=11 action={"action_type":"communicate","target":"status_page","parameters":{"message":"INCIDENT UPDATE: auth-service v3.1.0 memory leak identified. Rollback in progress. ETA 15 min."},"reasoning":"Stakeholders need status update."} reward=-0.12 done=true error=null
[END] success=true steps=11 score=0.93 rewards=0.07,0.04,0.02,0.01,0.08,0.18,0.04,0.01,-0.08,-0.10,-0.12
```

**Post-run grader breakdown (from `inference.py --verbose`):**

```
--- severity_classification ---
  Score: 0.99   Steps: 3   Cumulative reward: 0.29
  Feedback: ✓ Severity correct. ✓ Investigated relevant services. ✓ Efficient (3 steps).
    severity_accuracy:        0.5000
    investigation_quality:    0.2500
    efficiency:               0.2500

--- root_cause_analysis ---
  Score: 0.99   Steps: 5   Cumulative reward: 0.56
  Feedback: ✓ Root cause correct (Redis maxmemory → eviction of payment tokens). ✓ Investigated redis-session. ✓ Correct remediation. Efficiency: optimal 5-step path.
    severity_accuracy:               0.1500
    investigated_root_cause_service: 0.1500
    diagnosis_accuracy:              0.3000
    remediation_quality:             0.2000
    efficiency:                      0.2000

--- full_incident_management ---
  Score: 0.93   Steps: 11   Cumulative reward: 0.13
  Feedback: ✓ Root cause identified (auth v3.1.0 unbounded token cache). ✓ Rollback + scale remediations. ✓ Both teams escalated. ✓ Status communicated.
    severity_accuracy:          0.1200
    diagnosis_accuracy:         0.2000
    remediation_quality:        0.1800
    escalation_quality:         0.1500
    communication:              0.0600
    investigation_thoroughness: 0.1200
    investigation_precision:    0.0300
    efficiency:                 0.0700
```

---

## Validation Output

Running `python validate.py`:

```
============================================================
OpenEnv Pre-Submission Validation
============================================================
  [PASS] openenv.yaml valid: Found 3 tasks
  [PASS] 3+ tasks defined: 3 tasks defined
  [PASS] reset() works: All tasks reset successfully
  [PASS] step() returns StepResult: Step returns correct StepResult
  [PASS] state() works: State snapshot correct
  [PASS] Graders score [0.0-1.0]: All graders in [0.0, 1.0]
  [PASS] Baseline reproducible: Baseline scores: ['0.9900', '0.9900', '0.9300']
============================================================
ALL CHECKS PASSED
============================================================
```

---

## Test Suite (145 / 145 Passing)

Running `pytest tests/ -v`:

```
tests/test_env.py::TestReset::test_reset_returns_observation[severity_classification] PASSED
tests/test_env.py::TestReset::test_reset_returns_observation[root_cause_analysis] PASSED
tests/test_env.py::TestReset::test_reset_returns_observation[full_incident_management] PASSED
tests/test_env.py::TestReset::test_reset_invalid_task PASSED
tests/test_env.py::TestReset::test_reset_clears_state PASSED
tests/test_env.py::TestStep::test_step_investigate PASSED
tests/test_env.py::TestStep::test_step_classify PASSED
tests/test_env.py::TestStep::test_step_without_reset_raises PASSED
tests/test_env.py::TestStep::test_step_after_done_raises PASSED
tests/test_env.py::TestStep::test_investigate_invalid_service PASSED
tests/test_env.py::TestState::test_state_after_reset PASSED
tests/test_env.py::TestState::test_state_tracks_actions PASSED
tests/test_env.py::TestRewards::test_relevant_investigation_positive PASSED
tests/test_env.py::TestRewards::test_irrelevant_investigation_negative PASSED
tests/test_env.py::TestRewards::test_correct_classification_positive PASSED
tests/test_env.py::TestRewards::test_wrong_classification_negative PASSED
tests/test_env.py::TestGraders::test_grader_score_range[severity_classification] PASSED
tests/test_env.py::TestGraders::test_grader_score_range[root_cause_analysis] PASSED
tests/test_env.py::TestGraders::test_grader_score_range[full_incident_management] PASSED
tests/test_env.py::TestGraders::test_perfect_easy_score PASSED
tests/test_env.py::TestGraders::test_zero_score_on_no_action PASSED
tests/test_env.py::TestBaseline::test_rule_based_baseline_all_tasks PASSED
tests/test_env.py::TestBaseline::test_rule_based_scores_reproducible PASSED
tests/test_env.py::TestBaseline::test_difficulty_progression PASSED
tests/test_env.py::TestEpisodeBoundaries::test_easy_ends_on_classify PASSED
tests/test_env.py::TestEpisodeBoundaries::test_medium_ends_on_diagnose_and_remediate PASSED
tests/test_env.py::TestEpisodeBoundaries::test_max_steps_terminates PASSED
tests/test_env.py::TestConcurrency::test_parallel_episodes_do_not_share_state PASSED
tests/test_env.py::TestConcurrency::test_many_sequential_resets_same_instance PASSED
tests/test_env.py::TestScenarioVariants::test_variant_seed_0_is_deterministic PASSED
tests/test_env.py::TestScenarioVariants::test_different_seeds_may_return_different_scenarios PASSED
tests/test_env.py::TestScenarioVariants::test_variant_seed_wraps_gracefully PASSED
# ... + 89 extended quality tests in tests/test_quality.py (all 7 scenarios,
#     every reward component, temporal degradation, blast radius, partial credit,
#     grader feedback quality, observation contract, YAML structure, inference script)
# ... + 12 Prometheus live metrics endpoint tests
# ... + 12 TSDB ring-buffer range-query tests
145 passed in 104.91s
```

---

## Prometheus-Compatible Live Metrics

Beyond the `investigate` action, the environment exposes **passive Prometheus observability** that an agent can query at any step without consuming an action slot — exactly as SREs query Grafana dashboards during incidents.

### Text scrape (Prometheus exposition format)
```bash
curl -H "X-Session-ID: $SID" $BASE_URL/prometheus/metrics
# HELP irt_error_rate HTTP error rate fraction 0.0-1.0
# TYPE irt_error_rate gauge
irt_error_rate{service="user-api",scenario="db-conn-pool-001",incident="severity_classification"} 0.13
irt_error_rate{service="postgres-primary",scenario="db-conn-pool-001",incident="severity_classification"} 0.02
...
```
Metrics **worsen automatically** with blast radius as the episode progresses — the agent observes real-time degradation, not a static snapshot.

### Instant query (PromQL selector)
```bash
curl -H "X-Session-ID: $SID" \
  "$BASE_URL/prometheus/query?query=irt_error_rate{service=\"payment-api\"}"
{
  "status": "success",
  "data": {
    "resultType": "vector",
    "result": [
      {
        "metric": {"__name__": "irt_error_rate", "service": "payment-api", ...},
        "value": [1712500000.0, "0.47"]
      }
    ]
  }
}
```

### Range query — time-series matrix (TSDB ring buffer)
```bash
curl -H "X-Session-ID: $SID" \
  "$BASE_URL/prometheus/query_range?query=irt_error_rate&start=0&end=9999999999"
{
  "status": "success",
  "data": {
    "resultType": "matrix",
    "result": [
      {
        "metric": {"__name__": "irt_error_rate", "service": "postgres-primary", ...},
        "values": [
          [1712500000.0, "0.12"],
          [1712500001.3, "0.19"],
          [1712500002.7, "0.31"]
        ]
      }
    ]
  }
}
```
The ring buffer holds up to **64 samples per service** (one per episode step). Metrics worsen with blast radius as the episode progresses — the matrix response shows the real degradation trend and lets agents detect early warning signals.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/prometheus/metrics` | GET | Prometheus text scrape (`?fmt=json` for JSON) |
| `/prometheus/query?query=<selector>` | GET | Instant query — standard Prometheus JSON envelope (`resultType: vector`) |
| `/prometheus/query_range?query=<selector>&start=<ts>&end=<ts>` | GET | Range query — Prometheus matrix response backed by TSDB ring buffer (64 samples / service) |

Agents using `prometheus-api-client` or `requests-prom` work against these endpoints unchanged.

---

## Two Inference Files — Why?

There are intentionally **two** inference files that serve different purposes. They are NOT duplicates:

| File | Purpose | Used by |
|------|---------|---------|
| **`inference.py`** (root) | Competition-mandatory entry point. Standalone script with `[START]/[STEP]/[END]` structured stdout format, HTTP client calling the running server, env var credentials, exit code 0. | OpenEnv evaluator, `python inference.py` |
| **`baseline/inference.py`** | Reusable Python module. Supports `rules` and `llm` modes, `--direct` flag (in-process, no server needed), CLI args. Imported by unit tests via `from baseline.inference import run_all_tasks`. | Unit tests, `python -m baseline.inference` |

**The competition spec only runs `inference.py` at the root.** The `baseline/` module exists so tests can validate environment logic without needing a running HTTP server — which makes tests fast, hermetic, and runnable in CI without starting Uvicorn.

---

## Architecture

```
openEnv/
├── .github/
│   └── workflows/
│       └── ci.yml                # CI: tests + Docker E2E + lint on push/PR
├── inference.py                  # Root inference script (competition-compliant, [START]/[STEP]/[END] logs)
├── app.py                        # FastAPI server with all endpoints + WebSocket + web UI
├── openenv.yaml                  # OpenEnv metadata spec
├── Dockerfile                    # Single-stage container definition (non-root, health check, python:3.12-slim)
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

## Special Design Inventions

These design choices distinguish this environment from typical benchmark toys:

### 1. Dynamic Blast Radius

The environment state *worsens in real-time* as the agent delays. Every step increases error rates, fills queues, and degrades metrics — revealed when the agent investigates a service. An agent that wastes steps on red herring services will see dramatically worse metrics when it finally investigates the correct service. This mirrors real production incidents where the damage compounds.

```
Step 1 → auth-service CPU: 67%,  memory: 45%
Step 3 → auth-service CPU: 78%,  memory: 62%    (still not investigated)
Step 6 → auth-service CPU: 93%,  memory: 97%    (OOMKill imminent)
```

### 2. Progressive Information Disclosure

Logs and metrics are **hidden until the agent explicitly investigates** a service. The initial observation shows only alert titles — the agent must choose which services to investigate, and each investigation costs a step. This prevents trivial pattern-matching and forces genuine reasoning about symptom-cause chains.

```json
// Before investigate:
"logs": {},  "metrics": {}

// After INVESTIGATE auth-service:
"logs": {"auth-service": [{"timestamp": "14:03:11", "level": "ERROR", ...}]},
"metrics": {"auth-service": {"memory_percent": 97, "cpu_percent": 93, ...}}
```

### 3. Red Herrings

The hard task has **8 available services** but only **4 are relevant**. `cdn-static` and `postgres-primary` show completely normal metrics (no alerts, no anomalies). An agent that investigates them wastes steps and incurs a lower precision score in the grader. This tests the ability to reason about causality rather than exhaustively sampling all services.

### 4. Temporal Degradation (Scales with Difficulty)

A per-step penalty is applied automatically, scaling by task difficulty:
- Easy: `−0.005/step` (slow-burning, low urgency)
- Medium: `−0.010/step` (payment revenue impact)
- Hard: `−0.015/step` (cascading outage, every second counts)

This creates a real pressure to be efficient — an agent that diagnoses correctly but takes 20 steps scores lower than one that diagnoses in 6.

### 5. 7 Scenario Variants (Anti-Memorization)

Each task has multiple scenario variants to prevent agents from memorizing a single scenario:

| Task | Variants | Distinct Incidents |
|------|----------|-------------------|
| Easy | 2 | DB connection pool exhaustion, Elasticsearch disk full |
| Medium | 2 | Redis session eviction / payment failure, Worker pool OOM / memory leak |
| Hard | 3 | Auth memory leak cascading outage, K8s HPA eviction loop / node pressure, PostgreSQL split-brain / pgbouncer misconfig |

The variant is selected by seed (`variant_seed % len(variants)`). Seed 0 always returns the primary scenario for reproducibility. Evaluation under random seeds tests true generalization.

### 6. Content-Aware Reasoning Bonus

The reward function reads the agent's free-text `reasoning` field and pattern-matches for relevant keywords. An agent that writes `"redis eviction causing session token loss"` gets a +0.01 bonus vs one that writes `"checking this service"`. This incentivizes agents to explain their reasoning — improving interpretability as a side effect.

### 7. Multi-Dimensional Graders

Each grader measures **orthogonal dimensions** independently, not just binary success:

| Task | Dimensions | Example |
|------|-----------|---------|
| Easy | 3 | severity_accuracy, investigation_quality, efficiency |
| Medium | 5 | severity_accuracy, investigated_root_cause_service, diagnosis_accuracy, remediation_quality, efficiency |
| Hard | 8 | severity_accuracy, diagnosis_accuracy, remediation_quality, escalation_quality, communication, investigation_thoroughness, investigation_precision, efficiency |

An agent that identifies the root cause perfectly but escalates to the wrong team loses only the `escalation_quality` component — it still gets full credit on diagnosis. This gives richer gradient signal for agent training.

### 8. Rich Actionable Grader Feedback

Graders return detailed, actionable feedback with ✓/~/✗ prefixes explaining *why* each component scored as it did and *what the correct action was*:

```
✓ Root cause correctly identified: auth-service v3.1.0 introduced an unbounded in-memory token cache causing OOMKill and cascading failures across all auth-dependent services.
✓ Comprehensive remediation: rolled back auth-service AND scaled order-service to drain the 15k+ message backlog.
✗ No escalation. This is a P1 cascading outage — escalate to platform-team (urgent) and auth-team (owns the buggy deployment).
~ Investigation spread too wide. cdn-static and postgres-primary are red herrings — normal metrics, no alerts. Focus on auth-dependent services.
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

## Curriculum Learning

Tasks are designed as a **3-stage curriculum** following the bootcamp principle: *start with simple variants and progressively increase complexity so the model receives non-zero reward from the very first episode.*

| Stage | Task | Difficulty | Reward Dims | Max Steps | Degradation/Step | Variants |
|-------|------|-----------|-------------|-----------|-----------------|----------|
| 1 | `severity_classification` | Easy | 3 | 10 | 0.005 | 2 |
| 2 | `root_cause_analysis` | Medium | 5 | 15 | 0.010 | 2 |
| 3 | `full_incident_management` | Hard | 8 | 20 | 0.015 | 3 |

The rule-based baseline scores **0.99 / 0.99 / 0.93** on stages 1-3 respectively, confirming that even a minimal agent receives strong reward signal from day one. Query the `/curriculum` endpoint for machine-readable stage metadata.

---

## Process Supervision

This environment implements **per-step shaped rewards** — every action receives immediate feedback, not just a binary end-of-episode signal. This is a prerequisite for training agents with RL using intermediate reward:

| Component | Value | Signal |
|-----------|-------|--------|
| `relevant_investigation` | +0.06 | Positive immediately when the right service is investigated |
| `irrelevant_investigation` | −0.02 | Negative feedback on wasted steps |
| `duplicate_investigation` | −0.03 | Penalises redundant actions |
| `correct_classification` | +0.15 | Rewards correct P-level assignment |
| `correct_diagnosis` | +0.15 | Rewards identifying the real root cause |
| `correct_remediation` | +0.12 | Rewards applying the right fix |
| `escalation_quality` | +0.08 | Rewards notifying the expected teams |
| `communication` | +0.03 | Rewards status updates to any channel |
| `reasoning_bonus` | +0.005–0.02 | **Process token supervision** — agent's free-text `reasoning` field is scored; mentioning relevant services or causal keywords earns higher bonus |
| `temporal_degradation` | −0.005 to −0.015/step | Urgency pressure; scales with task difficulty |

The `reasoning_bonus` component implements lightweight **thinking-token supervision**: an agent that writes `"redis eviction is causing session token loss"` gets +0.02 vs +0.005 for uninformative reasoning. This incentivises the agent to explain its reasoning at every step, not just at the end.

---

## Reward Hacking Mitigations

Several design choices directly prevent the agent from exploiting the reward function:

1. **12 independent reward components** — optimising one component (e.g., always escalating) does not help any other. An agent must genuinely perform the correct action in the correct sequence to maximise total reward.

2. **7 scenario variants** (`variant_seed`) — the agent cannot memorise correct answers for a single scenario. Evaluation with random seeds (`variant_seed = random.randint(0, 99)`) tests generalisation.

3. **Deterministic grader** — scoring is keyword-matched and rule-based (no neural judge that can be exploited with prompt injection or adversarial text).

4. **Red herring services** — in the hard task, 4 of 8 services are irrelevant. Investigating them incurs `−0.02` and reduces `investigation_precision` in the final grade. An agent that investigates everything scores lower than one that investigates strategically.

5. **Temporal degradation** — rewards decay per step. An agent that repeats actions (e.g., re-investigating known services) scores significantly lower.

6. **Negative rewards for invalid / duplicate / wrong actions** — the reward function actively penalises bad behaviour, not just withholds positive reward.

7. **Score clamped to (0.01, 0.99)** — prevents degenerate all-zero or all-one gradients during training.

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
- [x] `ENABLE_WEB_INTERFACE=true` set in `Dockerfile` — visual testing at `/web`
- [x] `/curriculum` endpoint exposes ordered learning stages
- [x] Curriculum progression: easy (3 dims) → medium (5 dims) → hard (8 dims)
- [x] Process supervision: `reasoning_bonus` scores agent thinking tokens per step
- [x] Reward hacking mitigated: 12 independent components, 7 variants, deterministic grader

---

## License

MIT

