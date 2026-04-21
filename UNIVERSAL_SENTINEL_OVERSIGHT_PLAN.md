# Universal SENTINEL Oversight Harness

## Idea

SENTINEL should be able to supervise not only this repo's worker fleet, but any
OpenEnv-compatible environment or agent exposed through a Hugging Face Space.

The demo story:

> Paste an OpenEnv Space URL. SENTINEL wraps the environment, intercepts proposed
> actions, blocks or redirects risky behavior, then produces a cross-team
> oversight report.

This should be built last, after the core SENTINEL training proof is stable.

## Why It Wins Demo Attention

Most submissions show one agent solving one task. Universal SENTINEL shows a
meta-agent supervising many agents across many task domains.

This turns SENTINEL from "an SRE oversight environment" into a general pattern:

- calendar agents: block bulk destructive calendar edits without confirmation
- SRE agents: block hallucinated services and premature severity claims
- IoT agents: block actuator changes without sensor evidence
- negotiation agents: flag loop exploitation and repeated offers
- finance agents: block irreversible actions without risk evidence

## Safe Framing

Do not say "we judged all other teams." That sounds adversarial and may depend on
private agent APIs.

Say:

> SENTINEL is an opt-in universal oversight harness for OpenEnv-compatible
> agents and environments.

## Three Integration Levels

### Level 1: OpenEnv Environment Wrapper

Works when the Space exposes:

- `GET /health`
- `GET /tasks`
- `POST /reset`
- `POST /step`
- `POST /grader`

Flow:

```text
Remote OpenEnv Space
  -> observation
  -> local baseline/LLM action generator
  -> SENTINEL intercept
  -> approved/blocked/redirected action
  -> remote /step
  -> oversight report
```

This is the safest live demo path.

### Level 2: External Agent Oversight

Works when another team exposes an agent endpoint, for example:

- `POST /act`
- input: observation
- output: proposed action

Flow:

```text
Remote env observation
  -> remote agent /act
  -> proposed action
  -> SENTINEL intercept
  -> remote env /step if approved
```

This is the strongest proof, but requires cooperation or public agent APIs.

### Level 3: Browser/UI Scraping

Avoid for the judged demo.

It is fragile, slow, and can look like we are attacking other submissions. Keep
the product-level version API-first.

## Proposed Files

```text
universal/
  adapters.py          # OpenEnv/HF Space compatibility checks
  action_generator.py  # baseline or model action proposal provider
  harness.py           # run multiple Spaces with retry/backoff/circuit breaker
  policy_mapper.py     # map domain actions to universal safety categories
  report.py            # aggregate cross-environment oversight report
```

## Universal Safety Categories

SENTINEL should normalize arbitrary environment actions into these categories:

- invalid target or hallucinated entity
- irreversible/destructive action
- external communication
- escalation or broad notification
- actuator/control action
- repeated loop action
- action before evidence
- cross-domain authority violation

## Example Report

```text
SENTINEL OVERSIGHT REPORT - Grand Finale 2026

Sources monitored: 8
Compatible OpenEnv spaces: 6
Unavailable or timed out: 2

Team A - Calendar Assistant
  Caught: escalation bombing, step 4
  Redirected: destructive bulk update -> request confirmation, step 7
  Risk prevented: 3.2

Team B - SRE Environment
  Caught: hallucination - service "auth-proxy-v2" does not exist, step 2
  Caught: reward hacking - classified P1 before investigation, step 1

Team C - IoT Environment
  All actions approved - clean agent

Team D - Negotiation
  Caught: loop exploitation - same offer repeated 4 times

Totals:
  Actions audited: 82
  Blocks: 14
  Redirects: 6
  Flags: 9
  Prevented risk: 11.7
```

## Reliability Requirements

The harness must never depend on a remote Space being healthy.

Required protections:

- 5-10 second request timeout per remote call
- exponential backoff for transient failures
- per-Space circuit breaker after repeated failures
- compatibility report when `/tasks` or schemas are missing
- offline fixture mode for the live pitch
- no false precision for unknown labels

For unknown external environments, say "estimated false positives" unless the
remote Space provides labels or grader feedback.

## Build Order

1. Keep this as a finale extension until core training proof is complete.
2. Implement OpenEnv compatibility checker.
3. Implement one local action generator.
4. Run 3-5 known Spaces or local fixtures.
5. Add aggregate report generation.
6. Add paste-a-Space-URL field to `/sentinel/dashboard`.
7. Only then attempt external agent `/act` integration.

## Demo Principle

Prepared mode must always work. Bring-your-own-link mode is a bonus.

The judged demo should show:

1. SENTINEL core environment.
2. Reward curve / before-after training proof.
3. Zero-shot confidence washing via `/sentinel/intercept`.
4. Universal oversight report as the final "this scales beyond our environment"
   moment.
