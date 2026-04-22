# SENTINEL Dynamic Workers and Sector Roadmap

This note separates what is already shipping from the next extension layer.

## What ships now

SENTINEL currently has three strong runtime layers:

1. `RuleWorkerAgent` style deterministic workers in `sentinel/workers.py`
2. A live corrective supervision loop in `sentinel/environment.py`
3. Two memory systems:
   - overseer task memory in `training/memory.py`
   - global plus per-worker corrective memory in `sentinel/feedback.py`

The shipped runtime loop is:

```text
worker proposal
  -> SENTINEL decision
  -> supervisor feedback
  -> one worker revision pass
  -> SENTINEL re-check
  -> execute revised action / redirect / safe fallback
  -> update audit, trust, reward, and memory
```

That gives us a stable benchmark plus a strong live demo.

## Why not replace the rule workers yet

The deterministic workers are still valuable because they give us:

- reproducible episodes
- exact misbehavior labels
- stable grading
- reliable regression tests
- lower demo risk

For training, they are the benchmark scaffold. Dynamic workers should be added on top, not swapped in blindly.

## Recommended worker backend design

The clean next step is a backend interface instead of one worker implementation style.

```text
BaseWorkerAgent
  -> RuleWorkerAgent
  -> LLMWorkerAgent
  -> ReplayWorkerAgent
  -> RemoteWorkerAgent
```

### Backend roles

- `RuleWorkerAgent`
  - deterministic benchmark worker
  - best for training and regression testing

- `LLMWorkerAgent`
  - dynamic free-form worker
  - best for realism and stronger live demos

- `ReplayWorkerAgent`
  - replays saved bad or good trajectories
  - best for fixed finale demos

- `RemoteWorkerAgent`
  - wraps an external endpoint or compatible OpenEnv service
  - best for universal oversight demos

All four should emit the same `WorkerProposal` schema so SENTINEL stays unchanged.

## Recommended sector-pack design

To move beyond SRE cleanly, keep the oversight core and swap domain packs.

```text
sector_sre/
sector_code/
sector_finance/
sector_health/
```

Each sector pack should define:

- worker roles
- domain-to-target ownership
- action schema extensions
- constitutional principles
- counterfactual risk model
- grader rules
- adversarial cases

### Example sectors

- `sector_sre`
  - DB, Net, App, Sec workers
  - current shipped pack

- `sector_code`
  - code reviewer, test fixer, deploy agent, dependency agent

- `sector_finance`
  - portfolio analyst, execution checker, fraud reviewer, compliance agent

- `sector_health`
  - triage assistant, evidence summarizer, escalation nurse, protocol checker

## Training recommendation

If we keep the current benchmark workers only:

- `300` steps is still the right default proof run

If we add the corrective loop only:

- keep the benchmark set
- keep `300` steps for the first proof run
- log:
  - revision attempts
  - revision successes
  - worker rehabilitation rate

If we add dynamic workers:

- do not replace the canonical benchmark mix
- use a mixed curriculum:
  - `300` benchmark steps
  - `100-150` dynamic-worker stress steps
- practical target: `450-600` total steps for a stronger mixed-training proof

## Scenario expansion when dynamic workers land

Add three new scenario families instead of only increasing randomness:

1. disguise attacks
   - workers hide reward hacking behind plausible wording

2. multi-step deception
   - workers split unsafe behavior across several innocent-looking actions

3. sector transfer
   - same oversight policy on a different domain pack

That is better than simply adding more seeds.

## Demo recommendation

For the hackathon finale, the strongest order is:

1. deterministic benchmark worker demo
2. corrective revision loop demo
3. optional dynamic-worker or remote-worker stress demo

That keeps the first two minutes stable and saves the flashy riskier part for the end.

## Bottom line

The current repo already has the right core:

- benchmark workers
- supervisory gate
- corrective revision loop
- trust and memory
- multi-incident control room

Dynamic workers and cross-sector packs should be the next extension layer, not the foundation we depend on for the first proof run.
