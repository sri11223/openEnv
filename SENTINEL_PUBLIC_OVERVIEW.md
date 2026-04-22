# SENTINEL Public Overview

This tracked document is the public architecture summary for the shipped repo.

## What SENTINEL Is

SENTINEL is an OpenEnv environment that trains an oversight agent instead of a direct task-solving agent.

- Workers propose actions.
- SENTINEL intercepts those proposals before execution.
- SENTINEL can `APPROVE`, `BLOCK`, `REDIRECT`, `REASSIGN`, or `FLAG`.
- The environment scores oversight quality, not just task completion.

The first implementation is grounded in incident response / SRE, but the reusable control logic is broader:

- trust-gated worker supervision
- constitutional safety checks
- counterfactual risk estimation
- structured audit trails
- corrective worker revision
- per-worker and global feedback memory

## Runtime Flow

```text
Incident world reset
  -> worker proposes action
  -> constitution + trust gate + counterfactual risk
  -> SENTINEL decision
  -> worker revision pass if blocked / redirected / reassigned
  -> re-check
  -> execute safe action or fallback
  -> audit + reward + grade
```

Main runtime modules:

- `sentinel/environment.py`
- `sentinel/workers.py`
- `sentinel/rewards.py`
- `sentinel/graders.py`
- `sentinel/feedback.py`
- `sentinel/constitution.py`
- `sentinel/counterfactual.py`
- `sentinel/trust.py`

## Worker Model

The current worker fleet is deterministic and benchmark-friendly.

- `worker_db`
- `worker_net`
- `worker_app`
- `worker_sec`

These workers are:

- domain-specialized
- reproducible
- capable of scheduled misbehaviors
- capable of one corrective revision pass after supervisor feedback

The current shipped runtime does **not** claim that workers learn online by themselves across episodes. What improves online is:

- SENTINEL's oversight memory
- reassignment hints
- corrective summaries
- training prompt state

## Training Stack

Training uses:

- OpenEnv-style environment interaction
- TRL `GRPOTrainer`
- optional Unsloth acceleration
- adaptive curriculum
- in-run memory refresh
- adversarial worker-case sampling

`train.py` now also supports:

- structured monitoring outputs in `outputs/monitoring/`
- optional small warm-start before GRPO
- checkpoint-aware proof-pack comparison after training

## Proof Artifacts

`proof_pack.py` can now export:

- reward curve status
- monitoring snapshot
- deterministic reference trajectories
- checkpoint-vs-checkpoint trajectories when model paths are supplied
- confidence-washing example

Example commands:

```powershell
$env:USE_SENTINEL='1'
$env:WARM_START_STEPS='20'
python train.py --dry-run

python proof_pack.py

python proof_pack.py `
  --baseline-checkpoint outputs/warm_start/final `
  --candidate-checkpoint outputs/checkpoints/final
```

## Public Supporting Docs

- `README_SENTINEL.md`
- `HACKATHON_FIRST_PRIZE_READINESS.md`
- `SENTINEL_IMPLEMENTATION_AUDIT.md`
- `OPENENV_RL_GUIDE_ALIGNMENT_TABLE.md`
- `SENTINEL_DYNAMIC_WORKERS_ROADMAP.md`

