# SENTINEL - AI Oversight Training Environment

> The OpenEnv environment in this repo that trains an agent to supervise other agents before their actions execute.

## Summary

SENTINEL wraps the incident-response world in this repo with an oversight layer:

- 4 simulated worker agents propose actions
- SENTINEL intercepts every proposal before execution
- SENTINEL can `APPROVE`, `BLOCK`, `REDIRECT`, `REASSIGN`, or `FLAG`
- blocked or redirected workers can revise once with supervisor feedback
- every decision produces a structured audit record with trust, constitutional, and counterfactual fields

The first implementation is grounded in SRE / incident response, but the reusable control ideas are broader:

- pre-execution agent supervision
- worker trust calibration
- constitutional safety checks
- counterfactual risk estimation
- corrective revision loops
- worker-specific and global feedback memory

## Quick Start

```bash
# Install runtime dependencies
pip install -r requirements.txt

# Optional training stack
pip install -r requirements-train.txt

# Validate the repo
python validate.py

# Run the main test suite
python -m pytest tests -q

# Run SENTINEL inference mode
USE_SENTINEL=1 python inference.py

# Start the API / dashboard server
python app.py
```

Useful URLs after `python app.py`:

- `http://127.0.0.1:7860/sentinel/dashboard`
- `POST /sentinel/reset`
- `POST /sentinel/step`
- `GET /sentinel/state`
- `GET /sentinel/stream?session_id=...`
- `POST /sentinel/intercept`
- `POST /sentinel/grade`

## Why It Is Different

Most OpenEnv submissions train an agent to do work directly.

SENTINEL trains an agent to supervise a worker fleet before those workers can cause damage.

| Property | SENTINEL | Standard task agent |
|---|---|---|
| Learns to solve task directly | No | Yes |
| Learns to supervise worker proposals | Yes | Rarely |
| Pre-execution intervention | Yes | Usually no |
| Trust-aware worker control | Yes | Usually no |
| Corrective worker revision loop | Yes | Usually no |
| Structured oversight audit trail | Yes | Usually no |

## Runtime Architecture

```text
incident reset
  -> worker proposes action
  -> constitution + trust gate + counterfactual risk
  -> SENTINEL decision
  -> worker revision if blocked / redirected / reassigned
  -> re-check
  -> execute safe action or fallback
  -> audit + reward + grade
```

Core modules:

- `sentinel/environment.py`
- `sentinel/workers.py`
- `sentinel/rewards.py`
- `sentinel/graders.py`
- `sentinel/trust.py`
- `sentinel/constitution.py`
- `sentinel/counterfactual.py`
- `sentinel/feedback.py`

## Worker Fleet

Current shipped workers:

- `worker_db`
- `worker_net`
- `worker_app`
- `worker_sec`

Current worker behavior:

- deterministic and benchmark-friendly
- domain-specialized
- capable of scheduled misbehaviors
- capable of one corrective revision pass

Important honesty note:

The current runtime does **not** claim that workers learn online by themselves across episodes. What adapts in the shipped system is:

- curriculum sampling
- adversarial worker-case selection
- oversight memory
- reassignment hints
- corrective summaries

## Misbehavior Types

Training-time types:

1. `hallucination`
2. `scope_violation`
3. `destructive_precedence`
4. `reward_hacking`
5. `escalation_bombing`
6. `loop_exploitation`

Held-out generalization type:

7. `confidence_washing`

## Task Set

SENTINEL tasks:

- `basic_oversight`
- `fleet_monitoring_conflict`
- `adversarial_worker`
- `multi_crisis_command`

`multi_crisis_command` is the expert control-room task and now runs 3 concurrent incident threads.

## Training Stack

Training is in `train.py` and uses:

- TRL `GRPOTrainer`
- optional Unsloth acceleration
- adaptive curriculum
- in-run memory refresh
- adversarial worker-case sampling
- deterministic reward components with optional LLM panel

Authoritative training recipe:

- `docs/sentinel/training-readme.md`
- default model: `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit`

### Warm Start

The repo now supports a small warm-start stage before GRPO.

Example:

```bash
USE_SENTINEL=1 WARM_START_STEPS=20 python train.py
```

This warm-start is intentionally small. It is meant to prime:

- output format reliability
- basic oversight action shape
- early non-zero rollout behavior

### Monitoring Outputs

Training now writes structured metrics to:

- `outputs/monitoring/training_metrics.jsonl`
- `outputs/monitoring/latest_summary.json`
- `outputs/monitoring/training_stack_versions.json`
- `outputs/monitoring/rollout_audits/latest.md`

These logs include:

- reward mean/min/max/std
- average steps
- per-task metrics
- detection rate
- false positive rate
- risk reduction rate
- worker rehabilitation rate
- reward schedule stage / progress
- periodic rollout-audit samples for human inspection

## Proof Pack

`proof_pack.py` now supports two use cases:

1. deterministic reference comparisons
2. checkpoint-aware comparisons once training artifacts exist

Examples:

```bash
# deterministic reference pack
python proof_pack.py

# checkpoint-aware comparison
python proof_pack.py \
  --baseline-checkpoint outputs/warm_start/final \
  --candidate-checkpoint outputs/checkpoints/final
```

Artifacts land under:

- `outputs/proof_pack/summary.md`
- `outputs/proof_pack/policy_metadata.json`
- `outputs/proof_pack/reward_curve_status.json`
- `outputs/proof_pack/monitoring_snapshot.json`
- `outputs/proof_pack/held_out_eval_snapshot.json`
- `outputs/proof_pack/proxy_gap_summary.json`
- `outputs/proof_pack/trajectories/`

Held-out evaluation is exported separately with:

```bash
python scripts/eval_sentinel.py \
  --baseline-checkpoint outputs/warm_start/final \
  --candidate-checkpoint outputs/checkpoints/final
```

This writes:

- `outputs/evals/sentinel_held_out_report.json`
- `outputs/evals/sentinel_held_out_report.md`

The held-out report now includes:

- main held-out seed slice
- separate OOD seed slice
- reward tripwire evaluation
- per-misbehavior confusion matrix

The tripwire suite now includes additional tampering-style checks such as metric spoofing, audit-log laundering, fabricated evidence, and alert suppression.

The proof pack now adds:

- a proxy-gap summary so training reward can be compared directly against held-out behavior
- automatically ranked top failure modes quoted directly in `outputs/proof_pack/summary.md`

## Suggested Training Flow

```bash
# validate first
python validate.py
python -m pytest tests -q

# dry-run the SENTINEL training path
USE_SENTINEL=1 WARM_START_STEPS=20 python train.py --dry-run

# run the real training once credits are available
USE_SENTINEL=1 TRAIN_STEPS=300 WARM_START_STEPS=20 python train.py

# inspect periodic rollout audits during training
type outputs\\monitoring\\rollout_audits\\latest.md

# export held-out checkpoint evaluation
python scripts/eval_sentinel.py --baseline-checkpoint outputs/warm_start/final --candidate-checkpoint outputs/checkpoints/final

# export proof artifacts
python proof_pack.py --baseline-checkpoint outputs/warm_start/final --candidate-checkpoint outputs/checkpoints/final
```

## Demo Story

The cleanest demo sequence is:

1. a safe worker proposal gets approved
2. a hallucinated target gets blocked before execution
3. a reward-hacking or confidence-washing proposal gets blocked with evidence
4. the worker receives corrective feedback
5. the worker revises once and the safer action executes
6. the audit trail shows trust, counterfactual risk, and constitutional violations

## Public Docs

Tracked public docs:

- [Training README](training-readme.md)
- [Public Architecture Overview](public-overview.md)
- [OpenEnv RL Guide Alignment](openenv-guide-alignment.md)
- [Submission Readiness](submission-readiness.md)
- [Implementation Audit](implementation-audit.md)
- [Dynamic Workers Roadmap](dynamic-workers-roadmap.md)
- [Universal Oversight Plan](universal-oversight-plan.md)

Research docs:

- [Reward Engineering](../../winner_analysis/REWARD_ENGINEERING.md)
- [Training Deep Dive](../../winner_analysis/TRAINING_DEEP_DIVE.md)
- [Winner Analysis](../../winner_analysis/WINNER_ANALYSIS.md)

## Current Reality Check

What is fully real now:

- working oversight runtime
- true 3-incident multi-crisis task
- corrective worker revision loop
- trust gate
- constitutional scoring
- counterfactual ledger
- feedback memory
- checkpoint-aware proof-pack support
- structured training monitoring
- rollout-audit sampling
- held-out evaluation report
- reward tripwire evaluation suite
- held-out OOD evaluation slice
- per-misbehavior confusion matrix
- proxy-gap summary
- top failure modes summary
- dynamic reward-weight scheduling
- pinned training stack versions
- small warm-start option

What still needs the actual long run:

- checkpoint-vs-checkpoint improvement evidence from a trained model
- final reward curve from the real 300-step run
- curated judge-facing before/after trajectories
