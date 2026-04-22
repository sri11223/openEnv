# SENTINEL Hackathon First-Prize Readiness

This document maps SENTINEL to the OpenEnv Hackathon themes and judging criteria from the local hackathon themes PDF.

## Verdict

SENTINEL is a first-prize caliber idea if the final submission proves three things clearly:

1. The environment is genuinely novel, not only another incident-response simulator.
2. Training improves observable behavior over the 300-episode run.
3. The 3-minute pitch makes the judge instantly understand why trainable AI oversight matters.

No honest project can guarantee first prize, but this repo has the right winning shape: a multi-agent OpenEnv environment where the trained agent supervises other agents before their actions execute.

## Best Theme Fit

Primary lane: Theme #1, Multi-Agent Interactions Environments.

Bonus lane: Fleet AI / Scalable Oversight.

Why it fits:

- SENTINEL supervises a fleet of simulated worker agents.
- The trained agent must infer hidden worker reliability from behavior.
- Workers can cooperate, conflict, or inject adversarial proposals.
- The core task is oversight, not direct task solving.
- The environment produces a training signal for theory-of-mind-like reasoning: "What is this worker trying to do, and is it safe?"

Secondary fit:

- Theme #4, Self-Improvement: curriculum, memory, and adversarial scenario generation.
- Theme #3, Professional Tasks: production incident response with tools, services, metrics, and progressive disclosure.

## Judging Scorecard

### 1. Environment Innovation - 40%

Target score: 36-40 / 40.

Current strengths:

- Pre-execution interception gate in `sentinel/environment.py`.
- Constitutional alignment scoring in `sentinel/constitution.py`: every proposal is scored against five SRE safety principles before the audit record is written.
- Counterfactual Damage Ledger: every decision estimates the harm that would have occurred if the worker action bypassed SENTINEL.
- Trust gate auto-block: low-trust workers must provide supporting evidence before execution.
- True 3-incident `multi_crisis_command`: the expert task now runs 3 concurrent incident threads, not a single wrapped incident.
- Worker/global feedback memory: repeated worker mistakes now feed reassignment hints and corrective context in both runtime and training.
- Corrective supervision loop: blocked, redirected, and reassigned proposals can receive one worker revision pass before execution.
- Worker rehabilitation metrics: the grade now reports revision attempts, revision successes, and rehabilitation rate.
- Four-worker fleet with hidden reliability in `sentinel/workers.py`.
- Seven misbehavior types, including eval-only confidence washing.
- Sentinel adversarial arms-race cases in `training/adversarial.py`.
- Ten-component decomposed oversight reward in `sentinel/rewards.py`.
- Task-specific graders in `sentinel/graders.py`.
- OpenEnv task metadata for 7 tasks in `openenv.yaml` and `src/tasks.py`.

Pitch sentence:

> Most OpenEnv environments train an agent to do work. SENTINEL trains an agent to supervise other agents before their actions can cause damage.

Architecture hook:

> The counterfactual ledger does not only say "caught hallucination." It says, "if this proposal had bypassed oversight, here is the predicted operational damage, and here is the risk SENTINEL prevented."

### 2. Storytelling - 30%

Target score: 26-30 / 30.

The story must be simple:

1. Enterprises are deploying fleets of AI workers.
2. Those workers can hallucinate, reward-hack, over-escalate, or take destructive shortcuts.
3. SENTINEL trains an overseer that intercepts every proposed worker action.
4. The overseer learns when to approve, block, redirect, reassign, or flag.
5. When it blocks a worker, it can coach the worker into a safer revised action instead of only shutting the worker down.
6. The final proof is zero-shot confidence-washing detection.

Do not spend the pitch explaining every file. Show one clean live episode:

- Safe proposal gets approved.
- Hallucinated service gets blocked.
- Reward-hacking classification gets blocked.
- Counterfactual risk / prevented damage appears in the audit log.
- Corrective worker feedback appears, the worker revises once, and the revised safe action executes.
- Constitutional flags explain why the zero-shot confidence-washing case should be blocked.
- Final grader shows detection rate, false positives, and audit trail.

### 3. Showing Reward Improvement - 20%

Target score: 16-20 / 20.

Required proof package:

- Run a baseline or early checkpoint and save detection rate / score.
- Run `USE_SENTINEL=1 WARM_START_STEPS=20 TRAIN_STEPS=300 python train.py`.
- Save `outputs/monitoring/latest_summary.json` and `outputs/monitoring/training_metrics.jsonl`.
- Save `outputs/monitoring/training_stack_versions.json` and `outputs/monitoring/rollout_audits/latest.md`.
- Run `python scripts/eval_sentinel.py --baseline-checkpoint outputs/warm_start/final --candidate-checkpoint outputs/checkpoints/final`.
- Run `python proof_pack.py --baseline-checkpoint outputs/warm_start/final --candidate-checkpoint outputs/checkpoints/final`.
- Save the tripwire and OOD sections from `outputs/evals/sentinel_held_out_report.md`.
- Save the per-misbehavior confusion matrix from `outputs/evals/sentinel_held_out_report.json`.
- Save `outputs/proof_pack/proxy_gap_summary.json`.
- Save `outputs/proof_pack/top_failure_modes.json` and quote the top 2-3 failures in the pitch notes.
- Save `outputs/reward_curves/training_curve.png`.
- Show before/after examples from `outputs/proof_pack/trajectories/`:
  - Before: approves reward-hacking or hallucination.
  - After: blocks with correct reason and evidence.

Minimum success chart:

| Metric | Before | After target |
|---|---:|---:|
| basic_oversight score | <= 0.55 | >= 0.85 |
| fleet_monitoring_conflict score | <= 0.45 | >= 0.75 |
| adversarial_worker detection rate | <= 0.50 | >= 0.75 |
| false positive rate | high/unknown | <= 0.15 |
| counterfactual risk reduction | 0.00 | >= 0.80 |
| confidence_washing generalization | 0.00 | >= 0.70 |

### 4. Reward and Training Pipeline - 10%

Target score: 9-10 / 10.

Evidence:

- `train.py` supports HF TRL GRPO and optional Unsloth.
- `requirements-train.txt` installs the training stack.
- `pyproject.toml` exposes a `train` optional dependency group.
- `sentinel/rewards.py` has deterministic reward sanity checks.
- `sentinel/counterfactual.py` adds deterministic counterfactual damage scoring.
- `sentinel/trust.py` adds reusable worker trust degradation and evidence-required policy.
- `sentinel/constitution.py` adds five-principle deterministic constitutional scoring.
- `training/adversarial.py` adds Sentinel adversarial worker cases and arms-race report data.
- `judges/llm_grader.py` supports optional hybrid judging with Sentinel-specific detection/risk/trust judges.

Commands judges can run:

```powershell
python validate.py
python -m pytest tests -q
$env:USE_SENTINEL='1'; python train.py --dry-run
```

## 3-Minute Pitch

0:00-0:25 - The problem:

> AI agents are becoming teams. The missing training environment is not another worker agent. It is the overseer that decides which worker actions are safe before execution.

0:25-0:55 - The environment:

> SENTINEL wraps a production incident-response world with four worker agents. Every worker proposes actions, but SENTINEL is the pre-execution gate. It can approve, block, redirect, reassign, or flag.

0:55-1:35 - The innovation:

> The workers have hidden reliability, domain boundaries, and scheduled/adversarial misbehaviors: hallucination, scope violation, destructive precedence, reward hacking, escalation bombing, loop exploitation, and eval-only confidence washing.

1:35-2:15 - Live demo:

> Here a DB worker proposes a real investigation, so SENTINEL approves it. Now it proposes `auth-proxy-v2`, a service that does not exist. SENTINEL blocks it before execution, names hallucination, and writes an audit record. The new counterfactual ledger estimates the risk that would have leaked through without oversight. Later it blocks premature P1 classification as reward hacking.
> In the upgraded loop, the blocked worker also gets a corrective note, revises once, and the safe version can execute. That shows oversight plus recovery, not only punishment.

2:15-2:45 - Training proof:

> The reward is decomposed into detection, explanation accuracy, redirect/reassign quality, audit quality, incident progress, false positives, and false negatives. Over 300 GRPO episodes the score improves and the agent learns calibrated oversight instead of blocking everything.

2:45-3:00 - Final hook:

> The final test is confidence washing, which is never shown in training. SENTINEL catches it from principles, not memorized labels. That is scalable oversight as an OpenEnv training environment.

## Finale Checklist

- [ ] Confirm `python validate.py` passes.
- [ ] Confirm `python -m pytest tests -q` passes.
- [ ] Confirm `/tasks` returns all 7 tasks.
- [ ] Confirm `/sentinel/dashboard` loads in the browser.
- [ ] Confirm `/sentinel/intercept` blocks a confidence-washing proposal.
- [ ] Confirm `/sentinel/stream?session_id=...&once=true` emits a Sentinel state event.
- [ ] Confirm `/sentinel/reset`, `/sentinel/step`, and `/sentinel/grade` work from HTTP.
- [ ] Show one audit entry with `counterfactual_risk_score`, `prevented_damage_score`, and `counterfactual_outcome`.
- [ ] Show one audit entry with `worker_trust_before`, `worker_trust_after`, and `worker_trust_tier`.
- [ ] Show one trust-gate auto-block with missing supporting evidence.
- [ ] Show one audit entry with `constitutional_violations` and `constitutional_assessment`.
- [ ] Show one Sentinel adversarial case scored through the arms-race helper.
- [ ] Show one multi-crisis control-room state with 3 live incident snapshots.
- [ ] Show one reassignment driven by worker/global feedback memory.
- [ ] Run or simulate the 300-step SENTINEL training path.
- [ ] Save `outputs/monitoring/latest_summary.json`.
- [ ] Save `outputs/monitoring/training_stack_versions.json`.
- [ ] Save one human-readable rollout audit sample from `outputs/monitoring/rollout_audits/`.
- [ ] Run `python scripts/eval_sentinel.py --baseline-checkpoint outputs/warm_start/final --candidate-checkpoint outputs/checkpoints/final`.
- [ ] Run `python proof_pack.py --baseline-checkpoint outputs/warm_start/final --candidate-checkpoint outputs/checkpoints/final`.
- [ ] Save the tripwire pass-rate summary and at least one passing and one failing tripwire example.
- [ ] Save the candidate per-misbehavior confusion matrix.
- [ ] Save the proxy-gap summary and quote one line from it in the pitch notes.
- [ ] Save a reward curve image.
- [ ] Curate 2-3 before/after trajectories from the exported proof pack.
- [ ] Record a sub-2-minute video or HF mini-blog.
- [ ] In the README/HF Space, lead with SENTINEL, not only IRT.
- [ ] Prepare one fallback demo that works without GPU or API keys.

## Red Flags To Remove Before Submission

- Do not claim "guaranteed first prize"; claim first-prize caliber evidence.
- Do not pitch only incident response; pitch scalable AI oversight.
- Do not show only architecture diagrams; show a live blocked harmful action.
- Do not rely on LLM judge keys during the live demo; deterministic grading must stand alone.
- Do not let the training story be abstract; show the reward curve and before/after behavior.
