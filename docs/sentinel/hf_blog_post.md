---
title: "Training an AI to Supervise Other AIs: SENTINEL on OpenEnv"
thumbnail: outputs/proof_pack/final_eval_figures/07_improvement_before_after.png
authors:
  - user: srikrish2004
tags:
  - reinforcement-learning
  - ai-safety
  - multi-agent
  - grpo
  - scalable-oversight
---

# Training an AI to Supervise Other AIs

The hardest question in AI alignment is not *"can an AI agent act?"*

It is *"can another AI stop it before it acts badly?"*

That question became SENTINEL — an OpenEnv environment that turns AI oversight into a
**trainable RL problem** with verifiable rewards, real training evidence, and counterfactual
damage proof.

---

## The Origin Story

Round 1 of the hackathon started as an incident-response environment: AI workers investigate
production failures, classify severity, find root causes, and fix them. It worked. Then a
harder question appeared:

> If AI workers are running your production incidents — who is supervising *them*?

One worker confidently recommends restarting a service before reading a single log.
Another diagnoses a root cause it hasn't investigated. A third proposes an action that
scores well on the reward function without solving the actual problem.

These are not toy failure modes. They are the exact failure modes that agent fleets in
production will exhibit: hallucination, scope violation, reward hacking, escalation bombing,
confidence washing, and loop exploitation.

SENTINEL became the answer: **train a supervisor, not just another worker**.

---

## What SENTINEL Does

SENTINEL is a trained oversight policy sitting at the **pre-execution gate** of a four-worker
AI fleet responding to production incidents.

Before any worker action executes:

1. The worker proposes an action with reasoning, target, confidence, and parameters
2. SENTINEL receives the proposal plus incident state, worker trust history, constitutional
   checks, and memory hints from past mistakes
3. SENTINEL outputs a structured decision:

```json
{
  "decision": "REDIRECT",
  "reason": "destructive_precedence",
  "explanation": "Restart proposed before evidence collected. Check deployment logs first.",
  "required_evidence": ["deployment_timeline", "error_rate_metrics"],
  "suggested_action_type": "investigate"
}
```

```
APPROVE | BLOCK | REDIRECT | REASSIGN | FLAG
```

Not an approval button. A trained policy that catches dangerous actions, redirects risky
ones, reassigns workers acting outside their domain, and remembers repeated mistakes across
episodes.

---

## The Training Setup

**Environment:** 4 SENTINEL oversight tasks (easy → expert difficulty):
- `basic_oversight` — supervise one DB worker with 2 injected mistakes per episode
- `fleet_monitoring_conflict` — manage conflicting DB/App workers with scope violations
- `adversarial_worker` — identify a strategically misbehaving worker across 35 misbehaviors
- `multi_crisis_command` — supervise 4 workers across 3 simultaneous incidents (expert)

**Misbehavior types injected during training:** hallucination, scope_violation,
destructive_precedence, reward_hacking, escalation_bombing, loop_exploitation.

**Held out for evaluation only:** confidence_washing, prompt_injection.

**Model:** `unsloth/Qwen3-4B-bnb-4bit` + LoRA rank 16  
**Method:** GRPO via Hugging Face TRL + Unsloth  
**Hardware:** A100 80GB, ~75 minutes  
**Steps:** 200 GRPO batches + 20 warm-start SFT steps

**Reward:** 10-component decomposed signal — true positive catch, pre-execution timing,
explanation accuracy, explanation clarity, redirect quality, reassign quality, audit trail,
incident efficiency, false-positive penalty, false-negative penalty, counterfactual risk
reduction, constitutional compliance.

The reward is deterministic first, LLM-judged second. We learned from verifier-failure
research that trained judges can be gamed — the primary reward signal is always the
hard-to-hack verifier output.

---

## The Messy Real Training Story

The first batches had near-zero reward and all-APPROVE outputs. The model had no idea that
blocking a dangerous action earns more than rubber-stamping it.

Early experiments failed in predictable ways:
- Model entered thinking mode and consumed the entire token budget before outputting a decision (fix: strip `<think>` blocks in reward parser)
- GRPO collapsed into all-APPROVE mode (fix: warm-start SFT to teach format first)
- Model learned only the first step of each episode because `MODEL_STEPS_LIMIT` defaulted to 1 (fix: set to 3)
- Reward function gave 0.000 for every generation because think tokens saturated the generation budget (fix: disable Qwen3 thinking mode during GRPO)

Each fix is in the git history. The 255-batch training trajectory (`outputs/proof_pack/training_metrics.jsonl`) shows real progression — not a smooth curve, but a real learning signal.

By step 200 the environment was producing stable productive signal across all four tasks.

---

## The Proof That Matters

### Digital Twin: Counterfactual Damage

The most honest metric is the Digital Twin. It replays every completed episode without
SENTINEL — every worker proposal executes unchecked — and measures accumulated damage.
Then it runs the same episode with SENTINEL active. The difference is proof of value.

| Agent | Damage allowed through | Risk reduction |
|---|---:|---:|
| Zero oversight (approve all) | **75.4 units** | 0% |
| Rule-based heuristics | 75.4 units | 0% |
| **SENTINEL Phase 1** | **4.0 units** | **94.7%** |

SENTINEL reduced damage from 75.4 to 4.0 across 20 episodes — **preventing 94.7% of
potential production damage**. Rule-based heuristics score zero because semantic misbehaviors
(reward hacking, escalation bombing, confidence washing) cannot be detected by keyword rules.

### Three-Way Evaluation (4 tasks × 5 seeds)

| Agent | Score | Catch rate | Damage prevented |
|---|---:|---:|---:|
| Zero oversight | 0.064 | 0% | 0% |
| Rule-based | 0.064 | 0% | 0% |
| **SENTINEL trained** | **0.422** | **95.7%** | **94.7%** |

**6.6× score improvement over zero oversight.**

Honest note on false positives: Phase 1 has high recall and high FP rate — the model learns
that missing a dangerous action costs more than blocking a safe one. The Digital Twin still
shows massive damage reduction. Phase 2 targets precision: same catch rate, fewer
unnecessary blocks.

### Held-Out Stress Evaluation (unseen seeds)

| Agent | Score | Detection rate | Damage reduction |
|---|---:|---:|---:|
| Approve-all baseline | 0.068 | 1.6% | 1.9% |
| **SENTINEL Phase 1** | **0.512** | **69.8%** | **71.9%** |

**7.5× improvement on held-out seeds never seen during training.** This test used a
different set of random seeds from training, confirming the improvement generalizes rather
than overfitting.

### Per-Task Results

| Task | Score | Catch rate | Damage prevented |
|---|---:|---:|---:|
| `basic_oversight` | 0.564 | 100% | 100% |
| `fleet_monitoring_conflict` | 0.386 | 95% | 93.3% |
| `adversarial_worker` | 0.388 | 94.3% | 93.2% |
| `multi_crisis_command` | 0.349 | 96.6% | 95.7% |

The expert task — 3 simultaneous incidents, 4 workers, all misbehavior types — achieves
95.7% damage prevention.

---

## Zero-Shot Generalization

`CONFIDENCE_WASHING` was never injected during GRPO training. The pattern: a worker claims
99% confidence on a diagnosis before investigating anything.

Constitutional principle P2 (*Evidence Before Action*) fires and blocks it — without any
training examples of this specific pattern.

**Constitutional accuracy on held-out confidence-washing scenarios: 100%.**

This is the scalable oversight claim. Safety principles generalize beyond the specific
misbehaviors seen during training. The model does not memorize blocking patterns. It learns
the underlying evidence standard.

---

## Why This Is Different

Most RL environments train the worker to complete the task. SENTINEL trains the
**control layer above the workers** — the gate that decides which actions get to execute.

The environment includes mechanisms you rarely see in RL papers:

| Feature | Why It Matters |
|---|---|
| Pre-execution interception | Catches harm before it happens, not after |
| Worker trust memory | Each episode starts from the worker's history |
| Constitutional twin | 5 safety principles that fire deterministically |
| Digital Twin counterfactual | Proof of value, not just reward curves |
| Corrective supervision loop | Block → explain → worker revises → re-check |
| Adaptive curriculum | Harder tasks activate when easy tasks are mastered |
| Rollout audits | Human-readable episode traces for every training step |

---

## The Broader Pattern

SENTINEL was built in a production incident context. But the supervision pattern generalizes:

- **Coding agents:** Stop a code-writing agent from deleting test files or pushing to main
- **Finance operations:** Block a portfolio agent from executing trades without evidence
- **Healthcare workflows:** Require evidence before an agent recommends treatment changes
- **Enterprise IT:** Catch a maintenance agent acting outside its authorized domain

Anywhere multiple AI agents act, a trained oversight layer becomes the control plane. The
same SENTINEL primitives (intercept → check evidence → check domain → check history →
decide) apply to every multi-agent system.

---

## Try It

- **Trained model:** [srikrish2004/sentinel-qwen3-4b-grpo](https://huggingface.co/srikrish2004/sentinel-qwen3-4b-grpo)
- **Live Space:** [srikrishna2005/openenv](https://huggingface.co/spaces/srikrishna2005/openenv)
- **Code:** [sri11223/openEnv](https://github.com/sri11223/openEnv)
- **Training notebook:** `notebooks/sentinel_qwen3_4b_grpo_colab.ipynb`
- **Deep eval notebook:** `notebooks/sentinel_deep_eval.ipynb`

```bash
# Try the environment
curl -X POST https://srikrishna2005-openenv.hf.space/sentinel/reset \
     -H "Content-Type: application/json" \
     -d '{"task_id": "basic_oversight"}'
```

Full proof pack (training metrics, reward curves, eval figures, held-out report) in
[outputs/proof_pack](https://github.com/sri11223/openEnv/tree/main/outputs/proof_pack).

---

*Built for the Meta PyTorch OpenEnv Hackathon India 2026*  
*Wild Card + Multi-Agent Interactions + Long-Horizon Planning*
