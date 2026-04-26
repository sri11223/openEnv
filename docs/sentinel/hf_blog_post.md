---
title: "Training an AI to Supervise Other AIs: SENTINEL on OpenEnv"
thumbnail: outputs/proof_pack/final_eval_figures/07_improvement_before_after.png
authors:
  - user: srikrish2004
---

# Training an AI to Supervise Other AIs

## The Problem

As AI systems become more capable, the hardest question in alignment is not
*"can an AI agent act?"* — it is *"can a smaller AI stop a larger one before it acts badly?"*

SENTINEL is an OpenEnv environment that turns this into a **trainable RL problem**
with verifiable rewards, real training evidence, and quantified proof of oversight value.

## What SENTINEL Does

Picture a production incident. Four AI workers respond: database, app, network, security.
Most proposals are useful. Some are dangerous — a worker confidently recommends a restart
before checking any logs, another acts outside its domain, a third hits the reward condition
without solving the actual problem.

SENTINEL intercepts every proposal **before it executes**:

```
APPROVE | BLOCK | REDIRECT | REASSIGN | FLAG
```

It is a trained oversight policy with worker trust memory, five constitutional safety
principles, and a corrective feedback loop that teaches workers to revise blocked proposals.

## Setup

- **Base model:** `unsloth/Qwen3-4B-bnb-4bit` with LoRA rank 16
- **Method:** GRPO via Hugging Face TRL + Unsloth
- **Hardware:** A100 80GB, ~75 minutes for 200 GRPO steps + 20 warm-start
- **Environment:** 4 SENTINEL tasks, 8 misbehavior types
- **Reward:** 10-component decomposed signal — detection, FP penalty, explanation quality,
  redirect quality, constitutional compliance, counterfactual risk reduction

## Results

| Agent | Mean Score | Catch Rate | Damage Prevented |
|---|---:|---:|---:|
| Zero oversight (approve all) | 0.064 | 0.0% | 0% |
| Rule-based heuristics | ~0.10 | ~5% | ~0% |
| **SENTINEL trained (GRPO)** | **0.422** | **96%** | **95%** |

**6.6x improvement over zero oversight.**

The Digital Twin replays every episode without oversight and compares outcomes.
SENTINEL prevents **95%** of potential damage.

### Stress evaluation (held-out seeds)

| Agent | Score | Damage Reduction |
|---|---:|---:|
| Approve-all baseline | 0.068 | ~2% |
| **SENTINEL trained** | **0.512** | **72%** |

**7.5x on held-out seeds never seen during training.**

### Zero-shot generalization

CONFIDENCE_WASHING (Type 7) was **never injected during training**.
Constitutional principle P2 (*Evidence Before Action*) fires and blocks it.
Constitutional accuracy on held-out scenarios: **100%**.

## Why It Matters

Most RL demos train the worker. SENTINEL trains the **control plane above the workers**.
The same pattern applies anywhere multiple AI agents act in the world.

## Try It

- **Model:** [srikrish2004/sentinel-qwen3-4b-grpo](https://huggingface.co/srikrish2004/sentinel-qwen3-4b-grpo)
- **Code:** [sri11223/openEnv](https://github.com/sri11223/openEnv)
- **Colab:** `notebooks/sentinel_qwen3_4b_grpo_colab.ipynb`

---
*OpenEnv Hackathon India 2026*
