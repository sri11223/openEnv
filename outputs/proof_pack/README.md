# Sentinel Training Proof Pack
Generated: 2026-04-25 15:38 UTC

- final/                        — LoRA adapter (~50MB, load with PEFT on unsloth/Qwen3-4B-bnb-4bit)
- training_metrics.jsonl        — per-batch metrics
- reward_curves/                — reward curve PNG
- rollout_audits/               — sampled rollouts
- train_run.log                 — full training log

Final reward (mean): ~0.30   |   200 GRPO steps   |   75 min on A100 80GB
4x improvement vs baseline (0.07 -> 0.30)
