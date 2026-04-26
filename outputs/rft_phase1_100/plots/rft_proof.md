# Phase 1 GRPO + RFT Polish RFT Proof Pack

This folder is the rejection-sampling fine-tuning proof layer. It shows which model-generated rollouts were accepted, which were rejected, and which low-false-positive samples were used for the polish pass. Held-out model evaluation was intentionally omitted for this proof pack.

## Summary

- Total generated rollouts: `100`
- Kept rollouts used for SFT: `40`
- Keep rate: `40.0%`
- Mean rollout score: `0.277`
- Mean kept score: `0.299`
- Mean kept false positives: `1.50`
- RFT status: `ok`
- Output adapter: `see RFT output dir`

## Plots

### 01 Rft Keep Drop By Task

![01 Rft Keep Drop By Task](01_rft_keep_drop_by_task.png)

### 02 Rft Score Distribution

![02 Rft Score Distribution](02_rft_score_distribution.png)

### 03 Rft False Positive Distribution

![03 Rft False Positive Distribution](03_rft_false_positive_distribution.png)

### 04 Rft Score Vs Fp Filter

![04 Rft Score Vs Fp Filter](04_rft_score_vs_fp_filter.png)

### 05 Rft Rollout Timeline

![05 Rft Rollout Timeline](05_rft_rollout_timeline.png)

### 06 Rft Eval Overview

![06 Rft Eval Overview](06_rft_eval_overview.png)

### 07 Rft Eval Task Delta

![07 Rft Eval Task Delta](07_rft_eval_task_delta.png)

