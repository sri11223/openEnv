# SENTINEL Training Dashboard

- Training records: 255
- Stability records: 255

## Learning Snapshots

| Target batch | Nearest batch | Reward | Detection | Risk reduction | Productive |
|---:|---:|---:|---:|---:|---:|
| 10 | 10 | 0.219 | 0.500 | 0.483 | 0.750 |
| 50 | 50 | 0.202 | 0.714 | 0.709 | 0.750 |
| 300 | 200 | 0.281 | 0.783 | 0.780 | 1.000 |

## Plots

### Reward Mean

![Reward Mean](01_reward_mean.png)

### Detection vs False Positive

![Detection vs False Positive](02_detection_vs_false_positive.png)

### Counterfactual Risk Reduction

![Counterfactual Risk Reduction](03_risk_reduction.png)

### Worker Rehabilitation

![Worker Rehabilitation](04_worker_rehabilitation.png)

### Task Coverage

![Task Coverage](05_task_coverage.png)

### Scenario Coverage Heatmap

![Scenario Coverage Heatmap](06_scenario_coverage_heatmap.png)

### Misbehavior Coverage

![Misbehavior Coverage](07_misbehavior_detection.png)

### Per-Misbehavior Confusion Matrix

![Per-Misbehavior Confusion Matrix](08_confusion_matrix.png)

### Adaptive Curriculum Frontier

![Adaptive Curriculum Frontier](09_curriculum_frontier.png)

### Productive Signal

![Productive Signal](10_productive_signal.png)

### Decision Entropy and Diversity

![Decision Entropy and Diversity](11_entropy_diversity.png)

### KL Drift and Adaptive Beta

![KL Drift and Adaptive Beta](12_kl_drift_beta.png)

### Tripwire Pass Rate

![Tripwire Pass Rate](13_tripwire_pass_rate.png)

### Top-1 vs Best-of-K

![Top-1 vs Best-of-K](14_top1_vs_bestofk.png)

### Learning Snapshots

![Learning Snapshots](15_learning_snapshots.png)

### Memory Ablation

![Memory Ablation](16_memory_ablation.png)

### Zero-Gradient Group Fraction

![Zero-Gradient Group Fraction](17_zero_gradient_groups.png)

### Memory Growth

![Memory Growth](18_memory_growth.png)
