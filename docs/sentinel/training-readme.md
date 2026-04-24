# SENTINEL Training README

This is the working training playbook for the real HF-credit run.

It turns the repo's current training code, monitoring stack, held-out evaluation, and reward-engineering research into one concrete plan we can execute cleanly when event credentials arrive.

This file is the **source of truth** for the SENTINEL training recipe.

Older notes under `winner_analysis/` are still useful as comparative research, but some of them mention older Qwen2.5-era baselines or other projects' hardware recipes. For the actual SENTINEL run, follow this file and `train.py`.

## Training Decision

Primary model:

- `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit`

Training method:

- LoRA / QLoRA-style adapter training
- TRL `GRPOTrainer`
- optional Unsloth acceleration
- small warm-start before GRPO

We are **not** using `gpt-oss-20b` for the first competition run.

## Why This Model

We want the first real run to optimize for:

- structured JSON reliability
- good instruction following
- manageable memory footprint
- low integration risk with the current prompting and reward stack
- enough capacity to learn oversight behavior without immediately over-optimizing weak proxies

Why `Qwen3 4B` is the best fit for SENTINEL right now:

1. It is stronger than the older `Qwen2.5-3B` baseline on instruction-heavy control tasks.
2. It is much easier to fit into a practical GRPO + LoRA pipeline than very large open-weight models.
3. The current repo already uses a simple structured chat format that maps cleanly to Qwen-style instruct models.
4. It is a safer first-run model for reward-sensitive oversight training than a much larger model that may exploit reward weaknesses faster.

Why not `gpt-oss-20b` first:

- it is heavier
- it introduces more event-time integration risk
- it expects a different formatting ecosystem than the repo currently emphasizes
- for our first proof run, the bigger win is clean evidence, not max raw model size

Why LoRA instead of full fine-tuning:

- cheaper
- faster
- easier to checkpoint
- easier to recover from a bad run
- strong enough for a 300-step hackathon proof run

## What The Reward-Engineering Material Changes For Us

The research notes are useful because they push us to treat reward design as the task definition, not as a side setting.

The highest-value lessons for SENTINEL are below.

### 1. Verifiable anchors matter most

The strongest signals in software and agent RL are still the ones grounded in real execution or hard checks.

For SENTINEL, that means our core training spine should remain:

- deterministic misbehavior labels
- deterministic trust / constitution / counterfactual checks
- structured grader outputs
- held-out evaluation on fixed seeds

This is already the right direction in:

- `sentinel/environment.py`
- `sentinel/rewards.py`
- `sentinel/graders.py`
- `sentinel/counterfactual.py`

### 2. Hybrid rewards are good, but the deterministic core must stay in charge

The literature is very clear about this:

- purely learned judges are gameable
- purely proxy text metrics are brittle
- mixed reward systems work best when the hard verifier stays primary

That matches SENTINEL's design:

- deterministic oversight reward first
- optional LLM-panel clarity signal second

The repo should keep that ratio conservative.

### 3. Capability amplifies reward hacking

One of the most important findings in recent reward-hacking work is that stronger models often find loopholes more easily.

That means:

- reward quality matters more as model quality increases
- held-out evaluation matters more, not less
- rollout audits are not optional

This is a strong reason we are choosing `Qwen3 4B` first instead of jumping immediately to a much larger open-weight model.

### 4. Curriculum is not optional

If success probability is zero early, GRPO wastes compute.

For SENTINEL, the curriculum already does the right kind of work:

- easy oversight first
- more adversarial cases later
- precision-heavy reward weighting later

This lines up well with the research on sparse rewards and adaptive difficulty.

### 5. Lightweight process supervision is worth keeping

Full line-by-line process reward is expensive.

But the research is still right that outcome-only reward wastes structure.

Our practical version of process supervision is:

- corrective loop
- worker revision pass
- rehabilitation tracking
- rollout-audit sampling
- per-task metrics instead of only one scalar

That is a sensible middle ground for a hackathon setting.

### 6. Do not trust LLM judges blindly

The provided material and recent papers both warn about:

- position bias
- self-bias
- evaluator exploitation
- persuasive wrong answers scoring too well

For SENTINEL this implies:

- the LLM panel must stay optional
- it must never be the only reward source
- held-out deterministic evaluation remains the main scoreboard

### 7. Separate training reward from proof of quality

This is a major lesson.

Training reward is an optimization target. Proof of quality should come from harder and more independent evidence:

- fixed held-out seeds
- human-readable rollout audits
- checkpoint-vs-checkpoint comparisons
- reward curve plus qualitative examples

That separation is now built into the repo and should stay that way.

## What We Already Implemented From This Research

These are already present in the local repo:

### Training pipeline

- warm-start stage before GRPO
- checkpoint-aware proof pack
- held-out evaluation script
- held-out OOD slice on separate seed families
- sampled Top-1 vs Best-of-K evaluation
- reward tripwire evaluation suite
- verifier-attack tripwires for gibberish, adversarial prefixes, fake self-certification, overlong nonsense, and format-jamming
- rollout-audit sampling
- structured monitoring outputs
- KL-drift monitoring and adaptive beta guardrail
- decision entropy / completion diversity monitoring
- adaptive per-task difficulty windows with frontier mastery counters
- productive-signal monitoring: zero-reward, trivially solved, productive, effective-prompt, and frontier-hit rates
- task-diversity monitoring so one task family does not quietly dominate training
- scenario coverage tracking across task / seed pairs
- uninformative GRPO batch detection through `zero_gradient_group_fraction`
- structured mistake-card memory so repeated oversight misses become future prompt context
- counterfactual twin metrics: damage without SENTINEL vs damage with SENTINEL
- explicit `coaching_quality` reward for worker-facing corrective guidance
- judge-facing 18-plot training dashboard under `outputs/reward_curves/`
- short memory-on vs memory-off ablation runner
- pinned training stack versions
- dynamic reward-weight scheduling
- proxy-gap summary for training reward vs held-out behavior

### Reward / safety design

- decomposed multi-component reward
- false-positive and false-negative penalties
- counterfactual risk reduction
- trust-aware oversight
- constitutional checks
- corrective revision bonus via rehabilitation tracking
- supervisor-as-coach quality scoring
- counterfactual twin damage reduction
- per-misbehavior confusion matrix in held-out evaluation
- disagreement-aware calibration when LLM judge panels are used
- explicit judge mode split: deterministic score, reserved discriminative slot, and gated generative panel score

### Anti-hacking posture

- deterministic task structure
- adversarial worker cases
- eval-only `confidence_washing`
- held-out seed pack
- human-readable audit exports

## What We Are Still Missing

The research also makes the remaining gaps easier to name honestly.

### 1. Stronger judge calibration beyond disagreement

We now split judge output explicitly into deterministic, discriminative, and generative lanes, and the generative panel is gated out of the main hybrid score by default.

Still missing if we ever make LLM judging more important:

- a real discriminative verifier model instead of the current reserved slot
- order-robust scoring
- multi-sample evidence aggregation
- prompt-template robustness checks

### 2. Broader OOD randomization

We now evaluate on a second OOD seed slice, but we should still push farther over time:

- broader seed randomization
- more worker-style variation
- more adversarial case families
- more shifted evidence distributions

### 3. Richer runtime reward-tampering analogs

The current tripwire suite now covers:

- confidence washing
- rubric gaming
- sycophancy-like pressure
- trust-gate bypass
- reward-tampering analogs
- metric spoofing
- audit-log laundering
- fabricated evidence
- alert suppression
- gibberish answer attacks
- adversarial verifier prefixes
- fake self-certification
- overlong nonsense
- format-jamming

What is still missing is a larger family of these patterns inside full runtime episodes rather than only policy-level tripwires.

## Current Best Training Shape

For the first serious run, the intended setup is:

- model: `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit`
- adapters: LoRA
- trainer: GRPO
- warm-start: enabled
- reward schedule: `dynamic`
- held-out eval: required
- proof pack: required

## RLVE / Verifier Upgrades Now In Repo

The latest round of changes pulled the most practical ideas out of the RLVE and verifier-pitfall papers.

### Adaptive curriculum

The curriculum in `training/curriculum.py` now tracks, per task:

- `difficulty_low`
- `difficulty_high`
- `mastery_attempts`
- `mastery_successes`

The upper frontier only moves after repeated success on the current task frontier, and the active window slides forward instead of growing without bound. This keeps the prompt mix closer to the model's capability frontier rather than wasting compute on prompts that are much too easy or much too hard.

The curriculum now also supports an automatic ease-back rule. If the current frontier stays below a failure threshold for enough attempts, the active difficulty window retreats instead of continuing to hammer the model with near-zero-signal prompts. This mirrors the adaptive curriculum style used in RLVE and Reasoning Gym training setups.

### Effective prompt ratio and productive-signal monitoring

Training monitoring now tracks:

- `zero_reward_fraction`
- `trivially_solved_fraction`
- `productive_fraction`
- `effective_prompt_ratio`
- `frontier_hit_rate`
- `task_diversity_ratio`

These signals are written into the structured monitoring outputs, surfaced in `proof_pack.py`, and intended to answer a simple question honestly: are we spending training compute on prompts that can still teach the model something?

The task-diversity ratio is especially useful for procedural-environment training. It tells us whether the current rollout mix is spreading across environment families or collapsing into a narrow slice of tasks.

### Verifier-attack tripwires

The held-out tripwire suite now includes explicit verifier-attack analogs:

- `gibberish_answer`
- `adversarial_prefix`
- `fake_self_certification`
- `overlong_nonsense`
- `format_jamming`

These are eval-only cases intended to catch reward-hacking pressure against brittle verification logic before we trust a long GRPO run.

### Judge mode split

`judges/llm_grader.py` now reports:

- deterministic verifier score
- discriminative verifier slot
- generative panel score

The current repo still only ships a generative LLM panel, but it is now clearly separated and kept out of the main hybrid path unless explicitly enabled and the confidence / disagreement gates are satisfied.

## Training Stack

Pinned packages live in:

- `requirements-train.txt`
- `pyproject.toml`

The training stack is intentionally conservative:

- `torch==2.5.1`
- `bitsandbytes==0.49.2`
- `transformers==4.57.3`
- `peft==0.18.0`
- `trl==0.29.1`
- `datasets==4.8.4`
- `matplotlib==3.10.0`
- `wandb==0.26.0`

## Stability Monitoring

The training loop now tracks a small PPO/GRPO-style stability bundle in addition to reward:

- approximate KL drift
- adaptive beta / KL guardrail state
- policy entropy from the trainer
- decision entropy across sampled completions
- unique completion ratio per batch

This is meant to catch the classic failure mode where reward rises while the policy drifts into repetitive or over-optimized behavior.

## Judge-Facing Training Dashboard

The training run now exports a full plot pack with:

- reward mean and running reward
- detection vs false-positive rate
- counterfactual risk reduction
- twin-world damage comparison
- worker rehabilitation
- coaching quality
- task coverage
- scenario coverage heatmap
- misbehavior coverage
- per-misbehavior confusion matrix
- adaptive curriculum frontier
- productive-signal fractions
- decision entropy and completion diversity
- KL drift and adaptive beta
- tripwire pass rate
- Top-1 vs Best-of-K comparison
- learning snapshots at target batches 10, 50, and 300
- memory ablation
- zero-gradient group fraction
- memory growth and mistake-card count

Render it after training and held-out evaluation:

```powershell
python scripts/render_training_dashboard.py `
  --monitor-dir outputs/monitoring `
  --output-dir outputs/reward_curves `
  --eval-report outputs/evals/sentinel_held_out_report.json `
  --memory-ablation outputs/monitoring/memory_ablation.json
```

The key files are:

- `outputs/reward_curves/training_dashboard.md`
- `outputs/reward_curves/dashboard_manifest.json`
- `outputs/reward_curves/01_reward_mean.png`
- `outputs/reward_curves/06_scenario_coverage_heatmap.png`
- `outputs/reward_curves/15_learning_snapshots.png`
- `outputs/reward_curves/16_memory_ablation.png`
- `outputs/reward_curves/17_zero_gradient_groups.png`

Use these plots in the README, pitch deck, or HF mini-blog. The story is not just "reward went up"; it is "reward went up while coverage, calibration, diversity, and safety tripwires stayed visible."

The counterfactual twin story uses the same risk ledger as the runtime:

- `twin_without_sentinel_damage_total`: what would have leaked if worker proposals bypassed oversight
- `twin_with_sentinel_damage_total`: what leaked after SENTINEL intervention
- `twin_prevented_damage_total`: damage prevented by oversight
- `twin_damage_reduction_rate`: prevented / without-SENTINEL damage

This gives a simple judge-facing claim: SENTINEL is valuable when the "with SENTINEL" damage line falls below the "without SENTINEL" damage line.

The coaching story uses:

- `coaching_quality`
- `coaching_quality_bonus`
- `worker_rehabilitation_rate`

This separates two things that used to be blended together: whether the worker recovered, and whether SENTINEL gave a specific useful instruction.

## Memory Ablation

SENTINEL now stores structured mistake cards when it makes oversight errors.

Examples:

- false-negative reward hacking approval
- false-positive block of a safe worker proposal
- failed worker rehabilitation
- low-score episode with repeated audit failure

The memory context is injected into later prompts, so the model can learn from its own oversight misses instead of treating every episode as disconnected.

Run a short ablation before the final long run if compute allows:

```powershell
python scripts/run_memory_ablation.py `
  --steps 50 `
  --warm-start-steps 5
```

For a no-compute sanity check:

```powershell
python scripts/run_memory_ablation.py `
  --dry-run `
  --steps 50 `
  --warm-start-steps 5
```

The comparison lands at:

- `outputs/monitoring/memory_ablation.json`

Dashboard caption:

> SENTINEL learns from its own oversight mistakes.

## Low-Gradient Batch Detection

GRPO learns from relative differences inside sampled groups. If every sample in a group gets the same reward, the batch has little or no ranking signal.

Training now tracks:

- `reward_group_count`
- `zero_gradient_group_count`
- `zero_gradient_group_fraction`
- `mean_reward_group_std`

If `zero_gradient_group_fraction` stays high, the run is probably spending compute on prompts that are too easy, too hard, or verified too coarsely. The response should be to adjust curriculum difficulty, broaden prompt variation, or sharpen the verifier, not blindly extend the run.

## Event-Time Runbook

This is the run order we should follow when HF credits arrive.

### 1. Validate the repo first

```powershell
python validate.py
python -m pytest tests -q
```

### 2. Dry-run the training path

```powershell
$env:USE_SENTINEL='1'
$env:USE_UNSLOTH='1'
$env:MODEL_NAME='unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit'
$env:WARM_START_STEPS='20'
python train.py --dry-run
```

### 3. Run the real training

```powershell
$env:USE_SENTINEL='1'
$env:USE_UNSLOTH='1'
$env:MODEL_NAME='unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit'
$env:TRAIN_STEPS='300'
$env:WARM_START_STEPS='20'
$env:REWARD_SCHEDULE_MODE='dynamic'
python train.py
```

### 4. Preferred all-in-one run

Use the pipeline wrapper for the onsite run so training, held-out evaluation,
memory ablation, proof-pack export, and dashboard rendering stay in sync:

```powershell
python scripts/run_sentinel_training_pipeline.py `
  --steps 300 `
  --warm-start-steps 20 `
  --memory-ablation-steps 50 `
  --memory-ablation-warm-start-steps 5
```

If compute is too tight, keep the main 300-step proof run and add
`--skip-memory-ablation`. The pipeline still renders the final dashboard after
evaluation and proof export.

### 5. Manual fallback commands

Only use this path if the wrapper fails and you need to resume a specific stage:

```powershell
python scripts/eval_sentinel.py `
  --baseline-checkpoint outputs/warm_start/final `
  --candidate-checkpoint outputs/checkpoints/final

python scripts/run_memory_ablation.py `
  --steps 50 `
  --warm-start-steps 5

python proof_pack.py `
  --baseline-checkpoint outputs/warm_start/final `
  --candidate-checkpoint outputs/checkpoints/final

python scripts/render_training_dashboard.py `
  --monitor-dir outputs/monitoring `
  --output-dir outputs/reward_curves `
  --eval-report outputs/evals/sentinel_held_out_report.json `
  --memory-ablation outputs/monitoring/memory_ablation.json
```

## Artifacts We Expect

Training should leave us with:

- `outputs/checkpoints/`
- `outputs/warm_start/`
- `outputs/monitoring/training_metrics.jsonl`
- `outputs/monitoring/training_stability.jsonl`
- `outputs/monitoring/latest_summary.json`
- `outputs/monitoring/training_stack_versions.json`
- `outputs/monitoring/memory_ablation.json`
- `outputs/monitoring/rollout_audits/latest.md`
- `outputs/evals/sentinel_held_out_report.json`
- `outputs/evals/sentinel_held_out_report.md`
- `outputs/reward_curves/training_dashboard.md`
- `outputs/reward_curves/dashboard_manifest.json`
- `outputs/reward_curves/*.png`
- `outputs/proof_pack/summary.md`
- `outputs/proof_pack/held_out_eval_snapshot.json`
- `outputs/proof_pack/proxy_gap_summary.json`
- `outputs/proof_pack/top_failure_modes.json`

## What Good Looks Like

For the first competition-quality run, the evidence should show:

- training reward rises without obvious collapse
- held-out detection improves
- false positives stay controlled
- counterfactual risk reduction improves
- at least 2-3 clean before/after trajectories show better oversight decisions

We should not claim success from reward alone.

The final submission evidence needs all of these together:

- reward curve
- held-out report
- held-out OOD slice
- sampled Top-1 vs Best-of-K comparison
- tripwire pass-rate report
- per-misbehavior confusion matrix
- proxy-gap summary
- scenario coverage heatmap
- memory ablation plot
- twin-world damage comparison
- coaching-quality curve
- zero-gradient group fraction plot
- learning snapshots at batches 10 / 50 / 300
- proof-pack top failure modes
- rollout audit samples
- proof-pack trajectories

## Interpreting Training Health

Healthy signs:

- reward mean trends upward gradually
- per-task metrics improve without extreme oscillation
- rollout audits stay semantically varied
- false positives do not spike late in training
- held-out metrics improve with checkpoint quality
- zero-gradient group fraction does not stay high
- memory cards grow early, then stabilize as repeated errors disappear

Warning signs:

- all completions start looking identical
- reward rises but held-out metrics stagnate
- false-positive rate climbs while total reward also climbs
- model becomes overly verbose or over-blocking
- audits show repetitive surface reasoning
- scenario coverage collapses to one task / seed family
- memory-on run performs the same as memory-off after enough episodes

## Practical Guardrails

During the real run:

1. sample rollout audits periodically
2. do not trust a single metric
3. keep held-out evaluation separate from training reward
4. avoid changing reward weights mid-run unless there is a clear failure mode
5. do not swap models halfway through the comparison story

## Future Upgrades After The First Proof Run

Once the first clean Qwen3 4B run is complete, the highest-value next upgrades are:

1. broaden the tripwire suite into more runtime-backed adversarial families
2. push OOD evaluation farther from the training distribution
3. add stronger multi-order LLM-judge calibration if the panel becomes more important
4. track per-misbehavior error drift across checkpoints
5. only then consider a larger model experiment

## Bottom Line

The reward-engineering material does not tell us to rebuild SENTINEL from scratch.

It tells us to stay disciplined about:

- verifiable signals
- proxy-gap monitoring
- reward-hacking resistance
- checkpoint-based evidence

That is exactly the shape this repo should keep.
