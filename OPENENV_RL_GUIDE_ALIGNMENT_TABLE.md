# OpenEnv RL Guide Alignment Table

This table maps the self-serve hackathon guide / FAQ topics to the shipped SENTINEL repo.

| Q# | Focus | Meaning for SENTINEL | Status | Next move |
|---:|---|---|---|---|
| 1 | What RL is for LLMs | SENTINEL learns oversight decisions from rollouts plus verifiable reward | Done | Run long checkpointed training |
| 2 | Why rewards matter | Reward is the oversight spec; wrong reward means brittle supervision | Done | Keep auditing for loopholes |
| 3 | Rewards engineering | Multi-part reward, trust, constitution, counterfactuals are first-class | Done | Recalibrate after long runs |
| 4 | RLVR vs reward model | SENTINEL is verifier-driven RLVR, not only preference scoring | Done | Keep LLM judging optional |
| 5 | Why environments matter | Oversight requires multi-step interaction, not prompt-only scoring | Done | Expand held-out evals |
| 6 | What OpenEnv is | Repo follows OpenEnv-style `reset/step/state/grade` contract | Done | Keep API surfaces stable |
| 7 | How OpenEnv works | FastAPI + typed environment + local/remote execution path | Done | Continue demo hardening |
| 8 | Where TRL and Unsloth fit | `train.py` already targets TRL and optional Unsloth | Done | Pin versions for training run |
| 9 | PPO vs GRPO | GRPO is the current post-training path in `train.py` | Done | Compare settings after credits arrive |
| 10 | Why RL is inefficient | Sparse success is handled with curriculum and corrective loop | Done | Monitor plateau risk |
| 11 | Process supervision | Step-level audits and corrective loop provide lightweight process feedback | Mostly done | Add more held-out step audits |
| 12 | Reward hacking | Reward hacking is part of the benchmark itself | Done | Keep adversarial inspections live |
| 13 | Reduce reward hacking | Layered checks, trust gate, anti-shortcut logic, audits | Done | Continue spot-checking runs |
| 14 | Curriculum learning | Adaptive curriculum is implemented and now really steers prompt sampling | Done | Validate with long run |
| 15 | Is task suitable for RL | Oversight is stepwise, verifiable, and non-trivial | Done | Keep task mix balanced |
| 16 | SFT before RL | Repo now supports a small warm-start option before GRPO | Done | Tune warm-start size after first run |
| 17 | What to monitor during RL | Structured monitoring now writes reward, detection, FP, risk, rehab metrics | Done | Build judge-facing charts |
| 18 | Fast hackathon RL strategy | Environment -> verifier -> deploy -> warm-start -> RL -> proof pack | Done | Execute the long run cleanly |
| 19 | Starter resources | Public docs and roadmap now point to concrete repo artifacts | Done | Add notebook notes after training |
| 20 | One-sentence takeaway | SENTINEL trains an agent to supervise other agents safely | Done | Keep this as the pitch line |
| 21 | What RLVR is | Verifiable oversight reward, not soft preference only | Done | None |
| 22 | What RLVE is | Adaptive environment and curriculum keep tasks informative | Mostly done | Add more procedural task generation later |
| 23 | RLVE vs RLVR | SENTINEL is strong RLVR with partial RLVE behavior | Done | Dynamic workers later |
| 24 | Why RL environments help post-training | Oversight needs action consequences and audit state | Done | None |
| 25 | Where TRL, GRPO, Unsloth fit | Stack alignment is strong | Done | Run on credits |
| 26 | Why rewards matter so much | Reward defines safety and oversight quality | Done | Keep proof artifacts honest |
| 27 | What reward engineering is | Reward, verifier, shaping, monitoring, anti-cheat all exist | Done | Tighten after observing runs |
| 28 | What reward hacking is | Explicit benchmark behavior | Done | Keep examples in demo |
| 29 | Sparse reward problem | Curriculum and warm-start reduce zero-signal regimes | Done | Measure early rollout pass rate |
| 30 | Dense reward danger | Reward remains decomposed but grounded in outcome checks | Done | Watch for over-shaping |
| 31 | Common env mistake | Repo stays closer to real SRE workflow than toy string matching | Done | Add more real-world failure cases |
| 32 | Weak verifiers | Deterministic grading beats judge-only scoring | Done | Keep hard checks primary |
| 33 | Risk of LLM-as-judge | LLM panel is optional, not the only truth source | Done | Demo deterministically |
| 34 | Tool-using env pitfall | Environment models service/incident realism, not only happy paths | Done | Add more auth/permission style failures later |
| 35 | Static task difficulty | Curriculum avoids being stuck too easy or too hard | Done | Longer run will confirm |
| 36 | Environment diversity | 4 SENTINEL tasks + adversarial cases provide moderate diversity | Mostly done | Add broader domain packs later |
| 37 | Real-world transfer failures | Audit, trust, counterfactuals keep abstraction closer to deployment reality | Done | Add external-worker eval mode later |
| 38 | Biggest reward mistake | Repo avoids single proxy reward | Done | Continue manual review |
| 39 | Start with complicated reward? | Reward is rich but still decomposed and inspectable | Done | Keep components legible |
| 40 | Conflicting reward components | Current reward is aligned around safety + progress | Mostly done | Re-check weighting after long run |
| 41 | Binary reward appeal | Deterministic catch/miss signals remain visible | Done | Use binary submetrics in pitch |
| 42 | Binary reward limits | Dense shaping exists for long-horizon oversight | Done | Monitor for exploit patterns |
| 43 | How to detect hacked reward | Monitoring outputs + proof pack + rollout inspection | Done | Use held-out checkpoint eval |
| 44 | Safe reward pattern | Hard checks first, shaping second, audits always | Done | None |
| 45 | Common GRPO mistake | Base instruct model + optional warm-start now supported | Done | Tune warm-start + RL split |
| 46 | Why RL plateaus | Curriculum and adversarial cases help; long-run evidence still needed | Mostly done | Observe actual plateau points |
| 47 | Why more RL can hurt | Repo now separates deterministic reference proof from trained proof | Done | Compare checkpoints carefully |
| 48 | Static RLVR dataset pitfall | Dynamic prompt state and adversarial cases reduce staleness | Done | Add more procedural cases later |
| 49 | Why similar GRPO runs differ | Monitoring and structured artifacts now capture more of the pipeline state | Done | Freeze seeds and versions for demo |
| 50 | Mixing many environments badly | Curriculum keeps task mixture under control | Done | Watch easy-task dominance in logs |
| 51 | Long-horizon difficulty | Multi-crisis is the expert task; corrective loop helps keep signal alive | Done | Measure completion depth on long run |
| 52 | Monitoring only reward | Monitoring now includes detection/FP/risk/rehab, not reward alone | Done | Add screenshot-ready charts |
| 53 | Safest RL post-training pipeline | Warm-start -> validate -> small run -> inspect -> full run path now exists | Done | Execute it with credits |
| 54 | Best hackathon project shape | Stepwise, verifiable, medium-horizon oversight is a strong match | Done | None |
| 55 | What to avoid building | Repo avoids purely subjective or judge-only tasks | Done | Keep demo grounded |
| 56 | Best debugging order | Environment, verifier, scripted baselines, dry-run, proof pack | Done | Keep this discipline during final run |
| 57 | One rule to remember | Break your own reward before the model does | Done | Continue adversarial case design |
| 58 | Strong references | Public docs now map the main design choices to concrete repo artifacts | Done | Add video learnings after review |
| 59 | Unsloth recipe guidance | Repo is compatible with that training style; warm-start is now available | Done | Pick the smallest stable recipe first |

## Current Highest-Value Next Steps

| Priority | Move | Why |
|---:|---|---|
| 1 | Run checkpointed training with HF credits | Converts architecture into evidence |
| 2 | Use checkpoint-aware `proof_pack.py` | Gives honest before/after trajectories |
| 3 | Export monitoring screenshots | Makes the training story judge-friendly |
| 4 | Tune warm-start size | Improves early rollout formatting and sample efficiency |
| 5 | Add dynamic worker backend later | Strong future upgrade, not today’s blocker |

