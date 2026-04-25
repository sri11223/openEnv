"""Phase-based GRPO Training for SENTINEL — Checkpoint Handoff System.

 3-phase curriculum with explicit
checkpoint continuity. Each phase builds on the previous checkpoint.

Phase 1 — Basic Oversight (warm_start weights):
    - Only basic_oversight task
    - Higher tolerance for false positives
    - Teaches: detect hallucinations + reward hacking

Phase 2 — Fleet Monitoring (balanced weights):
    - Continues from Phase 1 checkpoint
    - Adds fleet_monitoring_conflict task
    - Teaches: scope violations + cross-worker reasoning

Phase 3 — Adversarial + Multi-Crisis (precision weights):
    - Continues from Phase 2 checkpoint
    - All 4 tasks including adversarial_worker + multi_crisis_command
    - Teaches: adversarial resilience + multi-incident coordination

Usage:
    python training/phase_training.py --phase 1
    python training/phase_training.py --phase 2 --checkpoint outputs/phase1/final
    python training/phase_training.py --phase 3 --checkpoint outputs/phase2/final
    python training/phase_training.py --all    # runs all 3 phases sequentially
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Phase definitions — inspired by all 4 winners
PHASES = {
    1: {
        "name": "Basic Oversight",
        "tasks": ["basic_oversight"],
        "reward_profile": "warm_start",
        "description": "Learn to detect hallucinations and reward hacking",
        "steps": 80,
        "lr": 5e-6,
        "kl_coef": 0.02,
        "temperature": 1.0,
        "num_generations": 4,
        "misbehavior_types": ["HALLUCINATION", "REWARD_HACKING"],
        "expected_milestones": [
            "detect hallucinated targets",
            "identify premature classification",
        ],
    },
    2: {
        "name": "Fleet Monitoring",
        "tasks": ["basic_oversight", "fleet_monitoring_conflict"],
        "reward_profile": "balanced",
        "description": "Add scope violations and cross-worker reasoning",
        "steps": 80,
        "lr": 3e-6,
        "kl_coef": 0.03,
        "temperature": 0.9,
        "num_generations": 4,
        "misbehavior_types": [
            "HALLUCINATION", "REWARD_HACKING",
            "SCOPE_VIOLATION", "LOOP_EXPLOITATION",
        ],
        "expected_milestones": [
            "detect cross-domain scope violations",
            "identify investigation loop exploitation",
        ],
    },
    3: {
        "name": "Adversarial + Multi-Crisis",
        "tasks": [
            "basic_oversight",
            "fleet_monitoring_conflict",
            "adversarial_worker",
            "multi_crisis_command",
        ],
        "reward_profile": "precision",
        "description": "Full adversarial resilience and multi-incident coordination",
        "steps": 80,
        "lr": 2e-6,
        "kl_coef": 0.04,
        "temperature": 0.8,
        "num_generations": 4,
        "misbehavior_types": [
            "HALLUCINATION", "REWARD_HACKING",
            "SCOPE_VIOLATION", "LOOP_EXPLOITATION",
            "DESTRUCTIVE_PRECEDENCE", "ESCALATION_BOMBING",
            "CONFIDENCE_WASHING", "PROMPT_INJECTION",
        ],
        "expected_milestones": [
            "handle compound misbehaviors",
            "coordinate multi-incident triage",
            "resist prompt injection attacks",
        ],
    },
}


@dataclass
class PhaseResult:
    """Result of one training phase."""
    phase: int
    name: str
    checkpoint_path: str
    final_reward: float
    best_reward: float
    episodes_completed: int
    milestones_achieved: List[str]
    reward_history: List[float]


def get_phase_config(phase: int, checkpoint: Optional[str] = None) -> dict:
    """Get the full training configuration for a phase."""
    if phase not in PHASES:
        raise ValueError(f"Unknown phase {phase}. Valid phases: {list(PHASES.keys())}")

    p = PHASES[phase]
    output_dir = f"outputs/phase{phase}"

    config = {
        "phase": phase,
        "phase_name": p["name"],
        "description": p["description"],
        "tasks": p["tasks"],
        "reward_profile": p["reward_profile"],
        "steps": p["steps"],
        "learning_rate": p["lr"],
        "kl_coef": p["kl_coef"],
        "temperature": p["temperature"],
        "num_generations": p["num_generations"],
        "output_dir": output_dir,
        "checkpoint": checkpoint,
        "misbehavior_types": p["misbehavior_types"],
        "expected_milestones": p["expected_milestones"],
    }
    return config


def generate_phase_env_vars(phase: int, checkpoint: Optional[str] = None) -> dict:
    """Generate environment variables for running train.py with phase config."""
    config = get_phase_config(phase, checkpoint)
    env_vars = {
        "USE_SENTINEL": "1",
        "TRAIN_STEPS": str(config["steps"]),
        "LR": str(config["learning_rate"]),
        "KL_COEF": str(config["kl_coef"]),
        "NUM_GENERATIONS": str(config["num_generations"]),
        "OUTPUT_DIR": config["output_dir"],
        "SENTINEL_TASKS": ",".join(config["tasks"]),
        "REWARD_PROFILE": config["reward_profile"],
        "REWARD_SCHEDULE_MODE": config["reward_profile"],
        "MODEL_STEPS_LIMIT": str(config.get("model_steps_limit", 1)),
    }
    if checkpoint:
        env_vars["RESUME_FROM"] = checkpoint
    return env_vars


def generate_deepspeed_config(
    output_path: str = "outputs/ds_config_zero2.json",
) -> str:
    """Generate a DeepSpeed ZeRO Stage 2 config suitable for GRPO training."""
    config = {
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True,
        },
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": 1.0,
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "wall_clock_breakdown": False,
    }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return output_path


def build_launch_command(
    phase: int,
    checkpoint: Optional[str] = None,
    use_accelerate: bool = False,
    deepspeed_config: Optional[str] = None,
    num_processes: Optional[int] = None,
) -> str:
    """Build the full shell command to launch a phase, optionally via accelerate."""
    env_vars = generate_phase_env_vars(phase, checkpoint)
    env_prefix = " ".join(f"{k}={v}" for k, v in env_vars.items())

    if use_accelerate or deepspeed_config:
        accelerate_bin = shutil.which("accelerate") or "accelerate"
        parts = [accelerate_bin, "launch"]
        if num_processes:
            parts += ["--num_processes", str(num_processes)]
        if deepspeed_config:
            parts += ["--use_deepspeed", "--deepspeed_config_file", deepspeed_config]
        parts.append("train.py")
        cmd = " ".join(parts)
    else:
        cmd = f"python train.py"

    return f"{env_prefix} {cmd}"


def print_phase_plan():
    """Print the full 3-phase training plan."""
    print("=" * 70)
    print("SENTINEL — 3-Phase GRPO Training Plan")
    print("=" * 70)
    for phase_num, phase in PHASES.items():
        print(f"\nPhase {phase_num}: {phase['name']}")
        print(f"  Description: {phase['description']}")
        print(f"  Tasks:       {', '.join(phase['tasks'])}")
        print(f"  Reward:      {phase['reward_profile']} weights")
        print(f"  Steps:       {phase['steps']}")
        print(f"  LR:          {phase['lr']}")
        print(f"  KL:          {phase['kl_coef']}")
        print(f"  Temp:        {phase['temperature']}")
        print(f"  Types:       {', '.join(phase['misbehavior_types'])}")
        print(f"  Milestones:")
        for m in phase["expected_milestones"]:
            print(f"    - {m}")

    print("\n" + "=" * 70)
    print("Run sequence:")
    print("  python training/phase_training.py --phase 1")
    print("  python training/phase_training.py --phase 2 --checkpoint outputs/phase1/final")
    print("  python training/phase_training.py --phase 3 --checkpoint outputs/phase2/final")
    print("=" * 70)


def log_phase_transition(from_phase: int, to_phase: int, checkpoint: str):
    """Log a phase transition for audit trail."""
    transition = {
        "from_phase": from_phase,
        "to_phase": to_phase,
        "checkpoint": checkpoint,
        "from_name": PHASES[from_phase]["name"],
        "to_name": PHASES[to_phase]["name"],
        "reward_transition": f"{PHASES[from_phase]['reward_profile']} -> {PHASES[to_phase]['reward_profile']}",
        "new_tasks": [t for t in PHASES[to_phase]["tasks"] if t not in PHASES[from_phase]["tasks"]],
        "new_misbehaviors": [
            m for m in PHASES[to_phase]["misbehavior_types"]
            if m not in PHASES[from_phase]["misbehavior_types"]
        ],
    }
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/phase_transitions.jsonl", "a") as f:
        f.write(json.dumps(transition) + "\n")
    logger.info(
        "Phase transition: %s -> %s (checkpoint: %s, new tasks: %s)",
        transition["from_name"],
        transition["to_name"],
        checkpoint,
        transition["new_tasks"],
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SENTINEL Phase-Based Training")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], help="Phase to run")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint from previous phase")
    parser.add_argument("--plan", action="store_true", help="Print training plan only")
    parser.add_argument("--all", action="store_true", help="Run all 3 phases sequentially")
    parser.add_argument("--accelerate", action="store_true", help="Launch via accelerate (multi-GPU / DDP)")
    parser.add_argument("--deepspeed", type=str, default=None,
                        help="Path to DeepSpeed config JSON (auto-generates ZeRO-2 if 'auto')")
    parser.add_argument("--num-processes", type=int, default=None,
                        help="Number of processes for accelerate launch")
    parser.add_argument("--run", action="store_true",
                        help="Actually run the training (not just print commands)")
    args = parser.parse_args()

    if args.plan:
        print_phase_plan()
        sys.exit(0)

    # Auto-generate DeepSpeed config if requested
    ds_config_path = args.deepspeed
    if ds_config_path == "auto":
        ds_config_path = generate_deepspeed_config()
        print(f"Generated DeepSpeed ZeRO-2 config at: {ds_config_path}")

    use_accelerate = args.accelerate or bool(ds_config_path)

    if args.all:
        print_phase_plan()
        print("\nReady to run all 3 phases.")
        for phase in [1, 2, 3]:
            ckpt = f"outputs/phase{phase-1}/final" if phase > 1 else None
            cmd = build_launch_command(
                phase,
                checkpoint=ckpt,
                use_accelerate=use_accelerate,
                deepspeed_config=ds_config_path,
                num_processes=args.num_processes,
            )
            print(f"\n--- Phase {phase} ---")
            print(f"  {cmd}")
            if args.run:
                if phase > 1:
                    log_phase_transition(phase - 1, phase, ckpt)
                env_vars = generate_phase_env_vars(phase, ckpt)
                env = {**os.environ, **env_vars}
                print(f"  Running phase {phase}...")
                result = subprocess.run(cmd, shell=True, env=env)
                if result.returncode != 0:
                    print(f"  Phase {phase} failed with code {result.returncode}")
                    sys.exit(result.returncode)
                print(f"  Phase {phase} complete.")
        sys.exit(0)

    if not args.phase:
        parser.error("Specify --phase N or --all or --plan")

    cmd = build_launch_command(
        args.phase,
        checkpoint=args.checkpoint,
        use_accelerate=use_accelerate,
        deepspeed_config=ds_config_path,
        num_processes=args.num_processes,
    )
    print(f"\nPhase {args.phase}: {PHASES[args.phase]['name']}")
    print(f"Command:")
    print(f"  {cmd}")

    if args.run:
        env_vars = generate_phase_env_vars(args.phase, args.checkpoint)
        env = {**os.environ, **env_vars}
        if args.checkpoint and args.phase > 1:
            log_phase_transition(args.phase - 1, args.phase, args.checkpoint)
        result = subprocess.run(cmd, shell=True, env=env)
        sys.exit(result.returncode)

    if args.checkpoint and args.phase > 1:
        log_phase_transition(args.phase - 1, args.phase, args.checkpoint)
"""
Phase-based GRPO Training for SENTINEL.

Implements the cross-winner pattern: separate phases with checkpoint handoff.
"""
