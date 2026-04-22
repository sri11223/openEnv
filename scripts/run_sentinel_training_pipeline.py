from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str], *, env: dict[str, str]) -> None:
    subprocess.run(command, cwd=str(ROOT), env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SENTINEL warm-start -> train -> eval -> proof pipeline.")
    parser.add_argument("--steps", type=int, default=300, help="GRPO training steps.")
    parser.add_argument("--warm-start-steps", type=int, default=20, help="Warm-start steps before GRPO.")
    parser.add_argument("--output-dir", type=str, default="outputs/checkpoints", help="Training checkpoint directory.")
    parser.add_argument("--baseline-checkpoint", type=str, default="", help="Optional baseline checkpoint for eval/proof.")
    parser.add_argument("--candidate-checkpoint", type=str, default="", help="Optional candidate checkpoint override.")
    parser.add_argument("--base-model", type=str, default="", help="Optional base model for adapter checkpoints.")
    parser.add_argument("--proof-seed", type=int, default=0, help="Seed for proof-pack trajectory export.")
    parser.add_argument("--held-out-seeds", type=str, default="100-104", help="Held-out seeds for evaluation.")
    parser.add_argument("--dry-run", action="store_true", help="Validate commands without executing full training.")
    args = parser.parse_args()

    env = os.environ.copy()
    env["USE_SENTINEL"] = "1"
    env["TRAIN_STEPS"] = str(args.steps)
    env["WARM_START_STEPS"] = str(args.warm_start_steps)

    train_cmd = [
        sys.executable,
        "train.py",
        "--steps",
        str(args.steps),
        "--warm-start-steps",
        str(args.warm_start_steps),
        "--output",
        args.output_dir,
    ]
    if args.dry_run:
        train_cmd.append("--dry-run")
    _run(train_cmd, env=env)

    candidate_checkpoint = args.candidate_checkpoint or str(Path(args.output_dir) / "final")

    eval_cmd = [
        sys.executable,
        "scripts/eval_sentinel.py",
        "--seeds",
        args.held_out_seeds,
        "--candidate-checkpoint",
        candidate_checkpoint,
    ]
    if args.baseline_checkpoint:
        eval_cmd.extend(["--baseline-checkpoint", args.baseline_checkpoint])
    if args.base_model:
        eval_cmd.extend(["--base-model", args.base_model])
    if args.dry_run:
        eval_cmd.append("--dry-run")
    _run(eval_cmd, env=env)

    if args.dry_run:
        _run([sys.executable, "proof_pack.py", "--seed", str(args.proof_seed)], env=env)
        return

    proof_cmd = [
        sys.executable,
        "proof_pack.py",
        "--seed",
        str(args.proof_seed),
        "--candidate-checkpoint",
        candidate_checkpoint,
    ]
    if args.baseline_checkpoint:
        proof_cmd.extend(["--baseline-checkpoint", args.baseline_checkpoint])
    if args.base_model:
        proof_cmd.extend(["--base-model", args.base_model])
    _run(proof_cmd, env=env)


if __name__ == "__main__":
    main()
