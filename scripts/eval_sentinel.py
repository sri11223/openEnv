from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import proof_pack
from sentinel.evaluation import (
    DEFAULT_EVAL_OUTPUT_DIR,
    DEFAULT_HELD_OUT_TASK_IDS,
    DEFAULT_OOD_EVAL_SEEDS,
    build_eval_report,
    evaluate_tripwire_policy,
    parse_seed_spec,
    write_eval_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run held-out SENTINEL evaluation.")
    parser.add_argument("--seeds", type=str, default="100-104", help="Comma list or range of held-out seeds.")
    parser.add_argument("--baseline-checkpoint", type=str, default="", help="Optional baseline checkpoint.")
    parser.add_argument("--candidate-checkpoint", type=str, default="", help="Optional candidate checkpoint.")
    parser.add_argument("--base-model", type=str, default="", help="Optional base model for adapter checkpoints.")
    parser.add_argument("--baseline-label", type=str, default="", help="Display label for the baseline policy.")
    parser.add_argument("--candidate-label", type=str, default="", help="Display label for the candidate policy.")
    parser.add_argument("--ood-seeds", type=str, default="200-204", help="Comma list or range of OOD held-out seeds.")
    parser.add_argument("--skip-tripwires", action="store_true", help="Skip the policy-level tripwire evaluation suite.")
    parser.add_argument("--best-of-k", type=int, default=4, help="Sample K first-step decisions and score the best one separately.")
    parser.add_argument("--sampling-temperature", type=float, default=0.8, help="Temperature used for sampled Best-of-K evaluation.")
    parser.add_argument("--skip-best-of-k", action="store_true", help="Skip the sampled Top-1 vs Best-of-K comparison.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_EVAL_OUTPUT_DIR), help="Where to write the eval report.")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and exit without executing episodes.")
    args = parser.parse_args()

    seeds = parse_seed_spec(args.seeds)
    ood_seeds = parse_seed_spec(args.ood_seeds) if args.ood_seeds else list(DEFAULT_OOD_EVAL_SEEDS)
    if args.dry_run:
        print(
            {
                "seeds": seeds,
                "ood_seeds": ood_seeds,
                "baseline_checkpoint": args.baseline_checkpoint or None,
                "candidate_checkpoint": args.candidate_checkpoint or None,
                "base_model": args.base_model or None,
                "tripwires": not args.skip_tripwires,
                "best_of_k": None if args.skip_best_of_k else max(1, int(args.best_of_k)),
                "sampling_temperature": float(args.sampling_temperature),
                "output_dir": args.output_dir,
            }
        )
        return

    baseline_spec = proof_pack._resolve_policy_spec(
        label=args.baseline_label or None,
        checkpoint=args.baseline_checkpoint or None,
        base_model=args.base_model or None,
        fallback_name="approve_all",
        fallback_policy=proof_pack._approve_all_policy,
    )
    candidate_spec = proof_pack._resolve_policy_spec(
        label=args.candidate_label or None,
        checkpoint=args.candidate_checkpoint or None,
        base_model=args.base_model or None,
        fallback_name="corrective_policy",
        fallback_policy=proof_pack._corrective_policy,
    )

    baseline_runs = []
    candidate_runs = []
    baseline_sampling_top1_runs = []
    candidate_sampling_top1_runs = []
    baseline_best_of_k_runs = []
    candidate_best_of_k_runs = []
    baseline_ood_runs = []
    candidate_ood_runs = []
    for task_id in DEFAULT_HELD_OUT_TASK_IDS:
        for seed in seeds:
            baseline_runs.append(
                proof_pack.run_episode(
                    task_id=task_id,
                    variant_seed=seed,
                    policy_name=baseline_spec.name,
                    policy=baseline_spec.policy,
                    eval_mode=True,
                )
            )
            candidate_runs.append(
                proof_pack.run_episode(
                    task_id=task_id,
                    variant_seed=seed,
                    policy_name=candidate_spec.name,
                    policy=candidate_spec.policy,
                    eval_mode=True,
                )
            )
            if not args.skip_best_of_k and args.best_of_k > 1:
                baseline_sampled = proof_pack.evaluate_policy_best_of_k(
                    task_id=task_id,
                    variant_seed=seed,
                    policy_spec=baseline_spec,
                    num_samples=args.best_of_k,
                    temperature=args.sampling_temperature,
                    eval_mode=True,
                )
                candidate_sampled = proof_pack.evaluate_policy_best_of_k(
                    task_id=task_id,
                    variant_seed=seed,
                    policy_spec=candidate_spec,
                    num_samples=args.best_of_k,
                    temperature=args.sampling_temperature,
                    eval_mode=True,
                )
                baseline_sampling_top1_runs.append(baseline_sampled["top1"])
                candidate_sampling_top1_runs.append(candidate_sampled["top1"])
                baseline_best_of_k_runs.append(baseline_sampled["best"])
                candidate_best_of_k_runs.append(candidate_sampled["best"])
        for seed in ood_seeds:
            baseline_ood_runs.append(
                proof_pack.run_episode(
                    task_id=task_id,
                    variant_seed=seed,
                    policy_name=baseline_spec.name,
                    policy=baseline_spec.policy,
                    eval_mode=True,
                )
            )
            candidate_ood_runs.append(
                proof_pack.run_episode(
                    task_id=task_id,
                    variant_seed=seed,
                    policy_name=candidate_spec.name,
                    policy=candidate_spec.policy,
                    eval_mode=True,
                )
            )

    baseline_tripwire = None
    candidate_tripwire = None
    if not args.skip_tripwires:
        baseline_tripwire = evaluate_tripwire_policy(baseline_spec.name, baseline_spec.policy)
        candidate_tripwire = evaluate_tripwire_policy(candidate_spec.name, candidate_spec.policy)

    report = build_eval_report(
        baseline_runs=baseline_runs,
        candidate_runs=candidate_runs,
        baseline_label=baseline_spec.name,
        candidate_label=candidate_spec.name,
        seeds=seeds,
        best_of_k=args.best_of_k,
        sampling_temperature=args.sampling_temperature,
        baseline_sampling_top1_runs=baseline_sampling_top1_runs if baseline_sampling_top1_runs else None,
        candidate_sampling_top1_runs=candidate_sampling_top1_runs if candidate_sampling_top1_runs else None,
        baseline_best_of_k_runs=baseline_best_of_k_runs if baseline_best_of_k_runs else None,
        candidate_best_of_k_runs=candidate_best_of_k_runs if candidate_best_of_k_runs else None,
        ood_seeds=ood_seeds,
        baseline_ood_runs=baseline_ood_runs,
        candidate_ood_runs=candidate_ood_runs,
        baseline_tripwire=baseline_tripwire,
        candidate_tripwire=candidate_tripwire,
    )
    paths = write_eval_report(report, output_dir=args.output_dir)
    print(f"Held-out evaluation written to {paths['json_path']} and {paths['markdown_path']}")


if __name__ == "__main__":
    main()
