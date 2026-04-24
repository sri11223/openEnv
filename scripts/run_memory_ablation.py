from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]


def _summary_path(root: Path, label: str) -> Path:
    return root / label / "monitoring" / "latest_summary.json"


def _load_summary(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def collect_ablation(root: Path, output_path: Path) -> Dict[str, Any]:
    runs: List[Dict[str, Any]] = []
    for label, enabled in (("memory_off", False), ("memory_on", True)):
        summary = _load_summary(_summary_path(root, label))
        runs.append(
            {
                "label": label,
                "memory_enabled": enabled,
                "summary_path": str(_summary_path(root, label)),
                "summary": summary,
            }
        )

    off = runs[0]["summary"]
    on = runs[1]["summary"]
    comparison = {
        "reward_mean_delta": round(float(on.get("reward_mean", on.get("running_reward_mean", 0.0))) - float(off.get("reward_mean", off.get("running_reward_mean", 0.0))), 4),
        "detection_rate_delta": round(float(on.get("detection_rate", 0.0)) - float(off.get("detection_rate", 0.0)), 4),
        "risk_reduction_rate_delta": round(float(on.get("risk_reduction_rate", 0.0)) - float(off.get("risk_reduction_rate", 0.0)), 4),
    }
    payload = {
        "caption": "SENTINEL learns from its own oversight mistakes.",
        "runs": runs,
        "comparison": comparison,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _run(command: List[str], env: Dict[str, str], dry_run: bool) -> None:
    if dry_run:
        print(" ".join(command))
        return
    subprocess.run(command, cwd=str(ROOT), env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or collect a short SENTINEL memory ablation.")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warm-start-steps", type=int, default=5)
    parser.add_argument("--root", default="outputs/ablation")
    parser.add_argument("--output", default="outputs/monitoring/memory_ablation.json")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = ROOT / args.root
    output_path = ROOT / args.output

    if not args.collect_only:
        for label, enabled in (("memory_off", "0"), ("memory_on", "1")):
            env = os.environ.copy()
            env.update(
                {
                    "USE_SENTINEL": "1",
                    "USE_AGENT_MEMORY": enabled,
                    "USE_FEEDBACK_MEMORY": enabled,
                    "TRAIN_STEPS": str(args.steps),
                    "WARM_START_STEPS": str(args.warm_start_steps),
                    "OUTPUT_DIR": str(root / label / "checkpoints"),
                    "TRAIN_MONITOR_DIR": str(root / label / "monitoring"),
                    "WARM_START_OUTPUT_DIR": str(root / label / "warm_start"),
                    "ROLLOUT_AUDIT_DIR": str(root / label / "monitoring" / "rollout_audits"),
                }
            )
            _run(
                [
                    sys.executable,
                    "train.py",
                    "--steps",
                    str(args.steps),
                    "--warm-start-steps",
                    str(args.warm_start_steps),
                    "--output",
                    str(root / label / "checkpoints"),
                ],
                env,
                args.dry_run,
            )

    payload = collect_ablation(root, output_path)
    print(json.dumps(payload["comparison"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
