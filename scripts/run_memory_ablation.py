"""Collect memory-on vs memory-off summaries for the SENTINEL proof dashboard.

The training pipeline can be run twice:

  outputs/ablation/memory_off/monitoring/latest_summary.json
  outputs/ablation/memory_on/monitoring/latest_summary.json

This helper reads those summaries and writes a compact
``memory_ablation.json`` consumed by ``render_training_dashboard.py``.
It is intentionally lightweight so CI can validate the proof-pack contract
without running training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


DEFAULT_CAPTION = "SENTINEL learns from its own oversight mistakes."
DEFAULT_METRICS = ("reward_mean", "detection_rate", "risk_reduction_rate")


def collect_ablation(
    root: str | Path = "outputs/ablation",
    output_path: str | Path = "outputs/monitoring/memory_ablation.json",
) -> Dict[str, Any]:
    """Read memory-off/on summaries, compute deltas, and write dashboard JSON."""
    root_path = Path(root)
    output = Path(output_path)

    runs = [
        _load_run(root_path, "memory_off"),
        _load_run(root_path, "memory_on"),
    ]
    comparison = _compare_summaries(runs[0].get("summary", {}), runs[1].get("summary", {}), DEFAULT_METRICS)

    payload: Dict[str, Any] = {
        "caption": DEFAULT_CAPTION,
        "root": str(root_path),
        "runs": runs,
        "comparison": comparison,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _load_run(root: Path, label: str) -> Dict[str, Any]:
    summary_path = root / label / "monitoring" / "latest_summary.json"
    summary = _read_json(summary_path)
    return {
        "label": label,
        "summary_path": str(summary_path),
        "summary": summary,
        "available": bool(summary),
    }


def _compare_summaries(
    baseline: Dict[str, Any],
    candidate: Dict[str, Any],
    metrics: Iterable[str],
) -> Dict[str, Optional[float]]:
    comparison: Dict[str, Optional[float]] = {}
    for metric in metrics:
        base = _as_float(baseline.get(metric))
        cand = _as_float(candidate.get(metric))
        comparison[f"{metric}_delta"] = None if base is None or cand is None else round(cand - base, 4)
        comparison[f"{metric}_memory_off"] = base
        comparison[f"{metric}_memory_on"] = cand
    return comparison


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _as_float(value: Any) -> Optional[float]:
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect SENTINEL memory ablation proof data.")
    parser.add_argument("--root", default="outputs/ablation", help="Directory containing memory_off/ and memory_on/ runs.")
    parser.add_argument(
        "--output",
        default="outputs/monitoring/memory_ablation.json",
        help="Output JSON path for the dashboard renderer.",
    )
    args = parser.parse_args()

    payload = collect_ablation(args.root, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
