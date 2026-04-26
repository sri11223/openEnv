from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _get(payload: Dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    cur: Any = payload
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _ensure_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save_placeholder(path: Path, title: str, message: str) -> None:
    plt = _ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.axis("off")
    ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=17, fontweight="bold")
    ax.text(0.5, 0.42, message, ha="center", va="center", fontsize=11, wrap=True)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _task_groups(rollouts: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rollouts:
        grouped[str(row.get("task_id") or "unknown")].append(row)
    return dict(sorted(grouped.items(), key=lambda item: item[0]))


def _save_keep_drop(path: Path, rollouts: List[Dict[str, Any]]) -> None:
    if not rollouts:
        _save_placeholder(path, "RFT Keep/Drop By Task", "No RFT rollouts found.")
        return
    plt = _ensure_matplotlib()
    groups = _task_groups(rollouts)
    labels = list(groups)
    kept = [sum(1 for row in groups[label] if row.get("kept")) for label in labels]
    dropped = [len(groups[label]) - kept[index] for index, label in enumerate(labels)]

    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.bar(labels, kept, color="#2ca25f", label="kept for RFT")
    ax.bar(labels, dropped, bottom=kept, color="#d95f02", label="rejected")
    ax.set_title("RFT Rejection Sampling: Kept vs Rejected Rollouts")
    ax.set_ylabel("rollouts")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    for i, label in enumerate(labels):
        total = len(groups[label])
        rate = kept[i] / total if total else 0.0
        ax.text(i, kept[i] + dropped[i] + 0.25, f"{rate:.0%}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _save_score_by_task(path: Path, rollouts: List[Dict[str, Any]], min_score: Optional[float]) -> None:
    if not rollouts:
        _save_placeholder(path, "RFT Score Distribution", "No RFT rollouts found.")
        return
    plt = _ensure_matplotlib()
    groups = _task_groups(rollouts)
    labels = list(groups)
    fig, ax = plt.subplots(figsize=(12, 5.8))
    for index, label in enumerate(labels):
        rows = groups[label]
        scores = [_as_float(row.get("score")) for row in rows]
        colors = ["#2ca25f" if row.get("kept") else "#d95f02" for row in rows]
        xs = [index + ((i % 7) - 3) * 0.025 for i in range(len(rows))]
        ax.scatter(xs, scores, c=colors, alpha=0.8, s=36, edgecolors="white", linewidths=0.4)
    if min_score is not None:
        ax.axhline(min_score, color="#333333", linestyle="--", linewidth=1.4, label=f"keep score >= {min_score:g}")
        ax.legend()
    ax.set_title("RFT Rollout Scores By Task")
    ax.set_ylabel("filter score")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _save_fp_by_task(path: Path, rollouts: List[Dict[str, Any]], max_fp: Optional[float]) -> None:
    if not rollouts:
        _save_placeholder(path, "RFT False Positive Distribution", "No RFT rollouts found.")
        return
    plt = _ensure_matplotlib()
    groups = _task_groups(rollouts)
    labels = list(groups)
    fig, ax = plt.subplots(figsize=(12, 5.8))
    for index, label in enumerate(labels):
        rows = groups[label]
        fps = [_as_float(row.get("fp")) for row in rows]
        colors = ["#2ca25f" if row.get("kept") else "#d95f02" for row in rows]
        xs = [index + ((i % 7) - 3) * 0.025 for i in range(len(rows))]
        ax.scatter(xs, fps, c=colors, alpha=0.8, s=36, edgecolors="white", linewidths=0.4)
    if max_fp is not None:
        ax.axhline(max_fp, color="#333333", linestyle="--", linewidth=1.4, label=f"keep fp <= {max_fp:g}")
        ax.legend()
    ax.set_title("RFT False Positives By Task")
    ax.set_ylabel("false positives / episode")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _save_score_vs_fp(path: Path, rollouts: List[Dict[str, Any]], min_score: Optional[float], max_fp: Optional[float]) -> None:
    if not rollouts:
        _save_placeholder(path, "RFT Score vs False Positives", "No RFT rollouts found.")
        return
    plt = _ensure_matplotlib()
    groups = _task_groups(rollouts)
    palette = ["#1b9e77", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d"]
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    for index, (task_id, rows) in enumerate(groups.items()):
        kept_rows = [row for row in rows if row.get("kept")]
        drop_rows = [row for row in rows if not row.get("kept")]
        color = palette[index % len(palette)]
        if drop_rows:
            ax.scatter(
                [_as_float(row.get("fp")) for row in drop_rows],
                [_as_float(row.get("score")) for row in drop_rows],
                marker="x",
                s=50,
                color=color,
                alpha=0.55,
                label=f"{task_id} rejected",
            )
        if kept_rows:
            ax.scatter(
                [_as_float(row.get("fp")) for row in kept_rows],
                [_as_float(row.get("score")) for row in kept_rows],
                marker="o",
                s=60,
                color=color,
                edgecolors="black",
                linewidths=0.4,
                label=f"{task_id} kept",
            )
    if min_score is not None:
        ax.axhline(min_score, color="#111111", linestyle="--", linewidth=1.2)
    if max_fp is not None:
        ax.axvline(max_fp, color="#111111", linestyle="--", linewidth=1.2)
    ax.set_title("RFT Filter Boundary: Keep High Score, Low False Positives")
    ax.set_xlabel("false positives / episode")
    ax.set_ylabel("filter score")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _save_timeline(path: Path, rollouts: List[Dict[str, Any]]) -> None:
    if not rollouts:
        _save_placeholder(path, "RFT Rollout Timeline", "No RFT rollouts found.")
        return
    plt = _ensure_matplotlib()
    xs = list(range(1, len(rollouts) + 1))
    scores = [_as_float(row.get("score")) for row in rollouts]
    kept_x = [xs[i] for i, row in enumerate(rollouts) if row.get("kept")]
    kept_y = [scores[i] for i, row in enumerate(rollouts) if row.get("kept")]
    drop_x = [xs[i] for i, row in enumerate(rollouts) if not row.get("kept")]
    drop_y = [scores[i] for i, row in enumerate(rollouts) if not row.get("kept")]
    rolling_keep = []
    for index in range(len(rollouts)):
        start = max(0, index - 9)
        window = rollouts[start : index + 1]
        rolling_keep.append(sum(1 for row in window if row.get("kept")) / len(window))

    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.plot(xs, scores, color="#6b7280", linewidth=1.1, alpha=0.65, label="score")
    ax.scatter(kept_x, kept_y, color="#2ca25f", s=45, label="kept")
    ax.scatter(drop_x, drop_y, color="#d95f02", marker="x", s=42, label="rejected")
    ax2 = ax.twinx()
    ax2.plot(xs, rolling_keep, color="#2563eb", linewidth=2, label="rolling keep rate")
    ax.set_title("RFT Rollout Timeline")
    ax.set_xlabel("generated rollout")
    ax.set_ylabel("filter score")
    ax2.set_ylabel("rolling keep rate")
    ax.grid(True, axis="y", alpha=0.25)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _save_eval_overview(path: Path, eval_report: Dict[str, Any]) -> None:
    if not eval_report:
        _save_placeholder(path, "Held-Out Eval After RFT", "No eval report provided yet.")
        return
    plt = _ensure_matplotlib()
    metrics = [
        ("Mean score", "mean_score"),
        ("Detection", "detection_rate"),
        ("Risk reduction", "risk_reduction_rate"),
        ("Worker rehab", "worker_rehabilitation_rate"),
        ("False positive", "false_positive_rate"),
    ]
    baseline = _get(eval_report, "overall.baseline", {})
    candidate = _get(eval_report, "overall.candidate", {})
    labels = [label for label, _ in metrics]
    base_values = [_as_float(baseline.get(key)) for _, key in metrics]
    cand_values = [_as_float(candidate.get(key)) for _, key in metrics]
    xs = list(range(len(labels)))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.bar([x - width / 2 for x in xs], base_values, width=width, color="#d95f02", label=str(eval_report.get("baseline_label") or "baseline"))
    ax.bar([x + width / 2 for x in xs], cand_values, width=width, color="#2ca25f", label=str(eval_report.get("candidate_label") or "candidate"))
    ax.set_title("Held-Out Evaluation: Baseline vs RFT Candidate")
    ax.set_ylabel("rate / score")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _save_eval_task_delta(path: Path, eval_report: Dict[str, Any]) -> None:
    per_task = _get(eval_report, "per_task", {})
    if not isinstance(per_task, dict) or not per_task:
        _save_placeholder(path, "RFT Held-Out Score Delta By Task", "No per-task eval rows found.")
        return
    labels = []
    deltas = []
    for task_id, payload in sorted(per_task.items()):
        baseline_score = _as_float(_get(payload, "baseline.mean_score"))
        candidate_score = _as_float(_get(payload, "candidate.mean_score"))
        labels.append(str(task_id))
        deltas.append(candidate_score - baseline_score)
    plt = _ensure_matplotlib()
    colors = ["#2ca25f" if value >= 0 else "#d95f02" for value in deltas]
    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.bar(labels, deltas, color=colors)
    ax.axhline(0.0, color="#111111", linewidth=1)
    ax.set_title("Held-Out Score Delta By Task")
    ax.set_ylabel("candidate mean score - baseline mean score")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _write_markdown(
    path: Path,
    label: str,
    rollouts: List[Dict[str, Any]],
    kept: List[Dict[str, Any]],
    summary: Dict[str, Any],
    eval_report: Dict[str, Any],
    images: List[str],
) -> None:
    total = len(rollouts)
    kept_count = len(kept)
    keep_rate = kept_count / total if total else 0.0
    mean_score_total = mean([_as_float(row.get("score")) for row in rollouts]) if rollouts else 0.0
    mean_score_kept = mean([_as_float(row.get("score")) for row in kept]) if kept else 0.0
    mean_fp_kept = mean([_as_float(row.get("fp")) for row in kept]) if kept else 0.0
    eval_overall = _get(eval_report, "overall", {})

    if eval_overall:
        intro = (
            "This folder is the rejection-sampling fine-tuning proof layer. "
            "It shows which model-generated rollouts were accepted, which were rejected, "
            "and what the held-out evaluation says after the polish pass."
        )
    else:
        intro = (
            "This folder is the rejection-sampling fine-tuning proof layer. "
            "It shows which model-generated rollouts were accepted, which were rejected, "
            "and which low-false-positive samples were used for the polish pass. "
            "Held-out model evaluation was intentionally omitted for this proof pack."
        )

    lines = [
        f"# {label} RFT Proof Pack",
        "",
        intro,
        "",
        "## Summary",
        "",
        f"- Total generated rollouts: `{total}`",
        f"- Kept rollouts used for SFT: `{kept_count}`",
        f"- Keep rate: `{keep_rate:.1%}`",
        f"- Mean rollout score: `{mean_score_total:.3f}`",
        f"- Mean kept score: `{mean_score_kept:.3f}`",
        f"- Mean kept false positives: `{mean_fp_kept:.2f}`",
    ]
    if summary:
        lines.extend([
            f"- RFT status: `{_get(summary, 'sft.status', summary.get('status', 'unknown'))}`",
            f"- Output adapter: `{_get(summary, 'output.final_dir', summary.get('final_dir', 'see RFT output dir'))}`",
        ])
    if eval_overall:
        lines.extend([
            "",
            "## Held-Out Eval",
            "",
            f"- Baseline mean score: `{_as_float(eval_overall.get('baseline_mean_score')):.3f}`",
            f"- Candidate mean score: `{_as_float(eval_overall.get('candidate_mean_score')):.3f}`",
            f"- Mean score delta: `{_as_float(eval_overall.get('mean_score_delta')):.3f}`",
            f"- Candidate risk reduction: `{_as_float(eval_overall.get('candidate_risk_reduction_rate')):.1%}`",
            f"- Candidate false-positive rate: `{_as_float(eval_overall.get('candidate_false_positive_rate')):.1%}`",
        ])
    lines.extend(["", "## Plots", ""])
    for image in images:
        title = Path(image).stem.replace("_", " ").title()
        lines.extend([f"### {title}", "", f"![{title}]({image})", ""])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_rft_proof(
    rft_dir: Path,
    output_dir: Path,
    eval_report_path: Optional[Path],
    label: str,
    min_score: Optional[float],
    max_fp: Optional[float],
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rollouts = _load_jsonl(rft_dir / "rollouts.jsonl")
    kept = [row for row in rollouts if row.get("kept")]
    summary = _load_json(rft_dir / "rft_summary.json")
    eval_report = _load_json(eval_report_path) if eval_report_path else {}

    if min_score is None:
        min_score = _as_float(_get(summary, "config.MIN_SCORE"), default=float("nan"))
        if min_score != min_score:
            min_score = None
    if max_fp is None:
        max_fp = _as_float(_get(summary, "config.MAX_FP"), default=float("nan"))
        if max_fp != max_fp:
            max_fp = None

    image_names = [
        "01_rft_keep_drop_by_task.png",
        "02_rft_score_distribution.png",
        "03_rft_false_positive_distribution.png",
        "04_rft_score_vs_fp_filter.png",
        "05_rft_rollout_timeline.png",
        "06_rft_eval_overview.png",
        "07_rft_eval_task_delta.png",
    ]
    _save_keep_drop(output_dir / image_names[0], rollouts)
    _save_score_by_task(output_dir / image_names[1], rollouts, min_score)
    _save_fp_by_task(output_dir / image_names[2], rollouts, max_fp)
    _save_score_vs_fp(output_dir / image_names[3], rollouts, min_score, max_fp)
    _save_timeline(output_dir / image_names[4], rollouts)
    _save_eval_overview(output_dir / image_names[5], eval_report)
    _save_eval_task_delta(output_dir / image_names[6], eval_report)

    manifest = {
        "label": label,
        "rft_dir": str(rft_dir),
        "eval_report_path": str(eval_report_path) if eval_report_path else "",
        "total_rollouts": len(rollouts),
        "kept_rollouts": len(kept),
        "keep_rate": len(kept) / len(rollouts) if rollouts else 0.0,
        "images": image_names,
    }
    (output_dir / "rft_plot_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_markdown(output_dir / "rft_proof.md", label, rollouts, kept, summary, eval_report, image_names)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Render proof plots for a SENTINEL RFT polish run.")
    parser.add_argument("--rft-dir", default="/data/sentinel_outputs_rft_phase1_100", help="Directory containing rollouts.jsonl and rft_summary.json.")
    parser.add_argument("--eval-report", default="/data/rft_eval/sentinel_held_out_report.json", help="Optional held-out eval JSON report.")
    parser.add_argument("--output-dir", default="outputs/rft_phase1_100/plots", help="Where to write PNG plots and markdown.")
    parser.add_argument("--label", default="Phase 1 + RFT", help="Label used in the markdown report.")
    parser.add_argument("--min-score", type=float, default=None, help="Override score threshold line.")
    parser.add_argument("--max-fp", type=float, default=None, help="Override false-positive threshold line.")
    args = parser.parse_args()

    eval_report = Path(args.eval_report) if args.eval_report else None
    manifest = render_rft_proof(
        rft_dir=Path(args.rft_dir),
        output_dir=Path(args.output_dir),
        eval_report_path=eval_report,
        label=args.label,
        min_score=args.min_score,
        max_fp=args.max_fp,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
