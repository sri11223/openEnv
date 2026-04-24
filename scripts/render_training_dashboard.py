from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


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
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
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


def _steps(records: List[Dict[str, Any]]) -> List[int]:
    return [int(record.get("batch_index") or record.get("global_step") or index + 1) for index, record in enumerate(records)]


def _series(records: List[Dict[str, Any]], key: str) -> List[float]:
    return [_as_float(_get(record, key)) for record in records]


def _sum_counter(records: Iterable[Dict[str, Any]], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for record in records:
        value = _get(record, key, {})
        if not isinstance(value, dict):
            continue
        for label, count in value.items():
            counts[str(label)] = counts.get(str(label), 0) + int(count or 0)
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def _ensure_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save_placeholder(path: Path, title: str, message: str) -> None:
    plt = _ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.axis("off")
    ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=16, fontweight="bold")
    ax.text(0.5, 0.42, message, ha="center", va="center", fontsize=11, wrap=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_line_plot(
    path: Path,
    title: str,
    x: List[int],
    series: List[Tuple[str, List[float]]],
    ylabel: str,
) -> None:
    if not x or not any(values for _, values in series):
        _save_placeholder(path, title, "No training records found yet.")
        return
    plt = _ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 5.2))
    plotted = False
    for label, values in series:
        if not values:
            continue
        usable = values[: len(x)]
        ax.plot(x[: len(usable)], usable, marker="o", linewidth=1.8, markersize=3, label=label)
        plotted = True
    if not plotted:
        _save_placeholder(path, title, "Metric is not present in the current run.")
        return
    ax.set_title(title)
    ax.set_xlabel("training batch / step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_bar_plot(path: Path, title: str, counts: Dict[str, int], ylabel: str = "count") -> None:
    if not counts:
        _save_placeholder(path, title, "No coverage records found yet.")
        return
    plt = _ensure_matplotlib()
    labels = list(counts)
    values = [counts[label] for label in labels]
    fig_width = max(9, min(16, 0.65 * len(labels) + 5))
    fig, ax = plt.subplots(figsize=(fig_width, 5.2))
    ax.bar(labels, values, color="#2f6f9f")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_heatmap(path: Path, title: str, scenario_counts: Dict[str, int]) -> None:
    if not scenario_counts:
        _save_placeholder(path, title, "No task/variant coverage records found yet.")
        return
    tasks = sorted({label.split(":seed", 1)[0] for label in scenario_counts})
    seeds = sorted({label.split(":seed", 1)[1] for label in scenario_counts if ":seed" in label}, key=lambda x: int(x))
    if not tasks or not seeds:
        _save_placeholder(path, title, "Scenario labels were not parseable.")
        return
    matrix = []
    for task in tasks:
        row = []
        for seed in seeds:
            row.append(scenario_counts.get(f"{task}:seed{seed}", 0))
        matrix.append(row)

    plt = _ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(max(8, len(seeds) * 0.8 + 4), max(4, len(tasks) * 0.55 + 2)))
    image = ax.imshow(matrix, cmap="YlGnBu")
    ax.set_title(title)
    ax.set_xlabel("variant seed")
    ax.set_ylabel("task")
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels(seeds)
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels(tasks)
    for y, row in enumerate(matrix):
        for x, value in enumerate(row):
            ax.text(x, y, str(value), ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="samples")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _candidate_confusion_rows(eval_report: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return (
        _get(eval_report, "confusion_matrix.candidate.rows", {})
        or _get(eval_report, "confusion_matrix.rows", {})
        or {}
    )


def _save_confusion_plot(path: Path, eval_report: Dict[str, Any]) -> None:
    rows = _candidate_confusion_rows(eval_report)
    if not rows:
        _save_placeholder(path, "Per-Misbehavior Confusion Matrix", "No held-out confusion matrix found yet.")
        return
    labels = list(rows)
    caught = [_as_float(rows[label].get("caught")) for label in labels]
    missed = [_as_float(rows[label].get("missed")) for label in labels]
    misclassified = [_as_float(rows[label].get("misclassified")) for label in labels]
    plt = _ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 0.8 + 4), 5.2))
    xs = list(range(len(labels)))
    ax.bar([x - 0.25 for x in xs], caught, width=0.25, label="caught", color="#238b45")
    ax.bar(xs, missed, width=0.25, label="missed", color="#cb181d")
    ax.bar([x + 0.25 for x in xs], misclassified, width=0.25, label="wrong reason", color="#fb6a4a")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("cases")
    ax.set_title("Per-Misbehavior Confusion Matrix")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _curriculum_frontier_series(records: List[Dict[str, Any]]) -> Tuple[List[float], List[float]]:
    lows: List[float] = []
    highs: List[float] = []
    for record in records:
        per_task = _get(record, "curriculum.adaptive_difficulty.per_task", {}) or {}
        if not isinstance(per_task, dict) or not per_task:
            lows.append(0.0)
            highs.append(0.0)
            continue
        low_values = [_as_float(item.get("difficulty_low")) for item in per_task.values() if isinstance(item, dict)]
        high_values = [_as_float(item.get("difficulty_high")) for item in per_task.values() if isinstance(item, dict)]
        lows.append(sum(low_values) / len(low_values) if low_values else 0.0)
        highs.append(sum(high_values) / len(high_values) if high_values else 0.0)
    return lows, highs


def _save_learning_snapshots(path: Path, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    checkpoints = [10, 50, 300]
    snapshots: List[Dict[str, Any]] = []
    for checkpoint in checkpoints:
        if not records:
            snapshots.append({"target_batch": checkpoint, "found": False})
            continue
        nearest = min(records, key=lambda item: abs(int(item.get("batch_index", 0) or 0) - checkpoint))
        snapshots.append(
            {
                "target_batch": checkpoint,
                "found": True,
                "batch_index": nearest.get("batch_index"),
                "reward_mean": nearest.get("reward_mean"),
                "detection_rate": nearest.get("detection_rate"),
                "false_positive_rate": nearest.get("false_positive_rate"),
                "risk_reduction_rate": nearest.get("risk_reduction_rate"),
                "effective_prompt_ratio": nearest.get("effective_prompt_ratio"),
            }
        )

    plt = _ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.axis("off")
    ax.set_title("Learning Snapshots: 10 vs 50 vs 300 Batches", fontweight="bold", pad=16)
    rows = []
    for snap in snapshots:
        rows.append(
            [
                snap["target_batch"],
                snap.get("batch_index", "missing"),
                _fmt(snap.get("reward_mean")),
                _fmt(snap.get("detection_rate")),
                _fmt(snap.get("risk_reduction_rate")),
                _fmt(snap.get("effective_prompt_ratio")),
            ]
        )
    table = ax.table(
        cellText=rows,
        colLabels=["target", "nearest", "reward", "detect", "risk red.", "productive"],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return snapshots


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def _save_memory_ablation_plot(path: Path, ablation: Dict[str, Any]) -> None:
    runs = ablation.get("runs") or []
    if not runs:
        _save_placeholder(path, "Memory Ablation", "No memory ablation JSON found yet.")
        return
    labels = [str(run.get("label", f"run_{index}")) for index, run in enumerate(runs)]
    rewards = [_as_float(_get(run, "summary.reward_mean", _get(run, "summary.running_reward_mean"))) for run in runs]
    detection = [_as_float(_get(run, "summary.detection_rate")) for run in runs]
    plt = _ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(9, 5))
    xs = list(range(len(labels)))
    ax.bar([x - 0.18 for x in xs], rewards, width=0.36, label="reward", color="#3182bd")
    ax.bar([x + 0.18 for x in xs], detection, width=0.36, label="detection", color="#31a354")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title("Memory Ablation: SENTINEL Learns From Its Own Oversight Mistakes")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def render_dashboard(
    monitor_dir: str = "outputs/monitoring",
    output_dir: str = "outputs/reward_curves",
    eval_report_path: str = "outputs/evals/sentinel_held_out_report.json",
    memory_ablation_path: str = "outputs/monitoring/memory_ablation.json",
) -> Dict[str, Any]:
    monitor = Path(monitor_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    records = _load_jsonl(monitor / "training_metrics.jsonl")
    stability = _load_jsonl(monitor / "training_stability.jsonl")
    eval_report = _load_json(Path(eval_report_path))
    memory_ablation = _load_json(Path(memory_ablation_path))
    x = _steps(records)
    sx = _steps(stability)

    images: List[Dict[str, str]] = []

    def add_image(filename: str, title: str) -> Path:
        path = output / filename
        images.append({"file": filename, "title": title})
        return path

    _save_line_plot(add_image("01_reward_mean.png", "Reward Mean"), "Reward Mean", x, [
        ("reward_mean", _series(records, "reward_mean")),
        ("running_reward_mean", _series(records, "running_reward_mean")),
        ("best_reward_mean", _series(records, "best_reward_mean")),
    ], "reward")
    _save_line_plot(add_image("02_detection_vs_false_positive.png", "Detection vs False Positive"), "Detection vs False Positive", x, [
        ("detection_rate", _series(records, "detection_rate")),
        ("false_positive_rate", _series(records, "false_positive_rate")),
    ], "rate")
    _save_line_plot(add_image("03_risk_reduction.png", "Counterfactual Risk Reduction"), "Counterfactual Risk Reduction", x, [
        ("risk_reduction_rate", _series(records, "risk_reduction_rate")),
        ("twin_damage_reduction_rate", _series(records, "twin_damage_reduction_rate")),
        ("without_sentinel_damage", _series(records, "twin_without_sentinel_damage_total")),
        ("with_sentinel_damage", _series(records, "twin_with_sentinel_damage_total")),
    ], "rate / damage")
    _save_line_plot(add_image("04_worker_rehabilitation.png", "Worker Rehabilitation"), "Worker Rehabilitation", x, [
        ("worker_rehabilitation_rate", _series(records, "worker_rehabilitation_rate")),
        ("coaching_quality", _series(records, "coaching_quality")),
        ("revision_attempts", _series(records, "revision_attempts")),
        ("revision_successes", _series(records, "revision_successes")),
    ], "rate / count")
    _save_bar_plot(add_image("05_task_coverage.png", "Task Coverage"), "Task Coverage", _sum_counter(records, "task_counts"))
    _save_heatmap(add_image("06_scenario_coverage_heatmap.png", "Scenario Coverage Heatmap"), "Scenario Coverage Heatmap", _sum_counter(records, "scenario_counts"))
    _save_bar_plot(add_image("07_misbehavior_detection.png", "Misbehavior Coverage"), "Misbehavior Coverage", _sum_counter(records, "misbehavior_counts"))
    _save_confusion_plot(add_image("08_confusion_matrix.png", "Per-Misbehavior Confusion Matrix"), eval_report)

    lows, highs = _curriculum_frontier_series(records)
    _save_line_plot(add_image("09_curriculum_frontier.png", "Adaptive Curriculum Frontier"), "Adaptive Curriculum Frontier", x, [
        ("difficulty_low", lows),
        ("difficulty_high", highs),
    ], "difficulty rank")
    _save_line_plot(add_image("10_productive_signal.png", "Productive Signal"), "Productive Signal", x, [
        ("zero_reward_fraction", _series(records, "zero_reward_fraction")),
        ("trivially_solved_fraction", _series(records, "trivially_solved_fraction")),
        ("productive_fraction", _series(records, "productive_fraction")),
        ("effective_prompt_ratio", _series(records, "effective_prompt_ratio")),
    ], "fraction")
    _save_line_plot(add_image("11_entropy_diversity.png", "Decision Entropy and Diversity"), "Decision Entropy and Diversity", x, [
        ("decision_entropy", _series(records, "decision_entropy")),
        ("unique_completion_ratio", _series(records, "unique_completion_ratio")),
    ], "value")
    _save_line_plot(add_image("12_kl_drift_beta.png", "KL Drift and Adaptive Beta"), "KL Drift and Adaptive Beta", sx, [
        ("approx_kl", _series(stability, "approx_kl")),
        ("adaptive_beta", [_as_float(_get(row, "kl_guardrail.current_beta", row.get("adaptive_beta"))) for row in stability]),
        ("policy_entropy", _series(stability, "policy_entropy")),
    ], "value")
    tripwire = _get(eval_report, "tripwire", {}) or {}
    _save_bar_plot(add_image("13_tripwire_pass_rate.png", "Tripwire Pass Rate"), "Tripwire Pass Rate", {
        "baseline": _as_float(_get(tripwire, "baseline.overall.pass_rate", _get(tripwire, "baseline.pass_rate"))) * 100,
        "candidate": _as_float(_get(tripwire, "candidate.overall.pass_rate", _get(tripwire, "candidate.pass_rate"))) * 100,
    }, ylabel="pass rate (%)")
    sampling = _get(eval_report, "sampling_eval", {}) or {}
    _save_bar_plot(add_image("14_top1_vs_bestofk.png", "Top-1 vs Best-of-K"), "Top-1 vs Best-of-K", {
        "candidate_top1": _as_float(sampling.get("candidate_top1_mean_score")),
        "candidate_best_of_k": _as_float(sampling.get("candidate_best_of_k_mean_score")),
        "baseline_top1": _as_float(sampling.get("baseline_top1_mean_score")),
        "baseline_best_of_k": _as_float(sampling.get("baseline_best_of_k_mean_score")),
    }, ylabel="score")
    snapshots = _save_learning_snapshots(add_image("15_learning_snapshots.png", "Learning Snapshots"), records)
    _save_memory_ablation_plot(add_image("16_memory_ablation.png", "Memory Ablation"), memory_ablation)
    _save_line_plot(add_image("17_zero_gradient_groups.png", "Zero-Gradient Group Fraction"), "Zero-Gradient Group Fraction", x, [
        ("zero_gradient_group_fraction", _series(records, "zero_gradient_group_fraction")),
        ("mean_reward_group_std", _series(records, "mean_reward_group_std")),
    ], "fraction / std")
    _save_line_plot(add_image("18_memory_growth.png", "Memory Growth"), "Memory Growth", x, [
        ("memory_total_episodes", _series(records, "memory.total_episodes")),
        ("mistake_cards", _series(records, "memory.mistake_cards_stored")),
        ("mistakes_stored", _series(records, "memory.mistakes_stored")),
    ], "count")

    manifest = {
        "records": len(records),
        "stability_records": len(stability),
        "images": images,
        "learning_snapshots": snapshots,
        "inputs": {
            "monitor_dir": str(monitor),
            "eval_report_path": eval_report_path,
            "memory_ablation_path": memory_ablation_path,
        },
    }
    (output / "dashboard_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_markdown_report(output / "training_dashboard.md", manifest)
    return manifest


def _write_markdown_report(path: Path, manifest: Dict[str, Any]) -> None:
    lines = [
        "# SENTINEL Training Dashboard",
        "",
        f"- Training records: {manifest.get('records', 0)}",
        f"- Stability records: {manifest.get('stability_records', 0)}",
        "",
        "## Learning Snapshots",
        "",
        "| Target batch | Nearest batch | Reward | Detection | Risk reduction | Productive |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for snap in manifest.get("learning_snapshots", []):
        lines.append(
            "| {target} | {nearest} | {reward} | {detect} | {risk} | {productive} |".format(
                target=snap.get("target_batch"),
                nearest=snap.get("batch_index", "missing"),
                reward=_fmt(snap.get("reward_mean")),
                detect=_fmt(snap.get("detection_rate")),
                risk=_fmt(snap.get("risk_reduction_rate")),
                productive=_fmt(snap.get("effective_prompt_ratio")),
            )
        )
    lines.extend(["", "## Plots", ""])
    for image in manifest.get("images", []):
        lines.append(f"### {image['title']}")
        lines.append("")
        lines.append(f"![{image['title']}]({image['file']})")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render SENTINEL training dashboard plots.")
    parser.add_argument("--monitor-dir", default="outputs/monitoring")
    parser.add_argument("--output-dir", default="outputs/reward_curves")
    parser.add_argument("--eval-report", default="outputs/evals/sentinel_held_out_report.json")
    parser.add_argument("--memory-ablation", default="outputs/monitoring/memory_ablation.json")
    args = parser.parse_args()
    manifest = render_dashboard(
        monitor_dir=args.monitor_dir,
        output_dir=args.output_dir,
        eval_report_path=args.eval_report,
        memory_ablation_path=args.memory_ablation,
    )
    print(json.dumps({"images": len(manifest["images"]), "records": manifest["records"]}, indent=2))


if __name__ == "__main__":
    main()
