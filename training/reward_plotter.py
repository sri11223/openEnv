"""Reward Curve Visualization — Auto-generate training plots.

Judges NEED to see reward curves.

Features:
  - Per-episode reward with rolling average
  - Trend line (slope shows learning rate)
  - Phase transitions marked with vertical lines
  - Milestone achievement annotations
  - Component-level breakdown sub-plots
  - Auto-saves PNG to training output directory

Usage:
    from training.reward_plotter import plot_reward_curves, log_episode_reward

    # During training:
    log_episode_reward(csv_path, episode=1, reward=0.42, breakdown={...})

    # After training:
    plot_reward_curves("outputs/reward_log.csv", "outputs/reward_plot.png")
"""

from __future__ import annotations

import csv
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def log_episode_reward(
    csv_path: str | Path,
    episode: int,
    total_reward: float,
    tp_rate: float = 0.0,
    fp_rate: float = 0.0,
    fn_rate: float = 0.0,
    exp_accuracy: float = 0.0,
    terminal_bonus: float = 0.0,
    milestones: int = 0,
    phase: int = 1,
    task_id: str = "basic_oversight",
    breakdown: Optional[Dict[str, Any]] = None,
) -> None:
    """Append one episode reward to the CSV log.

    This is called after each GRPO episode to build the reward curve data.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not csv_path.exists() or csv_path.stat().st_size == 0

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "episode", "total_reward", "tp_rate", "fp_rate", "fn_rate",
                "exp_accuracy", "terminal_bonus", "milestones", "phase",
                "task_id", "timestamp", "breakdown_json",
            ])
        writer.writerow([
            episode,
            round(total_reward, 4),
            round(tp_rate, 4),
            round(fp_rate, 4),
            round(fn_rate, 4),
            round(exp_accuracy, 4),
            round(terminal_bonus, 4),
            milestones,
            phase,
            task_id,
            datetime.now().isoformat(),
            json.dumps(breakdown) if breakdown else "",
        ])


def plot_reward_curves(
    csv_path: str | Path,
    out_path: Optional[str | Path] = None,
    title: str = "SENTINEL Oversight Agent — GRPO Training",
) -> Optional[str]:
    """Generate reward curve plots from training CSV log.

    Returns the path to the saved plot, or None if plotting failed.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib/numpy not available — skipping reward plot")
        return None

    csv_path = Path(csv_path)
    if not csv_path.exists():
        logger.warning("No reward log at %s", csv_path)
        return None

    # Read CSV
    episodes, totals, tp_rates, fp_rates, fn_rates = [], [], [], [], []
    exp_accuracies, terminal_bonuses, milestones_list, phases = [], [], [], []

    with open(csv_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) < 9:
                continue
            episodes.append(int(row[0]))
            totals.append(float(row[1]))
            tp_rates.append(float(row[2]))
            fp_rates.append(float(row[3]))
            fn_rates.append(float(row[4]))
            exp_accuracies.append(float(row[5]))
            terminal_bonuses.append(float(row[6]))
            milestones_list.append(int(row[7]))
            phases.append(int(row[8]))

    if not episodes:
        logger.warning("No episodes in %s", csv_path)
        return None

    # Rolling average
    window = min(10, len(episodes))
    def rolling_avg(vals):
        return [
            sum(vals[max(0, i - window):i + 1]) / min(i + 1, window)
            for i in range(len(vals))
        ]

    rolling = rolling_avg(totals)

    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), height_ratios=[3, 2, 2])
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

    # --- Plot 1: Total Reward Curve ---
    ax1.plot(episodes, totals, alpha=0.25, color="#6366f1", marker="o",
             markersize=3, label="Per episode")
    ax1.plot(episodes, rolling, color="#6366f1", linewidth=2.5,
             label=f"Rolling avg ({window})")

    # Trend line
    z = np.polyfit(episodes, totals, 1)
    trend = np.poly1d(z)
    direction = "↑" if z[0] > 0 else "↓"
    ax1.plot(episodes, trend(episodes), color="#ef4444", linewidth=1.5,
             linestyle="--", label=f"Trend ({direction} {abs(z[0]):.4f}/ep)")

    # Phase transitions
    phase_changes = []
    for i in range(1, len(phases)):
        if phases[i] != phases[i - 1]:
            phase_changes.append(episodes[i])
            ax1.axvline(x=episodes[i], color="#f59e0b", linestyle="--",
                       alpha=0.7, linewidth=1.5)
            ax1.text(episodes[i], max(totals) * 0.95,
                    f"Phase {phases[i]}",
                    rotation=90, fontsize=8, color="#f59e0b", ha="right")

    ax1.set_ylabel("Total Reward")
    ax1.set_title("Oversight Quality Over Training")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.3)

    # Stats annotation
    mean_all = sum(totals) / len(totals)
    last10 = totals[-10:]
    mean_last10 = sum(last10) / len(last10)
    ax1.text(0.02, 0.02,
             f"Episodes: {len(episodes)} | Mean: {mean_all:.3f} | "
             f"Last-10 avg: {mean_last10:.3f} | Best: {max(totals):.3f}",
             transform=ax1.transAxes, fontsize=9, verticalalignment="bottom",
             bbox=dict(boxstyle="round", facecolor="#1e1e2e", edgecolor="#6366f1",
                      alpha=0.8),
             color="white")

    # --- Plot 2: Detection Quality ---
    ax2.plot(episodes, tp_rates, color="#10b981", linewidth=1.5,
             alpha=0.7, label="TP Rate (detection)")
    ax2.plot(episodes, rolling_avg(tp_rates), color="#10b981", linewidth=2.5)
    ax2.plot(episodes, fp_rates, color="#ef4444", linewidth=1.5,
             alpha=0.7, label="FP Rate (over-blocking)")
    ax2.plot(episodes, rolling_avg(fp_rates), color="#ef4444", linewidth=2.5)
    ax2.plot(episodes, fn_rates, color="#f59e0b", linewidth=1.5,
             alpha=0.7, label="FN Rate (missed)")
    ax2.plot(episodes, rolling_avg(fn_rates), color="#f59e0b", linewidth=2.5)

    ax2.set_ylabel("Rate")
    ax2.set_title("Detection Quality: TP vs FP vs FN")
    ax2.legend(loc="center right")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, 1.05)

    # --- Plot 3: Terminal Bonus + Milestones ---
    ax3.bar(episodes, terminal_bonuses, alpha=0.4, color="#a855f7",
            label="Terminal Bonus")
    ax3_twin = ax3.twinx()
    ax3_twin.plot(episodes, milestones_list, color="#ec4899", linewidth=2,
                  marker="s", markersize=3, label="Milestones (of 8)")
    ax3_twin.set_ylabel("Milestones Achieved", color="#ec4899")
    ax3_twin.set_ylim(-0.5, 8.5)
    ax3_twin.tick_params(axis="y", labelcolor="#ec4899")

    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Terminal Bonus")
    ax3.set_title("Terminal Reward & Milestone Progression")
    ax3.legend(loc="upper left")
    ax3_twin.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = Path(out_path) if out_path else csv_path.with_suffix(".png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor="#0a0a0f", edgecolor="none")
    plt.close()

    logger.info("Reward plot saved to %s", save_path)
    return str(save_path)


def plot_component_breakdown(
    csv_path: str | Path,
    out_path: Optional[str | Path] = None,
) -> Optional[str]:
    """Generate a heatmap of reward component evolution."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None

    csv_path = Path(csv_path)
    if not csv_path.exists():
        return None

    # Read breakdowns
    episodes = []
    breakdowns = []

    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) < 12 or not row[11]:
                continue
            episodes.append(int(row[0]))
            breakdowns.append(json.loads(row[11]))

    if not breakdowns:
        return None

    # Extract component values
    components = [
        "true_positive_catch", "explanation_accuracy", "correct_redirect",
        "audit_trail_quality", "incident_efficiency",
        "false_positive_penalty", "false_negative_penalty",
    ]

    data = np.zeros((len(components), len(breakdowns)))
    for j, bd in enumerate(breakdowns):
        for i, comp in enumerate(components):
            data[i, j] = bd.get(comp, 0.0)

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=-0.3, vmax=1.0)

    ax.set_yticks(range(len(components)))
    ax.set_yticklabels([c.replace("_", " ").title() for c in components])
    ax.set_xlabel("Episode")
    ax.set_title("Reward Component Evolution — 10-Component Breakdown")

    plt.colorbar(im, ax=ax, label="Component Score")
    plt.tight_layout()

    save_path = Path(out_path) if out_path else csv_path.with_name("component_heatmap.png")
    plt.savefig(save_path, dpi=150)
    plt.close()

    return str(save_path)
