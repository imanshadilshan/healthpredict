"""Shared plotting style and figure-saving helpers.

Colors follow a validated colorblind-safe categorical order. Text always uses ink
colors; series colors are reserved for data marks.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .data_loader import PROJECT_ROOT

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# Categorical slots, fixed order (never cycled or reordered by rank).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
LOW_COST_COLOR = SERIES[0]   # class 0
HIGH_COST_COLOR = SERIES[1]  # class 1

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
NEUTRAL_MID = "#f0efec"

# Sequential (one hue, light -> dark) and diverging (blue <-> gray <-> red).
SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
DIVERGING = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#184f95", "#3987e5", "#9ec5f4", NEUTRAL_MID, "#f2a3a2", "#e34948", "#a92b2b"]
)


# One fixed color per algorithm, used in every modelling chart. The 9th entry
# (Voting) and the baselines use ink colors instead of a generated hue.
MODEL_ORDER = ["Logistic Regression", "KNN", "Decision Tree", "Random Forest", "SVM",
               "Naive Bayes", "Gradient Boosting", "XGBoost"]
MODEL_COLORS = {**dict(zip(MODEL_ORDER, SERIES)), "Voting": INK}


def model_color(name: str) -> str:
    return MODEL_COLORS.get(name, INK_MUTED)


def apply_style() -> None:
    """Set matplotlib defaults: light surface, recessive grid and axes, thin marks."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "legend.frameon": False,
        "legend.labelcolor": INK_SECONDARY,
        "lines.linewidth": 2,
        "axes.prop_cycle": plt.cycler(color=SERIES),
    })


def save_fig(fig: plt.Figure, name: str) -> Path:
    """Save a figure as PNG to reports/figures/<name>.png and return the path."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    return path


def rate_bar(ax, labels, rates, counts, overall: float, color: str = HIGH_COST_COLOR, title: str = "") -> None:
    """Horizontal bars of a per-category rate (%), with the overall rate as a reference line
    and n shown next to each bar."""
    y = range(len(labels))
    ax.barh(y, rates, color=color, height=0.62, edgecolor=SURFACE, linewidth=2)
    ax.axvline(overall, color=INK_SECONDARY, linewidth=1, linestyle="--", zorder=1)
    ax.set_yticks(list(y), labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(max(rates) * 1.25, overall * 1.3))
    for yi, r, n in zip(y, rates, counts):
        ax.text(r + ax.get_xlim()[1] * 0.01, yi, f"{r:.0f}%  (n={n:,})", va="center",
                fontsize=8.5, color=INK_SECONDARY, zorder=3,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=0.5))
    if title:
        ax.set_title(title)


def plot_variant_scores(table, algorithm: str, metrics=("roc_auc", "pr_auc", "f1"), best: str | None = None) -> plt.Figure:
    """One panel per metric: cross-validated mean (dot) and fold spread (± 1 std line)
    for each variant of one algorithm. The chosen variant is solid, the others lighter.
    Each panel is zoomed to its own range so small differences stay visible."""
    labels = {"roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "f1": "F1 (high-cost)",
              "recall": "Recall", "precision": "Precision", "accuracy": "Accuracy"}
    color = model_color(algorithm)
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.6 * len(metrics), 0.55 * len(table) + 1.5),
                             sharey=True)
    for ax, m in zip(axes, metrics):
        vals = table[m].astype(float)
        errs = table[f"{m}_std"].astype(float) if f"{m}_std" in table else vals * 0
        lo = np.nanmin(vals - errs.fillna(0)); hi = np.nanmax(vals + errs.fillna(0))
        pad = max((hi - lo) * 0.25, 0.01)
        ax.set_xlim(lo - pad, hi + pad * 2.2)
        for i, (name, v) in enumerate(vals.items()):
            if np.isnan(v):
                ax.text(0.02, i, "n/a (no scores)", va="center", fontsize=8.5, color=INK_MUTED,
                        transform=ax.get_yaxis_transform())
                continue
            alpha = 1.0 if name == best else 0.4
            e = 0 if np.isnan(errs.iloc[i]) else errs.iloc[i]
            ax.plot([v - e, v + e], [i, i], color=color, alpha=alpha, linewidth=2, solid_capstyle="round")
            ax.plot(v, i, "o", color=color, alpha=alpha, markersize=9, markeredgecolor=SURFACE, markeredgewidth=1.5)
            ax.text(v + e + pad * 0.15, i, f"{v:.3f}", va="center", fontsize=8.5, color=INK_SECONDARY)
        ax.set_yticks(range(len(table)), table.index)
        ax.set_ylim(len(table) - 0.5, -0.5)
        ax.grid(axis="y", visible=False)
        ax.set_title(labels.get(m, m), fontsize=11)
    title = f"{algorithm}: variants compared (5-fold CV mean ± 1 std)"
    if best:
        title += f"  ·  chosen: {best}"
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig
