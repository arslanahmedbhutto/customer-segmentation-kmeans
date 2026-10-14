"""Figures for the report and notebook."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from segmentation.data import AGE, INCOME, SPEND

SURFACE, INK, INK_2, GRID, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#b9b8b2"
# Categorical slots 1-5 in fixed order (validated palette)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "^", "D", "P", "X", "v", "*"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 13,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.color": INK_2, "ytick.color": INK_2, "font.size": 10.5,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _finish(fig, path: Path | None):
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def distributions(df: pd.DataFrame, path=None):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, col, title in zip(axes, [AGE, INCOME, SPEND], ["Age", "Annual income (k$)", "Spending score (1-100)"]):
        ax.hist(df[col], bins=20, color=SERIES[0], edgecolor=SURFACE, linewidth=1.5)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Customers")
    fig.suptitle("Customer base: 200 customers", x=0.01, ha="left", fontweight="bold", fontsize=13, color=INK)
    fig.tight_layout()
    return _finish(fig, path)


def k_selection(scores: pd.DataFrame, chosen_k: int, path=None):
    panels = [("inertia", "Elbow: within-cluster SS", "lower is better"),
              ("silhouette", "Silhouette", "higher is better"),
              ("davies_bouldin", "Davies–Bouldin", "lower is better"),
              ("calinski_harabasz", "Calinski–Harabasz", "higher is better")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, (col, title, hint) in zip(axes, panels):
        ax.plot(scores["k"], scores[col], "-o", color=SERIES[0], lw=2, ms=6)
        row = scores[scores["k"] == chosen_k].iloc[0]
        ax.plot(chosen_k, row[col], "o", ms=13, mfc="none", mec=SERIES[1], mew=2.5)
        ax.set_title(f"{title}\n({hint})", fontsize=10.5)
        ax.set_xlabel("K")
        ax.set_xticks(scores["k"])
        ax.grid(color=GRID, lw=0.8)
    fig.suptitle(f"Choosing the number of segments (orange ring = K={chosen_k})", x=0.01, ha="left",
                 fontweight="bold", fontsize=13, color=INK)
    fig.tight_layout()
    return _finish(fig, path)


def segments_scatter(df: pd.DataFrame, labels, profiles: pd.DataFrame, centers=None, path=None):
    fig, ax = plt.subplots(figsize=(9, 6.2))
    order = profiles["cluster"].tolist()
    for i, c in enumerate(order):
        m = np.asarray(labels) == c
        name = profiles.loc[profiles["cluster"] == c, "segment"].iloc[0]
        ax.scatter(df.loc[m, INCOME], df.loc[m, SPEND], s=55, color=SERIES[i], marker=MARKERS[i],
                   edgecolor=SURFACE, linewidth=1, label=f"{name} ({m.sum()})")
    if centers is not None:
        ax.scatter(centers[:, 0], centers[:, 1], s=260, marker="X", color=INK, edgecolor=SURFACE,
                   linewidth=1.5, label="Segment centre", zorder=5)
    ax.set_xlabel("Annual income (k$)")
    ax.set_ylabel("Spending score (1-100)")
    ax.set_title("Customer segments (K-Means, K=5)")
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    return _finish(fig, path)


def profile_bars(profiles: pd.DataFrame, path=None):
    cols = [("avg_income_k", "Avg income (k$)"), ("avg_spending_score", "Avg spending score"),
            ("avg_age", "Avg age"), ("share_%", "Share of customers (%)")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True)
    y = np.arange(len(profiles))[::-1]
    for ax, (col, title) in zip(axes, cols):
        ax.barh(y, profiles[col], color=SERIES[: len(profiles)], height=0.6)
        for yi, v in zip(y, profiles[col]):
            ax.text(v, yi, f" {v:.0f}", va="center", fontsize=9.5, color=INK)
        ax.set_title(title, fontsize=11)
        ax.set_xlim(0, profiles[col].max() * 1.25)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y, profiles["segment"])
    fig.suptitle("Segment profiles", x=0.01, ha="left", fontweight="bold", fontsize=13, color=INK)
    fig.tight_layout()
    return _finish(fig, path)


def stability(aris: np.ndarray, path=None):
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    lo = min(0.8, np.floor(aris.min() * 20) / 20)
    ax.hist(aris, bins=20, range=(lo, 1), color=SERIES[0], edgecolor=SURFACE, linewidth=1.5)
    ax.axvline(np.median(aris), color=SERIES[1], lw=2, ls="--")
    ax.text(np.median(aris), ax.get_ylim()[1] * 0.92, f"median ARI {np.median(aris):.2f}  ", color=SERIES[1],
            ha="right")
    ax.set_xlabel("Adjusted Rand Index vs the full-data segmentation (1 = identical)")
    ax.set_ylabel("Bootstrap resamples")
    ax.set_title(f"Segment stability over {len(aris)} bootstrap resamples")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    return _finish(fig, path)


def algorithm_comparison(df: pd.DataFrame, labels: dict, path=None):
    fig, axes = plt.subplots(1, len(labels), figsize=(5 * len(labels), 4.4), sharey=True)
    for ax, (name, lab) in zip(axes, labels.items()):
        for i, c in enumerate(sorted(set(lab))):
            m = lab == c
            ax.scatter(df.loc[m, INCOME], df.loc[m, SPEND], s=30, color=SERIES[i], marker=MARKERS[i],
                       edgecolor=SURFACE, linewidth=0.8)
        ax.set_title(name, fontsize=11.5)
        ax.set_xlabel("Annual income (k$)")
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Spending score (1-100)")
    fig.suptitle("Same K=5, three algorithms (colours are per-algorithm cluster ids)", x=0.01, ha="left",
                 fontweight="bold", fontsize=13, color=INK)
    fig.tight_layout()
    return _finish(fig, path)
