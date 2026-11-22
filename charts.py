"""Matplotlib charts in the app's visual style (transparent background, warm palette)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

INK = "#1F2B24"
ACCENT = "#B4532A"
SAND = "#CDBE97"
TRACK = "#EAE2D0"
GRID = "#E4DAC6"
OLIVE = "#5B7A3A"
CLAY = "#B5502E"
MUTED = "#6B7466"


def _fig(width, height):
    fig, ax = plt.subplots(figsize=(width, height), dpi=200)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    return fig, ax


def hbar(items, xlabel="Confidence (%)", highlight=1):
    """Horizontal bars. items: list of (label, probability 0 to 1), best first."""
    n = len(items)
    labels = [label for label, _ in items][::-1]
    values = [p * 100 for _, p in items][::-1]
    fig, ax = _fig(6.4, 0.44 * n + 0.95)
    colors = [ACCENT if i >= n - highlight else SAND for i in range(n)]
    ax.barh(labels, values, color=colors, height=0.58)
    ax.set_xlim(0, 118)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    for y, v in enumerate(values):
        ax.text(v + 1.5, y, f"{v:.1f}%", va="center", fontsize=9, color=INK)
    fig.tight_layout()
    return fig


def range_profile(rows):
    """Where each input sits inside the typical range seen in the training data.
    rows: list of (label, value, typical_lo, typical_hi, value_text, lo_text, hi_text).
    Each track IS the typical range, so the dot position reads directly as low, middle or high."""
    n = len(rows)
    fig, ax = _fig(6.4, 0.66 * n + 0.9)
    labels = []
    for i, (label, value, lo, hi, text, lo_text, hi_text) in enumerate(rows):
        y = n - 1 - i
        pos = (value - lo) / (hi - lo) if hi > lo else 0.5
        outside = pos < 0 or pos > 1
        x = float(np.clip(pos, -0.04, 1.04))
        ax.barh(y, 1, color=TRACK, height=0.24, zorder=1)
        ax.barh(y, min(max(pos, 0), 1), color=SAND, height=0.24, zorder=2)
        ax.scatter([x], [y], s=80, color=CLAY if outside else ACCENT, edgecolor="white", linewidth=1.2, zorder=3)
        ax.text(x, y + 0.24, text + ("  (outside typical range)" if outside else ""), ha="center", va="bottom",
                fontsize=8.2, color=CLAY if outside else INK)
        ax.text(0, y - 0.2, lo_text, ha="left", va="top", fontsize=7.5, color=MUTED)
        ax.text(1, y - 0.2, hi_text, ha="right", va="top", fontsize=7.5, color=MUTED)
        labels.append(label)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels[::-1], fontsize=9, color=INK)
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.65, n - 0.2)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.legend(handles=[Patch(color=TRACK, label="Typical range in the training data"),
                       Line2D([0], [0], marker="o", color="none", markerfacecolor=ACCENT,
                              markeredgecolor="white", markersize=8, label="Your value")],
              loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2, frameon=False, fontsize=8.5, labelcolor=MUTED)
    fig.tight_layout()
    return fig


def recall_chart(labels, recall, ci_low, ci_high):
    """Per class recall with 95 percent intervals. Weakest class at the top."""
    order = np.argsort(recall)[::-1]
    labels = [labels[i] for i in order]
    r = np.asarray(recall)[order]
    lo = np.asarray(ci_low)[order]
    hi = np.asarray(ci_high)[order]
    colors = [CLAY if v < 0.90 else SAND if v < 0.95 else OLIVE for v in r]
    n = len(r)
    fig, ax = _fig(6.4, 0.27 * n + 1.1)
    ax.barh(range(n), r * 100, color=colors, height=0.62)
    ax.errorbar(r * 100, range(n), xerr=[(r - lo) * 100, (hi - r) * 100], fmt="none", ecolor=INK,
                elinewidth=0.9, capsize=2, alpha=0.7)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(60, 103)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Recall on the test set (%), with 95% interval", color=MUTED, fontsize=9)
    ax.legend(handles=[Patch(color=OLIVE, label="95% or higher"), Patch(color=SAND, label="90 to 95%"),
                       Patch(color=CLAY, label="Below 90%")],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=8, labelcolor=MUTED)
    fig.tight_layout()
    return fig


def threshold_chart(thresholds, coverage, accuracy):
    """Trade-off between confidence threshold, share of photos answered and accuracy."""
    fig, ax = _fig(6.4, 3.0)
    t = np.asarray(thresholds) * 100
    ax.plot(t, np.asarray(accuracy) * 100, color=ACCENT, marker="o", linewidth=2, label="Accuracy on answered photos")
    ax.plot(t, np.asarray(coverage) * 100, color=MUTED, marker="o", linewidth=2, linestyle="dashed",
            label="Share of photos answered")
    ax.set_ylim(0, 105)
    ax.set_xlabel("Minimum confidence required (%)", color=MUTED, fontsize=9)
    ax.set_ylabel("%", color=MUTED, fontsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", frameon=False, fontsize=8.5, labelcolor=MUTED)
    fig.tight_layout()
    return fig
