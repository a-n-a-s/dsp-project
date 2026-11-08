"""Shared plotting helpers for the class-separation figures.

Kept separate from :mod:`src.figures` (which drives the pipeline phases) because
these figures belong to the statistical exploration: they are run on demand from
``test/`` and read the cached metric table rather than re-deriving segments.

Two conventions are enforced here so the figures stay honest:

* Every distribution panel draws **one point per recording**, taken as the median
  of that recording's windows. 3549 overlapping windows are not 3549
  independent observations and plotting them raw would overstate the evidence.
* Any panel that mixes classes with a known confound (normal is ~15 dB quieter
  and at a different blade-pass frequency) is labelled as such rather than
  quietly presented as class evidence.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config as C  # noqa: E402

plt.rcParams.update(C.plt_rc)

BROKEN = [c for c in C.CLASSES if c != "normal"]

# Metrics whose class contrast is dominated by recording gain or by blade-pass
# frequency rather than by damage. Normal files sit ~15 dB quieter at ~227 Hz
# against ~129 Hz for the damaged classes, so a metric that responds strongly to
# either variable will separate normal from broken for free.
CONFOUND_NOTE = ("normal recordings are ~15 dB quieter and run a different "
                 "blade-pass frequency than the damaged classes")


def apply_style() -> None:
    """Re-apply the project rcParams after a figure has changed them."""
    plt.rcParams.update(C.plt_rc)


def save(fig, stem: str, caption: str | None = None):
    """Save to figures/<stem>.png with an optional footnote, return the path."""
    apply_style()
    path = C.figure_path(stem)
    if caption:
        fig.text(0.005, 0.004, caption, ha="left", va="bottom",
                 fontsize=7.0, style="italic", color="#444444", wrap=True)
    fig.savefig(path, dpi=C.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def recording_medians(metrics: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Collapse the window table to one row per recording (median within file)."""
    keep = ["cls", "relpath"] + [c for c in columns if c in metrics.columns]
    return metrics[keep].groupby(["cls", "relpath"], as_index=False).median(
        numeric_only=True)


def jitter(n: int, width: float = 0.16, seed: int = C.RANDOM_STATE) -> np.ndarray:
    """Deterministic horizontal offsets for a strip plot."""
    return np.random.RandomState(seed).uniform(-width, width, n)


def strip_panel(ax, rec: pd.DataFrame, metric: str, classes: list[str] | None = None,
                width: float = 0.16, marker: str = "o", ms: float = 3.4,
                alpha: float = 0.75, show_box: bool = True) -> dict:
    """Scatter every recording on a categorical x-axis, with a box overlay.

    Returns a dict of per-class quartiles for the caller to annotate.
    """
    classes = classes or C.CLASSES
    summary = {}
    for i, cls in enumerate(classes):
        v = pd.to_numeric(rec.loc[rec["cls"] == cls, metric], errors="coerce").dropna()
        if v.empty:
            summary[cls] = dict(n=0, median=np.nan)
            continue
        if show_box:
            bp = ax.boxplot([v.values], positions=[i], widths=0.46,
                            showfliers=False, patch_artist=True,
                            medianprops=dict(color="#111111", lw=1.4),
                            boxprops=dict(facecolor=C.CLASS_COLORS[cls], alpha=0.18,
                                          edgecolor="#555555", lw=0.9),
                            whiskerprops=dict(color="#777777", lw=0.9),
                            capprops=dict(color="#777777", lw=0.9))
            del bp
        ax.scatter(np.full(len(v), i) + jitter(len(v), width), v.values,
                   s=ms ** 2 * 2.2, marker=marker, alpha=alpha,
                   color=C.CLASS_COLORS[cls], edgecolors="none", zorder=3)
        summary[cls] = dict(n=int(len(v)),
                            median=float(np.median(v)),
                            q25=float(np.percentile(v, 25)),
                            q75=float(np.percentile(v, 75)))
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([C.CLASS_LABELS[c] for c in classes])
    ax.set_ylabel(metric)
    ax.set_xlim(-0.6, len(classes) - 0.4)
    return summary


def annotate_medians(ax, summary: dict[str, dict], classes: list[str],
                     dy: float = 0.012) -> None:
    """Print each class median just under its box."""
    vals = [summary[c]["median"] for c in classes if np.isfinite(summary[c]["median"])]
    if not vals:
        return
    span = max(vals) - min(vals)
    if not span:
        span = max(abs(vals[0]), 1.0)
    for i, cls in enumerate(classes):
        m = summary[cls]["median"]
        if np.isfinite(m):
            ax.text(i, m - dy * span, f"{m:.2f}", ha="center", va="top",
                    fontsize=7.5, color="#111111")


def class_legend_handles(counts: dict[str, int] | None = None,
                         classes: list[str] | None = None):
    """Legend entries in the canonical class order, optionally with recording counts."""
    from matplotlib.lines import Line2D

    classes = classes or C.CLASSES
    counts = counts or {}
    return [Line2D([], [], marker="o", ls="", ms=5,
                   markerfacecolor=C.CLASS_COLORS[c],
                   markeredgecolor="none",
                   label=C.CLASS_LABELS[c] if c not in counts
                   else f"{C.CLASS_LABELS[c]} (n={counts[c]})")
            for c in classes]


def roc_curve_binary(y: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ROC from midranks, matching how the AUC in src.stats is computed."""
    order = np.argsort(score, kind="mergesort")
    s_sorted = score[order]
    y_sorted = y[order]
    distinct = np.where(np.diff(s_sorted))[0]
    idx = np.r_[distinct, s_sorted.size - 1]

    tps = np.cumsum(y_sorted)[idx]
    fps = 1 + idx - tps
    n_pos = float(y_sorted.sum())
    n_neg = float(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    tpr = np.r_[0.0, tps / n_pos]
    fpr = np.r_[0.0, fps / n_neg]
    return fpr, tpr


def roc_band(y: np.ndarray, score: np.ndarray, n_boot: int = 300,
             alpha: float = 0.20, seed: int = C.RANDOM_STATE) -> tuple[np.ndarray, ...]:
    """Bootstrap ROC curves by resampling observations, returning an FPR grid band.

    Used for the figure only; the reported AUC intervals in ``src.stats`` resample
    recordings instead, which is the correct unit and gives wider intervals.
    """
    grid = np.linspace(0, 1, 101)
    curves = []
    rng = np.random.RandomState(seed)
    for _ in range(n_boot):
        idx = rng.randint(0, y.size, y.size)
        if len(np.unique(y[idx])) < 2:
            continue
        fpr, tpr = roc_curve_binary(y[idx], score[idx])
        curves.append(np.interp(grid, fpr, tpr))
    if not curves:
        return grid, grid, grid
    A = np.vstack(curves)
    return grid, np.percentile(A, 100 * alpha / 2, axis=0), \
        np.percentile(A, 100 * (1 - alpha / 2), axis=0)


def clustered_order(corr: pd.DataFrame) -> list[str]:
    """Order metrics so strongly correlated ones sit next to each other."""
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform

    c = ((corr.abs().fillna(0.0) + corr.abs().fillna(0.0).T) / 2.0).to_numpy(float)
    c = np.array(c, copy=True)
    np.fill_diagonal(c, 1.0)
    dist = 1.0 - c
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0
    dist[dist < 0] = 0.0
    dist = np.array(dist, copy=True)
    if dist.shape[0] < 3:
        return list(corr.index)
    try:
        link = linkage(squareform(dist, checks=False), method="average")
        return list(corr.index[dendrogram(link, no_plot=True)["leaves"]])
    except Exception:
        return list(corr.index)