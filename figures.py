"""Figure generation.

One function per figure, each saving to figures/ and returning the path so the
pipeline can log what it produced. All axes carry units, legends and captions in
the form required by the case brief.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from . import config as C
from .segment import load_audio, compute_track, longest_run, all_runs, \
    merge_runs, steady_mask_from_track, C as SC

plt.rcParams.update(C.plt_rc)


def _save(fig, stem: str, caption: str | None = None) -> "C.Path":
    path = C.figure_path(stem)
    if caption:
        fig.text(0.01, 0.005, caption, ha="left", va="bottom",
                 fontsize=7.5, style="italic", color="#444444")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _style_time_axis(ax, fs: int) -> None:
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (normalised, full scale = 1.0)")


# ------------------------------------------------------------------ Figure 1
def figure1_class_inventory(inv: pd.DataFrame) -> "C.Path":
    """Class counts and duration distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))

    labels = [C.CLASS_LABELS[c] for c in C.CLASSES]
    colors = [C.CLASS_COLORS[c] for c in C.CLASSES]
    counts = [int((inv["cls"] == c).sum()) for c in C.CLASSES]
    minutes = [inv.loc[inv["cls"] == c, "duration_s"].sum() / 60.0 for c in C.CLASSES]

    ax = axes[0]
    bars = ax.bar(labels, counts, color=colors, edgecolor="black", linewidth=0.6, width=0.62)
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_ylabel("Number of recordings")
    ax.set_title("(a) Recordings per class")
    ax.set_ylim(0, max(counts) * 1.18)

    ax = axes[1]
    bars = ax.bar(labels, minutes, color=colors, edgecolor="black", linewidth=0.6, width=0.62)
    ax.bar_label(bars, labels=[f"{m:.0f}" for m in minutes], padding=3, fontsize=9)
    ax.set_ylabel("Total duration (min)")
    ax.set_title("(b) Total duration per class")
    ax.set_ylim(0, max(minutes) * 1.18)

    fig.suptitle("Figure 1. Dataset inventory: 293 recordings, "
                 f"{inv['duration_s'].sum()/3600:.1f} h, mono 16-bit "
                 f"{C.FS_EXPECTED//1000} kHz", y=1.02, fontsize=11)
    fig.tight_layout()
    return _save(fig, "fig01_class_inventory",
                 "Counts are near-balanced, so raw accuracy is not inflated by class "
                 "imbalance. Class sizes 78/72/71/72.")


# ------------------------------------------------------------------ Figure 2
def figure2_segmentation(inv: pd.DataFrame, rels: list[str]) -> "C.Path":
    """Level and dominant-frequency tracks with extracted hover windows."""
    n = len(rels)
    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 5.4), sharex="col")
    axes = np.atleast_2d(axes)

    for j, rel in enumerate(rels):
        cls = inv.loc[inv["relpath"] == rel, "cls"].iloc[0]
        fs, x = load_audio(C.DATASET_DIR / rel)
        tr = compute_track(x, fs)
        dt = tr.frame_dt

        pmask = tr.prominence_db > SC.TONE_PROMINENCE_DB
        p_len, p_a, p_b = longest_run(pmask, SC.MIN_POWERED_FRAMES)
        if not p_len:
            p_a, p_b = 0, tr.level_db.size
        runs = merge_runs(
            all_runs(steady_mask_from_track(tr.f0) & pmask, min_len=SC.MIN_STEADY_FRAMES),
            gap=SC.STEADY_MERGE_GAP_FRAMES,
        )
        runs = [(a, b) for a, b in runs if (b - a) * dt >= SC.MIN_STEADY_SECONDS]

        ax = axes[0, j]
        ax.plot(tr.t, tr.level_db, lw=0.8, color="#555555", label="Broadband level")
        ax2 = ax.twinx()
        ax2.plot(tr.t, tr.prominence_db, lw=0.8, color="#e377c2", alpha=0.85,
                 label="Tone prominence")
        ax2.axhline(SC.TONE_PROMINENCE_DB, ls=":", lw=0.9, color="#c56db0")
        ax2.set_ylabel("Tone prominence (dB)", color="#c56db0")
        ax2.tick_params(axis="y", labelcolor="#c56db0", labelsize=8)
        ax2.grid(False)
        ax2.set_ylim(0, max(25.0, float(np.percentile(tr.prominence_db, 99)) * 1.1))
        for ra, rb in runs:
            ax.axvspan(tr.t[ra], tr.t[rb - 1], color="#2ca02c", alpha=0.22)
        ax.set_ylabel("Broadband level (dB, 40-3000 Hz)")
        ax.set_title(f"{C.CLASS_LABELS[cls]}  -  {rel.replace(os.sep, '/')}",
                     fontsize=8.5)
        if j == 0:
            ax.legend(loc="lower right", framealpha=0.9)

        ax = axes[1, j]
        ax.plot(tr.t, tr.f0, lw=0.8, color="#1f4e9c",
                label="Frame dominant $f_0$")
        for ra, rb in runs:
            ax.axvspan(tr.t[ra], tr.t[rb - 1], color="#2ca02c", alpha=0.22)
        ax.set_ylabel("Dominant frequency (Hz)")
        ax.set_xlabel("Time (s)")
        ax.set_xlim(tr.t[0], tr.t[-1])
        if j == 0:
            ax.legend(loc="upper right", framealpha=0.9)

        if runs:
            total = sum(rb - ra for ra, rb in runs) * dt
            f0m = np.median(tr.f0[runs[0][0]:runs[-1][1]])
            note = (f"{len(runs)} steady run(s), {total:.1f} s total\n"
                    f"median $f_0$ in steady region = {f0m:.0f} Hz")
        else:
            note = "no steady run detected\n(powered segment used instead)"
        ax.text(0.5, -0.34, note, transform=ax.transAxes, ha="center", va="top",
                fontsize=7.5, color="#2ca02c")

    handles = [Patch(facecolor="#2ca02c", alpha=0.22, label="Steady-RPM region"),
               Line2D([0], [0], color="#555555", lw=1.2, label="Broadband level"),
               Line2D([0], [0], color="#e377c2", lw=1.2, label="Tone prominence (right axis)"),
               Line2D([0], [0], color="#1f4e9c", lw=1.2, label="Dominant $f_0$"),
               Line2D([0], [0], color="#c56db0", lw=1.2, ls=":", label="Powered threshold")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Figure 2. Steady-hover segmentation: broadband level (top row) and "
                 "frame-dominant frequency (bottom row)", y=1.01, fontsize=11)
    fig.tight_layout()
    return _save(fig, "fig02_segmentation",
                 "Green = detected constant-RPM region, the only part used for feature "
                 "extraction. A full-file spectrum would average across the whole "
                 "take-off-to-landing throttle sweep.")
