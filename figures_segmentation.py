"""Phase 2 diagnostic figures.

The point of this module is to make segmentation verifiable. Segmentation is
the one step in the pipeline that can silently invalidate everything after it,
so the figures here show, per class:

  fig02  overview      full track: level, tone prominence, f0, with the detected
                       steady-RPM regions shaded
  fig03  steady zoom   a 20 s slice around the largest steady region, at frame
                       resolution, with the individual 4 s windows marked
  fig04  window spectra  one Welch PSD per analysis window, plus the band-
                       normalised class mean, showing that within a class the
                       windows agree and across classes they differ
  fig05  yield         steady duration, run count, fallback reasons, window count
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from . import config as C
from . import dsp
from .segment import load_audio, compute_track, longest_run, all_runs, \
    merge_runs, steady_mask_from_track, C as SC

plt.rcParams.update(C.plt_rc)

STEADY_FACE = "#2ca02c"
LEVEL_C = "#555555"
PROM_C = "#c56db0"
F0_C = "#1f4e9c"


def _steady_runs(tr) -> list[tuple[int, int]]:
    """The same steady-run logic the segmenter uses, for plotting."""
    dt = tr.frame_dt
    pmask = tr.prominence_db > SC.TONE_PROMINENCE_DB
    runs = merge_runs(
        all_runs(steady_mask_from_track(tr.f0) & pmask, min_len=SC.MIN_STEADY_FRAMES),
        gap=SC.STEADY_MERGE_GAP_FRAMES,
    )
    return [(a, b) for a, b in runs if (b - a) * dt >= SC.MIN_STEADY_SECONDS]


def _shade(ax, runs, tr, label_once: bool = False):
    for i, (a, b) in enumerate(runs):
        ax.axvspan(tr.t[a], tr.t[min(b, tr.t.size) - 1], color=STEADY_FACE,
                   alpha=0.20, lw=0,
                   label="Detected steady-RPM region" if (label_once and i == 0) else None)


def _save(fig, stem: str, caption: str) -> "C.Path":
    path = C.figure_path(stem)
    fig.text(0.01, -0.02, caption, ha="left", va="top",
             fontsize=7.5, style="italic", color="#444444", wrap=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _pick_representative(inv: pd.DataFrame, seg: pd.DataFrame, n_files: int = 1) -> list[str]:
    """Pick the file per class whose steady duration is closest to the class median.

    A median-length file is more representative than the longest one, which
    tends to have an unusually long stable hover.
    """
    picks = []
    for c in C.CLASSES:
        sub = seg[seg["cls"] == c]
        med = sub["hover_s"].median()
        sub = sub.iloc[(sub["hover_s"] - med).abs().argsort()[:n_files]]
        picks.extend(sub["relpath"].tolist())
    return picks


# ------------------------------------------------------------------ fig 02
def figure_segmentation_overview(inv: pd.DataFrame, seg: pd.DataFrame,
                                 n_files: int = 1) -> "C.Path":
    """Full-recording tracks with detected steady regions, per class."""
    rels = _pick_representative(inv, seg, n_files)
    ncol = len(rels)
    fig, axes = plt.subplots(3, ncol, figsize=(4.0 * ncol, 7.6), sharex="col")

    for j, rel in enumerate(rels):
        cls = inv.loc[inv["relpath"] == rel, "cls"].iloc[0]
        fs, x = load_audio(C.DATASET_DIR / rel)
        tr = compute_track(x, fs)
        runs = _steady_runs(tr)
        srow = seg[seg["relpath"] == rel].iloc[0]

        ax = axes[0, j]
        ax.plot(tr.t, tr.level_db, lw=0.7, color=LEVEL_C)
        _shade(ax, runs, tr, label_once=(j == 0))
        ax.set_ylabel("Broadband level (dB re 1 unit/Hz)\nband 40-3000 Hz")
        ax.set_title(f"{C.CLASS_LABELS[cls]}\n{rel.replace(chr(92), '/')}",
                     fontsize=8)

        ax = axes[1, j]
        ax.plot(tr.t, tr.prominence_db, lw=0.7, color=PROM_C, alpha=0.9)
        ax.axhline(SC.TONE_PROMINENCE_DB, ls="--", lw=0.9, color="#8c2f78")
        _shade(ax, runs, tr)
        ax.set_ylabel("Tone prominence (dB)\nabove local median")
        ax.text(0.02, 0.06, f"threshold {SC.TONE_PROMINENCE_DB:g} dB",
                transform=ax.transAxes, fontsize=7.5, color="#8c2f78")

        ax = axes[2, j]
        ax.plot(tr.t, tr.f0, lw=0.7, color=F0_C)
        _shade(ax, runs, tr)
        ax.set_ylabel("Frame dominant $f_0$ (Hz)")
        ax.set_xlabel("Time (s)")
        ax.set_xlim(tr.t[0], tr.t[-1])

        if runs:
            tot = sum(b - a for a, b in runs) * tr.frame_dt
            f0m = np.median(tr.f0[runs[0][0]:runs[-1][1]])
            note = (f"{len(runs)} region(s), {tot:.1f} s kept\n"
                    f"median $f_0$ kept = {f0m:.0f} Hz")
        else:
            note = ("no steady region found\n"
                    f"file excluded ({srow['fallback_reason']})")
        axes[2, j].text(0.5, -0.40, note, transform=axes[2, j].transAxes,
                        ha="center", va="top", fontsize=7.5,
                        color=STEADY_FACE if runs else "#d62728")

    handles = [
        Patch(facecolor=STEADY_FACE, alpha=0.20, label="Detected steady-RPM region (used for features)"),
        Line2D([0], [0], color=LEVEL_C, lw=1.2, label="Broadband level (left axis, top)"),
        Line2D([0], [0], color=PROM_C, lw=1.2, label="Tone prominence (middle)"),
        Line2D([0], [0], color="#8c2f78", lw=1.2, ls="--", label="Powered threshold"),
        Line2D([0], [0], color=F0_C, lw=1.2, label="Frame dominant $f_0$ (bottom)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.10))
    fig.suptitle("Figure 2. Segmentation of the steady-RPM region.\n"
                 "Each recording is a full flight (take-off to landing), so the rotor "
                 "speed sweeps; only the shaded constant-speed regions are used for "
                 "feature extraction.", y=1.0, fontsize=11)
    fig.tight_layout()
    return _save(fig, "fig02_segmentation_overview",
                 "Read down a column: the shaded regions are where the $f_0$ trace "
                 "(bottom) is flat and tone prominence (middle) is above threshold. "
                 "Those two conditions together define what 'steady' means here.")


# ------------------------------------------------------------------ fig 03
def figure_steady_zoom(inv: pd.DataFrame, seg: pd.DataFrame, primary: pd.DataFrame,
                       zoom_s: float = 20.0) -> "C.Path":
    """Frame-resolution slice around the largest steady region, with windows marked."""
    rels = _pick_representative(inv, seg, 1)
    n = len(rels)
    fig, axes = plt.subplots(2, n, figsize=(4.0 * n, 5.2), sharex="col")

    for j, rel in enumerate(rels):
        cls = inv.loc[inv["relpath"] == rel, "cls"].iloc[0]
        fs, x = load_audio(C.DATASET_DIR / rel)
        tr = compute_track(x, fs)
        runs = _steady_runs(tr)
        if not runs:
            continue
        a, b = max(runs, key=lambda r: r[1] - r[0])
        centre = (a + b) / 2.0
        half = zoom_s / 2.0 / tr.frame_dt
        lo, hi = max(0, int(centre - half)), min(tr.t.size, int(centre + half))
        t = tr.t[lo:hi]

        wins = primary[primary["relpath"] == rel]
        keep = wins[(wins["start_s"] >= t[0]) & (wins["start_s"] <= t[-1])]

        ax = axes[0, j]
        ax.plot(t, tr.f0[lo:hi], lw=1.1, color=F0_C, marker="o", ms=2.2,
                label="Frame dominant $f_0$")
        for _, w in keep.iterrows():
            ax.axvline(w["start_s"], color="#333333", lw=0.6, alpha=0.55)
            ax.axvline(w["start_s"] + w["duration_s"], color="#333333", lw=0.6,
                       alpha=0.55)
        ax.axvspan(tr.t[a], tr.t[min(b, tr.t.size) - 1], color=STEADY_FACE, alpha=0.15)
        ax.set_ylabel("Frame dominant $f_0$ (Hz)")
        ax.set_title(f"{C.CLASS_LABELS[cls]}: {zoom_s:g} s around the largest "
                     f"steady region\n(rel. span {tr.t[a]:.1f}-{tr.t[min(b, tr.t.size)-1]:.1f} s)",
                     fontsize=8.5)
        ax.text(0.5, -0.30, f"{len(keep)} analysis window(s) marked, "
                f"{C.SEGMENT_SECONDS:g} s each",
                transform=ax.transAxes, ha="center", va="top", fontsize=8,
                color="#333333")

        ax = axes[1, j]
        ax.plot(t, tr.level_db[lo:hi], lw=1.0, color=LEVEL_C, label="Broadband level")
        ax2 = ax.twinx()
        ax2.plot(t, tr.prominence_db[lo:hi], lw=1.0, color=PROM_C, alpha=0.9,
                 label="Tone prominence")
        ax2.axhline(SC.TONE_PROMINENCE_DB, ls="--", lw=0.9, color="#8c2f78")
        ax2.set_ylabel("Tone prominence (dB)", color="#8c2f78")
        ax2.tick_params(axis="y", labelcolor="#8c2f78", labelsize=8)
        ax2.grid(False)
        ax2.set_ylim(0, max(25.0, float(tr.prominence_db.max()) * 1.15))
        for _, w in keep.iterrows():
            ax.axvline(w["start_s"], color="#333333", lw=0.6, alpha=0.55)
            ax.axvline(w["start_s"] + w["duration_s"], color="#333333", lw=0.6,
                       alpha=0.55)
        ax.axvspan(tr.t[a], tr.t[min(b, tr.t.size) - 1], color=STEADY_FACE, alpha=0.15)
        ax.set_ylabel("Broadband level (dB)\n40-3000 Hz")
        ax.set_xlabel("Time (s)")
        ax.set_xlim(t[0], t[-1])

    fig.suptitle("Figure 3. Inside the steady region: where each "
                 f"{C.SEGMENT_SECONDS:g} s analysis window is placed", y=1.01,
                 fontsize=11)
    fig.tight_layout()
    return _save(fig, "fig03_steady_zoom",
                 "Black vertical lines delimit the analysis windows. Because $f_0$ is "
                 "flat inside the region, a window does not straddle an RPM change, so "
                 "the band and harmonic ratios measured from it describe one operating "
                 "condition rather than an average of several.")


# ------------------------------------------------------------------ fig 04
def figure_window_spectra(inv: pd.DataFrame, primary: pd.DataFrame,
                          n_per_class: int = 25, seed: int = C.RANDOM_STATE) -> "C.Path":
    """Welch PSD of individual windows plus the band-normalised class mean."""
    win = int(round(C.SEGMENT_SECONDS * C.FS_EXPECTED))
    fig, axes = plt.subplots(2, 4, figsize=(14.5, 6.6))

    rng = np.random.RandomState(seed)
    for i, c in enumerate(C.CLASSES):
        files = primary[primary["cls"] == c]["relpath"].unique()
        rows = []
        for rel in files:
            w = primary[(primary["relpath"] == rel) & (primary["cls"] == c)]
            if len(w):
                rows.append(w.iloc[rng.randint(len(w))])
        rows = pd.DataFrame(rows).head(n_per_class)
        if rows.empty:
            continue

        cache = {}
        for rel in rows["relpath"].unique():
            fs, x = load_audio(C.DATASET_DIR / rel)
            cache[rel] = x

        # (a) absolute PSD of each window, log-log
        ax = axes[0, i]
        for _, r in rows.iterrows():
            x = cache[r["relpath"]]
            s = int(r["start_sample"])
            f, P = dsp.welch_psd(x[s:s + win], C.FS_EXPECTED)
            m = (f >= 40) & (f <= 4000)
            if not m.any():
                continue
            ax.plot(f[m], P[m] + 1e-14, lw=0.55,
                    color=C.CLASS_COLORS[c], alpha=0.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(40, 4000)
        ax.set_xlabel("Frequency (Hz, log)")
        ax.set_ylabel("PSD (dB re 1 unit/Hz)")
        ax.set_title(f"(a) {C.CLASS_LABELS[c]}: {len(rows)} individual windows,\n"
                     f"absolute level (note the level differences)", fontsize=9)

        # (b) band-normalised class mean
        wins = [cache[r["relpath"]][int(r["start_sample"]):int(r["start_sample"]) + win]
                for _, r in rows.iterrows()]
        fref, mean_db, std_db = dsp.aggregate_psd(wins, C.FS_EXPECTED)
        ax = axes[1, i]
        m = fref <= 1000
        top = mean_db[m].max()
        ax.plot(fref[m], mean_db[m] - top, lw=1.7, color=C.CLASS_COLORS[c],
                label="Class mean")
        ax.fill_between(fref[m], mean_db[m] - top - std_db[m],
                        mean_db[m] - top + std_db[m],
                        color=C.CLASS_COLORS[c], alpha=0.22, label=r"$\pm 1$ std")
        ax.set_xlim(40, 1000)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Normalised PSD (dB, peak = 0 dB)")
        ax.set_title(f"(b) {C.CLASS_LABELS[c]}: normalised to total 40-4000 Hz power",
                     fontsize=9)
        if i == 0:
            ax.legend(loc="upper right", framealpha=0.9)

    fig.suptitle("Figure 4. Spectra of the analysis windows. Top: absolute PSD, which "
                 "shows the level confound.\nBottom: each window normalised by its own "
                 "40-4000 Hz total, which removes it and leaves spectral shape.",
                 y=1.0, fontsize=11)
    fig.tight_layout()
    return _save(fig, "fig04_window_spectra",
                 "In (a) the four classes sit at visibly different levels, which is a "
                 "property of the recording sessions rather than of the propeller. "
                 "After normalisation in (b) the curves differ in shape instead, which "
                 "is the part that can carry fault information.")


# ------------------------------------------------------------------ fig 05
def figure_segmentation_yield(seg: pd.DataFrame, primary: pd.DataFrame) -> "C.Path":
    """Steady duration, run count, fallback reasons and window yield per class."""
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.4))
    colours = [C.CLASS_COLORS[c] for c in C.CLASSES]
    xs = np.arange(len(C.CLASSES))

    ax = axes[0, 0]
    data = [seg.loc[seg["cls"] == c, "hover_s"].values for c in C.CLASSES]
    bp = ax.boxplot(data, positions=xs, widths=0.55, patch_artist=True,
                    medianprops=dict(color="black", lw=1.4))
    for patch, col in zip(bp["boxes"], colours):
        patch.set_facecolor(col); patch.set_alpha(0.55)
    for i, d in enumerate(data):
        ax.scatter(np.random.RandomState(0).normal(i, 0.07, len(d)), d,
                   s=6, color="black", alpha=0.35, zorder=3)
    ax.set_xticks(xs); ax.set_xticklabels([C.CLASS_LABELS[c] for c in C.CLASSES])
    ax.set_ylabel("Total steady-RPM time per file (s)")
    ax.set_title("(a) Steady-RPM time available per recording")

    ax = axes[0, 1]
    fb = seg[seg["fallback"]]
    reasons = fb["fallback_reason"].value_counts()
    ax.barh(range(len(reasons)), reasons.values, color="#d62728", alpha=0.8)
    ax.set_yticks(range(len(reasons)))
    ax.set_yticklabels([r if r else "(none)" for r in reasons.index], fontsize=8)
    for i, v in enumerate(reasons.values):
        ax.text(v + 0.6, i, str(int(v)), va="center", fontsize=9)
    ax.set_xlabel("Number of recordings")
    ax.set_title(f"(b) Excluded recordings: {len(fb)} of {len(seg)} "
                 f"({100*len(fb)/len(seg):.1f}%)")
    ax.set_xlim(0, reasons.max() * 1.18 if len(reasons) else 1)

    ax = axes[1, 0]
    per_file = primary.groupby(["cls", "relpath"]).size().reset_index(name="w")
    order = [c for c in C.CLASSES]
    data = [per_file.loc[per_file["cls"] == c, "w"].values for c in order]
    bp = ax.boxplot(data, positions=xs, widths=0.55, patch_artist=True,
                    medianprops=dict(color="black", lw=1.4))
    for patch, col in zip(bp["boxes"], colours):
        patch.set_facecolor(col); patch.set_alpha(0.55)
    ax.axhline(C.MAX_WINDOWS_PER_FILE, ls="--", color="#8c2f78", lw=1.1,
               label=f"per-file cap = {C.MAX_WINDOWS_PER_FILE}")
    ax.set_xticks(xs); ax.set_xticklabels([C.CLASS_LABELS[c] for c in C.CLASSES])
    ax.set_ylabel("Analysis windows kept per file")
    ax.set_title("(c) Windows per recording in the primary set")
    ax.legend(loc="upper left", framealpha=0.9)

    ax = axes[1, 1]
    wins = primary.groupby("cls").size().reindex(C.CLASSES, fill_value=0)
    bars = ax.bar([C.CLASS_LABELS[c] for c in C.CLASSES], wins.values,
                  color=colours, edgecolor="black", lw=0.6)
    ax.bar_label(bars, padding=3, fontsize=9)
    nfiles = primary.groupby("cls")["relpath"].nunique().reindex(C.CLASSES, fill_value=0)
    for i, c in enumerate(C.CLASSES):
        ax.text(i, wins.iloc[i] * 1.14, f"from {int(nfiles.iloc[i])} files",
                ha="center", fontsize=8, color="#444444")
    ax.set_ylabel("Analysis windows in the primary set")
    ax.set_ylim(0, wins.max() * 1.28)
    ax.set_title("(d) Total windows per class")

    fig.suptitle("Figure 5. Segmentation yield and exclusions", y=1.0, fontsize=11)
    fig.tight_layout()
    return _save(fig, "fig05_segmentation_yield",
                 "The per-file cap in (c) stops long flights from dominating training, "
                 "so every recording contributes a comparable number of windows. The "
                 "exclusions in (b) are not class-random, which is stated as a "
                 "limitation rather than hidden.")


def run(inv: pd.DataFrame, seg: pd.DataFrame, primary: pd.DataFrame,
        verbose: bool = True) -> list:
    """Produce all Phase 2 diagnostic figures."""
    if verbose:
        print("  Phase 2 figures:")
    out = [
        figure_segmentation_overview(inv, seg, n_files=1),
        figure_steady_zoom(inv, seg, primary),
        figure_window_spectra(inv, primary),
        figure_segmentation_yield(seg, primary),
    ]
    for p in out:
        if verbose:
            print(f"    {p.name}")
    return out
