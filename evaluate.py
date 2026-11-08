"""Phase 7 - evaluation extras the report needs but the model loop does not.

Two questions are answered here.

**f0 estimation error.** ``window_metrics`` estimates the blade-pass
fundamental as the refined Welch peak. That estimate is not ground truth: it
is what the system itself believes the RPM-related tone to be, so its accuracy
has to be measured against an independent estimator before any statement about
harmonics (``h2``..``h5``) or blade-pass bands can be trusted. The independent
reference used here is the Phase 2 STFT frame track: per-frame dominant
frequency inside ``config.F0_BAND``, median-smoothed over 5 frames to reject
single-frame spectral flips, then the median over the window's frames with
adequate tone prominence. Both estimators see the same samples, so any
disagreement is method error, not acoustic variation. Errors are reported in
Hz and as a percentage of f0, because 3 Hz at a 130 Hz blade tone is 2.3%
while at 230 Hz it is 1.3%.

**Binary confusion detail.** The four-class confusion matrix is already in
``model_report.json``. For a health monitor the operational question is the
normal-vs-abnormal cut, so the binary confusion counts and the recordings
behind every false alarm are written out for the report.

Outputs
-------
results/f0_error.csv           one row per window: welch f0, STFT f0, error
analysis/f0_error.json         per-class error summary
results/binary_confusion.csv   2x2 counts from the primary OOF predictions
analysis/evaluate.json         binary detail with the listed false alarms
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.signal import stft

from . import config as C
from . import dsp
from .segment import load_audio

ANALYSIS_DIR = C.ROOT / "analysis"

# Smooth window (in frames) for the reference f0 track. 5 frames x 64 ms hop
# = 320 ms, short enough to follow a hover, long enough to kill single-frame
# argmax flips into a harmonic.
REF_SMOOTH_FRAMES = 5

# A frame only enters the reference median when its blade tone stands out.
PROMINENCE_FLOOR_DB = 5.0


def stft_f0_track(x: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame dominant frequency in F0_BAND with 5-frame median smoothing.

    Returns (t, f0_raw, f0_smooth, prominence_db). The raw track is kept so
    the error table can show how much of the disagreement is single-frame
    noise.
    """
    f, t, Z = stft(
        x, fs=fs, nperseg=C.STFT_NFFT, noverlap=C.STFT_NFFT - C.STFT_HOP,
        window=C.STFT_WINDOW, padded=False, boundary=None,
    )
    P = np.abs(Z) ** 2
    fb = (f >= C.F0_BAND[0]) & (f < C.F0_BAND[1])
    if not fb.any():
        return t, np.full(t.shape, np.nan), np.full(t.shape, np.nan)
    Pf = P[fb]
    idx = np.argmax(Pf, axis=0)
    f0_raw = f[fb][idx]
    prominence_db = 10.0 * np.log10(
        Pf.max(axis=0) / (np.median(Pf, axis=0) + 1e-30) + 1e-30)

    k = REF_SMOOTH_FRAMES
    pad = k // 2
    padded = np.pad(f0_raw, (pad, pad), mode="edge")
    f0_smooth = np.array([
        np.median(padded[i:i + k]) for i in range(f0_raw.size)
    ])
    return t, f0_raw, f0_smooth, prominence_db


def window_f0_reference(x: np.ndarray, fs: int) -> float:
    """Median smoothed STFT f0 over the prominent frames of one window."""
    _, _, f0_s, prom = stft_f0_track(x, fs)
    m = np.isfinite(f0_s) & (prom >= PROMINENCE_FLOOR_DB)
    if not m.any():
        m = np.isfinite(f0_s)
    if not m.any():
        return float("nan")
    return float(np.median(f0_s[m]))


def f0_error_table(verbose: bool = True) -> dict:
    """Welch refined-peak f0 vs smoothed STFT reference, per primary window."""
    primary = pd.read_csv(C.result_path("segments_primary.csv"))
    # Reuse the stored Welch estimate rather than recomputing it, so the error
    # is exactly what the feature pipeline used.
    stored = pd.read_csv(C.result_path("window_metrics.csv"))
    stored = stored[["relpath", "segment_index", "f0_hz"]].dropna()

    rows = []
    cache_rel, cache_x, cache_fs = None, None, None
    n = len(primary)
    if verbose:
        print(f"  f0 estimation error on {n} primary windows")
    for i, row in enumerate(primary.itertuples(), start=1):
        if row.relpath != cache_rel:
            cache_fs, cache_x = load_audio(C.DATASET_DIR / row.relpath)
            cache_rel = row.relpath
        s = int(row.start_sample)
        nn = int(round(C.SEGMENT_SECONDS * cache_fs))
        x = cache_x[s:s + nn]
        if len(x) < nn:
            continue
        f0_ref = window_f0_reference(x, cache_fs)
        rows.append(dict(relpath=row.relpath, cls=row.cls, mission=row.mission,
                         segment_index=int(row.segment_index),
                         start_s=float(row.start_s),
                         f0_stft_ref_hz=f0_ref))
        if verbose and i % 500 == 0:
            print(f"    {i}/{n} windows")

    err = pd.DataFrame(rows).merge(stored, on=["relpath", "segment_index"],
                                   how="inner")
    err = err.rename(columns={"f0_hz": "f0_welch_hz"})
    err["abs_error_hz"] = (err["f0_welch_hz"] - err["f0_stft_ref_hz"]).abs()
    err["signed_error_hz"] = err["f0_welch_hz"] - err["f0_stft_ref_hz"]
    err["rel_error_pct"] = 100.0 * err["abs_error_hz"] / err["f0_stft_ref_hz"]
    err.to_csv(C.result_path("f0_error.csv"), index=False)

    summary = summarize(err)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    (ANALYSIS_DIR / "f0_error.json").write_text(json.dumps(dict(
        reference="5-frame median-smoothed STFT dominant frequency in "
                  "config.F0_BAND, median over frames with tone prominence "
                  f">= {PROMINENCE_FLOOR_DB:.0f} dB",
        estimate="refined Welch peak (dsp.refined_peak), the value stored as "
                 "window_metrics.f0_hz",
        n_windows=int(len(err)),
        overall=summary["overall"],
        per_class=summary["per_class"],
    ), indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else str(o)),
        encoding="utf-8")
    if verbose:
        o = summary["overall"]
        print(f"  median |error| {o['median_abs_error_hz']:.2f} Hz "
              f"({o['median_rel_error_pct']:.2f}% of f0), "
              f"p95 {o['p95_abs_error_hz']:.2f} Hz")
    return summary


def summarize(err: pd.DataFrame) -> dict:
    def one(df: pd.DataFrame) -> dict:
        a = df["abs_error_hz"].to_numpy(float)
        r = df["rel_error_pct"].to_numpy(float)
        return dict(
            n=int(np.isfinite(a).sum()),
            median_abs_error_hz=float(np.nanmedian(a)),
            mean_abs_error_hz=float(np.nanmean(a)),
            p95_abs_error_hz=float(np.nanpercentile(a, 95)),
            median_rel_error_pct=float(np.nanmedian(r)),
            p95_rel_error_pct=float(np.nanpercentile(r, 95)),
        )

    per = {c: one(err[err["cls"] == c]) for c in C.CLASSES
           if (err["cls"] == c).any()}
    return dict(overall=one(err), per_class=per)


def binary_confusion(verbose: bool = True) -> dict:
    """2x2 normal/abnormal confusion from the primary OOF predictions."""
    oof_path = C.result_path("oof_predictions.csv")
    if not oof_path.exists():
        if verbose:
            print("  oof_predictions.csv missing; run models first")
        return {}
    oof = pd.read_csv(oof_path)
    y = oof["cls"].map(lambda c: "normal" if c == "normal" else "abnormal")
    p = oof["pred"].map(lambda c: "normal" if c == "normal" else "abnormal")
    oof["true_binary"] = y
    oof["pred_binary"] = p

    tp = int(((y == "abnormal") & (p == "abnormal")).sum())
    fp = int(((y == "normal") & (p == "abnormal")).sum())
    fn = int(((y == "abnormal") & (p == "normal")).sum())
    tn = int(((y == "normal") & (p == "normal")).sum())
    matrix = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=pd.Index(["true normal", "true abnormal"]),
        columns=["pred normal", "pred abnormal"],
    )
    matrix.to_csv(C.result_path("binary_confusion.csv"))

    false_alarms = oof[(y == "normal") & (p == "abnormal")][
        ["relpath", "cls", "mission", "pred"]].to_dict("records")
    misses = oof[(y == "abnormal") & (p == "normal")][
        ["relpath", "cls", "mission", "pred"]].to_dict("records")

    detail = dict(
        true_positive=tp, false_positive=fp,
        false_negative=fn, true_negative=tn,
        false_alarm_rate=fp / max(fp + tn, 1),
        detection_rate=tp / max(tp + fn, 1),
        false_alarm_recordings=false_alarms,
        missed_recordings=misses,
    )
    (ANALYSIS_DIR / "evaluate.json").write_text(
        json.dumps(dict(binary=detail), indent=2), encoding="utf-8")
    if verbose:
        print(f"  binary confusion: TN {tn} FP {fp} FN {fn} TP {tp}")
    return detail


def run(verbose: bool = True) -> dict:
    if verbose:
        print("\nEVALUATION EXTRAS")
    f0 = f0_error_table(verbose=verbose)
    binary = binary_confusion(verbose=verbose)
    return dict(f0_error=f0, binary=binary)


if __name__ == "__main__":
    run()
