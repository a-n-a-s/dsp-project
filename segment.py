"""Phase 2 - steady-hover segmentation.

Every recording in this dataset runs from take-off to landing, so a full-file
spectrum averages across a wide RPM sweep. This module finds the interval where
the rotor is at roughly constant RPM and cuts fixed-length windows out of it.

Method:
  1. Short-time broadband level track (STFT, 40-3000 Hz) locates powered frames.
  2. Per-frame dominant frequency track locates the blade-pass tone.
  3. Frames whose dominant frequency stays within 5 Hz of a 9-frame moving
     average are marked steady.
  4. The longest run of steady frames inside the longest run of powered frames
     defines the hover region; fixed 4 s windows are cut with 50% overlap.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import stft

from . import config as C


@dataclass
class Track:
    """Per-frame level, dominant-frequency and tone-prominence tracks."""
    t: np.ndarray
    level_db: np.ndarray
    f0: np.ndarray
    prominence_db: np.ndarray
    floor_db: float

    @property
    def frame_dt(self) -> float:
        return float(self.t[1] - self.t[0])


def load_audio(path: Path) -> tuple[int, np.ndarray]:
    """Read a WAV file as float64 in [-1, 1) and return (fs, samples)."""
    fs, x = wavfile.read(str(path))
    if x.ndim > 1:
        x = x[:, 0]
    return int(fs), x.astype(np.float64) / 32768.0


def compute_track(x: np.ndarray, fs: int) -> Track:
    """Short-time level and dominant-frequency tracks over the whole file.

    Also returns the frame-by-frame tone prominence: the blade tone's level
    above the median of its own band. This is what decides whether a frame is
    powered, because these recordings have no usable silence and only 4-16 dB
    between their quietest and loudest second.
    """
    f, t, Z = stft(
        x, fs=fs, nperseg=C.STFT_NFFT, noverlap=C.STFT_NFFT - C.STFT_HOP,
        window=C.STFT_WINDOW, padded=False, boundary=None,
    )
    P = np.abs(Z) ** 2

    lb = (f >= C.LEVEL_BAND[0]) & (f < C.LEVEL_BAND[1])
    level_db = 10.0 * np.log10(P[lb].sum(axis=0) + 1e-20)

    fb = (f >= C.F0_BAND[0]) & (f < C.F0_BAND[1])
    Pf = P[fb]
    f0 = f[fb][np.argmax(Pf, axis=0)]
    prominence_db = 10.0 * np.log10(
        Pf.max(axis=0) / (np.median(Pf, axis=0) + 1e-30) + 1e-30
    )

    return Track(t=t, level_db=level_db, f0=f0, prominence_db=prominence_db,
                 floor_db=float(np.percentile(level_db, 5.0)))


def longest_run(mask: np.ndarray, min_len: int) -> tuple[int, int, int]:
    """Longest run of True as (length, start, end) in frames, or (0, 0, 0)."""
    if mask.size == 0:
        return (0, 0, 0)
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    starts, ends = edges[0::2], edges[1::2]
    if starts.size == 0:
        return (0, 0, 0)
    lens = ends - starts
    k = int(np.argmax(lens))
    if lens[k] < min_len:
        return (0, 0, 0)
    return (int(lens[k]), int(starts[k]), int(ends[k]))


def all_runs(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    """Every maximal run of True with length >= min_len, as (start, end) frames."""
    if mask.size == 0:
        return []
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    out = []
    for a, b in zip(edges[0::2], edges[1::2]):
        if b - a >= min_len:
            out.append((int(a), int(b)))
    return out


def merge_runs(runs: list[tuple[int, int]], gap: int) -> list[tuple[int, int]]:
    """Join runs separated by fewer than `gap` frames."""
    if not runs:
        return []
    out = [list(runs[0])]
    for a, b in runs[1:]:
        if a - out[-1][1] <= gap:
            out[-1][1] = b
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def steady_mask_from_track(f0: np.ndarray) -> np.ndarray:
    """Frames whose dominant frequency is flat against a 9-frame moving average.

    The first and last k//2 frames of a candidate run are dropped because their
    moving average is not centred, which would otherwise let the edges of a
    throttle ramp masquerade as steady.
    """
    k = C.STEADY_SMOOTH_FRAMES
    mask = np.zeros(f0.size, dtype=bool)
    if f0.size > k:
        smooth = np.convolve(f0, np.ones(k) / k, mode="valid")
        mask[k - 1:] = np.abs(f0[k - 1:] - smooth) <= C.STEADY_TOL_HZ
        edge = np.zeros(f0.size, dtype=bool)
        edge[: k - 1] = True
        mask &= ~edge
    return mask


@dataclass
class Segmentation:
    relpath: str
    cls: str
    duration_s: float
    floor_db: float
    n_frames: int
    powered: tuple[int, int]
    steady_span: tuple[int, int]
    powered_s: float
    hover_s: float
    hover_start_s: float
    n_steady_runs: int
    fallback: bool
    fallback_reason: str
    n_segments: int
    median_f0_hover: float
    f0_std_hover: float
    level_db_hover: float


def segment_one(relpath: str, cls: str, dataset_dir: Path,
                segment_seconds: float = C.SEGMENT_SECONDS,
                stride_fraction: float = C.SEGMENT_STRIDE_FRACTION,
                min_segment_seconds: float = C.MIN_SEGMENT_SECONDS,
                ) -> tuple[Segmentation, np.ndarray, Track, np.ndarray]:
    """Locate the hover region of one recording and cut analysis windows."""
    fs, x = load_audio(dataset_dir / relpath)
    if fs != C.FS_EXPECTED:
        raise ValueError(f"{relpath}: expected {C.FS_EXPECTED} Hz, got {fs} Hz")

    tr = compute_track(x, fs)
    dt = tr.frame_dt

    # Powered = blade tone present. Tone prominence, not absolute level.
    powered_mask = tr.prominence_db > C.TONE_PROMINENCE_DB
    p_len, p_a, p_b = longest_run(powered_mask, C.MIN_POWERED_FRAMES)
    reason = ""
    if p_len == 0:
        p_a, p_b = 0, tr.level_db.size
        reason = "no powered run"

    # Steady frames: dominant frequency flat against its own moving average,
    # restricted to powered frames. All maximal runs are kept, not just the
    # longest, so a recording with several constant-RPM stretches contributes
    # every one of them.
    steady_all = steady_mask_from_track(tr.f0) & powered_mask
    runs = merge_runs(all_runs(steady_all, min_len=C.MIN_STEADY_FRAMES),
                      gap=C.STEADY_MERGE_GAP_FRAMES)
    runs = [(a, b) for a, b in runs if (b - a) * dt >= C.MIN_STEADY_SECONDS]

    # Frame index -> sample index. Frames are one hop apart in samples, not fs.
    win = int(round(segment_seconds * fs))
    hop = int(round(win * stride_fraction))

    starts_list: list[np.ndarray] = []
    for ra, rb in runs:
        sa, sb = ra * C.STFT_HOP, rb * C.STFT_HOP
        if sb - sa >= win:
            starts_list.append(np.arange(sa, sb - win + 1, hop, dtype=np.int64))
    starts = np.concatenate(starts_list) if starts_list else np.array([], dtype=np.int64)

    if starts.size == 0:
        # No steady run long enough for a full window. Use the whole powered
        # region instead and mark the file so the fallback stays auditable.
        reason = reason or "no steady run >= window"
        if (p_b - p_a) * C.STFT_HOP >= win:
            starts = np.arange(p_a * C.STFT_HOP, p_b * C.STFT_HOP - win + 1,
                               hop, dtype=np.int64)
        elif (p_b - p_a) * C.STFT_HOP >= min_segment_seconds * fs:
            starts = np.array([p_a * C.STFT_HOP], dtype=np.int64)
        else:
            starts = np.array([0], dtype=np.int64)
            reason = "file shorter than minimum segment"

    if runs:
        h_a, h_b = runs[0][0], runs[-1][1]
    else:
        h_a, h_b = p_a, p_b

    hz = tr.f0[h_a:h_b]
    hl = tr.level_db[h_a:h_b]
    hover_s = sum(rb - ra for ra, rb in runs) * dt
    seg = Segmentation(
        relpath=relpath, cls=cls, duration_s=len(x) / fs, floor_db=tr.floor_db,
        n_frames=int(tr.level_db.size),
        powered=(int(p_a), int(p_b)), steady_span=(int(h_a), int(h_b)),
        powered_s=round((p_b - p_a) * dt, 3),
        hover_s=round(hover_s, 3),
        hover_start_s=round(h_a * dt, 3),
        n_steady_runs=len(runs),
        fallback=bool(reason),
        fallback_reason=reason,
        n_segments=int(starts.size),
        median_f0_hover=round(float(np.median(hz)), 2) if hz.size else np.nan,
        f0_std_hover=round(float(np.std(hz)), 2) if hz.size else np.nan,
        level_db_hover=round(float(np.mean(hl)), 2) if hl.size else np.nan,
    )
    return seg, x, tr, starts


def run(inventory: pd.DataFrame, verbose: bool = True,
        limit: int | None = None) -> pd.DataFrame:
    """Execute Phase 2 over the whole dataset and persist the segment index."""
    if verbose:
        print("\nPHASE 2 - steady-hover segmentation")

    df = inventory if limit is None else inventory.head(limit)
    segs, index = [], []

    fs = C.FS_EXPECTED
    for n, row in enumerate(df.itertuples(), start=1):
        seg, _x, _tr, starts = segment_one(row.relpath, row.cls, C.DATASET_DIR)
        segs.append(asdict(seg))

        # Uniform per-file cap keeps long flights from dominating, and keeps
        # every recording contributing a comparable number of windows.
        kept = starts[: C.MAX_WINDOWS_PER_FILE]
        for i, s in enumerate(starts):
            index.append(dict(
                relpath=seg.relpath, cls=seg.cls, config=row.config, group=row.group,
                mission=row.mission, take=int(row.take), segment_index=i,
                start_sample=int(s), start_s=round(s / fs, 3),
                duration_s=round(C.SEGMENT_SECONDS, 3),
                fallback=bool(seg.fallback),
                in_primary=bool(i < len(kept) and not (C.EXCLUDE_FALLBACK_WINDOWS
                                                       and seg.fallback)),
            ))
        if verbose and n % 40 == 0:
            print(f"    segmented {n}/{len(df)}")

    seg_df = pd.DataFrame(segs)
    idx_df = pd.DataFrame(index)
    primary = idx_df[idx_df["in_primary"]].copy()

    seg_df.to_csv(C.result_path("segmentation.csv"), index=False)
    idx_df.to_csv(C.result_path("segment_index.csv"), index=False)

    if verbose:
        report_segmentation(seg_df, idx_df)

    # The pipeline downstream consumes the primary set only.
    primary.to_csv(C.result_path("segments_primary.csv"), index=False)
    return seg_df, primary


def report_segmentation(seg_df: pd.DataFrame, idx_df: pd.DataFrame) -> None:
    """Print and persist the segmentation yield table (Table 2)."""
    g = (
        seg_df.groupby("cls")
        .agg(files=("relpath", "size"),
             fallbacks=("fallback", "sum"),
             steady_runs=("n_steady_runs", "median"),
             steady_total_s=("hover_s", "median"),
             longest_run_s=("hover_s", "max"),
             median_segments=("n_segments", "median"),
             total_segments=("n_segments", "sum"),
             median_f0_hz=("median_f0_hover", "median"),
             median_level_db=("level_db_hover", "median"))
        .reindex(C.CLASSES)
    )
    g.insert(0, "class_label", [C.CLASS_LABELS[c] for c in g.index])
    g["powered_frac"] = (seg_df.groupby("cls")["powered_s"].sum()
                         / seg_df.groupby("cls")["duration_s"].sum())
    g = g.round(3)
    g.to_csv(C.result_path("table2_segmentation_yield.csv"))

    # Fallback reasons, so every non-steady file is accounted for.
    fb = seg_df[seg_df["fallback"]]
    fb.groupby(["fallback_reason", "cls"]).size().rename("files").reset_index() \
        .to_csv(C.result_path("table2_fallback_reasons.csv"), index=False)

    prim = idx_df[idx_df["in_primary"]]
    print(f"  segmented {len(seg_df)} files -> {len(idx_df)} raw windows "
          f"({C.SEGMENT_SECONDS:g} s, {C.SEGMENT_STRIDE_FRACTION:.0%} overlap)")
    print(f"  steady-RPM detected in {int((~seg_df['fallback']).sum())}/{len(seg_df)} files; "
          f"{len(fb)} fell back to the whole powered segment")
    for reason, n in fb["fallback_reason"].value_counts().items():
        print(f"      {reason}: {n} files")
    print(f"  after the {C.MAX_WINDOWS_PER_FILE}-window-per-file cap and "
          f"excluding fallback windows: {len(prim)} windows "
          f"from {prim['relpath'].nunique()} files")
    print(f"  {'class':<11} {'files':>5} {'fallb':>5} {'steady s':>9} {'runs':>5} "
          f"{'raw win':>8} {'primary':>8} {'f0 Hz':>7} {'lvl dB':>7}")
    prim_per_class = prim.groupby("cls").size().reindex(C.CLASSES, fill_value=0)
    for c in C.CLASSES:
        r = g.loc[c]
        print(f"  {C.CLASS_LABELS[c]:<11} {int(r['files']):5d} "
              f"{int(r['fallbacks']):5d} {r['steady_total_s']:9.1f} "
              f"{r['steady_runs']:5.0f} {int(r['total_segments']):8d} "
              f"{int(prim_per_class[c]):8d} "
              f"{r['median_f0_hz']:7.1f} {r['median_level_db']:7.1f}")

    prim.groupby(["cls", "relpath"]).size().rename("windows").reset_index() \
        .to_csv(C.result_path("table2_primary_windows_per_file.csv"), index=False)


if __name__ == "__main__":
    inv = pd.read_csv(C.result_path("inventory.csv"))
    run(inv)
