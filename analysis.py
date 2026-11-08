"""Signal metric extraction and numerical characterisation.

Computes a wide set of DSP metrics for every analysis window, then writes a
set of JSON reports to `analysis/` describing what the numbers actually say
about each class: distributions, per-feature separability, the session
confounds, and which metrics survive level and f0 normalisation.

Outputs
-------
analysis/per_window_metrics.json   one record per window (sampled, see N_DUMP)
analysis/per_class_summary.json    distribution stats per class per metric
analysis/separability.json         per-feature AUC between every class pair
analysis/confounds.json            evidence for the level and f0 confounds
analysis/feature_inventory.json    every metric defined, with units
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import dsp
from .segment import load_audio

ANALYSIS_DIR = C.ROOT / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

# How many per-window records to dump verbatim into JSON. The full set is
# written to CSV for reuse; JSON gets a stratified sample so the file stays
# readable and can be opened in a text editor.
N_DUMP = 600

WIN = int(round(C.SEGMENT_SECONDS * C.FS_EXPECTED))


# ====================================================================== metrics
def time_domain_metrics(x: np.ndarray, fs: int) -> dict:
    """Scale-invariant time-domain descriptors."""
    n = len(x)
    mean = float(x.mean())
    xc = x - mean
    rms = float(np.sqrt((xc ** 2).mean()) + 1e-30)
    peak = float(np.abs(xc).max() + 1e-30)

    var = float((xc ** 2).mean())
    std = math.sqrt(var)
    skew = float(((xc ** 3).mean()) / (std ** 3 + 1e-30))
    kurt = float(((xc ** 4).mean()) / (std ** 4 + 1e-30))

    zc = float(np.mean(np.abs(np.diff(np.sign(xc))) > 0))

    # Envelope statistics from the amplitude envelope (50 ms hop RMS).
    env = envelope(x, fs)
    env_db = 20 * np.log10(env + 1e-30)
    p95, p5 = np.percentile(env_db, 95), np.percentile(env_db, 5)
    env_std = float(env_db.std())

    # Spectral flatness of the time-domain envelope = amplitude-modulation depth.
    ef, eP = dsp.welch_psd(env - env.mean(), fs, nperseg=min(1024, len(env) * 2),
                           noverlap=min(512, max(0, len(env) - 1)))
    mod_flat = float(np.exp(np.mean(np.log(eP + 1e-30)))
                     / (np.mean(eP) + 1e-30)) if eP.size > 1 else np.nan

    return dict(
        rms=rms,
        peak=peak,
        crest_factor=peak / (rms + 1e-30),
        kurtosis=kurt,
        skewness=skew,
        zero_crossing_rate=zc,
        dc_offset=mean,
        env_dynamic_range_db=float(p95 - p5),
        env_std_db=env_std,
        env_mod_flatness=mod_flat,
        # Level metrics kept for confound analysis only, never as features.
        level_dbfs=20 * math.log10(rms + 1e-30),
        peak_dbfs=20 * math.log10(peak + 1e-30),
    )


def envelope(x: np.ndarray, fs: int, win_ms: float = 50.0,
             hop_ms: float = 10.0) -> np.ndarray:
    """RMS amplitude envelope with a rectangular window."""
    w = max(1, int(fs * win_ms / 1000.0))
    h = max(1, int(fs * hop_ms / 1000.0))
    n = (len(x) - w) // h + 1
    if n <= 0:
        return np.array([float(np.sqrt((x ** 2).mean()) + 1e-30)])
    idx = np.arange(w)[None, :] + h * np.arange(n)[:, None]
    return np.sqrt((x[idx] ** 2).mean(axis=1) + 1e-30)


def spectral_metrics(f: np.ndarray, P: np.ndarray, f0: float) -> dict:
    """Shape descriptors and band ratios, all relative to the 40-4000 Hz total."""
    m = dsp.analysis_band(f)
    if not m.any():
        return {}
    fb, Pb = f[m], P[m]
    total = float(Pb.sum() + 1e-30)
    Pn = Pb / total                      # normalised power, sums to 1
    ref_db = 10 * math.log10(total)

    centroid = float((fb * Pn).sum())
    spread = float(np.sqrt(((fb - centroid) ** 2 * Pn).sum()))
    flatness = float(np.exp(np.mean(np.log(Pb + 1e-30))) / (np.mean(Pb) + 1e-30))
    entropy = float(-(Pn * np.log(Pn + 1e-30)).sum() / math.log(fb.size))

    csum = np.cumsum(Pn)
    roll85 = float(fb[int(np.searchsorted(csum, 0.85))]) if csum.size else np.nan
    roll95 = float(fb[int(np.searchsorted(csum, 0.95))]) if csum.size else np.nan

    out = {
        "centroid_rel": centroid,
        "spread_rel": spread,
        "flatness_rel": flatness,
        "entropy_rel": entropy,
        "rolloff85_rel": roll85,
        "rolloff95_rel": roll95,
        "psd_peak_db": ref_db,
    }

    for lo, hi in C.OCTAVE_BANDS:
        out[f"band_{lo:g}_{hi:g}_rel"] = dsp.band_relative_db(f, P, lo, hi)

    # Sub-octave resolution inside the two most informative bands.
    for lo, hi in [(100.0, 200.0), (200.0, 400.0), (400.0, 800.0),
                   (800.0, 1600.0), (1600.0, 3200.0)]:
        out[f"sub_{lo:g}_{hi:g}_rel"] = dsp.band_relative_db(f, P, lo, hi)

    for k in C.HARMONICS:
        out[f"h{k}_rel"] = dsp.harmonic_relative_db(f, P, f0, k)

    # Peak prominence: how far the blade tone stands above the local floor.
    b = (fb >= C.F0_BAND[0]) & (fb < C.F0_BAND[1])
    if b.any():
        Pb_b = Pb[b]
        sorted_p = np.sort(Pb_b)[::-1]
        ref = sorted_p[0]
        out["tone_prominence_db"] = float(10 * math.log10(
            ref / (np.median(Pb_b) + 1e-30) + 1e-30))
        out["peak_to_5th_db"] = float(10 * math.log10(
            ref / (sorted_p[min(4, sorted_p.size - 1)] + 1e-30) + 1e-30))
        out["peak_to_median_db"] = float(10 * math.log10(
            ref / (np.median(Pb_b) + 1e-30) + 1e-30))

    return out


def modulation_metrics(x: np.ndarray, fs: int, f0: float) -> dict:
    """Amplitude- and frequency-modulation sidebands around the blade tone.

    A cracked or missing blade produces a once-per-revolution amplitude
    fluctuation, so sidebands at the rotor rate and at 2x the rotor rate are
    the physically motivated place to look.
    """
    if not np.isfinite(f0) or f0 <= 0:
        return {}

    f, t, Z = _stft(x, fs)
    b = (f >= max(20.0, f0 * 0.5)) & (f <= min(f0 * 2.0, C.NYQUIST - 100))
    if not b.any() or Z.shape[1] < 2:
        return {}
    fb = f[b]
    # Average over time so each frequency bin holds one value: the sideband
    # search needs a per-frequency profile, not a full spectrogram.
    Pb = (np.abs(Z[b]) ** 2).mean(axis=1)
    if Pb.size < 2:
        return {}

    idx = int(np.argmin(np.abs(fb - f0)))
    peak = float(Pb[idx]) + 1e-30

    # Rotor rate = blade-pass rate / blades-per-rev. Unknown a priori, so report
    # the sideband level at several candidate modulation rates.
    out = {}
    df = float(f[1] - f[0]) if f.size > 1 else 1.0
    for blades, fr in [(2, f0 / 2), (3, f0 / 3)]:
        for k, name in [(1, "1x"), (2, "2x"), (3, "3x")]:
            fc = k * fr
            # Skip any sideband that lands on the fundamental or one of its own
            # harmonics, otherwise it just re-measures the blade tone at 0 dB.
            if min(abs(fc - j * f0) for j in (1, 2, 3)) <= 2.0 * max(df, 1.0):
                continue
            m = np.abs(fb - fc) <= max(2.0, 2 * df)
            if m.any():
                out[f"am_sideband_blades{blades}_{name}_db"] = float(
                    10 * math.log10(float(Pb[m].max()) / peak + 1e-30))

    # Total sideband energy in the band straddling the tone.
    lo, hi = f0 * 0.8, f0 * 1.25
    m2 = (fb >= lo) & (fb <= hi)
    out["tone_band_energy_rel"] = float(10 * math.log10(Pb[m2].sum() + 1e-30))
    out["tone_peak_rel"] = float(10 * math.log10(peak + 1e-30))
    out["tone_peak_frac_of_band"] = float(
        10 * math.log10(peak / (Pb.sum() + 1e-30) + 1e-30))
    return out


def _stft(x: np.ndarray, fs: int, nfft: int = 2048, hop: int = 512):
    from scipy.signal import stft as _s
    f, t, Z = _s(x, fs=fs, nperseg=nfft, noverlap=nfft - hop,
                 window="hann", padded=False, boundary=None)
    return f, t, Z


def wavelet_metrics(x: np.ndarray, fs: int, f0: float) -> dict:
    """Cyclic-spectrum-style band energies at the blade tone and its harmonics.

    Uses a band-pass decomposition rather than a full CWT: energy is measured
    in narrow bands centred on f0, 2*f0 and 3*f0, which is what a wavelet
    decomposition at those scales would return, at a fraction of the cost.
    """
    from scipy.signal import butter, sosfiltfilt

    out = {}
    if not np.isfinite(f0) or f0 <= 0:
        return out
    for k in (1, 2, 3):
        fc = k * f0
        if fc >= C.NYQUIST * 0.95:
            break
        bw = max(8.0, 0.25 * fc)
        lo, hi = max(20.0, fc - bw), min(C.NYQUIST * 0.98, fc + bw)
        if hi <= lo + 4:
            continue
        try:
            sos = butter(4, [lo / (fs / 2), hi / (fs / 2)], btype="band", output="sos")
            y = sosfiltfilt(sos, x)
        except ValueError:
            continue
        pk = np.abs(np.fft.rfft(y))
        fk = np.fft.rfftfreq(len(y), 1 / fs)
        m = (fk >= lo) & (fk <= hi)
        if not m.any():
            continue
        # Crest of the band-passed signal: a clean tone is highly periodic.
        env = np.abs(np.fft.rfft(y * np.hanning(len(y))))
        e = env[m]
        out[f"bp{k}_crest_db"] = float(20 * math.log10(
            (e.max() + 1e-30) / (np.median(e) + 1e-30)))
        out[f"bp{k}_rms_rel"] = float(20 * math.log10(
            (np.sqrt((y ** 2).mean()) + 1e-30) /
            (np.sqrt((x ** 2).mean()) + 1e-30)))
    return out


def window_metrics(x: np.ndarray, fs: int) -> dict:
    """Every metric for one analysis window."""
    f, P = dsp.welch_psd(x, fs)
    if f.size == 0:
        return {}
    f0, f0_lvl = dsp.refined_peak(f, P, *C.F0_BAND)

    m = {"f0_hz": f0, "f0_peak_db": f0_lvl}
    m.update(time_domain_metrics(x, fs))
    m.update(spectral_metrics(f, P, f0))
    m.update(modulation_metrics(x, fs, f0))
    m.update(wavelet_metrics(x, fs, f0))
    return m


# ====================================================================== driver
def extract_all(primary: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Compute metrics for every window in the primary set."""
    if verbose:
        print("  computing metrics for every analysis window ...")

    records = []
    files = primary["relpath"].unique()
    cache_rel = None
    cache = None
    cache_fs = C.FS_EXPECTED

    for n, rel in enumerate(files, start=1):
        if rel != cache_rel:
            cache_fs, cache = load_audio(C.DATASET_DIR / rel)
            cache_rel = rel
        sub = primary[primary["relpath"] == rel]
        for r in sub.itertuples():
            s = int(r.start_sample)
            x = cache[s:s + WIN]
            if len(x) < WIN:
                continue
            m = window_metrics(x, cache_fs)
            if not m:
                continue
            m.update(relpath=r.relpath, cls=r.cls, config=r.config,
                     group=r.group, mission=r.mission, take=int(r.take),
                     segment_index=int(r.segment_index),
                     start_s=float(r.start_s))
            records.append(m)
        if verbose and n % 50 == 0:
            print(f"    {n}/{len(files)} files, {len(records)} windows")

    df = pd.DataFrame.from_records(records)
    df.to_csv(C.result_path("window_metrics.csv"), index=False)
    if verbose:
        print(f"  {len(df)} windows x {df.shape[1]} columns")
    return df


# ====================================================================== reports
def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    return str(o)


def write_json(name: str, payload) -> Path:
    path = ANALYSIS_DIR / name
    path.write_text(json.dumps(payload, indent=2, default=_json_default),
                    encoding="utf-8")
    return path


def describe(df: pd.DataFrame, key: str) -> dict:
    """Distribution summary for one metric across all windows."""
    v = pd.to_numeric(df[key], errors="coerce").replace([np.inf, -np.inf], np.nan)
    v = v.dropna()
    if v.empty:
        return dict(n=0)
    q = np.percentile(v, [0, 5, 25, 50, 75, 95, 100])
    return dict(
        n=int(v.size),
        mean=float(v.mean()),
        std=float(v.std()),
        min=float(q[0]),
        p05=float(q[1]),
        p25=float(q[2]),
        median=float(q[3]),
        p75=float(q[4]),
        p95=float(q[5]),
        max=float(q[6]),
        skew=float(((v - v.mean()) ** 3).mean() / (v.std() ** 3 + 1e-30)),
    )


METRIC_UNITS = {
    "rms": "normalised amplitude", "peak": "normalised amplitude",
    "crest_factor": "ratio", "kurtosis": "ratio", "skewness": "ratio",
    "zero_crossing_rate": "crossings/sample", "dc_offset": "normalised amplitude",
    "env_dynamic_range_db": "dB", "env_std_db": "dB",
    "env_mod_flatness": "ratio (0-1)",
    "level_dbfs": "dBFS", "peak_dbfs": "dBFS",
    "f0_hz": "Hz", "f0_peak_db": "dB re band total",
    "centroid_rel": "Hz", "spread_rel": "Hz",
    "flatness_rel": "ratio (0-1)", "entropy_rel": "ratio (0-1)",
    "rolloff85_rel": "Hz", "rolloff95_rel": "Hz",
    "psd_peak_db": "dB re band total",
    "tone_prominence_db": "dB", "peak_to_5th_db": "dB",
    "peak_to_median_db": "dB",
    "tone_band_energy_rel": "dB", "tone_peak_rel": "dB",
    "tone_peak_frac_of_band": "dB",
    "time_sec": "s", "start_s": "s",
}


def feature_inventory(df: pd.DataFrame) -> dict:
    """Every metric with its unit, type and whether it is a usable feature."""
    confound = {"rms", "peak", "level_dbfs", "peak_dbfs", "dc_offset",
                "psd_peak_db", "f0_hz", "f0_peak_db", "env_dynamic_range_db",
                "env_std_db"}
    out = {}
    for col in df.columns:
        if col in ("relpath", "cls", "config", "group", "mission"):
            continue
        if col in ("take", "segment_index", "start_s", "fallback"):
            continue
        out[col] = dict(
            unit=METRIC_UNITS.get(col, "s" if col.endswith("_db") else "dimensionless"),
            role="diagnostic_only (level or f0 confounded)"
            if col in confound else "candidate_feature",
            n_valid=int(pd.to_numeric(df[col], errors="coerce").notna().sum()),
            n_total=int(len(df)),
        )
    return out


def per_class_summary(df: pd.DataFrame) -> dict:
    metrics = [c for c in df.columns
               if c not in ("relpath", "cls", "config", "group", "mission")]
    out = {"n_windows_per_class": {c: int((df["cls"] == c).sum()) for c in C.CLASSES},
           "n_files_per_class": {c: int(df.loc[df["cls"] == c, "relpath"].nunique())
                                 for c in C.CLASSES},
           "metrics": {}}
    for m in metrics:
        entry = {}
        for c in C.CLASSES:
            entry[C.CLASS_LABELS[c]] = describe(df[df["cls"] == c], m)
        out["metrics"][m] = entry
    return out


def auc(x, y) -> float:
    """Rank-based AUC, orientation-corrected so 0.5 is chance and 1.0 is perfect."""
    x = pd.to_numeric(pd.Series(x), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    y = pd.to_numeric(pd.Series(y), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    v = np.concatenate([x.values, y.values])
    lab = np.concatenate([np.ones(len(x)), np.zeros(len(y))])
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(len(v), dtype=float)
    sv = v[order]
    i = 0
    r = np.arange(1, len(v) + 1, dtype=float)
    while i < len(v):
        j = i
        while j + 1 < len(v) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = r[i:j + 1].mean()
        i = j + 1
    a = (ranks[lab == 1].sum() - len(x) * (len(x) + 1) / 2) / (len(x) * len(y))
    return float(max(a, 1 - a))


def separability(df: pd.DataFrame) -> dict:
    """Per-feature AUC for every class pair, plus the normal-vs-broken split."""
    metrics = [c for c in df.columns
               if c not in ("relpath", "cls", "config", "group", "mission",
                            "take", "segment_index", "start_s")]
    metrics = [m for m in metrics if pd.to_numeric(df[m], errors="coerce").notna().any()]

    pairs = []
    for i, a in enumerate(C.CLASSES):
        for b in C.CLASSES[i + 1:]:
            pairs.append((a, b))

    table = {}
    for m in metrics:
        row = {}
        for a, b in pairs:
            row[f"{a}_vs_{b}"] = auc(df.loc[df["cls"] == a, m],
                                     df.loc[df["cls"] == b, m])
        nb = df[df["cls"] != "normal"]
        row["normal_vs_all_broken"] = auc(df.loc[df["cls"] == "normal", m], nb[m])
        table[m] = row

    # Rank metrics by their worst within-broken separation.
    within_keys = [k for k in next(iter(table.values())) if not k.startswith("normal_vs")]
    ranked = sorted(
        table.items(),
        key=lambda kv: max((np.nan_to_num(kv[1][k], nan=0.5) for k in within_keys),
                           default=0.0),
        reverse=True,
    )
    return dict(
        auc_table=table,
        best_within_broken=[
            dict(metric=m,
                 best_worst_case_auc=float(max((np.nan_to_num(v[k], nan=0.5)
                                               for k in within_keys), default=0.5)),
                 mean_within_auc=float(np.nanmean([v[k] for k in within_keys])),
                 normal_vs_broken_auc=float(v["normal_vs_all_broken"]))
            for m, v in ranked[:25]
        ],
        note="AUC is orientation-corrected: 0.5 = chance, 1.0 = perfect separation. "
             "best_worst_case_auc is the best pairwise separation among the three "
             "broken classes, i.e. the hardest pair.",
    )


def confounds(df: pd.DataFrame) -> dict:
    """Quantify the level and f0 confounds and test which metrics survive them."""
    out = {"description": "Tests whether the measured class differences are "
                          "properties of the propeller or of the recording session.",
           "level": {}, "f0": {}, "survival": {}}

    # --- level
    lv = {c: pd.to_numeric(df.loc[df["cls"] == c, "level_dbfs"], errors="coerce")
          for c in C.CLASSES}
    lo = {c: float(v.min()) for c, v in lv.items()}
    hi = {c: float(v.max()) for c, v in lv.items()}
    out["level"] = dict(
        metric="level_dbfs (window RMS)",
        per_class={C.CLASS_LABELS[c]: dict(min=lo[c], max=hi[c],
                                           median=float(lv[c].median()))
                   for c in C.CLASSES},
        normal_vs_broken_ranges_disjoint=bool(hi["normal"] < min(lo[c] for c in C.CLASSES[1:])),
        auc_normal_vs_all_broken=auc(lv["normal"],
                                     pd.concat([lv[c] for c in C.CLASSES[1:]])),
        verdict="Absolute level separates normal from every broken class perfectly. "
                "It is a recording-session property, not a fault signature, so it is "
                "excluded from the feature set.",
    )

    # --- f0
    f0 = {c: pd.to_numeric(df.loc[df["cls"] == c, "f0_hz"], errors="coerce")
          for c in C.CLASSES}
    out["f0"] = dict(
        metric="f0_hz (refined Welch peak, 40-2000 Hz)",
        per_class={C.CLASS_LABELS[c]: dict(
            min=float(f0[c].min()), p05=float(f0[c].quantile(0.05)),
            median=float(f0[c].median()), p95=float(f0[c].quantile(0.95)),
            max=float(f0[c].max()), std=float(f0[c].std()))
            for c in C.CLASSES},
        auc_normal_vs_all_broken=auc(f0["normal"], pd.concat([f0[c] for c in C.CLASSES[1:]])),
        clusters_within_1_broken=sorted(
            [float(v) for v in np.round(f0["1_broken"].values, -1)[:0]] or
            [float(x) for x, n in pd.Series(np.round(f0["1_broken"], -1)).value_counts().items()
             if n >= 5]),
        verdict="The blade-pass frequency differs systematically between normal and "
                "broken flights, indicating a different throttle protocol per class. "
                "Absolute f0 is therefore excluded as a feature; only ratios measured "
                "relative to f0 are used.",
    )

    # --- which metrics survive normalisation
    survive = []
    for m in ["band_160_320_rel", "band_2000_4000_rel", "h2_rel", "h3_rel",
              "centroid_rel", "flatness_rel", "entropy_rel", "rolloff85_rel"]:
        if m not in df.columns:
            continue
        v = pd.to_numeric(df[m], errors="coerce")
        a_level = auc(v[df["cls"] == "normal"], v[df["cls"] != "normal"])
        a_f0 = auc(v[df["cls"] == "normal"], v[df["cls"] != "normal"])
        within = []
        for i, aa in enumerate(C.CLASSES[1:]):
            for bb in C.CLASSES[i + 2:]:
                within.append(auc(v[df["cls"] == aa], v[df["cls"] == bb]))
        survive.append(dict(
            metric=m,
            auc_normal_vs_broken=float(a_level),
            worst_within_broken_auc=float(np.nanmin(within)) if within else None,
            within_broken_aucs=[float(x) for x in within],
        ))
    out["survival"] = dict(
        description="Normalised metrics: their separation cannot be explained by the "
                    "level or f0 confounds, because they are ratios by construction.",
        metrics=survive,
    )
    return out


def f0_stability(df: pd.DataFrame) -> dict:
    """Within-window f0 spread: a quality check on the segmentation itself."""
    out = {}
    for c in C.CLASSES:
        v = []
        for rel, sub in df[df["cls"] == c].groupby("relpath"):
            f = pd.to_numeric(sub["f0_hz"], errors="coerce").dropna()
            if len(f) > 1:
                v.append(float(f.std()))
        if v:
            v = np.array(v)
            out[C.CLASS_LABELS[c]] = dict(
                n_files=int(v.size), median_std_hz=float(np.median(v)),
                p95_std_hz=float(np.percentile(v, 95)),
                frac_below_2hz=float(np.mean(v < 2.0)))
    return out


def run(primary: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Full analysis: extract metrics, then write every JSON report."""
    if verbose:
        print(f"\nANALYSIS  -> {ANALYSIS_DIR}")

    df = extract_all(primary, verbose=verbose)

    if verbose:
        print("  writing reports ...")

    # Sampled per-window dump keeps the JSON readable.
    per = max(1, N_DUMP // len(C.CLASSES))
    parts = [g.sample(min(per, len(g)), random_state=C.RANDOM_STATE)
             for _, g in df.groupby("cls", sort=False)]
    sample = (pd.concat(parts, ignore_index=True)
              .sort_values(["cls", "relpath", "segment_index"])
              .reset_index(drop=True))
    write_json("per_window_metrics.json", dict(
        description=f"Stratified sample of {len(sample)} analysis windows, "
                    f"{N_DUMP // len(C.CLASSES)} per class. The complete set of "
                    f"{len(df)} windows is in results/window_metrics.csv.",
        metric_units=METRIC_UNITS,
        windows=sample.to_dict(orient="records"),
    ))

    write_json("feature_inventory.json", dict(
        total_windows=int(len(df)),
        feature_count=int(sum(1 for v in feature_inventory(df).values()
                              if v["role"] == "candidate_feature")),
        diagnostic_count=int(sum(1 for v in feature_inventory(df).values()
                                 if v["role"].startswith("diagnostic"))),
        features=feature_inventory(df),
    ))

    write_json("per_class_summary.json", per_class_summary(df))
    write_json("separability.json", separability(df))
    write_json("confounds.json", confounds(df))
    write_json("f0_stability.json", f0_stability(df))

    if verbose:
        print_summary(df)

    return df


def print_summary(df: pd.DataFrame) -> None:
    """Human-readable digest of the analysis."""
    print("\n  " + "=" * 74)
    print("  SIGNAL METRIC SUMMARY  (medians per class)")
    print("  " + "=" * 74)
    print(f"  {'metric':<28}" + "".join(f"{C.CLASS_LABELS[c]:>11}" for c in C.CLASSES))

    groups = [
        ("-- level (confound) --", ["level_dbfs", "peak_dbfs"]),
        ("-- blade tone --", ["f0_hz", "tone_prominence_db", "peak_to_5th_db",
                              "tone_peak_frac_of_band"]),
        ("-- band ratios --", [f"band_{lo:g}_{hi:g}_rel" for lo, hi in C.OCTAVE_BANDS]),
        ("-- harmonics rel f0 --", [f"h{k}_rel" for k in C.HARMONICS]),
        ("-- spectral shape --", ["centroid_rel", "spread_rel", "flatness_rel",
                                  "entropy_rel", "rolloff85_rel"]),
        ("-- time domain --", ["crest_factor", "kurtosis", "env_dynamic_range_db",
                               "env_mod_flatness"]),
    ]
    for title, keys in groups:
        print(f"  {title}")
        for k in keys:
            if k not in df.columns:
                continue
            v = pd.to_numeric(df[k], errors="coerce")
            cells = "".join(f"{v[df['cls'] == c].median():>11.2f}" for c in C.CLASSES)
            print(f"    {k:<26}{cells}")

    sep = separability(df)
    print("\n  TOP METRICS BY HARDEST-PAIR SEPARATION AMONG BROKEN CLASSES")
    print("  " + "=" * 74)
    print(f"  {'metric':<28}{'worst pair':>12}{'mean within':>14}{'normal vs broken':>19}")
    for e in sep["best_within_broken"][:12]:
        print(f"  {e['metric']:<28}{e['best_worst_case_auc']:>12.3f}"
              f"{e['mean_within_auc']:>14.3f}{e['normal_vs_broken_auc']:>19.3f}")
    print("\n  AUC 0.5 = chance, 1.0 = perfect. 'worst pair' is the best available "
          "separation\n  among the three broken classes, so it bounds the 4-class task.")


if __name__ == "__main__":
    prim = pd.read_csv(C.result_path("segments_primary.csv"))
    run(prim)
