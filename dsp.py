"""Shared DSP primitives.

Used by the Phase 2 diagnostic figures and, later, by the Phase 4 feature
extractor. Every spectrum here is expressed relative to the total power in
``config.PSD_BAND`` so that the result does not depend on recording gain or on
the exact blade-pass frequency.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch

from . import config as C


def welch_psd(x: np.ndarray, fs: int, nperseg: int | None = None,
              noverlap: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Single-sided PSD of one analysis window."""
    nperseg = nperseg or C.WELCH_NPERSEG
    noverlap = noverlap if noverlap is not None else C.WELCH_NOVERLAP
    if len(x) < 256:
        return np.array([]), np.array([])
    nperseg = min(nperseg, len(x))
    f, P = welch(x, fs=fs, nperseg=nperseg,
                 noverlap=min(noverlap, npersergap(nperseg)),
                 window="hann", detrend="constant", scaling="density")
    return f, P


def npersergap(nperseg: int) -> int:
    """50% overlap, the setting used throughout the study."""
    return nperseg // 2


def preprocess(x: np.ndarray, fs: int, lo: float | None = None,
               hi: float | None = None, order: int = 4) -> np.ndarray:
    """Band-pass one window to the analysis band, with DC removed.

    Two different things are removed here:

    * Below ``lo`` the recording carries large low-frequency energy from
      airframe and handling noise, measured at roughly -39 dB relative to the
      40-4000 Hz total. It carries no blade-fault evidence and it skews the
      low end of every spectral-shape metric.
    * Above ``hi`` the spectrum is dominated by the recorder's own flat noise
      floor near -90 dB/Hz. Keeping it adds no information and only adds
      variance to the PSD estimate.

    A zero-phase Butterworth is used so the filter does not shift the timing of
    transients, applied twice by ``sosfiltfilt`` so the effective slope is halved
    and the passband stays flat.
    """
    from scipy.signal import butter, sosfiltfilt

    lo = C.PSD_BAND[0] if lo is None else lo
    hi = C.PSD_BAND[1] if hi is None else hi
    nyq = fs / 2.0
    if lo >= hi or lo <= 0 or hi >= nyq:
        return x - np.mean(x)

    sos = butter(order, [lo / nyq, hi / nyq], btype="band", output="sos")
    y = sosfiltfilt(sos, x)
    # Replace the filter's edge transients with the original samples.
    edge = min(len(y) // 10, fs // 20)
    if edge > 8:
        y[:edge] = x[:edge]
        y[-edge:] = x[-edge:]
    return y


def notch(x: np.ndarray, fs: int, freqs: list[float] | tuple[float, ...] = (),
          q: float = 30.0) -> np.ndarray:
    """Remove narrowband interference with zero-phase IIR notch filters.

    ``freqs`` is a list of centre frequencies in Hz. An empty list returns the
    signal unchanged, which is the default setting for this dataset: a scan of
    40 recordings found no persistent narrowband line anywhere between 40 and
    4000 Hz, so no notch is applied in the study as it stands. The parameter is
    kept so the finding can be re-checked, or a new recording set can be
    handled without touching the rest of the pipeline.

    A high Q is used deliberately: a low Q would remove a wide swath of
    spectrum and take genuine blade harmonics with it. Q=30 at 1 kHz gives
    roughly 33 Hz of attenuation width.
    """
    if not len(freqs):
        return x

    from scipy.signal import iirnotch, tf2sos, sosfiltfilt

    nyq = fs / 2.0
    y = x
    for f0 in freqs:
        if not 0 < f0 < nyq:
            continue
        b, a = iirnotch(f0, Q=q, fs=fs)
        y = sosfiltfilt(tf2sos(b, a), y)
    return y


def analysis_band(f: np.ndarray) -> np.ndarray:
    """Boolean mask for the 40-4000 Hz analysis band."""
    return (f >= C.PSD_BAND[0]) & (f <= C.PSD_BAND[1])


def band_relative_db(f: np.ndarray, P: np.ndarray, lo: float, hi: float) -> float:
    """Energy in [lo, hi) as dB relative to the 40-4000 Hz total.

    This is the single most important normalisation in the study: it removes
    both recording gain and the throttle-dependent blade-pass frequency, so what
    remains is the shape of the spectrum.
    """
    m = analysis_band(f)
    if not m.any():
        return np.nan
    ref = 10.0 * np.log10(P[m].sum() + 1e-30)
    b = (f >= lo) & (f < hi)
    if not b.any():
        return np.nan
    return float(10.0 * np.log10(P[b].sum() + 1e-30) - ref)


def refined_peak(f: np.ndarray, P: np.ndarray, lo: float, hi: float) -> tuple[float, float]:
    """Dominant frequency in [lo, hi) with parabolic interpolation, and its level.

    Returns (f_peak_hz, peak_db_relative_to_band_total).
    """
    b = (f >= lo) & (f < hi)
    if not b.any():
        return np.nan, np.nan
    fb, Pb = f[b], P[b]
    i = int(np.argmax(Pb))
    ref = 10.0 * np.log10(P[analysis_band(f)].sum() + 1e-30)
    peak_db = float(10.0 * np.log10(Pb[i] + 1e-30) - ref)

    if 0 < i < fb.size - 1:
        a, c, d = Pb[i - 1], Pb[i], Pb[i + 1]
        la, lc, ld = np.log(a + 1e-30), np.log(c + 1e-30), np.log(d + 1e-30)
        denom = la - 2 * lc + ld
        delta = 0.5 * (la - ld) / denom if abs(denom) > 1e-20 else 0.0
        fpk = float(fb[i] + np.clip(delta, -0.5, 0.5) * (fb[1] - fb[0]))
    else:
        fpk = float(fb[i])
    return fpk, peak_db


def harmonic_relative_db(f: np.ndarray, P: np.ndarray, f0: float, k: int,
                        tol_hz: float = C.HARMONIC_TOL_HZ) -> float:
    """Amplitude of the k-th harmonic as dB relative to the fundamental."""
    if not np.isfinite(f0) or f0 <= 0:
        return np.nan
    m = np.abs(f - k * f0) <= tol_hz
    if not m.any():
        return np.nan
    k0 = np.abs(f - f0) <= tol_hz
    ref = 10.0 * np.log10(P[k0].max() + 1e-30) if k0.any() else np.nan
    if not np.isfinite(ref):
        return np.nan
    return float(10.0 * np.log10(P[m].max() + 1e-30) - ref)


def window_spectrum(x: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Convenience wrapper: returns (f, P, f0_hz, f0_level_db) for one window."""
    f, P = welch_psd(x, fs)
    if f.size == 0:
        return f, P, np.nan, np.nan
    fpk, lvl = refined_peak(f, P, C.F0_BAND[0], C.F0_BAND[1])
    return f, P, fpk, lvl


def aggregate_psd(windows: list[np.ndarray], fs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and std of band-normalised PSDs across a list of analysis windows.

    Each window is normalised by its own 40-4000 Hz total before averaging, so
    the mean reflects spectral shape rather than loudness.
    """
    curves = []
    f_ref = None
    for x in windows:
        f, P = welch_psd(x, fs)
        if f.size == 0:
            continue
        m = analysis_band(f)
        f, P = f[m], P[m]
        P = P / (P.sum() + 1e-30)
        if f_ref is None:
            f_ref = f
        elif f.size != f_ref.size:
            continue
        curves.append(P)

    if not curves:
        return np.array([]), np.array([]), np.array([])
    A = np.vstack(curves)
    return f_ref, 10 * np.log10(A.mean(axis=0) + 1e-30), 10 * np.log10(A.std(axis=0) + 1e-30)
