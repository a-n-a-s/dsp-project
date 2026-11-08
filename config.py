"""Global configuration for Case 28 - UAV propeller fault detection from onboard audio.

Every path, hyperparameter and class definition used anywhere in the pipeline is
declared here so that a run is reproducible from a single module.
"""
from __future__ import annotations

import os
from pathlib import Path

RANDOM_STATE = 42

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "Dataset"
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
CACHE_DIR = RESULTS_DIR / "cache"

for _d in (FIGURES_DIR, RESULTS_DIR, MODELS_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- labels
# Folder order is the canonical class order used in every table, figure and model.
CLASSES = ["normal", "1_broken", "2_broken", "3_4_broken"]
CLASS_LABELS = {
    "normal": "Normal",
    "1_broken": "1 broken",
    "2_broken": "2 broken",
    "3_4_broken": "3-4 broken",
}
CLASS_COLORS = {
    "normal": "#1f77b4",
    "1_broken": "#ff7f0e",
    "2_broken": "#2ca02c",
    "3_4_broken": "#d62728",
}
CLASS_TO_INT = {c: i for i, c in enumerate(CLASSES)}
INT_TO_CLASS = {i: c for c, i in CLASS_TO_INT.items()}

# Binary task: healthy vs any damaged configuration.
BINARY_MAP = {c: ("normal" if c == "normal" else "abnormal") for c in CLASSES}
BINARY_LABELS = {"normal": "Normal", "abnormal": "Abnormal"}
BINARY_COLORS = {"normal": "#1f77b4", "abnormal": "#d62728"}

MISSIONS = ["cw_square", "ccw_square", "cw_f8", "ccw_f8"]

# ---------------------------------------------------------------- signal
# Verified by Phase 1 across all 293 files: mono, 16-bit, 16 kHz.
FS_EXPECTED = 16000
NYQUIST = FS_EXPECTED / 2.0

# ---------------------------------------------------------------- Phase 2 segmentation
# Level and dominant-frequency tracking.
STFT_NFFT = 4096
STFT_HOP = 1024              # 64 ms hop
STFT_WINDOW = "hann"

# Band used to decide whether a frame is powered and which frequency dominates.
LEVEL_BAND = (40.0, 3000.0)  # Hz, excludes rumble and near-Nyquist hiss
F0_BAND = (40.0, 2000.0)     # Hz, search range for the blade-pass fundamental

# Frame activity test. A fixed margin over the recording's own noise floor
# does not work here: these recordings are powered for 90-93% of their length
# with only 4-16 dB between quietest and loudest second, so a 10 dB margin
# rejects whole files. Instead a frame counts as powered when the blade tone
# stands out from its own local spectral median, which is scale-free and
# insensitive to recording gain.
TONE_PROMINENCE_DB = 8.0

# A frame is "steady" when the dominant frequency stays within this tolerance of
# a 9-frame (576 ms) moving average, i.e. the rotor is at roughly constant RPM.
STEADY_SMOOTH_FRAMES = 9
STEADY_TOL_HZ = 5.0

# Steady runs separated by less than this are merged, so a brief dip in tone
# prominence does not split one hover phase into several unusable pieces.
STEADY_MERGE_GAP_FRAMES = 16

# Minimum accepted run lengths, expressed in STFT frames.
MIN_POWERED_FRAMES = 5
MIN_STEADY_FRAMES = 3
MIN_STEADY_SECONDS = 1.0

# Segment length and stride for feature extraction.
SEGMENT_SECONDS = 4.0
SEGMENT_STRIDE_FRACTION = 0.5
MIN_SEGMENT_SECONDS = 2.0    # below this a window is discarded, not padded

# Cap on windows kept per recording. Longer flights would otherwise dominate,
# and the cap is applied uniformly so every file contributes equally. The cap
# is above the median yield, so it trims the tail rather than resampling.
MAX_WINDOWS_PER_FILE = 20

# Windows taken from a fallback region (no steady run was found) are excluded
# from the primary analysis set. Their blade tone is measurably weaker, so
# including them would blur the classes for reasons unrelated to damage.
EXCLUDE_FALLBACK_WINDOWS = True

# ---------------------------------------------------------------- Phase 4 features
# Analysis band for the Welch PSD. 40-4000 Hz covers blade-pass and its early
# harmonics while staying clear of the 6.5-8 kHz region used for the noise
# floor check and clear of the anti-alias rolloff.
PSD_BAND = (40.0, 4000.0)
WELCH_NPERSEG = 16384
WELCH_NOVERLAP = 8192

# Octave bands expressed as dB relative to total power in PSD_BAND. The chosen
# set is octave-spaced below 1280 Hz then jumps to the 2-4 kHz region where the
# blade-noise redistribution is strongest.
OCTAVE_BANDS = [
    (40.0, 80.0),
    (80.0, 160.0),
    (160.0, 320.0),
    (320.0, 640.0),
    (640.0, 1280.0),
    (2000.0, 4000.0),
]
BAND_FEATURE_NAMES = [f"band_{lo:g}_{hi:g}_rel" for lo, hi in OCTAVE_BANDS]

HARMONICS = [2, 3, 4, 5, 6]
HARMONIC_TOL_HZ = 4.0
HARMONIC_FEATURE_NAMES = [f"h{k}_rel" for k in HARMONICS]

SHAPE_FEATURE_NAMES = [
    "centroid_rel", "spread_rel", "flatness_rel",
    "entropy_rel", "rolloff85_rel", "peak_to_median_rel",
]
TIME_FEATURE_NAMES = ["kurtosis", "crest_factor", "zero_cross_rate"]

FEATURE_NAMES = (
    BAND_FEATURE_NAMES
    + HARMONIC_FEATURE_NAMES
    + SHAPE_FEATURE_NAMES
    + TIME_FEATURE_NAMES
)

# Reference feature for the conventional threshold baseline.
BASELINE_FEATURE = "band_160_320_rel"

# ---------------------------------------------------------------- Phase 6 models
RF_PARAMS = dict(
    n_estimators=50,
    max_depth=6,
    min_samples_leaf=3,
    max_features="sqrt",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
SVM_PARAMS = dict(C=1.0, max_iter=20000, random_state=RANDOM_STATE)
LR_PARAMS = dict(C=1.0, max_iter=5000, random_state=RANDOM_STATE)

# ---------------------------------------------------------------- Phase 8 robustness
SNR_LEVELS_DB = [20.0, 15.0, 10.0, 5.0, 0.0]
WINDOW_LENGTHS_SECONDS = [1.0, 2.0, 4.0, 8.0]

# ---------------------------------------------------------------- plotting
FIGURE_DPI = 150
plt_rc = {
    "figure.dpi": 110,
    "savefig.dpi": FIGURE_DPI,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
}


def figure_path(stem: str) -> Path:
    """Return the on-disk path for a figure and make sure its folder exists."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR / f"{stem}.png"


def result_path(stem: str):
    """Return the on-disk path for a JSON/CSV result artefact."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / stem


def set_seed() -> None:
    import random

    import numpy as np

    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    try:
        import sklearn

        sklearn.set_config(assume_finite=False)
    except Exception:
        pass


def describe_environment() -> dict:
    """Record library versions for the report appendix."""
    import platform
    import sys

    import matplotlib
    import numpy
    import scipy
    import sklearn

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
        "random_state": RANDOM_STATE,
    }
