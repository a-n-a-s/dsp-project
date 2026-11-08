"""Report figures for the model, baseline and robustness results.

These figures close the gaps left by ``figures.py`` and ``figures_segmentation.py``
(which describe the dataset and the segmentation) by showing the outcomes that
actually answer the case question:

fig_result_baseline     threshold baseline vs primary RF, binary and 4-class
fig_result_snr          macro-F1 of the primary model as noise is added
fig_result_windowlen    macro-F1 as the analysis window is shortened / lengthened
fig_result_importance   RF impurity importance of the four selected features
fig_result_confusion    binary (normal/abnormal) confusion matrix of the OOF run

Every figure reads only from artefacts already on disk, so re-running this
module after a pipeline run regenerates them without touching the models.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C

plt.rcParams.update(C.plt_rc)


def _load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def fig_result_baseline() -> plt.Figure:
    """Baseline threshold vs random forest, for both decision problems."""
    base = pd.read_csv(C.result_path("baseline_metrics.csv"))
    model = pd.read_csv(C.result_path("model_metrics.csv"))

    rf = model[model["configuration"].str.contains(
        "random_forest, leave-one-mission-out")].iloc[0]
    # The 4-class RF row is the primary one; binary RF is derived from the OOF
    # predictions if present, else from the report.
    report = _load_json(C.ROOT / "analysis" / "model_report.json")
    binary_health = report.get("binary_health", {})
    base_bin = base[base["configuration"].str.contains("binary")].iloc[0]
    base_4 = base[base["configuration"].str.contains("4") |
                  base["configuration"].str.contains("four") |
                  ~base["configuration"].str.contains("binary")].iloc[0]

    labels = ["Baseline threshold\n(binary)", "Random forest\n(binary)",
              "Baseline threshold\n(4-class)", "Random forest\n(4-class)"]
    bal = [
        float(base_bin["balanced_accuracy"]),
        float(binary_health.get("balanced_accuracy", np.nan)),
        float(base_4["balanced_accuracy"]),
        float(rf["balanced_accuracy"]),
    ]
    f1 = [
        float(base_bin["macro_f1"]),
        float(binary_health.get("macro_f1", np.nan)),
        float(base_4["macro_f1"]),
        float(rf["macro_f1"]),
    ]
    # Fall back: binary health may only carry accuracy/precision/recall.
    if not np.isfinite(bal[1]):
        # recompute binary macro-F1 from the OOF table if available
        oof_path = C.result_path("oof_predictions.csv")
        if oof_path.exists():
            from sklearn.metrics import f1_score, balanced_accuracy_score
            oof = pd.read_csv(oof_path)
            yb = oof["cls"].map(lambda c: "normal" if c == "normal" else "abnormal")
            pb = oof["pred"].map(lambda c: "normal" if c == "normal" else "abnormal")
            bal[1] = balanced_accuracy_score(yb, pb)
            f1[1] = f1_score(yb, pb, labels=["normal", "abnormal"],
                             average="macro", zero_division=0)

    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    w = 0.38
    b1 = ax.bar(x - w / 2, bal, w, label="balanced accuracy", color="#4c72b0")
    b2 = ax.bar(x + w / 2, f1, w, label="macro F1", color="#dd8452")
    ax.bar_label(b1, fmt="%.3f", fontsize=8, padding=2)
    ax.bar_label(b2, fmt="%.3f", fontsize=8, padding=2)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("score")
    ax.set_title("Single-threshold baseline vs primary random forest")
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    return fig


def fig_result_snr() -> plt.Figure:
    """4-class and binary scores as AWGN is added to the test fold."""
    table = pd.read_csv(C.result_path("robustness_snr.csv"))
    table = table.sort_values("snr_db", ascending=False)
    # clean sits at +inf; plot it at the leftmost position
    x = np.arange(len(table))
    labels = [
        "clean" if not np.isfinite(s) else f"{s:g} dB"
        for s in table["snr_db"]
    ]

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ax.plot(x, table["macro_f1"], "o-", color="#4c72b0", label="4-class macro F1")
    ax.plot(x, table["binary_accuracy"], "s--", color="#dd8452",
            label="binary accuracy")
    ax.plot(x, table["binary_far"], "^:", color="#c44e52",
            label="binary false-alarm rate")
    for xi, yi in zip(x, table["macro_f1"]):
        ax.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_xlabel("test-set SNR (train always clean)")
    ax.set_ylabel("score")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Robustness to additive white Gaussian noise")
    ax.legend(loc="lower left", frameon=True)
    fig.tight_layout()
    return fig


def fig_result_windowlen() -> plt.Figure:
    """Macro-F1 as the analysis window length changes."""
    table = pd.read_csv(C.result_path("robustness_window.csv"))
    table = table.sort_values("window_seconds")

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(table["window_seconds"], table["macro_f1"], "o-",
            color="#4c72b0", lw=2, label="4-class macro F1")
    ax.plot(table["window_seconds"], table["binary_accuracy"], "s--",
            color="#dd8452", label="binary accuracy")
    for xi, yi in zip(table["window_seconds"], table["macro_f1"]):
        ax.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    ax.axvline(C.SEGMENT_SECONDS, color="grey", ls=":", lw=1)
    ax.annotate("default 4 s", (C.SEGMENT_SECONDS, ax.get_ylim()[0]),
                xytext=(4, 6), textcoords="offset points", fontsize=8,
                color="grey")
    ax.set_xticks(table["window_seconds"])
    ax.set_xlabel("analysis window length (s)")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.08)
    ax.set_title("Effect of analysis window length (leave-one-mission-out)")
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    return fig


def fig_result_importance() -> plt.Figure:
    """Impurity-based importance of the selected features, and their per-fold
    presence, taken from a forest fitted on all recordings."""
    from sklearn.ensemble import RandomForestClassifier

    from . import features as F

    selected = json.loads((C.ROOT / "analysis" / "selected_features.json")
                          .read_text(encoding="utf-8"))["selected"]
    rec = F.recording_table()
    X, y = rec[selected], rec["cls"].to_numpy()
    m = RandomForestClassifier(**C.RF_PARAMS)
    m.fit(X, y)
    imp = pd.Series(m.feature_importances_, index=selected).sort_values()

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    colors = ["#55a868" if "crest" in n else "#4c72b0" for n in imp.index]
    bars = ax.barh(imp.index, imp.values, color=colors)
    ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=3)
    ax.set_xlabel("impurity importance (all recordings, not cross-validated)")
    ax.set_xlim(0, max(imp.values) * 1.25)
    ax.set_title("Selected feature importance (primary random forest)")
    # Blue = spectral shape / harmonic; green = blade-band crest.
    ax.text(0.99, 0.02,
            "green: blade-band crest   blue: spectral shape / harmonic",
            transform=ax.transAxes, ha="right", fontsize=7.5, color="grey")
    fig.tight_layout()
    return fig


def fig_result_confusion() -> plt.Figure:
    """Binary normal/abnormal confusion matrix of the primary OOF run."""
    path = C.result_path("binary_confusion.csv")
    if not path.exists():
        # derive from OOF if the dedicated csv is missing
        oof_path = C.result_path("oof_predictions.csv")
        if not oof_path.exists():
            raise FileNotFoundError("need binary_confusion.csv or "
                                   "oof_predictions.csv")
        oof = pd.read_csv(oof_path)
        y = oof["cls"].map(lambda c: "normal" if c == "normal" else "abnormal")
        p = oof["pred"].map(lambda c: "normal" if c == "normal" else "abnormal")
        cm = np.array([
            [int(((y == "normal") & (p == "normal")).sum()),
             int(((y == "normal") & (p == "abnormal")).sum())],
            [int(((y == "abnormal") & (p == "normal")).sum()),
             int(((y == "abnormal") & (p == "abnormal")).sum())],
        ])
        row_labels = ["true\nnormal", "true\nabnormal"]
        col_labels = ["pred\nnormal", "pred\nabnormal"]
    else:
        df = pd.read_csv(path, index_col=0)
        cm = df.to_numpy()
        row_labels = [str(i).replace("true ", "true\n") for i in df.index]
        col_labels = [str(c).replace("pred ", "pred\n") for c in df.columns]

    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}",
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() * 0.55 else "black",
                    fontsize=12, fontweight="bold")
    ax.set_xticks(range(cm.shape[1]), col_labels)
    ax.set_yticks(range(cm.shape[0]), row_labels)
    ax.set_title("Binary confusion (leave-one-mission-out, RF)")
    fig.colorbar(im, shrink=0.8, ax=ax)
    fig.tight_layout()
    return fig


def fig_result_confusion_4class() -> plt.Figure:
    """4-class confusion matrix of the primary RF out-of-fold predictions."""
    path = C.result_path("oof_predictions.csv")
    if not path.exists():
        raise FileNotFoundError("need results/oof_predictions.csv (run models)")
    from sklearn.metrics import confusion_matrix

    oof = pd.read_csv(path)
    cm = confusion_matrix(oof["cls"], oof["pred"], labels=C.CLASSES)
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}",
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() * 0.55 else "black",
                    fontsize=11, fontweight="bold")
    labels = [C.CLASS_LABELS[c] for c in C.CLASSES]
    ax.set_xticks(range(len(labels)), labels, rotation=20, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    acc = float((oof["cls"] == oof["pred"]).mean())
    ax.set_title(f"4-class, leave-one-mission-out (RF): accuracy {acc:.3f}")
    fig.colorbar(im, shrink=0.8, ax=ax)
    fig.tight_layout()
    return fig


def run(verbose: bool = True) -> list[Path]:
    """Draw every result figure. Missing inputs are skipped with a note."""
    jobs = [
        ("fig_result_baseline", "threshold baseline vs random forest",
         fig_result_baseline),
        ("fig_result_snr", "noise robustness", fig_result_snr),
        ("fig_result_windowlen", "window-length sweep", fig_result_windowlen),
        ("fig_result_importance", "feature importance", fig_result_importance),
        ("fig_result_confusion", "binary confusion matrix",
         fig_result_confusion),
        ("fig_result_confusion_4class", "4-class confusion matrix",
         fig_result_confusion_4class),
    ]
    written: list[Path] = []
    if verbose:
        print("\nRESULT FIGURES")
    for stem, title, fn in jobs:
        try:
            fig = fn()
            path = C.figure_path(stem)
            fig.savefig(path, dpi=C.FIGURE_DPI)
            plt.close(fig)
            written.append(path)
            if verbose:
                print(f"  {path.name}  ({title})")
        except FileNotFoundError as e:
            if verbose:
                print(f"  skipped {stem}: {e}")
    return written


if __name__ == "__main__":
    run()
