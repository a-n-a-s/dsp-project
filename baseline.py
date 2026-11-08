"""Phase 4 - conventional DSP threshold baseline.

The case brief requires one conventional DSP/statistical method before any ML,
so that improvement can be measured quantitatively. The baseline here is a
threshold on a single level-normalised band ratio, ``band_160_320_rel``: the
share of 40-4000 Hz energy that falls in the 160-320 Hz octave band. Normal
recordings concentrate energy around their ~227 Hz blade tone; damaged ones
redistribute it upward as broadband noise, so the band ratio drops.

Two variants are reported:

* ``binary``   one threshold, normal vs any damage. Tuned on the training fold
               only, by maximising balanced accuracy over candidate cut points.
* ``4-class``  three ordered thresholds on the same feature (a four-bin
               histogram classifier). Each bin is labelled by the majority
               class in the training fold. This is the strongest thing a single
               DSP feature can do, and its failure to separate the three damaged
               classes is the measured gap the ML models have to close.

Validation is leave-one-mission-out, identical to the primary ML configuration,
so the comparison in Table 5 is like-for-like.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support)

from . import config as C
from . import features as F

ANALYSIS_DIR = C.ROOT / "analysis"


# ====================================================================== 1-D tools
def candidate_cuts(v: np.ndarray) -> np.ndarray:
    """Midpoints between adjacent unique feature values, as threshold candidates."""
    u = np.unique(v)
    if u.size < 2:
        return np.array([])
    return (u[:-1] + u[1:]) / 2.0


def predict_regions(v: np.ndarray, cuts: np.ndarray, labels: np.ndarray
                    ) -> np.ndarray:
    """Assign each value to a region defined by ordered cuts, then a region label.

    ``labels[i]`` is the class assigned to region i (region i = values between
    cuts[i-1] and cuts[i], with region 0 below cuts[0]).
    """
    idx = np.searchsorted(cuts, v, side="right")
    return np.asarray(labels, dtype=object)[idx]


def fit_bin_labels(v: np.ndarray, y: np.ndarray, cuts: np.ndarray
                   ) -> list[str]:
    """Majority training class inside each region defined by ``cuts``."""
    idx = np.searchsorted(cuts, v, side="right")
    labels = []
    for r in range(len(cuts) + 1):
        sel = y[idx == r]
        if sel.size == 0:
            # Empty region: fall back to the nearest non-empty side so an
            # unseen feature range cannot produce an undefined label.
            labels.append(y[np.argmin(np.abs(idx - r))])
        else:
            counts = pd.Series(sel).value_counts()
            labels.append(str(counts.index[0]))
    return labels


def best_binary_cuts(v: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Single threshold maximising training balanced accuracy."""
    cand = candidate_cuts(v)
    if cand.size == 0:
        return np.array([]), float((y == pd.Series(y).mode()[0]).mean())
    best_cut, best_acc = cand[0], -1.0
    for t in cand:
        labels = fit_bin_labels(v, y, np.array([t]))
        pred = predict_regions(v, np.array([t]), labels)
        acc = balanced_accuracy_score(y, pred)
        if acc > best_acc:
            best_cut, best_acc = float(t), float(acc)
    return np.array([best_cut]), best_acc


def best_ordinal_cuts(v: np.ndarray, y: np.ndarray, n_regions: int = 4
                     ) -> tuple[np.ndarray, float]:
    """Ordered cut points maximising training accuracy with majority labels.

    Dynamic program over the sorted unique values: dp[k][i] is the best
    training accuracy obtainable with k regions covering the first i unique
    values. n is ~200 recordings, so the O(k * n^2) search is instantaneous.
    """
    u = np.unique(v)
    if u.size < n_regions:
        cuts = np.linspace(u.min(), u.max(), n_regions + 1)[1:-1]
        return cuts, float((y == pd.Series(y).mode()[0]).mean())

    # Region cost: accuracy of majority-labelling samples inside [a, b).
    idx_of = np.searchsorted(u, v, side="right")
    classes = np.array(sorted(set(y)), dtype=object)
    onehot = (y[:, None] == classes[None, :]).astype(np.int64)
    # counts[j] = class counts for unique values < j
    counts = np.zeros((u.size + 1, classes.size), dtype=np.int64)
    for j in range(u.size):
        counts[j + 1] = counts[j] + onehot[idx_of == j].sum(axis=0)

    def region_acc(a: int, b: int) -> float:
        """Accuracy of majority label on samples with unique index in [a, b)."""
        c = counts[b] - counts[a]
        n = c.sum()
        return (c.max() / n) if n else 0.0

    # dp[k][j] = best (sample-weighted) correct-count with k regions covering
    # uniques [0, j). Maximising correct-count == maximising accuracy because
    # the total sample count is fixed.
    neg = -1e18
    dp = np.full((n_regions + 1, u.size + 1), neg)
    back = np.full((n_regions + 1, u.size + 1), -1, dtype=np.int64)
    dp[0][0] = 0.0
    for k in range(1, n_regions + 1):
        for j in range(k, u.size + 1):
            for a in range(k - 1, j):
                n = int((counts[j] - counts[a]).sum())
                val = dp[k - 1][a] + region_acc(a, j) * n
                if val > dp[k][j]:
                    dp[k][j] = val
                    back[k][j] = a

    # Recover cuts.
    cuts_idx = []
    j = u.size
    for k in range(n_regions, 0, -1):
        a = int(back[k][j])
        cuts_idx.append(a)
        j = a
    cuts_idx = sorted(cuts_idx[:-1])          # drop the k=1 anchor at 0
    cuts = np.array([(u[i - 1] + u[i]) / 2.0 for i in cuts_idx if i > 0])
    total = float(len(y))
    acc = float(dp[n_regions][u.size] / total) if total else 0.0
    return cuts, acc


# ====================================================================== evaluation
def evaluate_binary(v: np.ndarray, y4: np.ndarray, groups: pd.Series) -> dict:
    """Leave-one-mission-out binary threshold baseline."""
    y = np.array(["normal" if c == "normal" else "abnormal" for c in y4])
    pred = np.empty_like(y)
    cuts_seen = []
    for g in sorted(groups.unique()):
        te = (groups == g).to_numpy()
        tr = ~te
        cuts, _ = best_binary_cuts(v[tr], y[tr])
        labels = fit_bin_labels(v[tr], y[tr], cuts)
        pred[te] = predict_regions(v[te], cuts, labels)
        cuts_seen.append(dict(mission=str(g), threshold=float(cuts[0]) if cuts.size else np.nan,
                              train_labels=labels))

    p, r, f, s = precision_recall_fscore_support(
        y, pred, labels=["normal", "abnormal"], zero_division=0)
    tp = int(((y == "abnormal") & (pred == "abnormal")).sum())
    fp = int(((y == "normal") & (pred == "abnormal")).sum())
    fn = int(((y == "abnormal") & (pred == "normal")).sum())
    tn = int(((y == "normal") & (pred == "normal")).sum())
    return dict(
        configuration=f"threshold baseline ({C.BASELINE_FEATURE}), binary, "
                      "leave-one-mission-out",
        task="binary normal vs abnormal",
        feature=C.BASELINE_FEATURE,
        n_recordings=int(len(y)),
        accuracy=float((pred == y).mean()),
        balanced_accuracy=float(balanced_accuracy_score(y, pred)),
        macro_f1=float(f1_score(y, pred, labels=["normal", "abnormal"],
                                average="macro", zero_division=0)),
        precision_abnormal=float(p[1]), recall_abnormal=float(r[1]),
        f1_abnormal=float(f[1]),
        false_positive=fp, false_negative=fn, true_positive=tp, true_negative=tn,
        false_alarm_rate=float(fp / max(fp + tn, 1)),
        confusion=confusion_matrix(y, pred, labels=["normal", "abnormal"]).tolist(),
        confusion_labels=["normal", "abnormal"],
        per_class=[dict(cls=c, label=C.BINARY_LABELS[c], support=int(n),
                        precision=float(pi), recall=float(ri), f1=float(fi))
                   for c, pi, ri, fi, n in
                   zip(["normal", "abnormal"], p, r, f, s)],
        fold_thresholds=cuts_seen,
        note="threshold tuned on the training fold only; no test-fold statistics "
             "enter the cut point",
    )


def evaluate_4class(v: np.ndarray, y: np.ndarray, groups: pd.Series) -> dict:
    """Leave-one-mission-out three-threshold (four-bin) baseline."""
    pred = np.empty(len(y), dtype=object)
    folds = []
    for g in sorted(groups.unique()):
        te = (groups == g).to_numpy()
        tr = ~te
        cuts, train_acc = best_ordinal_cuts(v[tr], y[tr], n_regions=4)
        labels = fit_bin_labels(v[tr], y[tr], cuts)
        pred[te] = predict_regions(v[te], cuts, labels)
        folds.append(dict(mission=str(g), thresholds=[float(c) for c in cuts],
                          train_labels=labels, train_accuracy=train_acc))

    p, r, f, s = precision_recall_fscore_support(
        y, pred, labels=C.CLASSES, zero_division=0)
    return dict(
        configuration=f"threshold baseline ({C.BASELINE_FEATURE}), 4-class, "
                      "leave-one-mission-out",
        task="4-class fault category",
        feature=C.BASELINE_FEATURE,
        n_recordings=int(len(y)),
        accuracy=float((pred == y).mean()),
        balanced_accuracy=float(balanced_accuracy_score(y, pred)),
        macro_f1=float(f1_score(y, pred, labels=C.CLASSES,
                                average="macro", zero_division=0)),
        n_errors=int((pred != y).sum()),
        per_class=[dict(cls=c, label=C.CLASS_LABELS[c], support=int(n),
                        precision=float(pi), recall=float(ri), f1=float(fi))
                   for c, pi, ri, fi, n in zip(C.CLASSES, p, r, f, s)],
        confusion=confusion_matrix(y, pred, labels=C.CLASSES).tolist(),
        confusion_labels=C.CLASSES,
        fold_thresholds=folds,
        note="three ordered thresholds on one feature; bins labelled by "
             "training-fold majority class",
    )


# ====================================================================== driver
def run(verbose: bool = True) -> dict:
    rec = F.recording_table()
    v = rec[C.BASELINE_FEATURE].to_numpy(float)
    y = rec["cls"].to_numpy()
    groups = rec["mission"]

    if verbose:
        print("\nDSP THRESHOLD BASELINE")
        print(f"  feature {C.BASELINE_FEATURE}, {len(y)} recordings, "
              "leave-one-mission-out")

    binary = evaluate_binary(v, y, groups)
    four = evaluate_4class(v, y, groups)

    if verbose:
        print(f"  binary  acc {binary['accuracy']:.4f}  "
              f"bal.acc {binary['balanced_accuracy']:.4f}  "
              f"macroF1 {binary['macro_f1']:.4f}  "
              f"FAR {binary['false_alarm_rate']:.4f}")
        print(f"  4-class acc {four['accuracy']:.4f}  "
              f"bal.acc {four['balanced_accuracy']:.4f}  "
              f"macroF1 {four['macro_f1']:.4f}  errors {four['n_errors']}")

    # One CSV row per task, same columns as model_metrics.csv where they overlap.
    rows = []
    for r in (binary, four):
        n_err = r.get("n_errors",
                      r.get("false_positive", 0) + r.get("false_negative", 0))
        rows.append(dict(
            configuration=r["configuration"],
            balanced_accuracy=r["balanced_accuracy"],
            macro_f1=r["macro_f1"],
            accuracy=r["accuracy"],
            macro_auc_ovr=float("nan"),
            n_errors=n_err,
        ))
    table = pd.DataFrame(rows)
    table.to_csv(C.result_path("baseline_metrics.csv"), index=False)

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    (ANALYSIS_DIR / "baseline_report.json").write_text(json.dumps(dict(
        unit="one row per recording (median of its windows)",
        feature=C.BASELINE_FEATURE,
        validation="LeaveOneGroupOut on mission, identical to the primary ML "
                   "configuration",
        binary=binary,
        four_class=four,
    ), indent=2, default=str), encoding="utf-8")

    if verbose:
        print(f"  -> results/baseline_metrics.csv, analysis/baseline_report.json")
    return dict(table=table, binary=binary, four_class=four)


if __name__ == "__main__":
    run()
