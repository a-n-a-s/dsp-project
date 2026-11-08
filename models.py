"""Classification models and their metrics.

Four configurations are reported, all of them deliberately less flattering than
the obvious one:

``record-level``   one row per recording, the unit of independent information
``window-level``   one row per 4 s window, split by group so overlapping windows
                   still cannot straddle the boundary
``leave-one-mission-out``  hold out cw_f8 / ccw_f8 / cw_square / ccw_square in
                   turn, testing transfer to an unseen flight manoeuvre
``nested selection``  confound screen and feature selection re-run inside every
                   training fold - the primary configuration, because the case
                   brief requires preprocessing fitted without the test set

A naive random split of the overlapping windows measures 0.991 balanced
accuracy per window (0.995 aggregated per recording) against 0.979/0.986 for
the honest grouped split on the same features: the inflation is real but
small, measured in split_audit.json. Even leave-one-mission-out on recordings
cannot test generalisation across *propeller configurations*, because config is
a four-bit mask whose set-bit count is the class label itself, so no two
classes share a config.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support,
                             roc_auc_score)
from sklearn.model_selection import (LeaveOneGroupOut, StratifiedKFold,
                                     cross_val_predict, cross_val_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import config as C
from . import features as F
from . import stats as S

MODELS_DIR = C.ROOT / "analysis"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SEVERITY_GROUPS = ["1_broken", "2_broken", "3_4_broken"]

MODELS = {
    "random_forest": lambda: make_pipeline(StandardScaler(),
                                           RandomForestClassifier(**C.RF_PARAMS)),
    "linear_svm": lambda: make_pipeline(StandardScaler(),
                                        SVC(kernel="linear", probability=True,
                                            **C.SVM_PARAMS)),
    "logistic_regression": lambda: make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=C.LR_PARAMS["max_iter"],
                           random_state=C.RANDOM_STATE)),
    "knn_3": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(3)),
}


# ====================================================================== helpers
def macro_ovr_auc(y_true: np.ndarray, proba: np.ndarray,
                  classes: list[str]) -> float:
    """Macro-averaged one-vs-rest ROC AUC.

    Fold-wise accuracy is averaged across folds because the class balance
    changes between them, but AUC is pooled over the out-of-fold predictions:
    averaging AUCs across folds is not meaningful when folds differ in difficulty.
    """
    aucs = []
    for i, c in enumerate(classes):
        y = (np.asarray(y_true) == c).astype(int)
        if 0 < y.sum() < y.size:
            aucs.append(roc_auc_score(y, proba[:, i]))
    return float(np.mean(aucs)) if aucs else float("nan")


def per_class_report(y_true: np.ndarray, y_pred: np.ndarray) -> list[dict]:
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=C.CLASSES, zero_division=0)
    return [dict(cls=c, label=C.CLASS_LABELS[c], support=int(n),
                 precision=float(pi), recall=float(ri), f1=float(fi))
            for c, pi, ri, fi, n in zip(C.CLASSES, p, r, f, s)]


def _fold_predictions(model_fn, X, y, groups, cv):
    """Out-of-fold predictions and probabilities with explicit class alignment.

    ``cross_val_predict(method="predict_proba")`` cannot be used here. It stacks
    each fold's probability matrix, and each matrix is ordered by the classes that
    fold happened to contain in its training set. When a fold's training set is
    missing a class, the columns silently shift and the pooled AUC becomes
    meaningless (it came out at 0.40 for a model with 0.97 accuracy). Fitting fold
    by fold and re-indexing every row onto C.CLASSES removes that failure mode.
    """
    pred = np.empty(len(y), dtype=object)
    proba = np.full((len(y), len(C.CLASSES)), np.nan)
    for tr, te in cv.split(X, y, groups):
        model = model_fn()
        model.fit(X.iloc[tr], y[tr])
        pred[te] = model.predict(X.iloc[te])
        if hasattr(model, "predict_proba"):
            cols = list(model.classes_)
            p = model.predict_proba(X.iloc[te])
            for i, c in enumerate(cols):
                if c in C.CLASSES:
                    proba[te, C.CLASSES.index(c)] = p[:, i]
    return pred.astype(str), proba


def evaluate(model_fn, X: pd.DataFrame, y: np.ndarray,
             groups: pd.Series, name: str, description: str,
             cv=None) -> dict:
    """Fit out-of-fold and collect every metric for one configuration."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv = cv or LeaveOneGroupOut()
        pred, proba = _fold_predictions(model_fn, X, y, groups, cv)
        scores = cross_val_score(model_fn(), X, y, groups=groups, cv=cv,
                                 scoring="balanced_accuracy")

    out = dict(
        configuration=name, description=description,
        n_recordings=int(len(y)),
        balanced_accuracy=float(balanced_accuracy_score(y, pred)),
        fold_balanced_accuracy=[float(s) for s in scores],
        macro_f1=float(f1_score(y, pred, labels=C.CLASSES,
                                average="macro", zero_division=0)),
        accuracy=float((pred == y).mean()),
        n_errors=int((pred != y).sum()),
        per_class=per_class_report(y, pred),
        confusion=confusion_matrix(y, pred, labels=C.CLASSES).tolist(),
    )
    if proba is not None and proba.shape[1] == len(C.CLASSES):
        out["macro_auc_ovr"] = macro_ovr_auc(y, proba, C.CLASSES)
    return out


def evaluate_nested(model_fn, rec: pd.DataFrame, name_pool: list[str],
                    name: str, description: str,
                    groups: pd.Series, fold_keys: list[str] | None = None
                    ) -> tuple[dict, np.ndarray]:
    """Grouped evaluation with *every* statistic fitted inside the fold.

    The case brief requires preprocessing to be derived without using the test
    set, and this is the only configuration that satisfies it literally: the
    confound screen and the sequential feature selection are re-run on the
    training folds only, so nothing from the held-out group influences the
    feature set. The fold-to-fold variation in the chosen features is reported,
    because a feature set that changes completely when one group disappears is
    not a stable finding. ``groups`` is mission for the primary configuration
    and take_session for the repeat-flight generalisation check.
    """
    y = rec["cls"].to_numpy()
    groups = groups.reset_index(drop=True)
    cv = LeaveOneGroupOut()
    pred = np.empty(len(y), dtype=object)
    proba = np.full((len(y), len(C.CLASSES)), np.nan)
    feats_per_fold: dict[str, list[str]] = {}
    fold_ba = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, (tr, te) in enumerate(cv.split(rec, y, groups)):
            key = fold_keys[i] if fold_keys else str(groups.iloc[te[0]])
            sub = rec.iloc[tr]
            pool = F.confound_screen(sub, name_pool)[0]
            feats = F.select_features(sub, pool=pool, verbose=False)["selected"]
            feats_per_fold[key] = feats
            model = model_fn()
            model.fit(sub[feats], y[tr])
            pred[te] = model.predict(rec.iloc[te][feats])
            if hasattr(model, "predict_proba"):
                p = model.predict_proba(rec.iloc[te][feats])
                for j, c in enumerate(model.classes_):
                    if c in C.CLASSES:
                        proba[te, C.CLASSES.index(c)] = p[:, j]
            fold_ba.append(balanced_accuracy_score(y[te], pred[te]))

    pred = pred.astype(str)
    out = dict(
        configuration=name, description=description,
        selection="nested: confound screen + sequential selection fitted on "
                  "the training folds only",
        n_folds=len(fold_ba),
        features_per_fold=feats_per_fold,
        n_recordings=int(len(y)),
        balanced_accuracy=float(balanced_accuracy_score(y, pred)),
        fold_balanced_accuracy=[float(s) for s in fold_ba],
        macro_f1=float(f1_score(y, pred, labels=C.CLASSES,
                                average="macro", zero_division=0)),
        accuracy=float((pred == y).mean()),
        n_errors=int((pred != y).sum()),
        per_class=per_class_report(y, pred),
        confusion=confusion_matrix(y, pred, labels=C.CLASSES).tolist(),
    )
    if np.isfinite(proba).any(axis=1).all():
        out["macro_auc_ovr"] = macro_ovr_auc(y, proba, C.CLASSES)
    return out, pred


def severity_agreement(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Does the model get the *ordering* of damage right on damaged recordings?

    A model can score well on accuracy while consistently confusing 2_broken with
    3_4_broken, both of which are "bad". For a propeller-health use case the
    ordinal error is the one that matters.
    """
    order = {c: i for i, c in enumerate(SEVERITY_GROUPS)}
    pairs = [(y, p) for y, p in zip(y_true, y_pred) if y in order and p in order]
    if not pairs:
        return {}
    t = np.array([order[y] for y, _ in pairs])
    p = np.array([order[p] for _, p in pairs])
    exact = float((t == p).mean())
    mae = float(np.abs(t - p).mean())
    return dict(n=len(pairs), exact_match=exact, mean_absolute_error_in_steps=mae,
                note="restricted to recordings whose true label is a damaged class")


def binary_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Health-detection view: normal against everything damaged."""
    yb = np.array(["normal" if c == "normal" else "abnormal" for c in y_true])
    pb = np.array(["normal" if c == "normal" else "abnormal" for c in y_pred])
    tp = int(((yb == "abnormal") & (pb == "abnormal")).sum())
    fp = int(((yb == "normal") & (pb == "abnormal")).sum())
    fn = int(((yb == "abnormal") & (pb == "normal")).sum())
    tn = int(((yb == "normal") & (pb == "normal")).sum())
    return dict(
        accuracy=float((yb == pb).mean()),
        balanced_accuracy=float(balanced_accuracy_score(yb, pb)),
        macro_f1=float(f1_score(yb, pb, labels=["normal", "abnormal"],
                                average="macro", zero_division=0)),
        precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
        false_alarm_rate=fp / max(fp + tn, 1),
        false_positive=int(fp), false_negative=int(fn),
        note="damaged recorded as abnormal is a true positive; the "
             "confounded level/RPM protocol makes this task easy and it "
             "is not the claim being made")


# ====================================================================== drivers
def run(selected: list[str] | None = None, verbose: bool = True) -> dict:
    rec = F.recording_table()
    if selected is None:
        selected = F.run(verbose=False)["features"]
    X = rec[selected]
    y = rec["cls"].to_numpy()

    if verbose:
        print("\nCLASSIFICATION")
        print(f"  {len(selected)} features, {len(y)} recordings "
              f"({dict(pd.Series(y).value_counts())})")

    rows = []
    # --- reference: all features, leave-one-out on recordings. Deliberately the
    # most flattering configuration, printed to show how far the honest numbers
    # are from it.
    pool = F.candidate_pool(rec)
    rows.append(evaluate(MODELS["random_forest"], rec[pool], y, rec["relpath"],
                         f"upper bound: all {len(pool)} admissible metrics, "
                         "leave-one-recording-out",
                         "not a claim; shown to expose how much a random-style "
                         "split overstates the result"))

    # --- primary: leave-one-mission-out with selection nested per fold, so
    # the held-out mission never influences the feature set.
    name_pool = F._name_filtered_pool(rec)
    oof_by_model = {}
    for name, fn in MODELS.items():
        r, pred = evaluate_nested(
            fn, rec, name_pool, f"{name}, leave-one-mission-out",
            "primary configuration (confound screen and selection nested "
            "inside each training fold)", rec["mission"])
        rows.append(r)
        oof_by_model[name] = pred

    detail_rows = []
    for name in ("random_forest", "linear_svm"):
        r, _ = evaluate_nested(
            MODELS[name], rec, name_pool,
            f"{name}, leave-one-take-out",
            "unseen repeat flight of the same airframe (selection nested)",
            rec["take_session"])
        detail_rows.append(r)

    y_pred = oof_by_model["random_forest"]
    primary = dict(
        configuration="random_forest, leave-one-mission-out",
        severity=severity_agreement(y, y_pred),
        binary_health=binary_report(y, y_pred),
        features_per_fold=next(r["features_per_fold"] for r in rows
                               if r["configuration"]
                               == "random_forest, leave-one-mission-out"),
    )

    rows.extend(detail_rows)

    # Out-of-fold predictions for the primary model: the confusion-matrix and
    # per-class figures are drawn from these, not from a retrained artefact.
    oof = rec[["relpath", "cls", "mission", "take"]].copy()
    oof["pred"] = y_pred
    oof.to_csv(C.result_path("oof_predictions.csv"), index=False)

    table = pd.DataFrame([dict(
        configuration=r["configuration"],
        balanced_accuracy=r["balanced_accuracy"],
        macro_f1=r["macro_f1"],
        accuracy=r["accuracy"],
        macro_auc_ovr=r.get("macro_auc_ovr", float("nan")),
        n_errors=r["n_errors"],
    ) for r in rows])
    table = table.sort_values("balanced_accuracy", ascending=False)
    table.to_csv(C.result_path("model_metrics.csv"), index=False)

    deployment = run_deployment(selected, X, y)

    (MODELS_DIR / "model_report.json").write_text(json.dumps(dict(
        unit="one row per recording (median of its windows)",
        n_recordings=int(len(y)),
        n_windows_available=int(pd.read_csv(C.RESULTS_DIR / "window_metrics.csv")
                                .shape[0]),
        selected_features=selected,
        selection_protocol="headline rows are nested: the confound screen and "
                           "the sequential feature selection are re-fitted on "
                           "each training fold, so the held-out group never "
                           "influences the feature set. The deployed artefact "
                           "and the robustness sweeps use one feature set "
                           "selected on all recordings, which is what a "
                           "production model does at final training time.",
        features_per_fold=primary["features_per_fold"],
        validation_note="No window-level random split: 4 s windows overlap by "
                        "half, so adjacent windows are near-duplicates (the "
                        "naive split measures 0.991 against the honest 0.979, "
                        "split_audit.json). config is "
                        "a four-bit mask whose set-bit count equals the class "
                        "label, so cross-config generalisation cannot be tested.",
        severity_agreement=primary["severity"],
        binary_health=primary["binary_health"],
        deployment=deployment,
        results=rows,
    ), indent=2, default=str), encoding="utf-8")

    if verbose:
        print()
        print(f"  {'configuration':<52}{'bal acc':>9}{'macroF1':>9}"
              f"{'acc':>8}{'AUC':>8}{'err':>6}")
        print("  " + "-" * 92)
        for _, r in table.iterrows():
            print(f"  {r.configuration[:50]:<52}{r.balanced_accuracy:>9.4f}"
                  f"{r.macro_f1:>9.4f}{r.accuracy:>8.4f}"
                  f"{r.macro_auc_ovr:>8.4f}{r.n_errors:>6d}")
        print()
        print("  RF features selected inside each training fold:")
        for fold, feats in primary["features_per_fold"].items():
            print(f"    {fold:<12} {', '.join(feats)}")
        print()
        print("  severity ordering on damaged recordings: "
              f"exact {primary['severity']['exact_match']:.3f}, "
              f"mean |error| {primary['severity']['mean_absolute_error_in_steps']:.2f} "
              "steps")
        print(f"  deployment: model {deployment['model_size_kb']:.1f} KB, "
              f"predict {deployment['predict_per_recording_ms']:.2f} ms, "
              f"full 4 s window {deployment['window_to_decision_ms']:.0f} ms")
    return dict(table=table, results=rows, selected=selected, detail=primary,
                deployment=deployment, oof=oof)


def run_deployment(selected: list[str], X: pd.DataFrame,
                   y: np.ndarray) -> dict:
    """Model size and end-to-end inference cost of the primary random forest.

    Three numbers matter for a health monitor: how large the artefact is, how
    long one scoring call takes once features exist, and the total time from
    raw audio samples to a class label for one 4 s window. The last number
    includes feature extraction, which dominates: a 50-tree forest scores a
    4-feature row in tens of microseconds, while ``window_metrics`` costs
    roughly 30-40 ms. Detection latency is therefore set by the DSP, not by
    the model.
    """
    from .analysis import window_metrics as _wm
    from .segment import load_audio

    # Fit the saved artefact with n_jobs=1. Training uses n_jobs=-1 for speed,
    # but at inference a single 4-feature row cannot use the tree-level
    # parallelism: each predict pays a thread-pool dispatch cost that, on this
    # machine, is larger than the forest itself (about 20 ms against 6 ms for
    # the serial predict). The trees are identical for a fixed random_state.
    params = dict(C.RF_PARAMS)
    params["n_jobs"] = 1
    model = make_pipeline(StandardScaler(),
                          RandomForestClassifier(**params))
    model.fit(X, y)

    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = C.MODELS_DIR / "random_forest_primary.joblib"
    joblib.dump(model, model_path, compress=3)
    size_kb = model_path.stat().st_size / 1024.0

    # Scoring latency once features exist. Two numbers are reported because
    # they answer different questions: a single call pays Python/sklearn
    # dispatch overhead that does not scale with model size, while a batch
    # amortises it. A health monitor scores one recording every few seconds,
    # so the single-call figure is the operationally relevant one; the batch
    # figure shows the forest itself is trivial.
    row = X.iloc[[0]]
    for _ in range(20):
        model.predict(row)
    t0 = time.perf_counter()
    n_rep = 2000
    for _ in range(n_rep):
        model.predict(row)
    pred_ms = (time.perf_counter() - t0) / n_rep * 1e3
    t0 = time.perf_counter()
    model.predict(X)
    batch_ms = (time.perf_counter() - t0) * 1e3
    batch_per_row_ms = batch_ms / max(len(X), 1)

    # Full per-window cost: raw samples -> features -> label.
    wm = pd.read_csv(C.result_path("window_metrics.csv"), nrows=200)
    files = wm["relpath"].unique()[:10]
    cache_rel, cache_x, cache_fs = None, None, C.FS_EXPECTED
    n_win = 0
    t_feat = 0.0
    t_pred = 0.0
    for rel in files:
        if rel != cache_rel:
            cache_fs, cache_x = load_audio(C.DATASET_DIR / rel)
            cache_rel = rel
        starts = wm.loc[wm["relpath"] == rel, "start_s"].head(5)
        for s in starts:
            a = int(round(float(s) * cache_fs))
            x = cache_x[a:a + int(round(C.SEGMENT_SECONDS * cache_fs))]
            if len(x) < int(round(C.SEGMENT_SECONDS * cache_fs)):
                continue
            tf0 = time.perf_counter()
            m = _wm(x, cache_fs)
            t_feat += time.perf_counter() - tf0
            if not m:
                continue
            tp0 = time.perf_counter()
            model.predict(pd.DataFrame([{k: m[k] for k in selected}]))
            t_pred += time.perf_counter() - tp0
            n_win += 1

    window_ms = (t_feat + t_pred) / max(n_win, 1) * 1e3
    return dict(
        model="random_forest (primary, leave-one-mission-out configuration)",
        model_path=str(model_path.relative_to(C.ROOT)),
        model_size_kb=round(float(size_kb), 1),
        n_features=int(len(selected)),
        predict_per_recording_ms=round(float(pred_ms), 3),
        predict_batch_all_ms=round(float(batch_ms), 3),
        predict_batch_per_row_ms=round(float(batch_per_row_ms), 4),
        n_windows_timed=int(n_win),
        feature_extraction_ms=round(float(t_feat / max(n_win, 1) * 1e3), 2),
        window_to_decision_ms=round(float(window_ms), 2),
        note="single-call predict pays Python/sklearn dispatch overhead that "
             "does not scale with the forest; a batch of all 251 recordings "
             "amortises it to tens of microseconds per row. Either way the "
             "forest is not the bottleneck: window_metrics dominates "
             "window_to_decision, which is what sets detection latency.",
    )


if __name__ == "__main__":
    run()