"""Phase 8 - robustness experiments.

Three experiments, all answering a question a deployment reviewer would ask:

1. **Additive noise.** White Gaussian noise is added to the test windows at
   SNR 20/15/10/5/0 dB. Training always uses clean windows; only the test fold
   is corrupted, which is the honest model of "the field is noisier than the
   lab". Metric: 4-class macro-F1 and binary accuracy of the primary random
   forest, per SNR.

2. **Window length.** Features are recomputed on 1, 2, 4 and 8 s windows cut
   from the same steady-RPM regions. This quantifies how much audio the system
   actually needs: the 4 s default is a compromise between feature stability
   and detection latency. 1-2 s windows are prefixes of the stored 4 s
   primary windows; 8 s windows join consecutive primary windows where the
   steady region allows it.

3. **Trajectory transfer.** Train on the square missions and test on the
   figure-eight missions, and vice versa. A health monitor that only works on
   the manoeuvre it was trained on is not a health monitor.

Every experiment reuses the primary model configuration (random forest, leave-
one-mission-out folds where applicable) and the already-selected feature set,
so a change in the reported numbers can only come from the changed condition.

Outputs
-------
results/robustness_snr.csv        macro-F1 / accuracy per SNR level
results/robustness_window.csv     macro-F1 / accuracy per window length
results/robustness_trajectory.csv square -> f8 and f8 -> square transfer
analysis/robustness.json          full record including per-class detail
results/cache/snr_*.csv           cached noisy feature tables (re-runs)
results/cache/winlen_*.csv        cached window-length feature tables
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (balanced_accuracy_score, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from . import features as F
from .analysis import window_metrics
from .segment import load_audio

ANALYSIS_DIR = C.ROOT / "analysis"


# ====================================================================== helpers
def add_awgn(x: np.ndarray, snr_db: float, rng: np.random.RandomState) -> np.ndarray:
    """Additive white Gaussian noise at a target SNR relative to window power."""
    p_sig = float(np.mean(x ** 2)) + 1e-30
    p_noise = p_sig / (10.0 ** (snr_db / 10.0))
    return x + rng.normal(0.0, np.sqrt(p_noise), size=x.shape)


def primary_model() -> "object":
    """The primary configuration used in every robustness experiment."""
    return make_pipeline(
        StandardScaler(),
        RandomForestClassifier(**C.RF_PARAMS),
    )


def fit_score_fold(X: pd.DataFrame, y: np.ndarray, groups: pd.Series,
                   test_mask: np.ndarray | None = None) -> dict:
    """Leave-one-mission-out scores, optionally restricted to a test subset.

    When ``test_mask`` is given, the model is still trained on the complement
    of each test fold (never on the masked-in rows), but only the masked-in
    test rows contribute to the scored metrics. Used by the trajectory
    experiment, where the test side is a subset of missions.
    """
    classes = C.CLASSES
    pred = np.full(len(y), "", dtype=object)
    scored = np.zeros(len(y), bool)
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        if test_mask is not None:
            keep = test_mask[te]
            if not keep.any():
                continue
            te = te[keep]
        m = primary_model()
        m.fit(X.iloc[tr], y[tr])
        pred[te] = m.predict(X.iloc[te])
        scored[te] = True

    sel = scored & (pred != "")
    if not sel.any():
        return dict(accuracy=float("nan"), balanced_accuracy=float("nan"),
                    macro_f1=float("nan"), n_test=0)
    p, r, f, s = precision_recall_fscore_support(
        y[sel], pred[sel], labels=classes, zero_division=0)
    return dict(
        accuracy=float((pred[sel] == y[sel]).mean()),
        balanced_accuracy=float(balanced_accuracy_score(y[sel], pred[sel])),
        macro_f1=float(f1_score(y[sel], pred[sel], labels=classes,
                                average="macro", zero_division=0)),
        n_test=int(sel.sum()),
        n_errors=int((pred[sel] != y[sel]).sum()),
        per_class=[dict(cls=c, label=C.CLASS_LABELS[c], support=int(n),
                        precision=float(pi), recall=float(ri), f1=float(fi))
                   for c, pi, ri, fi, n in zip(classes, p, r, f, s)],
        confusion=pd.DataFrame(
            confusion_counts(y[sel], pred[sel]), index=classes,
            columns=classes).to_dict(),
    )


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> list[list[int]]:
    out = [[0] * len(C.CLASSES) for _ in C.CLASSES]
    for t, p in zip(y_true, y_pred):
        if t in C.CLASS_TO_INT and p in C.CLASS_TO_INT:
            out[C.CLASS_TO_INT[t]][C.CLASS_TO_INT[p]] += 1
    return out


def binary_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Normal vs abnormal view of any set of 4-class predictions."""
    yt = np.array(["normal" if c == "normal" else "abnormal" for c in y_true])
    yp = np.array(["normal" if c == "normal" else "abnormal" for c in y_pred])
    fp = int(((yt == "normal") & (yp == "abnormal")).sum())
    tn = int(((yt == "normal") & (yp == "normal")).sum())
    return dict(
        accuracy=float((yt == yp).mean()),
        balanced_accuracy=float(balanced_accuracy_score(yt, yp)),
        macro_f1=float(f1_score(yt, yp, labels=["normal", "abnormal"],
                                average="macro", zero_division=0)),
        false_alarm_rate=float(fp / max(fp + tn, 1)),
        false_positive=fp, true_negative=tn,
    )


def scoring(y: np.ndarray, pred: np.ndarray, tag: dict) -> dict:
    """Full metric bundle for one experimental condition."""
    out = scoring_base(y, pred)
    out.update(tag)
    out["binary"] = binary_scores(y, pred)
    return out


def scoring_base(y: np.ndarray, pred: np.ndarray) -> dict:
    p, r, f, s = precision_recall_fscore_support(
        y, pred, labels=C.CLASSES, zero_division=0)
    return dict(
        accuracy=float((pred == y).mean()),
        balanced_accuracy=float(balanced_accuracy_score(y, pred)),
        macro_f1=float(f1_score(y, pred, labels=C.CLASSES,
                                average="macro", zero_division=0)),
        n_test=int(len(y)),
        n_errors=int((pred != y).sum()),
        per_class=[dict(cls=c, label=C.CLASS_LABELS[c], support=int(n),
                        precision=float(pi), recall=float(ri), f1=float(fi))
                   for c, pi, ri, fi, n in zip(C.CLASSES, p, r, f, s)],
        confusion=confusion_counts(y, pred),
    )


def out_of_fold(model_fn, X: pd.DataFrame, y: np.ndarray,
                groups: pd.Series) -> np.ndarray:
    """Plain leave-one-mission-out predictions on whatever X is supplied."""
    pred = np.full(len(y), "", dtype=object)
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        m = model_fn()
        m.fit(X.iloc[tr], y[tr])
        pred[te] = m.predict(X.iloc[te])
    return pred.astype(str)


# ====================================================================== SNR sweep
def _cache_valid(path, selected: list[str]) -> bool:
    """A cache is usable only if it carries every requested feature column.

    The feature selection can change between runs (it did when the path
    parser bug was fixed), which silently turns an old cache into a table
    missing the new selection. Caches are therefore written with the full
    metric set and validated against ``selected`` before being reused.
    """
    if not path.exists():
        return False
    cols = pd.read_csv(path, nrows=0).columns
    return all(c in cols for c in selected)


def noisy_feature_table(primary: pd.DataFrame, selected: list[str],
                        snr_db: float, verbose: bool = True) -> pd.DataFrame:
    """Recompute the selected features on AWGN-corrupted primary windows.

    Cached per SNR under results/cache/, because a full sweep costs around
    ten minutes and is typically re-run several times while writing the report.
    """
    cache = C.CACHE_DIR / f"snr_{snr_db:g}dB.csv"
    if _cache_valid(cache, selected):
        return pd.read_csv(cache)

    if verbose:
        print(f"    extracting features at SNR {snr_db:g} dB "
              f"({len(primary)} windows) ...")
    rng = np.random.RandomState(C.RANDOM_STATE + int(round(snr_db * 10)))
    records = []
    cache_rel, cache_x, cache_fs = None, None, None
    t0 = time.perf_counter()
    for n, row in enumerate(primary.itertuples(), start=1):
        if row.relpath != cache_rel:
            cache_fs, cache_x = load_audio(C.DATASET_DIR / row.relpath)
            cache_rel = row.relpath
        s = int(row.start_sample)
        x = cache_x[s:s + int(round(C.SEGMENT_SECONDS * cache_fs))]
        if len(x) < int(round(C.SEGMENT_SECONDS * cache_fs)):
            continue
        m = window_metrics(add_awgn(x, snr_db, rng), cache_fs)
        if not m:
            continue
        m.update(relpath=row.relpath, cls=row.cls, mission=row.mission,
                 segment_index=int(row.segment_index))
        records.append(m)
        if verbose and n % 500 == 0:
            el = time.perf_counter() - t0
            print(f"      {n}/{len(primary)} windows, {el:.0f} s")

    df = pd.DataFrame.from_records(records)
    # Full metric set: a later re-selection must not invalidate the cache.
    df.to_csv(cache, index=False)
    return df


def sweep_snr(selected: list[str], verbose: bool = True) -> dict:
    """Train clean, test noisy. SNR levels from config.SNR_LEVELS_DB."""
    primary = pd.read_csv(C.result_path("segments_primary.csv"))
    clean = pd.read_csv(C.result_path("window_metrics.csv"))
    clean = clean[clean["relpath"].isin(primary["relpath"])]

    def rec_table(df: pd.DataFrame) -> pd.DataFrame:
        g = df.groupby(["cls", "relpath", "mission"], as_index=False).median(
            numeric_only=True)
        return g

    def evaluate(tag: str, test_df: pd.DataFrame, train_df: pd.DataFrame) -> dict:
        tr = rec_table(train_df)
        te = rec_table(test_df)
        # Align: train and test share recordings; test features are the noisy
        # version of the same files, so X_test rows are matched by relpath.
        common = te.merge(tr[["relpath"]], on="relpath", how="inner")
        y_tr = tr["cls"].to_numpy()
        y_te = common["cls"].to_numpy()
        groups_tr = tr["mission"]
        pred = np.full(len(common), "", dtype=object)
        for itr, ite in LeaveOneGroupOut().split(tr[selected], y_tr, groups_tr):
            m = primary_model()
            m.fit(tr.iloc[itr][selected], y_tr[itr])
            # test fold = recordings whose mission is held out
            held = groups_tr.iloc[ite].unique()
            sel_te = common["mission"].isin(held).to_numpy()
            if not sel_te.any():
                continue
            pred[sel_te] = m.predict(common.loc[sel_te, selected])
        ok = pred != ""
        out = scoring(y_te[ok], pred[ok].astype(str),
                      dict(condition=tag, snr_db=snr_db if tag != "clean" else np.inf))
        return out

    results = [evaluate("clean", clean, clean)]
    for snr_db in C.SNR_LEVELS_DB:
        if verbose:
            print(f"  SNR {snr_db:g} dB")
        noisy = noisy_feature_table(primary, selected, snr_db, verbose=verbose)
        results.append(evaluate(f"awgn_{snr_db:g}dB", noisy, clean))

    rows = [dict(condition=r["condition"], snr_db=r["snr_db"],
                 accuracy=r["accuracy"], balanced_accuracy=r["balanced_accuracy"],
                 macro_f1=r["macro_f1"],
                 binary_accuracy=r["binary"]["accuracy"],
                 binary_far=r["binary"]["false_alarm_rate"],
                 n_test=r["n_test"], n_errors=r["n_errors"])
            for r in results]
    table = pd.DataFrame(rows)
    table.to_csv(C.result_path("robustness_snr.csv"), index=False)
    if verbose:
        print(table.to_string(index=False,
                              float_format=lambda v: f"{v:.4f}"))
    return dict(table=table, detail=results)


# ============================================================ window-length sweep
def windows_of_length(primary: pd.DataFrame, length_s: float,
                      verbose: bool = True) -> pd.DataFrame:
    """Cut ``length_s`` windows from the steady regions of each recording.

    * length <= 4 s: the first ``length`` seconds of every primary 4 s window.
    * length 8 s: every primary window whose start plus 8 s still lies inside
      the file's primary window envelope (max start + 4 s). This guarantees
      the longer window stays inside the detected steady-RPM region rather
      than wandering back into the throttle sweep.
    """
    fs = C.FS_EXPECTED
    n = int(round(length_s * fs))
    n4 = int(round(C.SEGMENT_SECONDS * fs))

    if abs(length_s - C.SEGMENT_SECONDS) < 1e-9:
        out = primary.copy()
        out["length_s"] = length_s
        return out

    out = []
    if length_s < C.SEGMENT_SECONDS:
        for row in primary.itertuples():
            out.append(dict(relpath=row.relpath, cls=row.cls,
                            mission=row.mission, take=int(row.take),
                            start_sample=int(row.start_sample),
                            length_s=length_s))
    else:
        for rel, sub in primary.groupby("relpath"):
            sub = sub.sort_values("start_sample")
            starts = sub["start_sample"].to_numpy(np.int64)
            envelope_end = int(starts.max() + n4)
            cls = sub["cls"].iloc[0]
            mission = sub["mission"].iloc[0]
            take = int(sub["take"].iloc[0])
            for s in starts:
                if s + n <= envelope_end:
                    out.append(dict(relpath=rel, cls=cls, mission=mission,
                                    take=take, start_sample=int(s),
                                    length_s=length_s))
    return pd.DataFrame(out)


def features_at_length(wins: pd.DataFrame, selected: list[str],
                       verbose: bool = True) -> pd.DataFrame:
    """Window metrics for arbitrary-length cuts, cached per length."""
    length_s = float(wins["length_s"].iloc[0]) if len(wins) else 0.0
    cache = C.CACHE_DIR / f"winlen_{length_s:g}s.csv"
    if _cache_valid(cache, selected):
        return pd.read_csv(cache)

    # The default length is exactly the primary window: the stored metrics
    # already describe it, so re-extracting would be wasted work.
    if abs(length_s - C.SEGMENT_SECONDS) < 1e-9:
        stored = pd.read_csv(C.result_path("window_metrics.csv"))
        df = stored.merge(wins[["relpath", "cls", "mission", "segment_index",
                                "length_s"]],
                          on=["relpath", "cls", "mission", "segment_index"],
                          how="inner")
        df.to_csv(cache, index=False)
        return df

    records = []
    cache_rel, cache_x, cache_fs = None, None, None
    t0 = time.perf_counter()
    for n, row in enumerate(wins.itertuples(), start=1):
        if row.relpath != cache_rel:
            cache_fs, cache_x = load_audio(C.DATASET_DIR / row.relpath)
            cache_rel = row.relpath
        nn = int(round(row.length_s * cache_fs))
        x = cache_x[int(row.start_sample):int(row.start_sample) + nn]
        if len(x) < nn:
            continue
        m = window_metrics(x, cache_fs)
        if not m:
            continue
        m.update(relpath=row.relpath, cls=row.cls, mission=row.mission)
        records.append(m)
        if verbose and n % 500 == 0:
            print(f"      {n}/{len(wins)} windows, {time.perf_counter()-t0:.0f} s")

    df = pd.DataFrame.from_records(records)
    df["length_s"] = length_s
    df.to_csv(cache, index=False)
    return df


def sweep_window_length(selected: list[str], verbose: bool = True) -> dict:
    """Re-extract features at several window lengths and retrain."""
    primary = pd.read_csv(C.result_path("segments_primary.csv"))
    results = []
    for L in C.WINDOW_LENGTHS_SECONDS:
        if verbose:
            print(f"  window length {L:g} s")
        wins = windows_of_length(primary, L, verbose=verbose)
        df = features_at_length(wins, selected, verbose=verbose)
        rec = df.groupby(["cls", "relpath", "mission"], as_index=False).median(
            numeric_only=True)
        pred = out_of_fold(primary_model, rec[selected],
                           rec["cls"].to_numpy(), rec["mission"])
        r = scoring(rec["cls"].to_numpy(), pred,
                    dict(window_seconds=L, n_windows=int(len(df)),
                         n_recordings=int(len(rec))))
        results.append(r)
        if verbose:
            print(f"      macro-F1 {r['macro_f1']:.4f}  "
                  f"bal.acc {r['balanced_accuracy']:.4f}  "
                  f"({r['n_recordings']} recordings, {r['n_test']} scored)")

    table = pd.DataFrame([dict(window_seconds=r["window_seconds"],
                               accuracy=r["accuracy"],
                               balanced_accuracy=r["balanced_accuracy"],
                               macro_f1=r["macro_f1"],
                               binary_accuracy=r["binary"]["accuracy"],
                               binary_far=r["binary"]["false_alarm_rate"],
                               n_recordings=r["n_recordings"],
                               n_windows=r["n_windows"],
                               n_errors=r["n_errors"])
                          for r in results])
    table.to_csv(C.result_path("robustness_window.csv"), index=False)
    return dict(table=table, detail=results)


# ========================================================== trajectory transfer
def trajectory_transfer(selected: list[str], verbose: bool = True) -> dict:
    """Train on square, test on figure-eight, and the reverse."""
    rec = F.recording_table()
    y = rec["cls"].to_numpy()
    X = rec[selected]

    pairs = [
        ("train square -> test f8", ["cw_square", "ccw_square"],
         ["cw_f8", "ccw_f8"]),
        ("train f8 -> test square", ["cw_f8", "ccw_f8"],
         ["cw_square", "ccw_square"]),
        ("train cw -> test ccw", ["cw_square", "cw_f8"],
         ["ccw_square", "ccw_f8"]),
        ("train ccw -> test cw", ["ccw_square", "ccw_f8"],
         ["cw_square", "cw_f8"]),
    ]

    results = []
    for tag, train_m, test_m in pairs:
        tr = rec["mission"].isin(train_m).to_numpy()
        te = rec["mission"].isin(test_m).to_numpy()
        m = primary_model()
        m.fit(X[tr], y[tr])
        pred = m.predict(X[te])
        r = scoring(y[te], pred, dict(condition=tag,
                                      n_train=int(tr.sum()),
                                      train_missions=train_m,
                                      test_missions=test_m))
        results.append(r)
        if verbose:
            print(f"  {tag:<28} macro-F1 {r['macro_f1']:.4f}  "
                  f"bal.acc {r['balanced_accuracy']:.4f}  "
                  f"({r['n_test']} test recordings)")

    table = pd.DataFrame([dict(condition=r["condition"],
                               n_train=r["n_train"], n_test=r["n_test"],
                               accuracy=r["accuracy"],
                               balanced_accuracy=r["balanced_accuracy"],
                               macro_f1=r["macro_f1"],
                               binary_accuracy=r["binary"]["accuracy"],
                               binary_far=r["binary"]["false_alarm_rate"],
                               n_errors=r["n_errors"])
                          for r in results])
    table.to_csv(C.result_path("robustness_trajectory.csv"), index=False)
    return dict(table=table, detail=results)


# ====================================================================== driver
def run(selected: list[str] | None = None, verbose: bool = True) -> dict:
    if selected is None:
        selected = json.loads(
            (C.ROOT / "analysis" / "selected_features.json").read_text(
                encoding="utf-8"))["selected"]

    if verbose:
        print("\nROBUSTNESS")
        print(f"  features: {', '.join(selected)}")

    if verbose:
        print("\n  [1/3] additive noise (train clean, test noisy)")
    snr = sweep_snr(selected, verbose=verbose)

    if verbose:
        print("\n  [2/3] window-length sweep")
    win = sweep_window_length(selected, verbose=verbose)

    if verbose:
        print("\n  [3/3] trajectory transfer")
    traj = trajectory_transfer(selected, verbose=verbose)

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    (ANALYSIS_DIR / "robustness.json").write_text(json.dumps(dict(
        features=selected,
        snr=dict(levels_db=C.SNR_LEVELS_DB,
                 note="AWGN on test windows only; training always clean",
                 table=json.loads(snr["table"].to_json(orient="records")),
                 detail=snr["detail"]),
        window_length=dict(
            note="features recomputed on the same steady-RPM regions, "
                 "primary model retrained per length",
            table=json.loads(win["table"].to_json(orient="records")),
            detail=win["detail"]),
        trajectory=dict(
            note="single train/test transfer, no cross-validation: the test "
                 "missions never appear in training",
            table=json.loads(traj["table"].to_json(orient="records")),
            detail=traj["detail"]),
    ), indent=2, default=str), encoding="utf-8")

    if verbose:
        print(f"\n  -> results/robustness_{{snr,window,trajectory}}.csv, "
              "analysis/robustness.json")
    return dict(snr=snr, window=win, trajectory=traj, features=selected)


if __name__ == "__main__":
    run()
