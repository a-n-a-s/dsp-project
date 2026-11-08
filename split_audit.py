"""Phase 5 - cross-validation split audit.

The central methodological risk in this dataset is leakage: 4 s windows
overlap by half their length, so a naive random split puts near-duplicate
windows on both sides of the train/test boundary. Measured on this dataset,
the naive window split scores 0.991 per window and 0.995 per recording
against 0.979/0.986 for the honest grouped split on the same features - an
inflation of +0.012/+0.009 with this model family, and up to +0.06 compared
end to end with the primary confound-screened protocol. The same risk
reappears whenever a split ignores the hierarchical structure (mission,
take).

This phase does not train a model. It *proves* the splits used later are
clean, by construction, and writes the evidence to disk:

* per-split window and recording counts
* an explicit check that no recording (relpath) appears on both sides of any
  fold, and that no overlapping window pair straddles a fold boundary
* two shuffled-random-split controls, run once each: one splitting at the
  recording level (mild) and one splitting overlapping windows directly
  (leaky, quantifies the inflation the study avoids). Neither number is
  used for any claim.

Outputs
-------
results/split_audit.csv     one row per split configuration
analysis/split_audit.json   full record including the leak checks
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import (LeaveOneGroupOut, StratifiedKFold,
                                     train_test_split)

from . import config as C

ANALYSIS_DIR = C.ROOT / "analysis"


def _overlaps(df: pd.DataFrame) -> bool:
    """True if any two windows from the same recording overlap in samples."""
    for _, sub in df.groupby("relpath"):
        starts = np.sort(sub["start_sample"].to_numpy(np.int64))
        if starts.size < 2:
            continue
        # consecutive starts closer than one window length => overlap
        if np.any(np.diff(starts) < int(round(C.SEGMENT_SECONDS * C.FS_EXPECTED))):
            return True
    return False


def audit_group_split(df: pd.DataFrame, groups: pd.Series,
                      name: str) -> dict:
    """Leave-one-group-out: assert no recording crosses the boundary."""
    X = df[["start_sample"]]
    y = df["cls"].to_numpy()
    n_overlap_folds = 0
    n_rec_leak_folds = 0
    folds = []
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        tr_rel = set(df.iloc[tr]["relpath"])
        te_rel = set(df.iloc[te]["relpath"])
        leak = bool(tr_rel & te_rel)
        # Overlapping windows from the same recording cannot straddle the
        # boundary if the recording itself does not.
        n_rec_leak_folds += int(leak)
        # Additionally verify: within each fold, train windows from one
        # recording never overlap a test window from another (impossible by
        # relpath disjointness) -- the check above is therefore sufficient.
        folds.append(dict(
            held_out=str(groups.iloc[te].iloc[0]) if len(te) else "",
            n_train=int(len(tr)), n_test=int(len(te)),
            train_recordings=len(tr_rel), test_recordings=len(te_rel),
            recording_leak=leak,
            train_has_overlap=_overlaps(df.iloc[tr]),
        ))
    return dict(
        split=name,
        n_folds=len(folds),
        n_recordleak_folds=n_rec_leak_folds,
        clean=n_rec_leak_folds == 0,
        folds=folds,
    )


def audit_random_control(df: pd.DataFrame, n_repeat: int = 5) -> dict:
    """Stratified random split of *recordings*: the mild control.

    Each fold still sees the same missions on both sides, so this measures
    how much the grouped split costs, not how much leakage inflates.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from . import features as F

    selected = json.loads(
        (C.ROOT / "analysis" / "selected_features.json").read_text(
            encoding="utf-8"))["selected"]
    rec = F.recording_table()
    X, y = rec[selected], rec["cls"].to_numpy()

    scores = []
    for r in range(n_repeat):
        tr, te = train_test_split(
            np.arange(len(y)), test_size=0.25, random_state=C.RANDOM_STATE + r,
            stratify=y)
        m = make_pipeline(StandardScaler(),
                          RandomForestClassifier(**C.RF_PARAMS))
        m.fit(X.iloc[tr], y[tr])
        scores.append(balanced_accuracy_score(y[te], m.predict(X.iloc[te])))

    return dict(
        split="stratified random 75/25 on recordings",
        n_repeat=n_repeat,
        mean_balanced_accuracy=float(np.mean(scores)),
        std_balanced_accuracy=float(np.std(scores)),
        scores=[float(s) for s in scores],
        note="Recording-level random split shares missions across the "
             "boundary but never shares a recording. It is the mild control: "
             "the damaging split is window-level, where overlapping windows "
             "straddle the boundary.",
    )


def audit_window_random_control(df: pd.DataFrame, n_repeat: int = 5) -> dict:
    """Quantify what a naive window-level split fabricates.

    A careless analysis splits the overlapping primary windows at random and
    reports accuracy - per window, or aggregated per recording exactly like
    this study does. Both scorings are computed here on the full metric set
    (no confound filtering, no feature selection: the naive study uses
    everything), alongside the matched honest reference - the same windows,
    same features, same scorings, but leave-one-mission-out - so the reported
    gap isolates the split itself.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    wm = pd.read_csv(C.result_path("window_metrics.csv"))
    wm = wm[wm["relpath"].isin(df["relpath"])]
    num = [c for c in wm.select_dtypes("number").columns
           if c not in ("start_sample", "segment_index")]
    X, y = wm[num].to_numpy(), wm["cls"].to_numpy()
    rel, groups = wm["relpath"].to_numpy(), wm["mission"].to_numpy()

    def _fit_score(train, test):
        m = make_pipeline(StandardScaler(),
                          RandomForestClassifier(**C.RF_PARAMS))
        m.fit(X[train], y[train])
        p = m.predict(X[test])
        per_window = balanced_accuracy_score(y[test], p)
        d = pd.DataFrame({"relpath": rel[test], "pred": p, "true": y[test]})
        g = d.groupby("relpath").agg(
            pred=("pred", lambda s: s.value_counts().index[0]),
            true=("true", "first"))
        per_recording = balanced_accuracy_score(g["true"], g["pred"])
        return per_window, per_recording

    naive_w, naive_r = [], []
    for r in range(n_repeat):
        tr, te = train_test_split(
            np.arange(len(y)), test_size=0.25, random_state=C.RANDOM_STATE + r,
            stratify=y)
        pw, pr = _fit_score(tr, te)
        naive_w.append(pw)
        naive_r.append(pr)

    honest_w, honest_r = [], []
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        pw, pr = _fit_score(tr, te)
        honest_w.append(pw)
        honest_r.append(pr)

    return dict(
        n_windows=int(len(y)),
        n_features=int(X.shape[1]),
        n_repeat=n_repeat,
        feature_set="all numeric window metrics (naive analysis uses "
                    "everything; no confound filtering)",
        model=f"random forest {C.RF_PARAMS}",
        naive_random_split=dict(
            split="stratified random 75/25 on overlapping windows",
            per_window_mean=float(np.mean(naive_w)),
            per_window_std=float(np.std(naive_w)),
            per_recording_mean=float(np.mean(naive_r)),
            per_recording_std=float(np.std(naive_r)),
        ),
        honest_grouped_reference=dict(
            split="leave-one-mission-out on the same windows",
            per_window_mean=float(np.mean(honest_w)),
            per_recording_mean=float(np.mean(honest_r)),
        ),
        inflation=dict(
            per_window=float(np.mean(naive_w) - np.mean(honest_w)),
            per_recording=float(np.mean(naive_r) - np.mean(honest_r)),
        ),
        note="The naive split places every test window next to the training "
             "window it overlaps by 50%, from the same recording. The honest "
             "reference changes nothing but the split. inflation = naive - "
             "honest, same features, same scorings. No claim uses the naive "
             "numbers; they exist to quantify what the protocol avoids.",
    )


def run(verbose: bool = True) -> dict:
    primary = pd.read_csv(C.result_path("segments_primary.csv"))
    if verbose:
        print("\nSPLIT AUDIT")
        print(f"  {len(primary)} primary windows from "
              f"{primary['relpath'].nunique()} recordings")

    mission_audit = audit_group_split(primary, primary["mission"],
                                      "leave-one-mission-out")
    take_audit = audit_group_split(
        primary, primary["mission"] + "|" + primary["take"].astype(str),
        "leave-one-take-out")

    control = audit_random_control(primary)
    window_control = audit_window_random_control(primary)

    rows = [dict(split=a["split"], n_folds=a["n_folds"],
                 n_recordleak_folds=a["n_recordleak_folds"],
                 clean=a["clean"])
            for a in (mission_audit, take_audit)]
    rows.append(dict(split=control["split"], n_folds=control["n_repeat"],
                     n_recordleak_folds=-1,
                     clean=False))
    rows.append(dict(split="naive random split on windows (LEAKY control)",
                     n_folds=window_control["n_repeat"],
                     n_recordleak_folds=-2,
                     clean=False))
    rows.append(dict(split="leave-one-mission-out on windows (honest ref.)",
                     n_folds=4,
                     n_recordleak_folds=-3,
                     clean=True))
    table = pd.DataFrame(rows)
    table.to_csv(C.result_path("split_audit.csv"), index=False)

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    (ANALYSIS_DIR / "split_audit.json").write_text(json.dumps(dict(
        n_windows=int(len(primary)),
        n_recordings=int(primary["relpath"].nunique()),
        mission=mission_audit,
        take=take_audit,
        random_control=control,
        window_random_control=window_control,
    ), indent=2, default=str), encoding="utf-8")

    if verbose:
        for a in (mission_audit, take_audit):
            status = "CLEAN" if a["clean"] else f"{a['n_recordleak_folds']} LEAKY"
            print(f"  {a['split']:<32} {a['n_folds']} folds, "
                  f"recording-leak {status}")
        print(f"  random split on recordings (mild control):  "
              f"bal.acc {control['mean_balanced_accuracy']:.4f} "
              f"+/- {control['std_balanced_accuracy']:.4f}")
        nr = window_control["naive_random_split"]
        hr = window_control["honest_grouped_reference"]
        inf = window_control["inflation"]
        print(f"  naive window split (LEAKY):  per-window "
              f"{nr['per_window_mean']:.4f} +/- {nr['per_window_std']:.4f}, "
              f"per-recording {nr['per_recording_mean']:.4f} "
              f"+/- {nr['per_recording_std']:.4f}")
        print(f"  honest window LOGO (same features): per-window "
              f"{hr['per_window_mean']:.4f}, per-recording "
              f"{hr['per_recording_mean']:.4f}")
        print(f"  inflation from the leak:      per-window "
              f"{inf['per_window']:+.4f}, per-recording "
              f"{inf['per_recording']:+.4f}")
    return dict(mission=mission_audit, take=take_audit, control=control,
                window_control=window_control)


if __name__ == "__main__":
    run()
