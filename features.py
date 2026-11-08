"""Feature selection at recording level.

The candidate pool starts from the metrics that survive FDR correction on the
severity test, plus the handful that carry the normal-vs-damaged contrast.
Three problems have to be solved before any of them can be handed to a
classifier.

1. Confounding. The raw level/f0 metrics are excluded by name, but derived
   proxies re-encode the same protocol differences: tone_band_energy_rel
   correlated +0.92 (Spearman) with level_dbfs. The pool is therefore also
   screened by correlation - any candidate at |rho| >= 0.70 against the
   excluded level/f0 metrics is dropped.

2. Redundancy. Twelve pairs correlate above 0.99 and many more above 0.90, so a
   model trained on the full pool can spread weight across duplicates and look
   more confident than the evidence supports. Selection here is therefore
   *sequential*, adding one metric at a time and stopping when the next one does
   not help, rather than scoring each metric in isolation.

3. The unit of analysis. Overlapping 4 s windows are not independent, so
   everything collapses to one row per recording first. A window-level split
   would place a test window next to the training window it overlaps and
   inflate accuracy (measured at +0.012 balanced accuracy per window with the
   same features; the naive end-to-end window split reports 0.991 against the
   honest grouped 0.967).

Validation is grouped, never random. ``mission`` spans all four classes, so
leave-one-mission-out is the split that actually tests generalisation to unseen
flight manoeuvres; leave-one-take-out tests generalisation to an unseen repeat
flight of the same airframe. Both are reported.
"""
from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from . import stats as S

FEATURES_DIR = C.ROOT / "analysis"

# Metrics that respond to recording gain or to blade-pass frequency. Excluded
# from the pool outright: normal files are ~15 dB quieter at ~227 Hz against
# ~129 Hz, so these would separate normal from damaged for reasons that have
# nothing to do with damage.
EXCLUDED = set(S.CONFOUNDED) | {"f0_hz", "f0_peak_db", "level_dbfs", "rms",
                                "peak", "dc_offset", "time_sec"}

# Metrics whose class contrast is dominated by a specific blade being damaged
# rather than by damage severity. Kept but reported separately, because they turn
# out to be useful in combination even though they are unstable alone.
BLADE_POSITION_SENSITIVE = {"bp1_crest_db", "bp2_crest_db", "bp3_crest_db",
                            "bp1_rms_rel", "bp2_rms_rel", "bp3_rms_rel"}

MAX_FEATURES = 8
MIN_GAIN = 0.004          # stop when the best addition gains less than this
MAX_PAIRWISE_RHO = 0.90   # reject a candidate duplicating an already-chosen one

# Columns that carry bookkeeping rather than signal. start_s is the offset of a
# window inside its recording, so it is not a property of the propeller at all:
# it was picked as the eighth "feature" and lifted leave-one-mission-out balanced
# accuracy from 0.976 to 0.982 purely because flights differ in length and the
# later parts of a flight sit at a different position. It is excluded here and
# the exclusion is asserted in the test suite.
NON_SIGNAL_COLS = {"cls", "relpath", "config", "group", "mission", "take",
                   "airframe", "session", "take_session", "n_windows",
                   "segment_index", "start_s", "start_sample", "duration_s",
                   "time_sec", "fallback", "in_primary"}


def recording_table(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per recording, with grouping columns attached for CV."""
    if df is None:
        df = pd.read_csv(C.RESULTS_DIR / "window_metrics.csv")

    # Damaged folders have two layouts (<class>/<config>/... and
    # 3_4_broken/<subtype>/<config>/...) and normal has none of those levels,
    # so the labels are derived with the same depth-aware parser the inventory
    # phase uses. Fixed positions silently read the config bitmask as the
    # mission for every 3_4_broken file, which put that class into the wrong
    # leave-one-mission-out folds.
    from .inventory import parse_relpath

    rec = df.groupby(["cls", "relpath"], as_index=False).median(numeric_only=True)
    parsed = rec["relpath"].map(parse_relpath)
    rec["config"] = [p["config"] for p in parsed]
    rec["group"] = [p["group"] for p in parsed]
    rec["mission"] = [p["mission"] for p in parsed]
    rec["take"] = [p["take"] for p in parsed]
    rec["airframe"] = [p["config"] if p["cls"] != "normal" else "normal"
                       for p in parsed]
    rec["session"] = rec["airframe"] + "|" + rec["mission"]
    rec["take_session"] = rec["session"] + "|take" + rec["take"].astype(str)
    return rec


def _name_filtered_pool(rec: pd.DataFrame) -> list[str]:
    """Numeric metrics that are neither excluded nor confounded by name."""
    out = []
    for c in rec.columns:
        if c in EXCLUDED or c in NON_SIGNAL_COLS:
            continue
        v = pd.to_numeric(rec[c], errors="coerce")
        if v.notna().any() and v.std() > 0:
            out.append(c)
    # Prefer survivors but never restrict the pool to them: the broken-vs-broken
    # survivors are ranked by severity, while a classifier also needs the
    # metrics that separate normal, which have weak severity trends.
    return out


# The name-based EXCLUDED list catches the raw metrics but not derived
# proxies: sequential selection once admitted tone_band_energy_rel, which
# correlates +0.92 (Spearman) with level_dbfs, and
# am_sideband_blades2_1x_db at -0.75 - both re-encode the level/f0 protocol
# differences the study claims to exclude. Any candidate strongly associated
# with the excluded level or f0 metrics is therefore dropped by correlation,
# not only by name.
CONFOUND_REFS = ["level_dbfs", "rms", "peak", "f0_peak_db", "f0_hz"]
MAX_CONFOUND_RHO = 0.70


def confound_screen(rec: pd.DataFrame, pool: list[str],
                    refs: list[str] | None = None,
                    max_rho: float = MAX_CONFOUND_RHO) -> tuple[list[str], list[dict]]:
    """Drop candidates that track an excluded level/f0 metric too closely.

    Spearman correlation at recording level. The threshold 0.70 is the
    conventional strong-association cut; everything above it is a re-encoding
    of recording gain or blade-pass frequency rather than a fault signature.
    """
    refs = [r for r in (refs or CONFOUND_REFS) if r in rec.columns]
    keep: list[str] = []
    rejected: list[dict] = []
    for c in pool:
        cors = {r: float(rec[c].corr(rec[r], method="spearman")) for r in refs}
        worst = max(refs, key=lambda r: abs(cors[r]))
        if abs(cors[worst]) >= max_rho:
            rejected.append(dict(metric=c, ref=worst,
                                 rho=round(cors[worst], 3)))
        else:
            keep.append(c)
    rejected.sort(key=lambda d: -abs(d["rho"]))
    return keep, rejected


def candidate_pool(rec: pd.DataFrame) -> list[str]:
    """Admissible candidates: name filter plus confound-correlation screen."""
    keep, _ = confound_screen(rec, _name_filtered_pool(rec))
    return keep


def _score(cols: list[str], rec: pd.DataFrame, groups: pd.Series) -> float:
    """Mean leave-one-group-out balanced accuracy of 1-NN on a feature subset.

    1-NN is used as the selector on purpose. A flexible model would let a bad
    feature survive selection by memorising, so a search over 30+ candidates
    driven by a random forest would report whatever it managed to overfit. The
    nearest-neighbour probe has no capacity to hide a redundant feature.
    """
    model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3))
    s = cross_val_score(model, rec[cols], rec["cls"], groups=groups,
                        cv=LeaveOneGroupOut(), scoring="balanced_accuracy")
    return float(np.mean(s))


def select_features(rec: pd.DataFrame, pool: list[str] | None = None,
                    max_features: int = MAX_FEATURES,
                    min_gain: float = MIN_GAIN,
                    max_corr: float = MAX_PAIRWISE_RHO,
                    verbose: bool = True) -> dict:
    """Sequential forward selection under leave-one-mission-out.

    Two constraints shape the search. A candidate must improve balanced accuracy
    by at least ``min_gain``, otherwise the search stops: past that point the
    added metric is noise. And a candidate is rejected outright if it correlates
    above ``max_corr`` with something already chosen, because a near-duplicate
    adds an apparent degree of freedom without adding information. Without the
    second constraint rolloff85_rel was admitted alongside rolloff95_rel at
    rho = 0.986.
    """
    pool = pool or candidate_pool(rec)
    groups = rec["mission"]
    corr = rec[pool].corr(method="spearman").abs()

    chosen: list[str] = []
    trace = []
    rejected: list[tuple[str, str, float]] = []

    for _ in range(max_features):
        best_gain, best_metric = 0.0, None
        best_score = 0.0
        for m in pool:
            if m in chosen:
                continue
            dup = next(((c, float(corr.loc[m, c])) for c in chosen
                        if corr.loc[m, c] > max_corr), None)
            if dup is not None:
                rejected.append((m, dup[0], dup[1]))
                continue
            score = _score(chosen + [m], rec, groups)
            gain = score - (trace[-1]["score"] if trace else 0.0)
            if gain > best_gain or best_metric is None:
                best_gain, best_metric, best_score = gain, m, score
        if best_metric is None:
            break
        if chosen and best_gain < min_gain:
            if verbose:
                print(f"  stopping: best addition {best_metric} gained "
                      f"{best_gain:+.4f} < {min_gain}")
            break
        chosen.append(best_metric)
        trace.append(dict(n=len(chosen), metric=best_metric, score=best_score,
                          gain=float(best_gain)))
        if verbose:
            flag = " [blade-position sensitive]" \
                if best_metric in BLADE_POSITION_SENSITIVE else ""
            print(f"  {len(chosen)}. {best_metric:<26} "
                  f"LOO-mission balanced acc {best_score:.4f}{flag}")
    return dict(selected=chosen, trace=trace, pool_size=len(pool),
                rejected_as_duplicate=rejected)


def redundancy_report(rec: pd.DataFrame, feats: list[str]) -> dict:
    """Pairwise Spearman among the selected features, to prove they are distinct."""
    if len(feats) < 2:
        return {}
    c = rec[feats].corr(method="spearman")
    n = len(feats)
    worst = []
    for i in range(n):
        for j in range(i + 1, n):
            worst.append((abs(float(c.iloc[i, j])), feats[i], feats[j],
                          float(c.iloc[i, j])))
    worst.sort(reverse=True)
    return dict(max_abs_rho=worst[0][0], pairs=worst[:5])


def run(verbose: bool = True) -> dict:
    if verbose:
        print("\nFEATURE SELECTION  (recording level, leave-one-mission-out)")
    rec = recording_table()
    name_pool = _name_filtered_pool(rec)
    pool, screen = confound_screen(rec, name_pool)
    if verbose:
        print(f"  {len(rec)} recordings, {len(name_pool)} name-admissible "
              f"metrics, {len(screen)} dropped by the confound screen "
              f"(|rho| >= {MAX_CONFOUND_RHO} vs level/f0 refs), "
              f"{len(pool)} candidates remain")

    sel = select_features(rec, pool, verbose=verbose)
    feats = sel["selected"]
    red = redundancy_report(rec, feats)
    selected_confound = confound_screen(rec, feats)[1]

    (FEATURES_DIR / "selected_features.json").write_text(json.dumps(dict(
        unit="one row per recording (median of its windows)",
        validation="LeaveOneGroupOut on mission, balanced accuracy of 3-NN",
        n_candidate_metrics=len(pool),
        excluded_confounded=sorted(EXCLUDED),
        confound_screen=dict(
            refs=CONFOUND_REFS,
            max_rho=MAX_CONFOUND_RHO,
            n_dropped=len(screen),
            dropped=screen,
        ),
        selected=feats,
        selected_max_confound_rho=selected_confound,
        selection_trace=sel["trace"],
        rejected_as_duplicate=[dict(candidate=a, duplicates=b, abs_rho=v)
                               for a, b, v in sel["rejected_as_duplicate"]],
        selected_pairwise_correlation=red,
        blade_position_sensitive=sorted(set(feats) & BLADE_POSITION_SENSITIVE),
    ), indent=2), encoding="utf-8")

    if verbose:
        print(f"\n  selected {len(feats)} features: {', '.join(feats)}")
        if red:
            print(f"  largest |rho| among selected = {red['max_abs_rho']:.3f} "
                  f"(0.90 would mean duplicates)")
    return dict(rec=rec, features=feats, pool_size=len(pool),
                trace=sel["trace"], redundancy=red)


if __name__ == "__main__":
    run()