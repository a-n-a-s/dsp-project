"""Statistical testing of class separation, with recording-level uncertainty.

Everything here treats the *recording* as the unit of analysis, not the window.
A 4 s window with 50% overlap means consecutive windows share half their
samples, so 3549 windows carry far less independent information than 3549
observations would suggest. Only 251 recordings are independent, and all
confidence intervals are built by resampling recordings with replacement.

Outputs
-------
analysis/auc_table.json        per-metric AUC with bootstrap CI, every comparison
analysis/severity_tests.json   Kruskal-Wallis, pairwise, Spearman, FDR corrected
analysis/feature_redundancy.json  correlation matrix among surviving metrics
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import config as C

ANALYSIS_DIR = C.ROOT / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

N_BOOT = 2000
SEED = C.RANDOM_STATE

BROKEN = [c for c in C.CLASSES if c != "normal"]
SEVERITY = {"1_broken": 1, "2_broken": 2, "3_4_broken": 3}

# Metrics whose variation is dominated by recording gain or by the blade-pass
# frequency. They are reported for completeness but never treated as evidence.
CONFOUNDED = {"rms", "peak", "level_dbfs", "peak_dbfs", "dc_offset",
              "psd_peak_db", "f0_hz", "f0_peak_db", "env_dynamic_range_db",
              "env_std_db", "time_sec"}

META_COLS = {"relpath", "cls", "config", "group", "mission", "take",
             "segment_index", "start_s", "fallback", "in_primary",
             "start_sample", "duration_s"}


# ====================================================================== helpers
def auc_mw(x: np.ndarray, y: np.ndarray, orient: bool = True) -> float:
    """Rank AUC via the Mann-Whitney U statistic.

    With ``orient`` the result is folded so that 0.5 is chance and 1.0 is a
    perfect split regardless of which group sits higher. Use orient=False when
    the direction of the effect matters, or when converting to Cliff's delta.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size < 2 or y.size < 2:
        return float("nan")
    u = stats.mannwhitneyu(x, y, alternative="two-sided").statistic
    a = float(u / (x.size * y.size))
    return max(a, 1.0 - a) if orient else a


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Non-parametric effect size. |d| near 0.15 small, 0.5 medium, 0.8 large."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size < 2 or y.size < 2:
        return float("nan")
    u = stats.mannwhitneyu(x, y, alternative="two-sided").statistic
    return float(2.0 * u / (x.size * y.size) - 1.0)


def boot_auc_ci(df: pd.DataFrame, metric: str, cls_a: str, cls_b: str,
                n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Bootstrap AUC by resampling *recordings* within each class.

    Resampling windows instead would treat overlapping views of the same 4 s of
    audio as independent samples and produce intervals roughly 3-4x too narrow.
    """
    sub = df[df["cls"].isin([cls_a, cls_b])]
    files_a = sorted(sub.loc[sub["cls"] == cls_a, "relpath"].unique())
    files_b = sorted(sub.loc[sub["cls"] == cls_b, "relpath"].unique())
    if not files_a or not files_b:
        return dict(auc=float("nan"), ci_low=float("nan"), ci_high=float("nan"),
                    n_bootstrap=0, n_files_a=len(files_a), n_files_b=len(files_b),
                    delta=float("nan"))

    by_file = {rel: pd.to_numeric(g[metric], errors="coerce").dropna().values
               for rel, g in sub.groupby("relpath")}

    point = auc_mw(np.concatenate([by_file[f] for f in files_a]),
                   np.concatenate([by_file[f] for f in files_b]))

    rng = np.random.RandomState(seed)
    stats_auc = []
    for _ in range(n_boot):
        ra = rng.choice(len(files_a), len(files_a), replace=True)
        rb = rng.choice(len(files_b), len(files_b), replace=True)
        xa = np.concatenate([by_file[files_a[i]] for i in ra])
        xb = np.concatenate([by_file[files_b[i]] for i in rb])
        v = auc_mw(xa, xb, orient=True)
        if np.isfinite(v):
            stats_auc.append(v)
    if len(stats_auc) < n_boot * 0.5:
        return dict(auc=point, ci_low=float("nan"), ci_high=float("nan"),
                    n_bootstrap=len(stats_auc), n_files_a=len(files_a),
                    n_files_b=len(files_b), delta=float("nan"))

    lo, hi = np.percentile(stats_auc, [2.5, 97.5])
    return dict(auc=float(point), ci_low=float(lo), ci_high=float(hi),
                n_bootstrap=len(stats_auc), n_files_a=len(files_a),
                n_files_b=len(files_b),
                delta=float(cliffs_delta(np.concatenate([by_file[f] for f in files_a]),
                                         np.concatenate([by_file[f] for f in files_b]))))


def bh_fdr(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(pvals, float)
    ok = np.isfinite(p)
    out = np.full(p.shape, np.nan)
    if ok.sum() == 0:
        return out.tolist()
    out[ok] = stats.false_discovery_control(p[ok], method="bh")
    return out.tolist()


def recording_medians(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Per-recording median of one metric, with class labels.

    Aggregating within a recording removes the pseudo-replication before any
    test that assumes independent observations.
    """
    v = pd.to_numeric(df[metric], errors="coerce")
    t = pd.DataFrame({"relpath": df["relpath"], "cls": df["cls"], "v": v}) \
        .dropna()
    return t.groupby(["cls", "relpath"], as_index=False)["v"].median()


def dunn_test(groups: dict[str, np.ndarray]) -> list[dict]:
    """Dunn's test with ties correction, for post-hoc after Kruskal-Wallis.

    Compares every pair of groups using rank sums pooled across all groups,
    which is the appropriate post-hoc for a rank-based omnibus test.
    """
    names = list(groups)
    allv = np.concatenate([groups[n] for n in names])
    n_all = allv.size
    if n_all < 3:
        return []
    r = stats.rankdata(allv)
    counts = np.array([groups[n].size for n in names], float)
    n_eff = 1.0 - (np.sum(counts ** 3 - counts) / (n_all ** 3 - n_all)) \
        if n_all > 1 else 1.0
    if n_eff <= 0:
        return []

    # Cumulative rank sum per group, following the group order.
    rs, start = {}, 0
    for n, c in zip(names, counts):
        rs[n] = r[start:start + int(c)].sum()
        start += int(c)

    out = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            diff = abs(rs[a] / counts[i] - rs[b] / counts[j])
            se = np.sqrt((n_all * (n_all + 1) / 12.0) / n_eff
                         * (1.0 / counts[i] + 1.0 / counts[j]))
            z = diff / se if se > 0 else np.nan
            out.append(dict(group_a=a, group_b=b, z=float(z),
                            p=float(2 * stats.norm.sf(abs(z))) if np.isfinite(z)
                            else float("nan"),
                            delta=float(cliffs_delta(groups[a], groups[b]))))
    return out


def epsilon_squared_h(h: float, k: int, n: int) -> float:
    """Kruskal-Wallis effect size."""
    if n < 1 or k < 2:
        return float("nan")
    return float(max(0.0, (h - k + 1) / (n - k)))


# ====================================================================== drivers
def metric_columns(df: pd.DataFrame) -> list[str]:
    out = []
    for c in df.columns:
        if c in META_COLS or c in CONFOUNDED:
            continue
        v = pd.to_numeric(df[c], errors="coerce")
        if v.notna().any():
            out.append(c)
    return out


def run_auc_table(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Per-metric AUC with bootstrap CI for every comparison."""
    if verbose:
        print("  bootstrap AUC at recording level "
              f"({N_BOOT} resamples per comparison) ...")

    metrics = metric_columns(df)
    comparisons = [
        ("1_broken", "2_broken"),
        ("2_broken", "3_4_broken"),
        ("1_broken", "3_4_broken"),
        ("normal", "ALL_BROKEN"),
    ]

    rows = []
    for metric in metrics:
        for a, b in comparisons:
            if b == "ALL_BROKEN":
                # Pooling the three broken classes as one group is correct here:
                # the question is normal vs everything that is not normal.
                r = _boot_auc_pooled(df, metric, a, BROKEN)
            else:
                r = boot_auc_ci(df, metric, a, b)
            rows.append(dict(metric=metric, comparison=f"{a}_vs_{b}", **r))
        if verbose:
            print(f"    {len(rows) // len(comparisons)}/{len(metrics)} metrics")

    out = pd.DataFrame(rows)
    out.to_csv(C.result_path("auc_table.csv"), index=False)
    return out


def _boot_auc_pooled(df: pd.DataFrame, metric: str, cls_a: str,
                     classes_b: list[str], n_boot: int = N_BOOT,
                     seed: int = SEED) -> dict:
    """AUC for one class against a pooled set of classes, bootstrapped by file."""
    sub = df[df["cls"].isin([cls_a] + classes_b)]
    files_a = sorted(sub.loc[sub["cls"] == cls_a, "relpath"].unique())
    files_b = sorted(sub.loc[sub["cls"].isin(classes_b), "relpath"].unique())
    by_file = {rel: pd.to_numeric(g[metric], errors="coerce").dropna().values
               for rel, g in sub.groupby("relpath")}
    xa = np.concatenate([by_file[f] for f in files_a])
    xb = np.concatenate([by_file[f] for f in files_b])
    point = auc_mw(xa, xb)

    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        ra = rng.choice(len(files_a), len(files_a), replace=True)
        rb = rng.choice(len(files_b), len(files_b), replace=True)
        v = auc_mw(np.concatenate([by_file[files_a[i]] for i in ra]),
                   np.concatenate([by_file[files_b[i]] for i in rb]))
        if np.isfinite(v):
            vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5]) if len(vals) > 10 else (np.nan, np.nan)
    return dict(auc=float(point), ci_low=float(lo), ci_high=float(hi),
                n_bootstrap=len(vals), n_files_a=len(files_a),
                n_files_b=len(files_b), delta=float(cliffs_delta(xa, xb)))


def run_severity_tests(df: pd.DataFrame, verbose: bool = True) -> dict:
    """Kruskal-Wallis, Dunn post-hoc and Spearman across the broken classes."""
    if verbose:
        print("  Kruskal-Wallis + Dunn + Spearman on per-recording medians ...")

    metrics = metric_columns(df)
    kw_rows, dunn_rows, sp_rows = [], [], []

    for metric in metrics:
        rec = recording_medians(df, metric)
        groups = {c: rec.loc[rec["cls"] == c, "v"].values for c in BROKEN}
        if min(len(v) for v in groups.values()) < 3:
            continue

        h, p_kw = stats.kruskal(*[groups[c] for c in BROKEN])
        n_tot = sum(len(v) for v in groups.values())
        kw_rows.append(dict(metric=metric, H=float(h), p=float(p_kw),
                            n_recordings=int(n_tot),
                            k=len(BROKEN),
                            epsilon_sq=epsilon_squared_h(float(h), len(BROKEN),
                                                          n_tot)))

        for r in dunn_test(groups):
            dunn_rows.append(dict(metric=metric, **r))

        # Severity 1, 2, 3 against the same recording-level values.
        allv, sevs = [], []
        for c, s in SEVERITY.items():
            allv.append(groups[c])
            sevs.append(np.full(len(groups[c]), s))
        x = np.concatenate(allv)
        y = np.concatenate(sevs)
        rho, p_rho = stats.spearmanr(y, x)
        sp_rows.append(dict(metric=metric, rho=float(rho), p=float(p_rho),
                            n_recordings=int(x.size)))

    kw = pd.DataFrame(kw_rows)
    dn = pd.DataFrame(dunn_rows)
    sp = pd.DataFrame(sp_rows)

    # Multiple-comparison correction across the whole metric set. The seven
    # visually promising metrics were chosen after inspecting medians, so
    # testing only those would be circular; correcting over everything avoids
    # having to rely on an exploratory/exploratory label.
    if len(kw):
        kw["p_fdr"] = bh_fdr(kw["p"].tolist())
    if len(dn):
        dn["p_fdr"] = bh_fdr(dn["p"].tolist())
    if len(sp):
        sp["p_fdr"] = bh_fdr(sp["p"].tolist())

    kw.to_csv(C.result_path("severity_kruskal.csv"), index=False)
    dn.to_csv(C.result_path("severity_dunn.csv"), index=False)
    sp.to_csv(C.result_path("severity_spearman.csv"), index=False)

    return dict(kruskal=kw, dunn=dn, spearman=sp)


def run_redundancy(df: pd.DataFrame, survivors: list[str]) -> pd.DataFrame:
    """Spearman correlation among metrics that survived the severity tests."""
    data = {}
    for m in survivors:
        r = recording_medians(df, m).set_index(["cls", "relpath"])["v"]
        data[m] = r
    mat = pd.DataFrame(data)
    mat = mat.dropna()
    corr = mat.corr(method="spearman")
    corr.to_csv(C.result_path("feature_redundancy.csv"))
    return corr


# ====================================================================== summary
def print_auc_summary(auc: pd.DataFrame, n_show: int = 18) -> None:
    print("\n  AUC BY METRIC, 95% CI from recording-level bootstrap")
    print("  " + "=" * 108)
    cols = ["1_broken_vs_2_broken", "2_broken_vs_3_4_broken",
            "1_broken_vs_3_4_broken", "normal_vs_ALL_BROKEN"]
    header = (f"  {'metric':<28}{'1B vs 2B':>18}{'2B vs 3/4B':>18}"
              f"{'1B vs 3/4B':>18}{'normal vs all':>18}")
    print(header)
    print("  " + "-" * 106)

    piv = auc.pivot_table(index="metric", columns="comparison", values="auc")
    lows = auc.pivot_table(index="metric", columns="comparison", values="ci_low")
    highs = auc.pivot_table(index="metric", columns="comparison", values="ci_high")
    # Rank by the hardest pair among the broken classes.
    piv["_worst"] = piv[[c for c in cols[:3]]].min(axis=1)
    for metric in piv.sort_values("_worst", ascending=False).head(n_show).index:
        cells = ""
        for c in cols:
            v = piv.loc[metric, c]
            lo = lows.loc[metric, c]
            hi = highs.loc[metric, c]
            cells += f"{v:>7.3f} [{lo:.2f},{hi:.2f}]" if np.isfinite(v) \
                else f"{'-':>18}"
        print(f"  {metric:<28}{cells}")

    print("\n  Brackets are 95% CI. A CI containing 0.5 means the metric does not")
    print("  separate that pair at recording level.")


def print_severity_summary(kw: pd.DataFrame, dn: pd.DataFrame,
                           sp: pd.DataFrame, n_show: int = 15) -> None:
    print("\n  SEVERITY TREND ACROSS 1B < 2B < 3/4B  (per-recording medians)")
    print("  " + "=" * 108)
    sig = sp[sp["p_fdr"] < 0.05].copy()
    if sig.empty:
        print("  no metric survives FDR correction on the Spearman test")
        return
    sig = sig.sort_values("rho", key=lambda s: s.abs(), ascending=False)
    m = sig.head(n_show)
    kw_i = kw.set_index("metric") if len(kw) else None

    print(f"  {'metric':<28}{'rho':>8}{'p_fdr':>11}{'KW H':>9}{'KW p_fdr':>11}"
          f"{'eps2':>8}   trend")
    print("  " + "-" * 104)
    for _, row in m.iterrows():
        metric = row["metric"]
        if kw_i is not None and metric in kw_i.index:
            k = kw_i.loc[metric]
            h, pk = f"{k['H']:.1f}", f"{k['p_fdr']:.2e}"
            eps = f"{k['epsilon_sq']:.3f}"
        else:
            h = pk = eps = "-"
        arrow = "increases" if row["rho"] > 0 else "decreases"
        print(f"  {metric:<28}{row['rho']:>8.3f}{row['p_fdr']:>11.2e}"
              f"{h:>9}{pk:>11}{eps:>8}   {arrow} with severity")

    print("\n  Pairwise Dunn tests (FDR corrected), strongest metrics:")
    if len(dn):
        dn_sig = dn[dn["p_fdr"] < 0.05]
        for metric in m["metric"].head(6):
            rows = dn_sig[dn_sig["metric"] == metric]
            if rows.empty:
                continue
            parts = [f"{r['group_a'][:6]} vs {r['group_b'][:6]}: "
                     f"p={r['p_fdr']:.1e}, d={r['delta']:+.2f}"
                     for _, r in rows.iterrows()]
            print(f"    {metric:<26} " + " | ".join(parts))


def _json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def run(verbose: bool = True) -> dict:
    if verbose:
        print(f"\nSTATISTICS  -> {ANALYSIS_DIR}")
    df = pd.read_csv(C.RESULTS_DIR / "window_metrics.csv")
    if verbose:
        print(f"  {len(df)} windows from {df['relpath'].nunique()} recordings")

    auc = run_auc_table(df, verbose=verbose)
    tests = run_severity_tests(df, verbose=verbose)

    sp = tests["spearman"]
    survivors = sp.loc[sp["p_fdr"] < 0.05, "metric"].tolist()
    corr = run_redundancy(df, survivors) if len(survivors) > 2 else None

    (ANALYSIS_DIR / "auc_table.json").write_text(json.dumps(dict(
        n_windows=int(len(df)),
        n_recordings=int(df["relpath"].nunique()),
        bootstrap_unit="recording",
        n_bootstrap=N_BOOT,
        note="AUC orientation-corrected: 0.5 = chance. CIs from resampling "
             "recordings with replacement, because windows overlap in time.",
        rows=json.loads(auc.to_json(orient="records")),
    ), indent=2, default=_json), encoding="utf-8")

    (ANALYSIS_DIR / "severity_tests.json").write_text(json.dumps(dict(
        unit="per-recording medians",
        severity_encoding=SEVERITY,
        multiple_comparison_correction="Benjamini-Hochberg over all metrics",
        kruskal=json.loads(tests["kruskal"].to_json(orient="records")),
        dunn=json.loads(tests["dunn"].to_json(orient="records")),
        spearman=json.loads(tests["spearman"].to_json(orient="records")),
        n_surviving_fdr=len(survivors),
        survivors=survivors,
    ), indent=2, default=_json), encoding="utf-8")

    if corr is not None:
        (ANALYSIS_DIR / "feature_redundancy.json").write_text(json.dumps(
            corr.to_dict(orient="index"), indent=2, default=_json),
            encoding="utf-8")

    if verbose:
        print_auc_summary(auc)
        print_severity_summary(tests["kruskal"], tests["dunn"],
                               tests["spearman"])
        if corr is not None:
            print_redundancy(corr)
    return dict(auc=auc, **tests, corr=corr)


def print_redundancy(corr: pd.DataFrame) -> None:
    print("\n  FEATURE REDUNDANCY  (Spearman, among metrics passing FDR)")
    print("  " + "=" * 108)
    c = corr.round(2)
    print("  " + " " * 24 + "".join(f"{k[:9]:>10}" for k in c.columns))
    for idx, row in c.iterrows():
        print(f"  {idx:<24}" + "".join(f"{row[k]:>10.2f}" for k in c.columns))

    # Pairs above 0.9 are effectively duplicates.
    pairs = []
    cols = list(c.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = c.iloc[i, j]
            if np.isfinite(v) and abs(v) > 0.9:
                pairs.append((cols[i], cols[j], float(v)))
    pairs.sort(key=lambda p: -abs(p[2]))
    if pairs:
        print("\n  near-duplicate pairs |rho| > 0.9:")
        for a, b, v in pairs[:12]:
            print(f"    {a:<26} {b:<26} {v:+.3f}")


if __name__ == "__main__":
    run()