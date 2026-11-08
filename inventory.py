"""Phase 1 - dataset inventory.

Walks the Dataset tree, validates every WAV header, assigns each file a class /
config / mission / take label from its path, and writes an inventory table.
Nothing here interprets the signal; this phase only establishes what the data
is and states what it cannot tell us.
"""
from __future__ import annotations

import hashlib
import json
import wave
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


def parse_relpath(rel: str) -> dict:
    """Derive labels from the directory layout.

    normal/<mission>/<take>/ch1.wav
    <class>/<config>/<mission>/<take>/ch1.wav
    3_4_broken/3_broken/<config>/<mission>/<take>/ch1.wav
    3_4_broken/4_broken/<config>/<mission>/<take>/ch1.wav
    """
    p = rel.split(Path(rel).anchor and "\\" or "/")
    p = [x for x in rel.replace("\\", "/").split("/") if x]
    head = p[0]

    if head == "normal":
        return dict(cls="normal", group="normal", config="0000",
                    mission=p[1], take=int(p[2]))

    if head == "3_4_broken":
        return dict(cls=head, group=p[1], config=p[2], mission=p[3], take=int(p[4]))

    return dict(cls=head, group=head, config=p[1], mission=p[2], take=int(p[3]))


def read_header(path: Path) -> dict:
    with wave.open(str(path), "rb") as w:
        return dict(
            channels=w.getnchannels(),
            sampwidth_bytes=w.getsampwidth(),
            bits=w.getsampwidth() * 8,
            framerate=w.getframerate(),
            frames=w.getnframes(),
        )


def build_inventory(dataset_dir: Path | None = None, verbose: bool = True) -> pd.DataFrame:
    """Scan every WAV file and return one validated row per recording."""
    dataset_dir = Path(dataset_dir or C.DATASET_DIR)
    rows: list[dict] = []
    problems: list[str] = []

    for wav in sorted(dataset_dir.rglob("*.wav")):
        rel = str(wav.relative_to(dataset_dir))
        try:
            hdr = read_header(wav)
        except Exception as exc:  # unreadable header is a hard failure
            problems.append(f"{rel}: unreadable header ({exc})")
            continue

        if hdr["framerate"] != C.FS_EXPECTED:
            problems.append(f"{rel}: fs={hdr['framerate']} expected {C.FS_EXPECTED}")
        if hdr["channels"] != 1:
            problems.append(f"{rel}: {hdr['channels']} channels, expected mono")

        rows.append(dict(
            relpath=rel,
            filename=wav.name,
            **parse_relpath(rel),
            **hdr,
            duration_s=hdr["frames"] / hdr["framerate"],
            size_bytes=wav.stat().st_size,
        ))

    if problems:
        raise ValueError(
            "Dataset failed header validation:\n  " + "\n  ".join(problems)
        )

    df = pd.DataFrame(rows)
    df["class_label"] = df["cls"].map(C.CLASS_LABELS)
    df["n_broken"] = df["config"].apply(lambda s: s.count("1"))
    df = df.sort_values(["cls", "config", "mission", "take"]).reset_index(drop=True)

    if verbose:
        print(f"  scanned {len(df)} files from {dataset_dir}")
    return df


def check_duplicates(df: pd.DataFrame) -> dict:
    """Hash each file's sample data to detect exact duplicates."""
    hashes: dict[str, list[str]] = defaultdict(list)
    dataset_dir = C.DATASET_DIR
    for rel in df["relpath"]:
        raw = (dataset_dir / rel).read_bytes()
        h = hashlib.md5(raw).hexdigest()
        hashes[h].append(rel)
    groups = {h: v for h, v in hashes.items() if len(v) > 1}
    return dict(
        n_unique_hashes=len(hashes),
        n_duplicate_groups=len(groups),
        n_files_in_duplicate_groups=sum(len(v) for v in groups.values()),
        examples=[v for v in list(groups.values())[:5]],
    )


def summarise(df: pd.DataFrame) -> dict:
    """Headline counts, durations and structure of the dataset."""
    per_class = (
        df.groupby("class_label")
        .agg(files=("relpath", "size"),
             total_minutes=("duration_s", lambda s: s.sum() / 60.0),
             median_duration_s=("duration_s", "median"),
             min_duration_s=("duration_s", "min"),
             max_duration_s=("duration_s", "max"))
        .reindex([C.CLASS_LABELS[c] for c in C.CLASSES])
    )
    per_mission = pd.crosstab(df["mission"], df["cls"]).reindex(columns=C.CLASSES, fill_value=0)
    per_config = (
        df.groupby(["cls", "config"]).size()
        .rename("files").reset_index()
    )
    return dict(
        n_files=int(len(df)),
        total_hours=round(df["duration_s"].sum() / 3600.0, 2),
        all_mono=bool((df["channels"] == 1).all()),
        all_16bit=bool((df["bits"] == 16).all()),
        sample_rates=sorted({int(v) for v in df["framerate"]}),
        bit_depths=sorted({int(v) for v in df["bits"]}),
        channels=sorted({int(v) for v in df["channels"]}),
        filenames=sorted({v for v in df["filename"]}),
        per_class=per_class,
        per_mission=per_mission,
        per_config=per_config,
        duration_range_s=(float(df["duration_s"].min()), float(df["duration_s"].max())),
    )


def dataset_limitations() -> list[dict]:
    """What this dataset structurally cannot support, stated up front."""
    return [
        dict(limitation="No accelerometer / vibration channel",
             impact="Vibration-based fault detection cannot be evaluated; acoustic only."),
        dict(limitation="No RPM or throttle telemetry",
             impact="Blade-pass frequency must be estimated from the signal itself, so "
                    "frequency-estimation error is measured against a signal-derived "
                    "reference rather than ground truth."),
        dict(limitation="Indoor missions only, square and figure-eight trajectories",
             impact="No outdoor wind, rain or ambient-noise variation; robustness to "
                    "those conditions is untested."),
        dict(limitation="Single UAV airframe and single microphone position",
             impact="No airframe-to-airframe or sensor-position generalisation evidence."),
        dict(limitation="Takeoff-to-landing recordings with a full throttle sweep",
             impact="Absolute level and dominant frequency vary with flight phase; "
                    "features must come from segmented steady-RPM windows."),
        dict(limitation="Damage type is limited to a broken (fractured) blade",
             impact="No crack-only, erosion, imbalance or nicked-tip classes."),
        dict(limitation="3_4_broken merges three and four broken propellers",
             impact="The class is not acoustically homogeneous, which adds label noise "
                    "and a ceiling on achievable macro-F1."),
    ]


def run(verbose: bool = True) -> pd.DataFrame:
    """Execute Phase 1 and persist all artefacts."""
    if verbose:
        print("\nPHASE 1 - dataset inventory")

    df = build_inventory(verbose=verbose)
    dup = check_duplicates(df)
    summary = summarise(df)
    limits = dataset_limitations()

    df.to_csv(C.result_path("inventory.csv"), index=False)
    summary["per_class"].to_csv(C.result_path("table1_class_summary.csv"))
    summary["per_mission"].to_csv(C.result_path("table1_class_by_mission.csv"))
    summary["per_config"].to_csv(C.result_path("table1_config_counts.csv"), index=False)
    C.result_path("table1_dataset_limitations.csv").write_text(
        pd.DataFrame(limits).to_csv(index=False), encoding="utf-8"
    )
    (C.result_path("phase1_summary.json")).write_text(
        json.dumps(
            dict(
                environment=C.describe_environment(),
                summary={k: v for k, v in summary.items()
                         if k not in ("per_class", "per_mission", "per_config")},
                duplicates=dup,
                limitations=limits,
            ),
            indent=2, default=str,
        ),
        encoding="utf-8",
    )

    if verbose:
        s = summary
        print(f"  {s['n_files']} files, {s['total_hours']} h, "
              f"fs={s['sample_rates']} Hz, {s['bit_depths']}-bit, "
              f"channels={s['channels']}")
        print(f"  filenames: {s['filenames']}")
        print(f"  duplicates: {dup['n_duplicate_groups']} groups, "
              f"{dup['n_files_in_duplicate_groups']} files")
        print("  per class:")
        for lbl, row in s["per_class"].iterrows():
            print(f"    {lbl:<11} {int(row['files']):3d} files  "
                  f"{row['total_minutes']:6.1f} min  "
                  f"dur {row['min_duration_s']:5.1f}-{row['max_duration_s']:5.1f} s")
        print("  per mission:")
        for m, row in s["per_mission"].iterrows():
            print(f"    {m:<11} " + "  ".join(f"{C.CLASS_LABELS[c]}={int(row[c]):2d}"
                                             for c in C.CLASSES))
        print(f"  limitations recorded: {len(limits)}")

    return df


if __name__ == "__main__":
    run()
