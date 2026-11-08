"""Case 28 - UAV propeller fault detection from on-board audio.

Runs the full pipeline in order and writes every table and figure to disk.
Each phase can also be run on its own via `python -m src.<phase>`.

Phases
------
1  inventory             scan the dataset, one row per WAV
2  segmentation          steady-hover detection, 4 s windows, primary set
3  analysis              window metrics + statistical tests
4  features              selection + conventional threshold baseline
5  split audit           prove no window/mission leakage in the CV splits
6  models                four classifiers + deployment cost
7  evaluate              f0 estimation error, binary confusion detail
8  robustness            SNR sweep, window-length sweep, trajectory transfer
   figures_results       report figures drawn from the artefacts above
"""
from __future__ import annotations

import argparse
import json
import time

from . import config as C


def _banner(phase: str, title: str) -> None:
    print("\n" + "=" * 78)
    print(f"{phase}  {title}")
    print("=" * 78)


def _selected_features() -> list[str] | None:
    path = C.ROOT / "analysis" / "selected_features.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["selected"]


def main(phases: list[int]) -> None:
    C.set_seed()
    t0 = time.perf_counter()

    env = C.describe_environment()
    print(f"Case 28 pipeline  |  seed={C.RANDOM_STATE}  |  python {env['python']}")
    print(f"  numpy {env['numpy']}  scipy {env['scipy']}  "
          f"sklearn {env['scikit-learn']}  matplotlib {env['matplotlib']}")

    inventory = None

    # -------------------------------------------------------------- phase 1
    if 1 in phases:
        from . import inventory as inv_mod
        _banner("PHASE 1", "Dataset inventory")
        inventory = inv_mod.run()
        from . import figures as F
        print(f"  figure 1 -> {F.figure1_class_inventory(inventory).name}")
    else:
        inventory_path = C.result_path("inventory.csv")
        if inventory_path.exists():
            import pandas as pd
            inventory = pd.read_csv(inventory_path)
            print(f"PHASE 1  reusing {inventory_path.name}")
        else:
            raise SystemExit("Phase 1 output missing; run phase 1 first")

    # -------------------------------------------------------------- phase 2
    if 2 in phases:
        from . import figures as F
        from . import segment as seg_mod
        _banner("PHASE 2", "Steady-hover segmentation")
        seg_df, primary = seg_mod.run(inventory)
        print(f"  primary analysis set: {len(primary)} windows from "
              f"{primary['relpath'].nunique()} recordings")

        from . import figures_segmentation as FS
        FS.run(inventory, seg_df, primary)
    else:
        for stem in ("segmentation.csv", "segments_primary.csv"):
            if not C.result_path(stem).exists():
                raise SystemExit(f"Phase 2 output missing ({stem}); "
                                 "run phase 2 first")
        print("PHASE 2  reusing results/segment*.csv")

    # -------------------------------------------------------------- phase 3
    if 3 in phases:
        import pandas as pd
        from . import analysis as A
        _banner("PHASE 3", "Window metrics and statistical tests")
        primary = pd.read_csv(C.result_path("segments_primary.csv"))
        A.run(primary)

    # -------------------------------------------------------------- phase 4
    if 4 in phases:
        from . import baseline
        from . import features as F
        _banner("PHASE 4", "Feature selection and threshold baseline")
        F.run()
        baseline.run()

    # -------------------------------------------------------------- phase 5
    if 5 in phases:
        from . import split_audit
        _banner("PHASE 5", "Cross-validation split audit")
        split_audit.run()

    # -------------------------------------------------------------- phase 6
    if 6 in phases:
        from . import models as M
        _banner("PHASE 6", "Classification models and deployment cost")
        M.run(selected=_selected_features())

    # -------------------------------------------------------------- phase 7
    if 7 in phases:
        from . import evaluate
        _banner("PHASE 7", "f0 estimation error and binary confusion")
        evaluate.run()

    # -------------------------------------------------------------- phase 8
    if 8 in phases:
        from . import robustness
        _banner("PHASE 8", "Robustness: noise, window length, trajectory")
        robustness.run(selected=_selected_features())

    # -------------------------------------------------------------- figures
    needs = {3, 4, 6, 8}
    if needs & set(phases):
        from . import figures_results
        _banner("FIGURES", "Model / baseline / robustness figures")
        figures_results.run()

    dt = time.perf_counter() - t0
    print(f"\nPipeline finished in {dt:.1f} s")
    print(f"  figures -> {C.FIGURES_DIR}")
    print(f"  results -> {C.RESULTS_DIR}")
    print(f"  analysis -> {C.ROOT / 'analysis'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phases", nargs="+", type=int,
                    default=[1, 2, 3, 4, 5, 6, 7, 8],
                    help="phases to run (default: 1 2 3 4 5 6 7 8)")
    args = ap.parse_args()
    main(args.phases)
