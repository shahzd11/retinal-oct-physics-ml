"""v5 / Part C: conformal triage, a *stated* statistical guarantee instead of "100% guaranteed".

For each task the soft-voting ensemble is wrapped in class-conditional split conformal prediction
(retina/conformal.py). We report, per class, the fraction of scans whose true label is inside the
prediction set (must be >= 1 - alpha), the average set size and how often the model commits to one label.

Triage tasks are NORMAL vs ABNORMAL on the two single-scanner datasets (OCTDL, Kermany).
OCTID is used only disease-only, because its NORMAL scans come from a different device.

  python v5_conformal.py
Writes results/v5/conformal.json
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from retina.conformal import conformal_cv, summarize
from retina.ensemble import soft_vote
from retina.evaluation import add_band_summaries, add_row_summaries, feature_sets

warnings.filterwarnings("ignore")
OUT = Path("results/v5"); OUT.mkdir(parents=True, exist_ok=True)
FSET = "v2 band-summary + disease + NDT robust + fractal"
ALPHAS = (0.05, 0.10)
SEEDS = (0, 1)
TASKS = {
    "OCTDL triage NORMAL vs ABNORMAL": ("octdl", lambda d: d.label.notna(), True),
    "Kermany triage NORMAL vs ABNORMAL": ("kermany", lambda d: d.label.notna(), True),
    "OCTID CSR/DR/MH (500x750)": ("octid", lambda d: (d.label != "NORMAL") & (d.orig_h == 500) & (d.orig_w == 750), False),
    "Kermany CNV/DME/DRUSEN": ("kermany", lambda d: d.label != "NORMAL", False),
}


def main():
    res = []
    for tname, (ds, sel, binary) in TASKS.items():
        df = add_band_summaries(add_row_summaries(pd.read_csv(f"results/v4/{ds}_features.csv")))
        d = df[sel(df)].reset_index(drop=True)
        y = np.where(d.label == "NORMAL", "NORMAL", "ABNORMAL") if binary else d.label.values
        g = d.patient.values
        X = d[feature_sets(d)[FSET]].values.astype(float)
        X[~np.isfinite(X)] = np.nan
        for a in ALPHAS:
            runs = []
            for s in SEEDS:
                S, _, classes = conformal_cv(soft_vote, X, y, g, alpha=a, seed=s)
                runs.append(summarize(S, y, classes))
            agg = {k: float(np.mean([r[k] for r in runs])) for k in runs[0] if k != "coverage_per_class"}
            agg["coverage_per_class"] = {c: float(np.mean([r["coverage_per_class"][c] for r in runs]))
                                         for c in runs[0]["coverage_per_class"]}
            row = dict(task=tname, alpha=a, target=1 - a, n=len(d), **agg)
            res.append(row)
            pc = ", ".join(f"{c} {v:.3f}" for c, v in row["coverage_per_class"].items())
            print(f"{tname:36s} alpha={a:.2f}  coverage {row['coverage']:.3f} (per class: {pc})  "
                  f"set size {row['mean_set_size']:.2f}  singletons {row['singleton_rate']:.2f} "
                  f"(acc {row['singleton_accuracy']:.3f})", flush=True)
    json.dump(res, open(OUT / "conformal.json", "w"), indent=2)


if __name__ == "__main__":
    main()
