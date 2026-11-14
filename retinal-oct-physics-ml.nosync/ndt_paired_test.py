"""Paired test: does adding NDT features to v2 improve balanced accuracy?

Same model, same patient-grouped folds (identical seeds), with vs without NDT columns.
Difference in balanced accuracy is bootstrapped over patients (paired: the same resampled
patients are scored for both feature sets).

  python ndt_paired_test.py
Writes results/v3/ndt_paired_test.json
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from retina.evaluation import add_band_summaries, add_row_summaries, feature_sets, hgb, logreg, oof_predictions

warnings.filterwarnings("ignore")
OUT = Path("results/v3")
BASE = "v2 band-summary + disease"
ADDS = ["v2 band-summary + disease + NDT", "v2 band-summary + disease + NDT robust"]
REPEATS, N_BOOT = 3, 2000


def paired_boot(y, pa, pb, groups, rng):
    ug = np.unique(groups)
    idx = {g: np.nonzero(groups == g)[0] for g in ug}
    d = []
    for _ in range(N_BOOT):
        s = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        if len(np.unique(y[s])) < len(np.unique(y)):
            continue
        d.append(balanced_accuracy_score(y[s], pb[s]) - balanced_accuracy_score(y[s], pa[s]))
    d = np.array(d)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float((d <= 0).mean())


def main():
    tasks = {
        "OCTID CSR/DR/MH (500x750)": ("octid", lambda d: (d.label != "NORMAL") & (d.orig_h == 500) & (d.orig_w == 750)),
        "OCTID 4-class": ("octid", lambda d: d.label.notna()),
        "OCTDL 7-class": ("octdl", lambda d: d.label.notna()),
        "OCTDL 6 diseases": ("octdl", lambda d: d.label != "NORMAL"),
        "Kermany 4-class": ("kermany", lambda d: d.label.notna()),
        "Kermany CNV/DME/DRUSEN": ("kermany", lambda d: d.label != "NORMAL"),
    }
    rng = np.random.default_rng(0)
    res = []
    for tname, (ds, sel) in tasks.items():
        df = add_band_summaries(add_row_summaries(pd.read_csv(OUT / f"{ds}_features.csv")))
        d = df[sel(df)].reset_index(drop=True)
        y, g = d.label.values, d.patient.values
        fs = feature_sets(d)

        def X(name):
            a = d[fs[name]].values.astype(float)
            a[~np.isfinite(a)] = np.nan
            return a

        for mname, fn in (("Logistic regression", logreg), ("LightGBM", hgb)):
            pa = [oof_predictions(fn, X(BASE), y, g, s, 5) for s in range(REPEATS)]
            for add in ADDS:
                pb = [oof_predictions(fn, X(add), y, g, s, 5) for s in range(REPEATS)]
                diffs = [balanced_accuracy_score(y, b) - balanced_accuracy_score(y, a) for a, b in zip(pa, pb)]
                lo, hi, p = paired_boot(y, pa[0], pb[0], g, rng)
                r = dict(task=tname, model=mname, added=add.replace(BASE + " + ", ""),
                         base_bacc=float(np.mean([balanced_accuracy_score(y, a) for a in pa])),
                         with_bacc=float(np.mean([balanced_accuracy_score(y, b) for b in pb])),
                         diff_mean=float(np.mean(diffs)), diff_per_repeat=[round(x, 4) for x in diffs],
                         diff_ci=[lo, hi], p_one_sided=p)
                res.append(r)
                print(f"{tname:28s} {mname:20s} +{r['added']:10s} {r['base_bacc']:.3f} -> {r['with_bacc']:.3f} "
                      f"diff {r['diff_mean']:+.3f} [{lo:+.3f}, {hi:+.3f}] p={p:.3f}  repeats {r['diff_per_repeat']}",
                      flush=True)
    json.dump(res, open(OUT / "ndt_paired_test.json", "w"), indent=2)
    print("saved", OUT / "ndt_paired_test.json")


if __name__ == "__main__":
    main()
