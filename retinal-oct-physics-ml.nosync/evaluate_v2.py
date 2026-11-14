"""Model x feature-set comparison with patient-grouped repeated CV and cluster-bootstrap CIs.

  python evaluate_v2.py results/v2/octid_features.csv --repeats 3
Writes results/v2/<name>_model_grid.csv and .md
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from retina.evaluation import (MODELS, feature_sets, add_row_summaries, add_band_summaries, repeated_cv)

warnings.filterwarnings("ignore")


def tasks_for(df):
    labels = sorted(df.label.unique())
    t = {f"all classes ({'/'.join(labels)})": np.ones(len(df), bool)}
    normal = next((l for l in ("NORMAL", "NO") if l in labels), None)
    if normal:
        t["disease-only (" + "/".join(l for l in labels if l != normal) + ")"] = (df.label != normal).values
    if (df.dataset == "OCTID").all():
        t["disease-only, 500x750 scans only (no size confound)"] = (
            (df.label != "NORMAL") & (df.orig_h == 500) & (df.orig_w == 750)).values
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("features")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--sets", nargs="*", default=None, help="subset of feature-set names")
    ap.add_argument("--models", nargs="*", default=None)
    a = ap.parse_args()

    df = pd.read_csv(a.features)
    df = add_band_summaries(add_row_summaries(df))
    fsets = feature_sets(df)
    if a.sets:
        fsets = {k: v for k, v in fsets.items() if k in a.sets}
    models = {k: v for k, v in MODELS.items() if not a.models or k in a.models}
    rows = []
    for tname, mask in tasks_for(df).items():
        d = df[mask].reset_index(drop=True)
        y, g = d.label.values, d.patient.values
        print(f"\n=== {tname}: n={len(d)}, patients={len(set(g))}, chance bacc={1/len(set(y)):.3f}")
        for sname, cols in fsets.items():
            X = d[cols].values.astype(np.float64)
            X[~np.isfinite(X)] = np.nan
            for mname, fn in models.items():
                if mname.startswith("Two-stage") and not {"NORMAL", "NO"} & set(y):
                    continue
                r, _ = repeated_cv(fn, X, y, g, repeats=a.repeats)
                r.update(task=tname, features=sname, n_features=len(cols), model=mname, n=len(d))
                rows.append(r)
                print(f"  {sname:32s} {mname:22s} acc {r['acc_mean']:.3f}±{r['acc_sd']:.3f}  "
                      f"bacc {r['bacc_mean']:.3f} [{r['bacc_ci_lo']:.3f},{r['bacc_ci_hi']:.3f}]  f1 {r['f1_mean']:.3f}",
                      flush=True)
    res = pd.DataFrame(rows)
    out = Path(a.features).with_name(Path(a.features).stem.replace("_features", "") + "_model_grid")
    res.to_csv(out.with_suffix(".csv"), index=False)
    write_md(res, out.with_suffix(".md"))
    print("saved", out.with_suffix(".csv"))


def write_md(res, path):
    with open(path, "w", encoding="utf-8") as fh:
        for t, d in res.groupby("task", sort=False):
            fh.write(f"\n### {t}\n\n| Features | Model | Accuracy | Balanced acc [95% CI] | Macro-F1 |\n|---|---|---|---|---|\n")
            for _, r in d.sort_values("bacc_mean", ascending=False).iterrows():
                fh.write(f"| {r.features} ({r.n_features}) | {r.model} | {r.acc_mean:.3f} ± {r.acc_sd:.3f} | "
                         f"{r.bacc_mean:.3f} [{r.bacc_ci_lo:.3f}, {r.bacc_ci_hi:.3f}] | {r.f1_mean:.3f} |\n")


if __name__ == "__main__":
    main()
