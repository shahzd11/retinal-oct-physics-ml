"""Train on one scanner, test on another.

Train: Kermany OCT2017 (Heidelberg Spectralis), labels harmonised to NORMAL / DME / AMD
       (CNV and DRUSEN are both AMD).
Test:  OCTDL (Optovue RTVue) with the same 3 labels, and OCTID (Zeiss Cirrus) as
       NORMAL-vs-abnormal (OCTID has no DME/AMD classes).

  python cross_scanner.py --train results/v2/kermany_features.csv \
      --test results/v2/octdl_features.csv --octid results/v2/octid_features.csv
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix, recall_score

from retina.evaluation import (feature_sets, add_row_summaries, add_band_summaries, hgb, rf_paper, logreg,
                               repeated_cv)

warnings.filterwarnings("ignore")
HARMONISE = {"NORMAL": "NORMAL", "NO": "NORMAL", "DME": "DME", "CNV": "AMD", "DRUSEN": "AMD", "AMD": "AMD"}
SETS = ["paper (300 grid H+S)", "paper row-summary (40)", "v2 texture (band x sector)",
        "v2 disease-specific", "v2 band-summary + disease", "v2 all"]


def prep(path, keep=None):
    df = add_band_summaries(add_row_summaries(pd.read_csv(path)))
    if keep is not None:
        df["label"] = df.label.map(HARMONISE)
        df = df[df.label.isin(keep)].reset_index(drop=True)
    return df


def X_of(df, cols):
    X = df.reindex(columns=cols).values.astype(float)
    X[~np.isfinite(X)] = np.nan
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", help="external dataset with NORMAL/DME/AMD-type labels (e.g. OCTDL)")
    ap.add_argument("--octid", help="OCTID features for NORMAL-vs-abnormal transfer")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--sets", nargs="*", default=SETS)
    ap.add_argument("--out", default="results/v2/cross_scanner.json")
    a = ap.parse_args()
    classes = ["NORMAL", "DME", "AMD"]
    tr = prep(a.train, classes)
    te = prep(a.test, classes) if a.test else None
    oc = pd.read_csv(a.octid) if a.octid else None
    if oc is not None:
        oc = add_band_summaries(add_row_summaries(oc))
    fsets = {k: v for k, v in feature_sets(tr).items() if k in a.sets}
    res = []
    for sname, cols in fsets.items():
        for mname, fn in [("RF (paper config)", rf_paper), ("Gradient boosting", hgb), ("Logistic regression", logreg)]:
            Xtr, ytr = X_of(tr, cols), tr.label.values
            within, _ = repeated_cv(fn, Xtr, ytr, tr.patient.values, repeats=a.repeats)
            m = fn(0).fit(Xtr, ytr)
            row = {"features": sname, "model": mname, "within_bacc": within["bacc_mean"],
                   "within_ci": [within["bacc_ci_lo"], within["bacc_ci_hi"]]}
            if te is not None:
                p = m.predict(X_of(te, cols))
                row.update(ext_bacc=balanced_accuracy_score(te.label, p), ext_acc=accuracy_score(te.label, p),
                           ext_cm=confusion_matrix(te.label, p, labels=classes).tolist())
            if oc is not None:
                p = m.predict(X_of(oc, cols))
                truth = np.where(oc.label == "NORMAL", "NORMAL", "ABNORMAL")
                pred = np.where(p == "NORMAL", "NORMAL", "ABNORMAL")
                row.update(octid_normal_vs_abnormal_bacc=balanced_accuracy_score(truth, pred),
                           octid_normal_recall=recall_score(truth, pred, pos_label="NORMAL"),
                           octid_abnormal_recall=recall_score(truth, pred, pos_label="ABNORMAL"))
            res.append(row)
            print(f"{sname:30s} {mname:18s} within-Kermany bacc {row['within_bacc']:.3f}"
                  + (f" | OCTDL bacc {row['ext_bacc']:.3f}" if te is not None else "")
                  + (f" | OCTID normal-vs-abnormal bacc {row['octid_normal_vs_abnormal_bacc']:.3f}"
                     if oc is not None else ""), flush=True)
    out = Path(a.out)
    json.dump(res, open(out, "w"), indent=2, default=float)
    print("saved", out)


if __name__ == "__main__":
    main()
