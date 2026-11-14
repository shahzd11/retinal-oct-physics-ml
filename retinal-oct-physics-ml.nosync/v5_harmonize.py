"""v5 / Part D: scanner harmonisation (Kermany/Spectralis -> OCTDL/Optovue, labels NORMAL/DME/AMD).

The NDT analogue is calibrating an instrument on a reference block before inspecting parts.
Three feature-level calibrations, each a per-feature affine map z = (x - mu) / sd:

  none          raw features (the repo's cross_scanner.py setting)
  z-all         mu, sd from all scans of that scanner (unsupervised; needs the whole target batch)
  healthy-ref   mu, sd from k healthy eyes of that scanner (the "reference block");
                for the target scanner those k calibration eyes (whole patients) are removed from the test set

  python v5_harmonize.py
Writes results/v5/harmonize.json
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score as bacc

from retina.ensemble import V5_MODELS
from retina.evaluation import add_band_summaries, add_row_summaries, feature_sets

warnings.filterwarnings("ignore")
OUT = Path("results/v5"); OUT.mkdir(parents=True, exist_ok=True)
HARMONISE = {"NORMAL": "NORMAL", "NO": "NORMAL", "DME": "DME", "CNV": "AMD", "DRUSEN": "AMD", "AMD": "AMD"}
SETS = ["v2 band-summary + disease + NDT", "v2 band-summary + disease + NDT robust + fractal"]
KS, DRAWS = (10, 30, 100), 10


def prep(p):
    df = add_band_summaries(add_row_summaries(pd.read_csv(p)))
    df["label"] = df.label.map(HARMONISE)
    return df[df.label.notna()].reset_index(drop=True)


def X_of(d, c):
    a = d.reindex(columns=c).values.astype(float)
    a[~np.isfinite(a)] = np.nan
    return a


def z(A, ref):
    mu, sd = np.nanmean(ref, 0), np.nanstd(ref, 0)
    sd[~(sd > 1e-12)] = 1.0
    return (A - mu) / sd


def main():
    tr, te = prep("results/v4/kermany_features.csv"), prep("results/v4/octdl_features.csv")
    fs = feature_sets(tr)
    rng = np.random.default_rng(0)
    healthy_pats = te.loc[te.label == "NORMAL", "patient"].unique()
    res = []
    for sname in SETS:
        c = fs[sname]
        Xtr, Xte, ytr, yte = X_of(tr, c), X_of(te, c), tr.label.values, te.label.values
        for mname, fn in V5_MODELS.items():
            row = {"features": sname, "model": mname,
                   "none": bacc(yte, fn(0).fit(Xtr, ytr).predict(Xte)),
                   "z-all": bacc(yte, fn(0).fit(z(Xtr, Xtr), ytr).predict(z(Xte, Xte)))}
            m = fn(0).fit(z(Xtr, Xtr[ytr == "NORMAL"]), ytr)
            for k in KS:
                v = []
                for _ in range(DRAWS):
                    cal = te.patient.isin(rng.choice(healthy_pats, k, replace=False)).values
                    v.append(bacc(yte[~cal], m.predict(z(Xte, Xte[cal])[~cal])))
                row[f"healthy-ref k={k}"] = float(np.mean(v)); row[f"healthy-ref k={k} sd"] = float(np.std(v))
            res.append(row)
            print(f"{sname:48s} {mname:24s} none {row['none']:.3f} | z-all {row['z-all']:.3f} | "
                  + " | ".join(f"ref k={k} {row[f'healthy-ref k={k}']:.3f}" for k in KS), flush=True)
    json.dump(res, open(OUT / "harmonize.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
