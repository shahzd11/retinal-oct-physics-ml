"""Grouped permutation importance of NDT vs v2 feature families (held-out, patient-grouped folds).

Feature set: v2 band-summary + disease + NDT. Disease-only task per dataset
(OCTID restricted to 500x750 scans). Models: logistic regression and LightGBM.

  python ndt_importance.py
Writes results/v3/ndt_importance.json and results/v3/ndt_importance.png
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from retina import plotstyle as ps
from retina.evaluation import (add_band_summaries, add_row_summaries, feature_sets, grouped_permutation_importance,
                               hgb, logreg)

warnings.filterwarnings("ignore")
OUT = Path("results/v3")
SET = "v2 band-summary + disease + NDT"


def families(cols):
    f = {}
    for b in ("inner", "outer", "rpe", "choroid"):
        f[f"v2 band: {b}"] = [c for c in cols if c.startswith(f"band_{b}_")]
    f["v2 disease: thickness/fovea"] = [c for c in cols if c.startswith(("thk_", "fovea_", "gap_", "thin_"))]
    f["v2 disease: fluid"] = [c for c in cols if c.startswith("fluid_")]
    f["v2 disease: reflectivity/RPE"] = [c for c in cols if c.startswith(("hrf_", "retina_", "rpe_", "qc_"))]
    f["NDT: attenuation"] = [c for c in cols if c.startswith(("ndt_att_", "ndt_trans_"))]
    f["NDT: interface contrast"] = [c for c in cols if c.startswith(("ndt_ic_", "ndt_ilm_", "ndt_rpe_echo"))]
    f["NDT: half-max sizing"] = [c for c in cols if c.startswith("ndt_hm_")]
    f["NDT: shadowing"] = [c for c in cols if c.startswith(("ndt_sh_", "ndt_hypertrans"))]
    f["NDT: SNR/CNR"] = [c for c in cols if c.startswith("ndt_cnr_")]
    f["NDT: speckle"] = [c for c in cols if c.startswith("ndt_spk_")]
    f["ALL v2"] = [c for c in cols if not c.startswith("ndt_")]
    f["ALL NDT"] = [c for c in cols if c.startswith("ndt_")]
    return {k: v for k, v in f.items() if v}


def main():
    ps.apply()
    tasks = {
        "OCTID CSR/DR/MH (500x750)": ("octid", lambda d: (d.label != "NORMAL") & (d.orig_h == 500) & (d.orig_w == 750)),
        "OCTDL 6 diseases": ("octdl", lambda d: d.label != "NORMAL"),
        "Kermany CNV/DME/DRUSEN": ("kermany", lambda d: d.label != "NORMAL"),
    }
    res, panels = {}, []
    for tname, (ds, sel) in tasks.items():
        df = add_band_summaries(add_row_summaries(pd.read_csv(OUT / f"{ds}_features.csv")))
        d = df[sel(df)].reset_index(drop=True)
        cols = feature_sets(d)[SET]
        X = d[cols].values.astype(float)
        X[~np.isfinite(X)] = np.nan
        fam = families(cols)
        idx = {k: [cols.index(c) for c in v] for k, v in fam.items()}
        res[tname] = {}
        for mname, fn in (("Logistic regression", logreg), ("LightGBM", hgb)):
            imp = grouped_permutation_importance(fn, X, d.label.values, d.patient.values, idx)
            res[tname][mname] = imp.round(4).to_dict()
            print(f"\n{tname} | {mname}")
            print("  " + "\n  ".join(f"{k:32s} {v:+.3f}" for k, v in imp.items()), flush=True)
        mean = pd.DataFrame(res[tname]).mean(1).sort_values(ascending=False)
        panels.append((tname, mean))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.8))
    for ax, (t, imp) in zip(axes, panels):
        imp = imp.drop(["ALL v2", "ALL NDT"]).sort_values(ascending=False)
        colors = [ps.SERIES[2] if k.startswith("NDT") else ps.SERIES[0] for k in imp.index]
        ps.hbar(ax, list(imp.index), list(imp.values), color=colors)
        ax.set_title(t, loc="left")
        ax.set_xlabel("Drop in balanced accuracy when permuted\n(held-out, mean of LogReg and LightGBM)")
    fig.suptitle("Which feature families carry the disease signal?  green = NDT, blue = v2", weight="bold")
    plt.tight_layout(); plt.savefig(OUT / "ndt_importance.png", dpi=140); plt.close()
    json.dump(res, open(OUT / "ndt_importance.json", "w"), indent=2)
    print("saved", OUT / "ndt_importance.json")


if __name__ == "__main__":
    main()
