"""Collinearity test (entropy vs Hurst), Gini vs permutation importance, grouped permutation
importance by feature family.

  python analysis_v2.py results/v2/octid_features.csv
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from retina.evaluation import (hgb, rf_paper, grouped_permutation_importance, DISEASE_PREFIX,
                               TEX_KINDS, BANDS)
from retina import plotstyle as ps
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def vif(df):
    X = df.dropna().values
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    out = {}
    for i, c in enumerate(df.columns):
        o = np.delete(X, i, axis=1)
        r2 = LinearRegression().fit(o, X[:, i]).score(o, X[:, i])
        out[c] = 1 / max(1 - r2, 1e-6)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("features")
    a = ap.parse_args()
    ps.apply()
    df = pd.read_csv(a.features)
    name = Path(a.features).stem.replace("_features", "")
    out_dir = Path(a.features).parent
    res = {}

    # ---------------------------------------------------------------- 1. collinearity
    rh = []
    for g in range(150):
        h, e = df[f"grid_{g}_hurst"], df[f"grid_{g}_entropy"]
        if h.std() > 0 and e.std() > 0:
            rh.append(spearmanr(h, e).statistic)
    rh = np.abs(np.array(rh))
    res["paper_grid_|rho|(H,entropy)_median"] = float(np.nanmedian(rh))
    res["paper_grid_frac_|rho|>0.8"] = float(np.nanmean(rh > 0.8))
    regions = sorted({m.group(1) for c in df.columns if (m := re.match(r"tex_(.+)_H$", c))})
    rv, vifs = [], []
    for r in regions:
        h, e = df[f"tex_{r}_H"], df[f"tex_{r}_entropy"]
        ok = h.notna() & e.notna()
        if ok.sum() > 20:
            rv.append(abs(spearmanr(h[ok], e[ok]).statistic))
        sub = df[[f"tex_{r}_{k}" for k in ["H", "entropy", "std", "mean", "kt1", "nmsid1", "lac4", "coh"]]]
        vifs.append(vif(sub))
    res["v2_region_|rho|(H,entropy)_median"] = float(np.median(rv))
    v = pd.DataFrame(vifs).median()
    res["v2_median_VIF"] = {k.split("_")[-1] if not k.endswith("nmsid1") else "nmsid1": round(float(x), 2)
                            for k, x in v.items()}
    print("[1] |Spearman(H, entropy)| median: paper grids", round(res["paper_grid_|rho|(H,entropy)_median"], 3),
          "| v2 regions", round(res["v2_region_|rho|(H,entropy)_median"], 3))
    print("    median VIF per feature kind (v2 regions):", res["v2_median_VIF"])

    # ---------------------------------------------------------------- 2. Gini vs permutation (paper setup)
    pcols = [c for c in df.columns if re.fullmatch(r"grid_\d+_(hurst|kt1)", c)]
    X, y = df[pcols].values, df.label.values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    rf = rf_paper(42).fit(Xtr, ytr)
    gini = pd.Series(rf[-1].feature_importances_, index=pcols).sort_values(ascending=False)
    pi = permutation_importance(rf, Xte, yte, n_repeats=10, random_state=0, scoring="balanced_accuracy", n_jobs=1)
    perm = pd.Series(pi.importances_mean, index=pcols).sort_values(ascending=False)
    top_g, top_p = list(gini.index[:15]), list(perm.index[:15])
    res["gini_top15"] = top_g
    res["perm_top15"] = top_p
    res["gini_perm_top15_overlap"] = len(set(top_g) & set(top_p))
    res["perm_n_features_positive"] = int((perm > 0.002).sum())
    res["perm_top15_hurst_frac"] = float(np.mean([c.endswith("hurst") for c in top_p]))
    print(f"[2] Gini vs permutation top-15 overlap: {res['gini_perm_top15_overlap']}/15; "
          f"features with permutation importance > 0.002: {res['perm_n_features_positive']}/300")

    # ---------------------------------------------------------------- 3. grouped permutation (v2, held-out)
    tex = [c for c in df.columns if c.startswith("tex_")]
    dis = [c for c in df.columns if c.startswith(DISEASE_PREFIX)]
    cols = tex + dis
    fams = {f"texture: {k}": [c for c in tex if c.endswith("_" + k) and
                               (k != "H" or not c.endswith("_r2"))] for k in TEX_KINDS}
    fams["texture: H"] = [c for c in tex if c.endswith("_H")]
    fams.update({f"band: {b}": [c for c in tex if c.startswith(f"tex_{b}_")] for b in BANDS})
    fams.update({
        "disease: thickness/fovea": [c for c in dis if c.startswith(("thk_", "fovea_", "thin_", "gap_"))],
        "disease: fluid": [c for c in dis if c.startswith("fluid_")],
        "disease: hyper-reflective foci": [c for c in dis if c.startswith("hrf_")],
        "disease: RPE elevation": [c for c in dis if c.startswith("rpe_")],
        "disease: retina reflectivity": [c for c in dis if c.startswith("retina_")],
    })
    idx = {k: [cols.index(c) for c in v] for k, v in fams.items() if v}
    res["grouped_perm"] = {}
    panels = []
    for tname, mask in [("all classes", np.ones(len(df), bool)), ("disease-only", (df.label != "NORMAL").values)]:
        d = df[mask]
        Xv = d[cols].values.astype(float)
        imp = grouped_permutation_importance(hgb, Xv, d.label.values, d.patient.values, idx)
        res["grouped_perm"][tname] = imp.round(4).to_dict()
        panels.append((tname, imp))
        print(f"[3] grouped permutation importance ({tname}, drop in balanced acc):")
        print("    " + "\n    ".join(f"{k:36s} {v:+.3f}" for k, v in imp.items()))

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    for ax, (t, imp) in zip(axes, panels):
        ps.hbar(ax, list(imp.index), list(imp.values))
        ax.set_title(f"{name}: {t}")
        ax.set_xlabel("Drop in balanced accuracy when permuted (held-out)")
    plt.tight_layout(); plt.savefig(out_dir / f"{name}_grouped_permutation.png", dpi=150); plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    ps.hbar(axes[0], top_g, list(gini[top_g].values), fmt="{:.4f}")
    axes[0].set_title("Gini importance (training data)")
    ps.hbar(axes[1], top_p, list(perm[top_p].values), color=ps.SERIES[1], fmt="{:.4f}")
    axes[1].set_title("Permutation importance (held-out, balanced acc)")
    plt.suptitle(f"{name}: paper RF, top-15 features by two importance measures", weight="bold")
    plt.tight_layout(); plt.savefig(out_dir / f"{name}_gini_vs_permutation.png", dpi=150); plt.close()

    json.dump(res, open(out_dir / f"{name}_analysis.json", "w"), indent=2)
    print("saved", out_dir / f"{name}_analysis.json")


if __name__ == "__main__":
    main()
