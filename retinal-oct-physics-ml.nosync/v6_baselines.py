"""v6 / figure inputs that need a model run (no images needed; reads committed feature CSVs).

  1. Paper method, per-scan predictions for confusion matrices:
     (a) the paper's own protocol: OCTID 4-class, stratified 80/20 split, seed 42 (train_eval.py)
     (b) the honest task: OCTID CSR/DR/MH 500x750, patient-grouped 5-fold, seed-0 folds
  2. Grouped permutation importance of the v5 soft vote on OCTID CSR/DR/MH, one group per
     (texture family x anatomical band), for the "where does the model look" heat map.

  python v6_baselines.py [--quick]
Writes results/v6/oof_octid_4_confounded_paper_rf_8020.csv, oof_octid_dis750_paper_rf.csv,
       results/v6/baselines.json, importance_band_family.json
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

from retina.ensemble import soft_vote
from retina.evaluation import BANDS, rf_paper
from retina.v6data import feat_matrix, load_features, task_frame
from retina.v6eval import argmax_labels, folds, oof_proba, save_oof, summarise
from retina.v6run import dump

warnings.filterwarnings("ignore")
OUT = "results/v6"
FAMILIES = {"Hurst H": ("H", "H_r2"), "NMSID roughness": ("nmsid1", "nmsid_ratio"), "lacunarity": ("lac4", "lac8"),
            "entropy": ("entropy",), "reflectivity (mean, std)": ("mean", "std"), "KT-1 slope": ("kt1",),
            "coherence": ("coh",)}


def paper_cols(d):
    return [c for c in d.columns if c.endswith(("_hurst", "_kt1")) and c.startswith("grid_")]


def paper_8020():
    df = pd.read_csv("results/features.csv")                    # the v1 reproduction table
    cols = [c for c in df.columns if c.endswith(("_hurst", "_kt1"))]
    X, y = df[cols].values, df.label.values
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=42)
    m = rf_paper(42).fit(X[tr], y[tr])
    P = m.predict_proba(X[te])
    save_oof(f"{OUT}/oof_octid_4_confounded_paper_rf_8020.csv", df.file.values[te], y[te], P, m.classes_)
    p = argmax_labels(P, m.classes_)
    return {"protocol": "OCTID 4-class, stratified 80/20, seed 42 (scanner-confounded)", "n_test": len(te),
            "acc": float((p == y[te]).mean()), "bacc": float(balanced_accuracy_score(y[te], p))}


def paper_cv(task="OCTID CSR/DR/MH (500x750)", quick=False):
    d, _, _ = task_frame(task)
    X = d[paper_cols(d)].values.astype(float)
    y, g = d.label.values, d.patient.values
    n = 2 if quick else 5
    preds = []
    for s in ((0,) if quick else (0, 1, 2)):
        P, cl = oof_proba(lambda seed: rf_paper(seed), X, y, g, s, n)
        preds.append(argmax_labels(P, cl))
        if s == 0:
            save_oof(f"{OUT}/oof_octid_dis750_paper_rf.csv", d.path.values, y, P, cl)
    return {"task": task, "model": "RF (paper config), 300 grid H+S", **summarise(y, preds, g)}


def band_family_importance(task="OCTID CSR/DR/MH (500x750)", n_rep=5, quick=False):
    d, _, _ = task_frame(task)
    X, cols = feat_matrix(d)
    y, g = d.label.values, d.patient.values
    groups = {}
    for fam, kinds in FAMILIES.items():
        for b in BANDS:
            idx = [i for i, c in enumerate(cols) for k in kinds
                   if c in (f"band_{b}_{k}", f"band_{b}_{k}_cen_minus_peri")]
            if idx:
                groups[(fam, b)] = idx
    rng = np.random.default_rng(0)
    drops = {k: [] for k in groups}
    for tr, te in folds(y, g, 0, 2 if quick else 5):
        m = soft_vote(0).fit(X[tr], y[tr])
        base = balanced_accuracy_score(y[te], m.predict(X[te]))
        for k, idx in groups.items():
            for _ in range(n_rep):
                Xp = X[te].copy()
                Xp[:, idx] = Xp[rng.permutation(len(te))][:, idx]
                drops[k].append(base - balanced_accuracy_score(y[te], m.predict(Xp)))
    mat = {fam: {b: float(np.mean(drops[(fam, b)])) for b in BANDS if (fam, b) in drops} for fam in FAMILIES}
    return {"task": task, "model": "v5 soft vote (171 features)", "bands": list(BANDS), "families": list(FAMILIES),
            "importance": mat, "note": "drop in balanced accuracy when the group is permuted in held-out folds (seed-0 folds)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    res = {"paper_8020": paper_8020(), "paper_cv_octid_dis750": paper_cv(quick=a.quick)}
    print(res, flush=True)
    dump(res, f"{OUT}/baselines.json")
    imp = band_family_importance(quick=a.quick)
    print(pd.DataFrame(imp["importance"]).T.round(4), flush=True)
    dump(imp, f"{OUT}/importance_band_family.json")


if __name__ == "__main__":
    main()
