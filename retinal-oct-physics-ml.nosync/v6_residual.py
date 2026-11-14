"""v6 / Step 2: POD residual anomaly score (ROM-based defect detection, NDT style).

A POD basis is fitted on *healthy* training eyes only. A scan the healthy modes cannot rebuild is
anomalous:
    e(x) = ||x - x̄ - V_r V_r^T (x - x̄)||² / ||x - x̄||²
and the signed residual map x - x̂ shows *where* it departs from a healthy retina.

  (a) NORMAL-vs-ABNORMAL AUROC of e(x): OCTDL and Kermany (out-of-fold, healthy training eyes only),
      Kermany-healthy -> OCTDL (cross-scanner). OCTID is skipped: its NORMAL scans come from a
      different export (README Section 6), so any NORMAL-vs-disease score there is confounded.
  (b) e(x) (r = 25, fixed in advance) as one extra feature in the v5 soft vote, paired test vs v5 best.
  (c) residual maps for 2 typical scans per class (figures).

  python v6_residual.py [--quick]
Writes results/v6/residual.json, residual_maps_<task>.npz, oof_<task>_v5_best_e_x.csv
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from retina.image_rom import ImagePOD
from retina.v6data import CROSS, SLUG, TASKS
from retina.v6eval import argmax_labels, paired, residual_oof, save_oof
from retina.v6run import cross_data, cross_fit_predict, cross_summary, dump, run_models, task_data

warnings.filterwarnings("ignore")
OUT = "results/v6"
RS = [10, 25, 50]
R_FEAT = 25
AUROC_TASKS = ["OCTDL 7-class", "Kermany 4-class"]


def auroc_ci(y01, s, groups, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    ug, inv = np.unique(groups, return_inverse=True)
    idx = [np.nonzero(inv == i)[0] for i in range(len(ug))]
    v = []
    for _ in range(n):
        pick = np.concatenate([idx[i] for i in rng.integers(0, len(ug), len(ug))])
        if 0 < y01[pick].sum() < len(pick):
            v.append(roc_auc_score(y01[pick], s[pick]))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def per_class_auroc(y, e):
    return {k: float(roc_auc_score((y[(y == k) | (y == "NORMAL")] == k).astype(int), e[(y == k) | (y == "NORMAL")]))
            for k in np.unique(y) if k != "NORMAL"}


def typical(y, e, per=2):
    """Per class, the `per` scans whose e(x) is closest to the class median (not cherry-picked)."""
    out = []
    for k in np.unique(y):
        i = np.nonzero(y == k)[0]
        out += list(i[np.argsort(np.abs(e[i] - np.median(e[i])))[:per]])
    return np.array(sorted(out))


def save_maps(slug, T, y, paths, e, maps, picks, pod_all):
    filled = pod_all._fill(T[picks])
    np.savez_compressed(f"{OUT}/residual_maps_{slug}.npz", idx=picks, labels=y[picks], paths=paths[picks],
                        e=e[picks], tiles=filled.astype(np.float32),
                        resid=np.stack([maps[i] for i in picks]).astype(np.float32),
                        mean_abs_resid=np.stack([np.abs(np.stack([maps[i] for i in np.nonzero(y == k)[0] if i in maps])).mean(0)
                                                 for k in np.unique(y)]).astype(np.float32),
                        classes=np.unique(y))


def within_auroc(task, seeds, quick):
    data = task_data(task, ("tile",), quick)
    a, b = data["layout"]["tile"]
    T, y, g = data["X"][:, a:b], data["y"], data["groups"]
    y01 = (y != "NORMAL").astype(int)
    out = {"task": task, "n": len(y), "n_normal": int((y == "NORMAL").sum()), "by_r": {}}
    for r in RS:
        aucs, e0, maps = [], None, None
        for s in seeds:
            if s == seeds[0] and r == R_FEAT:
                e, maps = residual_oof(T, y, g, s, r, n_splits=data["n_splits"], maps_for=set(range(len(y))))
                e0 = e
            else:
                e = residual_oof(T, y, g, s, r, n_splits=data["n_splits"])
                e0 = e if e0 is None else e0
            aucs.append(roc_auc_score(y01, e))
        out["by_r"][str(r)] = {"auroc": float(np.mean(aucs)), "auroc_sd": float(np.std(aucs)),
                               "auroc_ci": auroc_ci(y01, e0, g), "auroc_per_class": per_class_auroc(y, e0),
                               "median_e": {k: float(np.median(e0[y == k])) for k in np.unique(y)}}
        print(f"  {task:20s} r={r:3d} AUROC {out['by_r'][str(r)]['auroc']:.3f} {out['by_r'][str(r)]['auroc_ci']}", flush=True)
        if r == R_FEAT:
            picks = typical(y, e0)
            pod_any = ImagePOD(1).fit(T[y == "NORMAL"])          # only for mean-filling masked display pixels
            save_maps(data["slug"], T, y, data["paths"], e0, maps, picks, pod_any)
    return out


def cross_auroc():
    tr = task_data("Kermany 4-class", ("tile",))
    te = task_data("OCTDL 7-class", ("tile",))
    a, b = tr["layout"]["tile"]
    Ttr = tr["X"][:, a:b][tr["y"] == "NORMAL"]
    Tte, y, g = te["X"][:, a:b], te["y"], te["groups"]
    y01 = (y != "NORMAL").astype(int)
    out = {"task": "Kermany healthy -> OCTDL (NORMAL vs ABNORMAL)", "n_train_healthy": len(Ttr), "n_test": len(y), "by_r": {}}
    for r in RS:
        e = ImagePOD(r).fit(Ttr).residual(Tte)
        out["by_r"][str(r)] = {"auroc": float(roc_auc_score(y01, e)), "auroc_ci": auroc_ci(y01, e, g),
                               "auroc_per_class": per_class_auroc(y, e)}
        print(f"  cross r={r} AUROC {out['by_r'][str(r)]['auroc']:.3f}", flush=True)
    return out


def classifier(seeds, quick):
    res = {}
    spec = {"v5 best + e(x)": dict(use=("feat", "resid"), r_resid=R_FEAT)}
    for task in TASKS:
        data = task_data(task, ("feat", "tile"), quick)
        r = run_models(data, spec, seeds, OUT, save=tuple(spec))
        base = pd.read_csv(f"{OUT}/oof_{data['slug']}_v5_best_soft_vote.csv")
        assert (base.path.values == data["paths"]).all()
        r["paired_vs_v5"] = paired(data["y"], base.pred.values, data["preds"]["v5 best + e(x)"][0], data["groups"])
        print(f"  paired e(x): {r['paired_vs_v5']}", flush=True)
        res[task] = r
    data = cross_data(("feat", "tile"))
    _, P, classes = cross_fit_predict(data, spec["v5 best + e(x)"])
    p = argmax_labels(P, classes)
    save_oof(f"{OUT}/oof_{SLUG[CROSS]}_v5_best_e_x.csv", data["paths"], data["yte"], P, classes)
    base = pd.read_csv(f"{OUT}/oof_{SLUG[CROSS]}_v5_best_soft_vote.csv")
    res[CROSS] = {"v5 best + e(x)": cross_summary(data["yte"], p, data["gte"]),
                  "paired_vs_v5": paired(data["yte"], base.pred.values, p, data["gte"])}
    print(f"  cross v5+e(x) {res[CROSS]}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--part", default="all", choices=["all", "auroc", "clf"])
    a = ap.parse_args()
    seeds = (0,) if a.quick else (0, 1, 2)
    out = {}
    if a.part in ("all", "auroc"):
        out["auroc_within"] = {t: within_auroc(t, seeds, a.quick) for t in AUROC_TASKS}
        out["auroc_cross"] = cross_auroc()
        out["octid_note"] = "skipped: OCTID NORMAL scans come from a different export (scanner confound)"
        dump(out, f"{OUT}/residual_auroc.json")
    if a.part in ("all", "clf"):
        dump(classifier(seeds, a.quick), f"{OUT}/residual_clf.json")


if __name__ == "__main__":
    main()
