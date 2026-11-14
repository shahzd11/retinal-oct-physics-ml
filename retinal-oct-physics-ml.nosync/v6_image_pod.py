"""v6 / Step 1: image-level POD ("eigen-retinas") on the RPE-flattened retina.

For every task: classifier on the POD coefficients (LR, LightGBM) for r in {10, 25, 50, 100, 200},
then v5 best (soft vote on the 171 v5 features) with and without the r = 50 image-POD coefficients,
on identical patient-grouped folds (seeds 0, 1, 2), with a paired patient-bootstrap test.
r = 50 for the concatenation is fixed in advance (not chosen on the test folds).

  python v6_image_pod.py                    # all tasks + cross-scanner
  python v6_image_pod.py --task octid_dis750 --quick     # 1 seed, 2 folds
Needs data/cache/v6/tiles_*.npz (python v6_tiles.py octid octdl kermany).
Writes results/v6/image_pod_<task>.json, oof_<task>_*.csv, image_pod_*_<task>.png/.npz
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import argparse
import warnings

import numpy as np

from retina.ensemble import gbm, lr
from retina.image_rom import ImagePOD, LO
from retina.v6data import CROSS, SLUG, TASKS
from retina.v6eval import argmax_labels, paired, pod_sweep_oof, save_oof, summarise
from retina.v6run import cross_data, cross_fit_predict, cross_summary, dump, run_models, task_data

warnings.filterwarnings("ignore")
OUT = "results/v6"
RS = [10, 25, 50, 100, 200]
R_CONCAT = 50
SPECS = {"v5 best (soft vote)": dict(use=("feat",)),
         "v5 best + image-POD r=50": dict(use=("feat", "pod"), r_pod=R_CONCAT)}
POD_MODELS = {"LR": lr, "LightGBM": gbm}


def describe(T, y, slug, n_modes=8):
    """Descriptive POD of all scans of the task (for figures only, never for evaluation)."""
    pod = ImagePOD(200).fit(T)
    classes = np.unique(y)
    means = np.stack([np.nanmean(T[y == k], 0) for k in classes])
    rec = pod.reconstruct(means, r=R_CONCAT)
    np.savez_compressed(f"{OUT}/image_pod_desc_{slug}.npz", energy=pod.energy_, sigma=pod.sigma_,
                        mean=pod.mean_.astype(np.float32), modes=pod.modes_[:n_modes].astype(np.float32),
                        classes=classes, class_means=means.astype(np.float32), class_rec=rec.astype(np.float32),
                        shape=np.array(LO))
    return {"energy_at": {str(r): float(pod.energy_[r - 1]) for r in RS},
            "modes_for_90": int(np.searchsorted(pod.energy_, 0.90) + 1) if pod.energy_[-1] >= 0.90 else None,
            "modes_for_95": int(np.searchsorted(pod.energy_, 0.95) + 1) if pod.energy_[-1] >= 0.95 else None}


def within(task, seeds, quick):
    data = task_data(task, ("feat", "tile"), quick)
    a, b = data["layout"]["tile"]
    T, y, g = data["X"][:, a:b], data["y"], data["groups"]
    out = {"task": task, "n": len(y), "classes": list(np.unique(y)), "seeds": list(seeds),
           "n_splits": data["n_splits"], "describe": describe(T, y, data["slug"])}
    sweep = {m: {r: [] for r in RS} for m in POD_MODELS}
    for s in seeds:
        P, classes = pod_sweep_oof(T, y, g, s, RS, POD_MODELS, data["n_splits"])
        for m in POD_MODELS:
            for r in RS:
                sweep[m][r].append(argmax_labels(P[m][r], classes))
                if s == seeds[0] and r == R_CONCAT:
                    save_oof(f"{OUT}/oof_{data['slug']}_image_pod_r50_{m.lower()}.csv", data["paths"], y,
                             P[m][r], classes)
    out["pod_sweep"] = {m: {str(r): summarise(y, sweep[m][r], g) for r in RS} for m in POD_MODELS}
    for m in POD_MODELS:
        print(f"  {task:32s} image-POD {m:8s} " + "  ".join(
            f"r={r}: {out['pod_sweep'][m][str(r)]['bacc']:.3f}" for r in RS), flush=True)
    out["models"] = run_models(data, SPECS, seeds, OUT, save=tuple(SPECS))
    names = list(SPECS)
    out["paired_pod_vs_v5"] = paired(y, data["preds"][names[0]][0], data["preds"][names[1]][0], g)
    out["paired_pod_vs_v5"]["diff_per_seed"] = [
        out["models"][names[1]]["bacc_per_seed"][i] - out["models"][names[0]]["bacc_per_seed"][i]
        for i in range(len(seeds))]
    print(f"  paired (v5+POD) - v5: {out['paired_pod_vs_v5']}", flush=True)
    dump(out, f"{OUT}/image_pod_{data['slug']}.json")


def cross(seeds):
    data = cross_data(("feat", "tile"))
    a, b = data["layout"]["tile"]
    Ttr, Tte = data["Xtr"][:, a:b], data["Xte"][:, a:b]
    y, g = data["yte"], data["gte"]
    out = {"task": CROSS, "n_train": len(data["ytr"]), "n_test": len(y), "pod_sweep": {}, "models": {}}
    pod = ImagePOD(max(RS)).fit(Ttr)
    Atr, Ate = pod.transform(Ttr), pod.transform(Tte)
    for m, fn in POD_MODELS.items():
        out["pod_sweep"][m] = {}
        for r in RS:
            p = fn(0).fit(Atr[:, :r], data["ytr"]).predict(Ate[:, :r])
            out["pod_sweep"][m][str(r)] = cross_summary(y, p, g)
        print(f"  {CROSS} image-POD {m}: " + "  ".join(
            f"r={r}: {out['pod_sweep'][m][str(r)]['bacc']:.3f}" for r in RS), flush=True)
    preds = {}
    for name, kw in SPECS.items():
        _, P, classes = cross_fit_predict(data, kw, seed=0)
        preds[name] = argmax_labels(P, classes)
        out["models"][name] = cross_summary(y, preds[name], g)
        save_oof(f"{OUT}/oof_{SLUG[CROSS]}_{name_slug(name)}.csv", data["paths"], y, P, classes)
        print(f"  {CROSS} {name:30s} bacc {out['models'][name]['bacc']:.3f}", flush=True)
    n = list(SPECS)
    out["paired_pod_vs_v5"] = paired(y, preds[n[0]], preds[n[1]], g)
    dump(out, f"{OUT}/image_pod_{SLUG[CROSS]}.json")


def name_slug(name):
    from retina.v6run import model_slug
    return model_slug(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="all", help="task slug (see retina.v6data.SLUG), 'cross' or 'all'")
    ap.add_argument("--quick", action="store_true", help="1 seed, 2 folds (smoke test)")
    a = ap.parse_args()
    seeds = (0,) if a.quick else (0, 1, 2)
    os.makedirs(OUT, exist_ok=True)
    for t in TASKS:
        if a.task in ("all", SLUG[t]):
            within(t, seeds, a.quick)
    if a.task in ("all", "cross"):
        cross(seeds)


if __name__ == "__main__":
    main()
