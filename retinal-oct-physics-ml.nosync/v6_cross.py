"""v6 / Step 5: cross-scanner robustness, Kermany (Spectralis) -> OCTDL (Optovue), NORMAL / DME / AMD.

Models: v5 best (soft vote on 171 features) and the v6 hybrids (ResNet-18 = headline, DINOv2-S), each
  * without calibration, or with healthy-reference harmonisation (v5_harmonize.py, k = 30 healthy
    OCTDL patients as the "reference block"; those patients are removed from the test set; 10 draws).
    Here every block is calibrated: per-column z-scores for the features and the embedding, and a
    per-pixel z-score for the tile, each using the healthy eyes of its own scanner.
  * without / with speckle-matched test-time augmentation (TTA): the OCTDL test scans are Gaussian
    blurred (sigma in {0, 0.5, 1} px), *every* feature is recomputed from the blurred image, and
    the class probabilities of the three versions are averaged.

  python v6_cross.py
Needs: v6_tiles.py octdl --blur 0.5 1 --features ; v6_embed.py octdl --blur 0.5 1
Writes results/v6/cross.json, oof_kermany_to_octdl_<config>.csv
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, recall_score

from retina.hybrid import HybridClassifier
from retina.v6data import CACHE, SLUG, CROSS
from retina.v6eval import argmax_labels, paired, save_oof
from retina.v6run import cross_data, cross_summary, dump, model_slug

warnings.filterwarnings("ignore")
OUT = "results/v6"
BLOCKS = ("feat", "tile", "emb_resnet", "emb_dino")
HEADLINE = "v6 hybrid (ResNet-18 + physics)"
MODELS = {"v5 best (soft vote)": dict(use=("feat",)),
          HEADLINE: dict(use=("feat", "pod", "resid", "emb_resnet")),
          "v6 hybrid (DINOv2-S + physics)": dict(use=("feat", "pod", "resid", "emb_dino"))}
SIGMAS = (0.0, 0.5, 1.0)
K, DRAWS = 30, 10


def zcal(A, ref):
    with np.errstate(all="ignore"):
        mu, sd = np.nanmean(ref, 0), np.nanstd(ref, 0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-9), sd, 1.0)
    return (A - mu) / sd


def calibrate(X, ref_mask, layout):
    Z = X.copy()
    for b in ("feat", "tile", "emb_resnet", "emb_dino"):
        if b in layout:
            a, c = layout[b]
            Z[:, a:c] = zcal(X[:, a:c], X[ref_mask, a:c])
    return Z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="2 draws, no TTA")
    a = ap.parse_args()
    sig = (0.0,) if a.quick else SIGMAS
    draws = 2 if a.quick else DRAWS
    views = {}
    for s in sig:
        fdf = pd.read_csv(CACHE / f"features_octdl_blur{s:g}.csv") if s else None
        views[s] = cross_data(BLOCKS, test_blur=s, test_feat_df=fdf)
    base = views[0.0]
    y, g, ytr = base["yte"], base["gte"], base["ytr"]
    lay = base["layout"]
    healthy_pats = np.unique(g[y == "NORMAL"])
    rng = np.random.default_rng(0)
    cal_sets = [np.isin(g, rng.choice(healthy_pats, K, replace=False)) for _ in range(draws)]
    res = {"task": CROSS, "n_test": len(y), "sigmas": list(sig), "k": K, "draws": draws, "configs": {}}
    preds0 = {}
    for mname, kw in MODELS.items():
        # ---- no calibration
        m = HybridClassifier(lay, seed=0, **kw).fit(base["Xtr"], ytr)
        cls = m.classes_
        Ps = {s: m.predict_proba(views[s]["Xte"]) for s in sig}
        for tta in (False, True) if len(sig) > 1 else (False,):
            P = np.mean([Ps[s] for s in sig], 0) if tta else Ps[0.0]
            p = argmax_labels(P, cls)
            key = f"{mname} | none | {'TTA' if tta else 'no TTA'}"
            r = cross_summary(y, p, g)
            r["recall_per_class"] = dict(zip(cls, recall_score(y, p, labels=cls, average=None).round(4).tolist()))
            r["confusion"] = confusion_matrix(y, p, labels=cls).tolist()
            r["classes"] = list(cls)
            res["configs"][key] = r
            preds0[key] = p
            save_oof(f"{OUT}/oof_{SLUG[CROSS]}_{model_slug(key)}.csv", base["paths"], y, P, cls)
            print(f"  {key:60s} bacc {r['bacc']:.3f} {r['recall_per_class']}", flush=True)
        # ---- healthy-reference harmonisation (k healthy OCTDL patients), averaged over draws
        Xtr_c = calibrate(base["Xtr"], ytr == "NORMAL", lay)
        mc = HybridClassifier(lay, seed=0, **kw).fit(Xtr_c, ytr)
        for tta in (False, True) if len(sig) > 1 else (False,):
            b, recs, cms = [], [], np.zeros((len(cls), len(cls)), int)
            for cal in cal_sets:
                P = np.mean([mc.predict_proba(calibrate(views[s]["Xte"], cal, lay)[~cal]) for s in (sig if tta else (0.0,))], 0)
                p = argmax_labels(P, mc.classes_)
                b.append(cross_summary(y[~cal], p, g[~cal])["bacc"])
                recs.append(recall_score(y[~cal], p, labels=cls, average=None))
                cms += confusion_matrix(y[~cal], p, labels=cls)
            key = f"{mname} | healthy-ref k={K} | {'TTA' if tta else 'no TTA'}"
            res["configs"][key] = {"bacc": float(np.mean(b)), "bacc_sd": float(np.std(b)),
                                   "recall_per_class": dict(zip(cls, np.mean(recs, 0).round(4).tolist())),
                                   "confusion_summed_over_draws": cms.tolist(), "classes": list(cls)}
            print(f"  {key:60s} bacc {np.mean(b):.3f} ± {np.std(b):.3f}", flush=True)
    # paired tests (uncalibrated configs, same test scans)
    k0 = list(preds0)
    hybs = [k for k in k0 if k.startswith("v6") and k.endswith("| no TTA")]
    v5 = [k for k in k0 if k.startswith("v5") and k.endswith("| no TTA")][0]
    res["headline"] = HEADLINE
    res["paired"] = {f"{h} vs {v5}": paired(y, preds0[v5], preds0[h], g) for h in hybs}
    for k in k0:
        if k.endswith("| TTA"):
            res["paired"][f"{k} vs no TTA"] = paired(y, preds0[k.replace("| TTA", "| no TTA")], preds0[k], g)
    cnn = f"{OUT}/oof_{SLUG[CROSS]}_cnn.csv"
    if os.path.exists(cnn):
        c = pd.read_csv(cnn)
        assert (c.path.values == base["paths"]).all()
        for h in hybs:
            res["paired"][f"{h} vs CNN"] = paired(y, c.pred.values, preds0[h], g)
    best = max(res["configs"], key=lambda k: res["configs"][k]["bacc"])
    res["best_config"] = best
    print("best:", best, res["configs"][best]["bacc"])
    dump(res, f"{OUT}/cross.json")


if __name__ == "__main__":
    main()
