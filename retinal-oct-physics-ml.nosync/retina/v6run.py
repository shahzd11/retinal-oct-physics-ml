"""v6 task runner shared by the v6_*.py scripts.

    data = task_data(task, blocks=("feat", "tile"))       # within-scanner task (patient-grouped CV)
    res  = run_models(data, {"v5 vote": dict(use=("feat",)), ...}, seeds=(0, 1, 2))
    data = cross_data(blocks=...)                           # Kermany (train) -> OCTDL (test)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from retina.hybrid import HybridClassifier, assemble
from retina.image_rom import to_lo
from retina.v6data import CROSS, HARMONISE, SLUG, TASKS, feat_matrix, load_emb, load_features, load_tiles
from retina.v6eval import argmax_labels, oof_proba, paired, save_oof, summarise

EMB = {"emb_resnet": "resnet18", "emb_dino": "dinov2s"}


def _blocks(ds, rows, d, blocks, blur=0.0, tiles=None, feat_df=None, cols=None):
    out = {}
    if "rowid" in blocks:
        out["rowid"] = np.arange(len(rows))
    out["group"] = pd.factorize(d.patient)[0]
    if "tile" in blocks:
        t = tiles if tiles is not None else load_tiles(ds, blur)
        out["tile"] = to_lo(t["hi"][rows] / 255.0, t["valid"][rows])
    if "feat" in blocks:
        out["feat"] = feat_matrix(feat_df if feat_df is not None else d, cols)[0]
    for b, net in EMB.items():
        if b in blocks:
            out[b] = load_emb(ds, net, blur)[rows]
    return out


def task_data(task, blocks=("feat", "tile"), quick=False):
    ds, sel = TASKS[task]
    df = load_features(ds)
    m = sel(df).values
    rows = np.nonzero(m)[0]
    d = df[m].reset_index(drop=True)
    tiles = load_tiles(ds) if ("tile" in blocks or "ref" in blocks) else None
    X, layout = assemble(**_blocks(ds, rows, d, blocks, tiles=tiles))
    ref = None
    if "NORMAL" not in set(d.label):                      # external healthy reference for e(x)
        nr = np.nonzero((df.label == "NORMAL").values)[0]
        if len(nr) and tiles is not None:
            ref = to_lo(tiles["hi"][nr] / 255.0, tiles["valid"][nr])
    return {"task": task, "slug": SLUG[task], "ds": ds, "paths": d.path.values, "y": d.label.values,
            "groups": d.patient.values, "X": X, "layout": layout, "ref_tiles": ref, "frame": d,
            "n_splits": 2 if quick else 5}


def run_models(data, specs, seeds=(0, 1, 2), out_dir=None, save=(), log=print):
    """specs: {name: HybridClassifier kwargs}. Returns {name: summary} and keeps seed-wise preds
    in data['preds'][name] (list of label arrays) and seed-0 probabilities in data['P'][name]."""
    y, g = data["y"], data["groups"]
    data.setdefault("preds", {}); data.setdefault("P", {})
    res = {}
    for name, kw in specs.items():
        kw = dict(kw)
        if "resid" in kw.get("use", ()) and data.get("ref_tiles") is not None:
            kw.setdefault("ref_tiles", data["ref_tiles"])
        preds = []
        for s in seeds:
            fn = lambda seed, kw=kw: HybridClassifier(data["layout"], seed=seed, **kw)
            P, classes = oof_proba(fn, data["X"], y, g, s, data["n_splits"])
            preds.append(argmax_labels(P, classes))
            if s == seeds[0]:
                data["P"][name] = (P, classes)
        data["preds"][name] = preds
        res[name] = summarise(y, preds, g)
        log(f"  {data['task']:32s} {name:34s} bacc {res[name]['bacc']:.3f} "
            f"[{res[name]['bacc_ci'][0]:.3f}, {res[name]['bacc_ci'][1]:.3f}] acc {res[name]['acc']:.3f} "
            f"F1 {res[name]['macro_f1']:.3f}")
        if out_dir is not None and name in save:
            P, classes = data["P"][name]
            save_oof(Path(out_dir) / f"oof_{data['slug']}_{model_slug(name)}.csv", data["paths"], y, P, classes)
    return res


def model_slug(name):
    return "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_").replace("__", "_")


def paired_vs(data, a, b):
    return paired(data["y"], data["preds"][a][0], data["preds"][b][0], data["groups"])


# ---------------------------------------------------------------------------- cross-scanner
def _harm(df):
    df = df.copy()
    df["label"] = df.label.map(HARMONISE)
    m = df.label.notna().values
    return df[m].reset_index(drop=True), np.nonzero(m)[0]


def cross_data(blocks=("feat", "tile"), test_blur=0.0, test_feat_df=None):
    """Train: all Kermany scans (labels NORMAL/DME/AMD). Test: all OCTDL scans with those labels."""
    tr_df, tr_rows = _harm(load_features("kermany"))
    te_df, te_rows = _harm(load_features("octdl"))
    fdf = None
    if test_feat_df is not None:                       # features recomputed on blurred test images
        fdf = te_df[["path"]].merge(test_feat_df, on="path", how="left")
        from retina.evaluation import add_band_summaries
        fdf = add_band_summaries(fdf)                  # (no paper-grid columns: not in the v5 set)
    from retina.evaluation import feature_sets
    from retina.v6data import FSET
    cols = feature_sets(tr_df)[FSET]                   # test columns in the training order
    Xtr, layout = assemble(**_blocks("kermany", tr_rows, tr_df, blocks, cols=cols))
    Xte, layout2 = assemble(**_blocks("octdl", te_rows, te_df, blocks, blur=test_blur, feat_df=fdf, cols=cols))
    assert layout == layout2
    return {"task": CROSS, "slug": SLUG[CROSS], "Xtr": Xtr, "ytr": tr_df.label.values, "Xte": Xte,
            "yte": te_df.label.values, "gte": te_df.patient.values, "layout": layout,
            "paths": te_df.path.values, "te_frame": te_df}


def cross_fit_predict(data, kw, seed=0):
    m = HybridClassifier(data["layout"], seed=seed, **kw).fit(data["Xtr"], data["ytr"])
    classes = m.classes_
    return m, m.predict_proba(data["Xte"]), classes


def cross_summary(y, p, g):
    from retina.evaluation import cluster_bootstrap_ci
    lo, hi = cluster_bootstrap_ci(y, p, g, "bacc")
    return {"bacc": float(balanced_accuracy_score(y, p)), "bacc_ci": [float(lo), float(hi)],
            "acc": float(accuracy_score(y, p)), "macro_f1": float(f1_score(y, p, average="macro"))}


def dump(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(obj, open(path, "w"), indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
