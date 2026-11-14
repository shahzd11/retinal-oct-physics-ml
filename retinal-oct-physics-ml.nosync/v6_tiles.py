"""v6 / cache: flattened-retina tiles for every scan (and blurred variants for test-time augmentation).

  python v6_tiles.py octid octdl kermany                   # tiles  -> data/cache/v6/tiles_<ds>.npz
  python v6_tiles.py octdl --blur 0.5 1 --features         # blurred tiles + recomputed v2/NDT/fractal
                                                           #   features of the blurred images (Step 5)
Row order = results/v4/<ds>_features.csv. data/ is never committed.
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import warnings

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from retina.image_rom import flat_tile
from retina.preprocess import load_gray
from retina.v6data import CACHE, cache_path, load_features, manifest


def _args(row):
    icon = row["icon_box"]
    icon = tuple(float(v) for v in icon.split(",")) if isinstance(icon, str) and icon else None
    ax = row["axial_um"]
    ax = float(ax) if ax is not None and np.isfinite(ax) else None
    return ax, icon


def load_blurred(path, blur):
    g = load_gray(path)
    if blur:
        g = cv2.GaussianBlur(g, (0, 0), blur)
    return g


def one(row, blur, features):
    warnings.filterwarnings("ignore")
    ax, icon = _args(row)
    g = load_blurred(row["path"], blur)
    out = {"path": row["path"]}
    try:
        t, v, info = flat_tile(g, ax, icon)
        out.update(tile=np.clip(t * 255 + 0.5, 0, 255).astype(np.uint8), valid=v, info=info, ok=True)
    except Exception as e:                         # keep the row; an all-invalid tile
        out.update(tile=np.zeros((128, 512), np.uint8), valid=np.zeros((128, 512), bool),
                   info={"error": repr(e)[:200]}, ok=False)
    if features:
        from retina.features_ndt import ndt_features
        from retina.features_v2 import extract_v2
        from retina.fractal import boundary_features
        from retina.layers import flatten
        feats = {}
        for fn in (lambda: extract_v2(g, ax, icon), lambda: ndt_features(flatten(g, ax, icon)),
                   lambda: boundary_features(flatten(g, ax, icon))):
            try:
                feats.update(fn())
            except Exception:
                pass
        out["features"] = feats
    return out


def build(ds, blur=0.0, features=False, jobs=6):
    order = load_features(ds)[["path", "label", "patient", "dataset", "orig_h", "orig_w"]]
    m = manifest(ds).set_index("path")
    rows = order.join(m[["axial_um", "icon_box"]], on="path")
    missing = [p for p in rows.path if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"{ds}: {len(missing)} images missing, e.g. {missing[:2]} - run download_data.py first")
    res = Parallel(n_jobs=jobs, verbose=1)(delayed(one)(r, blur, features) for r in rows.to_dict("records"))
    CACHE.mkdir(parents=True, exist_ok=True)
    hi = np.stack([r["tile"] for r in res])
    vb = np.packbits(np.stack([r["valid"] for r in res]).reshape(len(res), -1), axis=1)
    np.savez_compressed(cache_path(ds, blur), paths=np.array(rows.path), hi=hi, valid_bits=vb,
                        info=np.array([r["info"] for r in res], dtype=object))
    print(ds, f"blur={blur}", "tiles", hi.shape, "failed", sum(not r["ok"] for r in res), "->", cache_path(ds, blur))
    if features:
        f = pd.DataFrame([{"path": r["path"], **r["features"]} for r in res])
        f = order.merge(f, on="path", how="left")
        out = CACHE / f"features_{ds}_blur{blur:g}.csv"
        f.to_csv(out, index=False)
        print("features", f.shape, "->", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+", choices=["octid", "octdl", "kermany"])
    ap.add_argument("--blur", type=float, nargs="*", default=[0.0])
    ap.add_argument("--features", action="store_true", help="also recompute v2/NDT/fractal features (blurred TTA)")
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    for ds in a.datasets:
        for b in a.blur:
            build(ds, b, a.features, a.jobs)


if __name__ == "__main__":
    main()
