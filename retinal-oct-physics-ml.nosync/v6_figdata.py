"""v6 / figure data that needs the images: small arrays written to results/v6/figdata.npz so that
make_figures.py reads only results/ (and runs without the datasets).

  * one example scan per OCTID class: raw, the paper's 750x500 normalised image, the flattened canvas
    with ILM / RPE / band boundaries / sector edges
  * one raw example per class for Kermany and OCTDL
  * KT-1 concept: 1/N_max vs sigma^2 for a healthy and a diseased 50x50 cell (retina/features.py)
  * Hurst concept: NMSID(k) for a smooth and a rough tissue cell
  * the paper pipeline's upsampling factor for every OCTID scan (confound scatter)

  python v6_figdata.py
"""
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from retina.features import SIGMAS, K_MAX, _w2, hurst, nmsid_curves
from retina.layers import R0, SECTORS, flatten
from retina.preprocess import WHITE_THRESH, load_gray, micro_slice, normalize
from retina.v6data import load_features, manifest

warnings.filterwarnings("ignore")
OUT = Path("results/v6")


def small(a, w=420):
    h = int(round(a.shape[0] * w / a.shape[1]))
    return cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)


def upsampling_factor(raw):
    """Scale applied by retina.preprocess.normalize: 750 / (width after bbox + 3:2 crop)."""
    ys, xs = np.nonzero(raw < WHITE_THRESH)
    h, w = (ys.max() - ys.min() + 1, xs.max() - xs.min() + 1) if len(ys) else raw.shape
    if w / h > 1.5:
        w = int(round(h * 1.5))
    return 750.0 / w


def icon_of(row):
    return tuple(float(v) for v in row.icon_box.split(",")) if isinstance(row.icon_box, str) and row.icon_box else None


def axial(row):
    return float(row.axial_um) if np.isfinite(row.axial_um) else None


def kt1_curve(cell):
    from scipy.ndimage import gaussian_filter
    x = np.array([_w2(s) for s in SIGMAS])
    y = np.array([1.0 / max(np.percentile(gaussian_filter(cell, s, mode="reflect"), 99), 0.05) for s in SIGMAS])
    slope, icpt = np.polyfit(x, y, 1)
    return x, y, slope, icpt


def tissue_cell(norm):
    """The 50x50 cell of the central column (col 7) with the highest mean (= retinal tissue)."""
    grids = micro_slice(norm).astype(np.float64) / 255
    col = [r * 15 + 7 for r in range(10)]
    g = col[int(np.argmax([grids[i].mean() for i in col]))]
    return g, grids[g]


def main():
    out = {}
    man = manifest("octid")
    feats = load_features("octid")
    # ---------------- OCTID examples (disease scans from the 500x750 export)
    for c in ["NORMAL", "CSR", "DR", "MH"]:
        d = feats[(feats.label == c) & ((c == "NORMAL") | ((feats.orig_h == 500) & (feats.orig_w == 750)))]
        p = sorted(d.path)[0]
        row = man.set_index("path").loc[p]
        raw = load_gray(p)
        f = flatten(raw, axial(row), icon_of(row))
        norm = normalize(raw)
        out[f"ex_{c}_path"] = np.array(p)
        out[f"ex_{c}_raw"] = small(raw)
        out[f"ex_{c}_norm"] = norm
        out[f"ex_{c}_flat"] = (f.img * 255).astype(np.uint8)
        out[f"ex_{c}_ilm"] = np.where(f.col_ok, f.ilm, np.nan)
        out[f"ex_{c}_rpe"] = np.where(f.col_ok, f.rpe, np.nan)
        out[f"ex_{c}_fovea"] = np.array(f.fovea)
        # ---------------- KT-1 concept (healthy NORMAL cell vs diseased cell)
        g, cell = tissue_cell(norm)
        x, y, s, i0 = kt1_curve(cell)
        out[f"kt1_{c}"] = np.array([*x, *y, s, i0, g])
    out["R0"] = np.array(R0)
    out["sectors"] = np.array([[a, b] for a, b in SECTORS.values()])
    # ---------------- Hurst concept: smooth vs rough tissue cells of the NORMAL example
    grids = micro_slice(out["ex_NORMAL_norm"]).astype(np.float64) / 255
    tissue = np.nonzero(grids.reshape(150, -1).mean(1) > 0.25)[0]
    H = hurst(grids[tissue])
    rough, smooth = tissue[np.argmin(H)], tissue[np.argmax(H)]
    curves = nmsid_curves(grids[[smooth, rough]])
    out["hurst_k"] = np.arange(1, K_MAX + 1)
    out["hurst_curves"] = curves
    out["hurst_H"] = np.array([H[np.argmax(H)], H[np.argmin(H)]])
    out["hurst_cells"] = np.stack([grids[smooth], grids[rough]]).astype(np.float32)
    out["hurst_cell_ids"] = np.array([smooth, rough])
    # ---------------- Kermany / OCTDL gallery
    for ds in ("kermany", "octdl"):
        m = manifest(ds)
        for c in sorted(m.label.unique()):
            p = sorted(m.path[m.label == c])[0]
            out[f"gal_{ds}_{c}"] = small(load_gray(p), 320)
    # ---------------- upsampling factor per OCTID scan (confound)
    up = [upsampling_factor(load_gray(p)) for p in feats.path]
    out["octid_paths"] = feats.path.values.astype(str)
    out["octid_up"] = np.array(up)
    np.savez_compressed(OUT / "figdata.npz", **out)
    print("saved", OUT / "figdata.npz", len(out), "arrays")


if __name__ == "__main__":
    main()
