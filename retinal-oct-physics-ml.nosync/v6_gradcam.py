"""v6 / Step 4: does the deep model look where the physics looks?

Grad-CAM (Selvaraju et al. 2017) on the fine-tuned ResNet-18 fold models (each scan explained by the
fold model that did *not* train on it), for the true class:
    w_c = mean_{i,j} d y_c / d A_{ij},     CAM = ReLU( sum_c w_c A_c )   (layer4, 7x7 -> image)
The CAM is carried from the raw scan onto the RPE-flattened canvas with the *same* per-column shifts
as retina.layers.flatten, then its mass is summed inside each anatomical band (retina.layers.band_masks).

Physics side: per-class grouped permutation importance of the v5 soft vote, one group per band
(all band_<b>_* texture summaries): drop in that class's recall when the band's group is permuted
in held-out folds (seed-0 folds).

Agreement: Spearman rho across the 4 bands x K classes cells.

  python v6_gradcam.py [--task octid_dis750|kermany_4] [--quick]
Writes results/v6/gradcam_<task>.json, gradcam_examples_<task>.npz
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import argparse
import warnings

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.stats import spearmanr
from sklearn.metrics import recall_score

from retina.ensemble import soft_vote
from retina.layers import BANDS, DOWN, R0, UP, _robust_poly, band_masks, flatten, resample, trace
from retina.preprocess import load_gray
from retina.v6data import CACHE, SLUG, TASKS, manifest
from retina.v6eval import folds
from retina.v6run import dump, task_data
from v6_cnn import Scans, net_for

warnings.filterwarnings("ignore")
OUT = "results/v6"


def flat_shift(gray, axial_um, icon_box):
    """Per-column shift used by retina.layers.flatten (re-derived with the same steps)."""
    img, valid = resample(gray, axial_um, icon_box)
    H, W = img.shape
    rpe, _ = trace(img, valid)
    x = np.arange(W)
    ok = np.isfinite(rpe)
    if ok.sum() < W // 4:
        rpe[:] = np.nanmedian(rpe) if ok.any() else H * 0.7
        ok[:] = True
    coef = _robust_poly(x[ok].astype(float), rpe[ok], 2)
    return np.round(R0 - np.polyval(coef, x)).astype(int), (H, W)


def to_canvas(field, shift, shape):
    """Resample a raw-image field to the resampled grid, then apply the flatten column shifts."""
    H, W = shape
    f = cv2.resize(field.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    out = np.zeros((UP + DOWN, W), np.float32)
    for c in range(W):
        s = shift[c]
        d0, d1 = max(0, s), min(UP + DOWN, H + s)
        if d1 > d0:
            out[d0:d1, c] = f[d0 - s:d1 - s, c]
    return out


class GradCAM:
    def __init__(self, net):
        self.net, self.A, self.G = net, None, None
        net.layer4.register_forward_hook(lambda m, i, o: setattr(self, "A", o))
        net.layer4.register_full_backward_hook(lambda m, gi, go: setattr(self, "G", go[0]))

    def __call__(self, x, cls):
        self.net.zero_grad()
        out = self.net(x)
        out[torch.arange(len(cls)), cls].sum().backward()
        w = self.G.mean((2, 3), keepdim=True)
        cam = torch.relu((w * self.A).sum(1)).detach().cpu().numpy()
        return cam / (cam.reshape(len(cam), -1).max(1)[:, None, None] + 1e-12)


def cam_band_fractions(task, quick):
    data = task_data(task, (), quick)
    paths, y, g = data["paths"], data["y"], data["groups"]
    man = manifest(data["ds"]).set_index("path")
    wdir = CACHE / "cnn" / data["slug"]
    out = np.full((len(y), len(BANDS) + 1), np.nan)
    examples = {}
    for k, (tr, te) in enumerate(folds(y, g, 0, 2 if quick else 5)):
        ck = torch.load(wdir / f"fold{k}.pt", map_location="cpu", weights_only=False)
        assert (ck["test_idx"] == te).all(), "fold models must come from the same folds"
        classes = ck["classes"]
        net = net_for(len(classes)); net.load_state_dict(ck["state_dict"]); net.eval()
        gc = GradCAM(net)
        ds = Scans(paths[te], y[te], classes, False)
        for j, i in enumerate(te):
            x, c = ds[j]
            cam = gc(x[None], torch.tensor([c]))[0]
            raw = load_gray(paths[i])
            row = man.loc[paths[i]]
            ax = float(row.axial_um) if np.isfinite(row.axial_um) else None
            icon = tuple(float(v) for v in row.icon_box.split(",")) if isinstance(row.icon_box, str) and row.icon_box else None
            cam_raw = cv2.resize(cam, (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_LINEAR)
            shift, shape = flat_shift(raw, ax, icon)
            f = flatten(raw, ax, icon)
            cc = to_canvas(cam_raw, shift, shape) * f.valid
            bm = band_masks(f)
            tot = cc.sum() + 1e-12
            fr = [float(cc[bm[b]].sum() / tot) for b in BANDS]
            out[i] = fr + [1 - sum(fr)]
            if y[i] not in examples or len(examples[y[i]]) < 2:
                examples.setdefault(y[i], []).append((i, f.img, cc, raw, cam_raw))
    return data, out, examples


def physics_band_importance(data, n_rep=5, quick=False):
    from retina.v6data import feat_matrix
    X, cols = feat_matrix(data["frame"])
    y, g = data["y"], data["groups"]
    classes = np.unique(y)
    groups = {b: [i for i, c in enumerate(cols) if c.startswith(f"band_{b}_")] for b in BANDS}
    rng = np.random.default_rng(0)
    drop = {b: {k: [] for k in classes} for b in BANDS}
    for tr, te in folds(y, g, 0, 2 if quick else 5):
        m = soft_vote(0).fit(X[tr], y[tr])
        base = recall_score(y[te], m.predict(X[te]), labels=classes, average=None)
        for b, idx in groups.items():
            for _ in range(n_rep):
                Xp = X[te].copy()
                Xp[:, idx] = Xp[rng.permutation(len(te))][:, idx]
                r = recall_score(y[te], m.predict(Xp), labels=classes, average=None)
                for j, k in enumerate(classes):
                    drop[b][k].append(base[j] - r[j])
    return {b: {k: float(np.mean(v)) for k, v in d.items()} for b, d in drop.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="octid_dis750")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    task = [t for t in TASKS if SLUG[t] == a.task][0]
    data, frac, ex = cam_band_fractions(task, a.quick)
    y = data["y"]
    classes = list(np.unique(y))
    cam = {b: {k: float(np.nanmean(frac[y == k, j])) for k in classes} for j, b in enumerate(BANDS)}
    other = {k: float(np.nanmean(frac[y == k, -1])) for k in classes}
    phys = physics_band_importance(data, quick=a.quick)
    cv_ = [cam[b][k] for b in BANDS for k in classes]
    pv_ = [phys[b][k] for b in BANDS for k in classes]
    rho, p = spearmanr(cv_, pv_)
    # also within each class (ranking of the 4 bands)
    per_class = {k: float(spearmanr([cam[b][k] for b in BANDS], [phys[b][k] for b in BANDS])[0]) for k in classes}
    res = {"task": task, "bands": list(BANDS), "classes": classes, "cam_mass_fraction": cam,
           "cam_mass_outside_bands": other, "physics_band_importance": phys,
           "spearman_rho": float(rho), "spearman_p": float(p), "n_cells": len(cv_),
           "spearman_rho_per_class_over_bands": per_class}
    dump(res, f"{OUT}/gradcam_{data['slug']}.json")
    sel = [e for k in classes for e in ex.get(k, [])]
    np.savez_compressed(f"{OUT}/gradcam_examples_{data['slug']}.npz",
                        idx=np.array([e[0] for e in sel]), labels=y[[e[0] for e in sel]],
                        flat=np.stack([e[1] for e in sel]).astype(np.float16),
                        cam_flat=np.stack([e[2] for e in sel]).astype(np.float16))
    print(f"{task}: Spearman rho {rho:.3f} (p={p:.3f}); per class {per_class}")
    print("CAM", cam, "\nphys", phys)


if __name__ == "__main__":
    main()
