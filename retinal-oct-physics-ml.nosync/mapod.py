"""Model-assisted probability of detection (MAPOD) for the fluid detectors, NDT style.

Synthetic, fully anechoic fluid pockets of known size are inserted into HEALTHY scans and each
detector is scored hit / miss, exactly like flaw-insertion studies in ultrasonic NDT
(MIL-HDBK-1823A hit/miss analysis):

  lesion types   cyst     intraretinal pocket (DME / RVO cystoid fluid), round in um (assumed
                          lateral spacing), random depth, 25-110 columns from the fovea
                 cyst_px  same, but round on the display canvas (width in columns = height in
                          rows), i.e. how cysts look on screen; brackets the lateral-scale assumption
                 srf   subretinal lens on top of the RPE (CSR), width = 8 x height, within 110 columns
                       of the fovea
  size a         lesion height in um, 8 ... 160 um (log spaced)
  fill           a patch of this scan's own vitreous (keeps its speckle and noise floor),
                 blended with a 1 px soft edge
  detectors      NDT  half-max anechoic rule (retina/features_ndt.fluid_mask)
                 v2   fixed-fraction dark rule (retina/features_v2.v2_fluid_mask)
                 both keep connected components >= 20 px, as in the feature code
  hit            >= 30% of the inserted lesion's pixels are newly flagged (not flagged in the
                 clean scan)
  false calls    components flagged on the clean healthy scan

POD(a) = Phi((ln a - mu) / sigma), fitted by maximum likelihood (probit on ln a). a90/95 is the
95th percentile of a90 over a patient bootstrap (insertions in the same scan are correlated).

Assumptions: lateral spacing 6 mm / 512 columns = 11.7 um (not stored in the images); the
lesion is painted onto the flattened canvas, so retinal segmentation is taken as given and the
tissue is not displaced the way real fluid would displace it.

  python mapod.py            # 60 healthy scans per scanner (Kermany/Spectralis, OCTDL/Optovue)
Writes results/v3/mapod_hits.csv, mapod.json, mapod_pod.png, mapod_sizing.png, mapod_examples.png
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import json
import warnings
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter, label
from scipy.optimize import minimize
from scipy.stats import norm

from retina import plotstyle as ps
from retina.features_ndt import _refs, fluid_mask
from retina.features_v2 import v2_fluid_mask
from retina.layers import flatten
from retina.preprocess import load_gray

warnings.filterwarnings("ignore")
OUT = Path("results/v3")
LAT_UM = 11.7
SIZES = np.geomspace(8, 160, 14)
PLACEMENTS = 2
MIN_COMP = 20
HIT_COVER = 0.3
KINDS = {"cyst": "intraretinal cyst, round in um", "cyst_px": "intraretinal cyst, round on screen",
         "srf": "subretinal fluid"}
DATASETS = {"kermany": "Kermany (Spectralis)", "octdl": "OCTDL (Optovue)"}
DET_COLOR = {"NDT half-max": ps.SERIES[2], "v2 fixed-fraction": ps.SERIES[0]}


# ---------------------------------------------------------------------------- detectors
def detect(f, fracs=None):
    """Kept fluid components (bool masks) per detector on flattened scan f.
    fracs: {detector: [threshold fractions]} for a sweep; result keys are then (detector, frac)."""
    sm = cv2.GaussianBlur(f.img, (0, 0), sigmaX=3, sigmaY=1)
    sm[~f.valid] = np.nan
    vit = _refs(f, np.nan_to_num(sm, nan=0.0))[0]
    sweep = fracs is not None
    fracs = fracs or {"NDT half-max": [0.4], "v2 fixed-fraction": [0.2]}
    smz = np.nan_to_num(sm, nan=vit)

    Hc = f.img.shape[0]
    rows = np.arange(Hc)[:, None]
    ret = (rows > np.where(f.col_ok, f.ilm, np.inf)[None]) & \
          (rows < np.where(f.col_ok, f.rpe - 4, -np.inf)[None]) & f.valid
    sm2 = gaussian_filter(f.img, 2)
    darks = {}
    for fr in fracs["NDT half-max"]:
        darks[("NDT half-max", fr)] = fluid_mask(f, smz, vit, frac=fr)[1]
    for fr in fracs["v2 fixed-fraction"]:
        darks[("v2 fixed-fraction", fr)] = v2_fluid_mask(f, ret, sm2, fr) if ret.sum() > 200 else np.zeros_like(ret)

    out = {}
    for name, dark in darks.items():
        lab, n = label(dark)
        if n:
            sizes = np.bincount(lab.ravel())
            sizes[0] = 0
            keep = sizes >= MIN_COMP
            out[name] = (keep[lab], int(keep.sum()))
        else:
            out[name] = (np.zeros_like(dark), 0)
    return out if sweep else {k[0]: v for k, v in out.items()}


# ---------------------------------------------------------------------------- insertion
def vitreous_patch(f, h, w, rng):
    """A real h x w patch of vitreous from this scan (fallback: shuffled vitreous pixels)."""
    ilm = np.where(f.col_ok, f.ilm, np.nan)
    W = f.img.shape[1]
    for _ in range(30):
        c0 = int(rng.integers(0, max(1, W - w)))
        top = np.nanmin(ilm[c0:c0 + w]) if np.isfinite(ilm[c0:c0 + w]).any() else np.nan
        if np.isfinite(top) and top - 6 >= h + 2 and f.valid[2:int(top) - 6, c0:c0 + w].all():
            r0 = int(rng.integers(2, int(top) - 6 - h + 1))
            return f.img[r0:r0 + h, c0:c0 + w].copy()
    vit = (np.arange(f.img.shape[0])[:, None] < np.where(f.col_ok, f.ilm - 6, -np.inf)[None]) & f.valid
    pool = f.img[vit] if vit.sum() > 50 else np.array([np.percentile(f.img[f.valid], 2)])
    return gaussian_filter(rng.choice(pool, (h, w)), 0.6)


def lesion_mask(f, kind, a_um, um, rng):
    """Boolean lesion mask on the canvas, or None if it does not fit this retina."""
    Hc, W = f.img.shape
    h = a_um / um
    rows = np.arange(Hc)[:, None]
    cols = np.arange(W)[None]
    for _ in range(25):
        if kind in ("cyst", "cyst_px"):
            w = max(a_um / LAT_UM, 1.0) if kind == "cyst" else h
            c = f.fovea + rng.choice([-1, 1]) * rng.uniform(25, 110)
            span = np.arange(int(np.floor(c - w / 2)) - 1, int(np.ceil(c + w / 2)) + 2)
            if span.min() < 0 or span.max() >= W or not f.col_ok[span].all():
                continue
            top, bot = np.max(f.ilm[span]) + 4, np.min(f.rpe[span]) - 8
            if bot - top < h + 1:
                continue
            r = rng.uniform(top + h / 2, bot - h / 2)
            m = ((rows - r) / (h / 2)) ** 2 + ((cols - c) / (w / 2)) ** 2 <= 1
        else:  # srf: lens sitting on the RPE
            w = max(8 * a_um / LAT_UM, 3.0)
            c = f.fovea + rng.uniform(-110, 110)
            span = np.arange(int(np.floor(c - w / 2)), int(np.ceil(c + w / 2)) + 1)
            if span.min() < 0 or span.max() >= W or not f.col_ok[span].all():
                continue
            base = f.rpe[span] - 7
            if np.any(base - h < f.ilm[span] + 4):
                continue
            m = np.zeros((Hc, W), bool)
            x = (span - c) / (w / 2)
            hh = h * np.clip(1 - x ** 2, 0, None)
            for cc, b, t in zip(span, base, hh):
                if t >= 0.5:
                    m[int(round(b - t)):int(round(b)) + 1, cc] = True
        if m.sum() >= 3:
            return m
    return None


def insert(f, m, rng):
    rr, cc = np.nonzero(m)
    r0, r1, c0, c1 = rr.min() - 2, rr.max() + 3, cc.min() - 2, cc.max() + 3
    r0, c0 = max(r0, 0), max(c0, 0)
    r1, c1 = min(r1, f.img.shape[0]), min(c1, f.img.shape[1])
    patch = vitreous_patch(f, r1 - r0, c1 - c0, rng)
    alpha = gaussian_filter(m.astype(np.float32), 0.8)[r0:r1, c0:c1]
    g = f.img.copy()
    g[r0:r1, c0:c1] = g[r0:r1, c0:c1] * (1 - alpha) + patch * alpha
    return type(f)(g, f.valid, f.ilm, f.rpe, f.rpe_resid, f.fovea, f.col_ok, f.scale_known)


def run_scan(row, seed):
    rng = np.random.default_rng(seed)
    um = float(row.axial_um)
    f = flatten(load_gray(row.path), um, None)
    clean = detect(f)
    recs = [dict(dataset=row.dataset_key, patient=row.patient, kind="clean", detector=d, a_um=0.0,
                 hit=np.nan, cover=np.nan, meas_um=np.nan, false_calls=n) for d, (_, n) in clean.items()]
    for kind in KINDS:
        for a in SIZES:
            for _ in range(PLACEMENTS):
                m = lesion_mask(f, kind, a, um, rng)
                if m is None:
                    continue
                det = detect(insert(f, m, rng))
                for d, (k, _) in det.items():
                    new = k & ~clean[d][0]
                    cover = float((new & m).sum() / m.sum())
                    lab, _ = label(k)
                    ids = np.unique(lab[m & k])
                    ids = ids[ids > 0]
                    if len(ids):
                        rr = np.nonzero(np.isin(lab, ids))[0]
                        meas = float((rr.max() - rr.min() + 1) * um)
                    else:
                        meas = np.nan
                    recs.append(dict(dataset=row.dataset_key, patient=row.patient, kind=kind, detector=d,
                                     a_um=float(a), hit=float(cover >= HIT_COVER), cover=cover,
                                     meas_um=meas, false_calls=np.nan))
    return recs


# ---------------------------------------------------------------------------- hit/miss POD
def fit_probit(a, y, x0=(-3.0, 1.0)):
    x = np.log(a)

    def nll(b):
        p = np.clip(norm.cdf(b[0] + b[1] * x), 1e-9, 1 - 1e-9)
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

    b = minimize(nll, list(x0), method="L-BFGS-B").x
    return -b[0] / b[1], 1 / b[1]


def a_at(mu, sigma, p):
    return float(np.exp(mu + norm.ppf(p) * sigma))


def pod_summary(d, rng, n_boot=400):
    mu, sigma = fit_probit(d.a_um.values, d.hit.values)
    pats = d.patient.unique()
    by = {p: g for p, g in d.groupby("patient")}
    a90b = []
    for _ in range(n_boot):
        s = pd.concat([by[p] for p in rng.choice(pats, len(pats))])
        m_, s_ = fit_probit(s.a_um.values, s.hit.values, (-mu / sigma, 1 / sigma))
        if s_ > 0:
            a90b.append(a_at(m_, s_, 0.9))
    return dict(mu=mu, sigma=sigma, a50=a_at(mu, sigma, 0.5), a90=a_at(mu, sigma, 0.9),
                a90_95=float(np.percentile(a90b, 95)), n=int(len(d)), n_scans=int(len(pats)))


# ---------------------------------------------------------------------------- figures
def examples_figure(man):
    rng = np.random.default_rng(11)
    fig, axes = plt.subplots(2, 3, figsize=(16, 6.2))
    for i, (ds, kind) in enumerate([("kermany", "cyst"), ("kermany", "cyst_px"), ("kermany", "srf"),
                                    ("octdl", "cyst"), ("octdl", "cyst_px"), ("octdl", "srf")]):
        ax = axes.ravel()[i]
        a = [40, 40, 40, 100, 100, 100][i]
        m = None
        while m is None:
            row = man[ds].iloc[int(rng.integers(len(man[ds])))]
            um = float(row.axial_um)
            f = flatten(load_gray(row.path), um, None)
            m = lesion_mask(f, kind, a, um, rng)
        g = insert(f, m, rng)
        det = detect(g)
        ax.imshow(g.img, cmap="gray", vmin=0, vmax=1, aspect="auto")
        ov = np.zeros((*g.img.shape, 4))
        ov[det["NDT half-max"][0]] = (0.11, 0.69, 0.48, 0.55)
        ax.imshow(ov, aspect="auto")
        ax.contour(m, levels=[0.5], colors=[ps.SERIES[1]], linewidths=0.8)
        rr, cc = np.nonzero(m)
        pad = 60
        ax.set_xlim(max(cc.min() - 2 * pad, 0), min(cc.max() + 2 * pad, g.img.shape[1] - 1))
        ax.set_ylim(min(rr.max() + pad, g.img.shape[0] - 1), max(rr.min() - pad, 0))
        ax.set_title(f"{DATASETS[ds]}  {kind}, {a} um", fontsize=10, loc="left")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Synthetic lesions in healthy scans: orange outline = inserted lesion, green = NDT half-max detection",
                 fontsize=11)
    plt.tight_layout(); plt.savefig(OUT / "mapod_examples.png", dpi=130); plt.close()


def pod_figure(hits, summ):
    fig, axes = plt.subplots(len(KINDS), 2, figsize=(13, 12.5), sharex=True, sharey=True)
    xs = np.geomspace(SIZES[0], SIZES[-1], 200)
    for i, kind in enumerate(KINDS):
        for j, ds in enumerate(DATASETS):
            ax = axes[i, j]
            for det, col in DET_COLOR.items():
                d = hits[(hits.kind == kind) & (hits.dataset == ds) & (hits.detector == det)]
                emp = d.groupby("a_um").hit.mean()
                ax.scatter(emp.index, emp.values, color=col, s=36, edgecolor=ps.SURFACE, linewidth=1.5, zorder=3)
                s = summ[f"{ds}|{kind}|{det}"]
                ax.plot(xs, norm.cdf((np.log(xs) - s["mu"]) / s["sigma"]), color=col, lw=2,
                        label=f"{det}: a90 = {s['a90']:.0f} um (a90/95 {s['a90_95']:.0f})")
            ax.axhline(0.9, color=ps.INK2, lw=0.8, ls=":")
            ax.set_xscale("log")
            ax.set_xticks([10, 20, 40, 80, 160], ["10", "20", "40", "80", "160"])
            ax.set_ylim(-0.03, 1.03)
            ax.set_title(f"{DATASETS[ds]}: {KINDS[kind]}", loc="left")
            fc = summ[f"{ds}|false_calls"]
            ax.legend(frameon=False, loc="upper left", fontsize=8.5,
                      title=f"healthy scans with a false call: NDT {fc['NDT half-max']:.0%}, "
                            f"v2 {fc['v2 fixed-fraction']:.0%}", title_fontsize=8)
            if i == len(KINDS) - 1:
                ax.set_xlabel("Lesion height a (um, log scale)")
            if j == 0:
                ax.set_ylabel("Probability of detection")
    fig.suptitle("Hit/miss POD curves from synthetic fluid insertion (dots = observed hit rate, line = probit fit)",
                 weight="bold")
    plt.tight_layout(); plt.savefig(OUT / "mapod_pod.png", dpi=140); plt.close()


def sizing_figure(hits):
    fig, axes = plt.subplots(1, len(KINDS), figsize=(16, 5.2), sharey=True)
    for ax, kind in zip(axes, KINDS):
        for det, col in DET_COLOR.items():
            d = hits[(hits.kind == kind) & (hits.detector == det) & (hits.hit == 1)]
            g = d.groupby("a_um").meas_um.median()
            q = d.groupby("a_um").meas_um.quantile([0.25, 0.75]).unstack()
            ax.fill_between(g.index, q[0.25], q[0.75], color=col, alpha=0.18, lw=0)
            ax.plot(g.index, g.values, color=col, lw=2, marker="o", ms=5, label=det)
        ax.plot(SIZES, SIZES, color=ps.INK2, lw=1, ls="--", label="perfect sizing")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks([10, 20, 40, 80, 160], ["10", "20", "40", "80", "160"])
        ax.set_title(KINDS[kind][0].upper() + KINDS[kind][1:], loc="left")
        ax.set_xlabel("True lesion height (um)")
        ax.legend(frameon=False, fontsize=9)
    axes[0].set_ylabel("Measured height of the detected pocket (um)\nmedian and IQR over detected lesions")
    fig.suptitle("Sizing accuracy: height of the flagged component vs the inserted height", weight="bold")
    plt.tight_layout(); plt.savefig(OUT / "mapod_sizing.png", dpi=140); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=60)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--figs-only", action="store_true")
    args = ap.parse_args()
    ps.apply()
    man = {}
    for ds in DATASETS:
        m = pd.read_csv(f"manifests/{ds}.csv")
        m = m[m.label == "NORMAL"].groupby("patient").head(1).sample(frac=1, random_state=0)
        man[ds] = m.head(args.per).assign(dataset_key=ds).reset_index(drop=True)
    examples_figure(man)
    hits_path = OUT / "mapod_hits.csv"
    if not args.figs_only:
        rows = [r for ds in DATASETS for _, r in man[ds].iterrows()]
        recs = Parallel(n_jobs=args.jobs, verbose=5)(delayed(run_scan)(r, 1000 + i) for i, r in enumerate(rows))
        pd.DataFrame([x for r in recs for x in r]).to_csv(hits_path, index=False)
    hits = pd.read_csv(hits_path)
    rng = np.random.default_rng(0)
    summ = {}
    for ds in DATASETS:
        cl = hits[(hits.dataset == ds) & (hits.kind == "clean")]
        summ[f"{ds}|false_calls"] = {d: float((g.false_calls > 0).mean()) for d, g in cl.groupby("detector")}
        summ[f"{ds}|false_calls_per_scan"] = {d: float(g.false_calls.mean()) for d, g in cl.groupby("detector")}
        for kind in KINDS:
            for det in DET_COLOR:
                d = hits[(hits.dataset == ds) & (hits.kind == kind) & (hits.detector == det)]
                s = pod_summary(d, rng)
                ok = d[d.hit == 1]
                s["sizing_ratio_median"] = float(np.nanmedian(ok.meas_um / ok.a_um)) if len(ok) else np.nan
                summ[f"{ds}|{kind}|{det}"] = s
                print(f"{DATASETS[ds]:22s} {kind:7s} {det:18s} a50 {s['a50']:6.1f}  a90 {s['a90']:6.1f}  "
                      f"a90/95 {s['a90_95']:6.1f} um   sizing x{s['sizing_ratio_median']:.2f}  n={s['n']}",
                      flush=True)
        print(f"{DATASETS[ds]:22s} false-call rate (any call on a healthy scan): {summ[f'{ds}|false_calls']}")
    pod_figure(hits, summ)
    sizing_figure(hits)
    json.dump(summ, open(OUT / "mapod.json", "w"), indent=2)
    print("saved", OUT / "mapod.json")


if __name__ == "__main__":
    main()
