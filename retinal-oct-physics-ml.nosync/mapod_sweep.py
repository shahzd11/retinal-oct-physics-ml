"""POD vs false-call trade-off: sweep each fluid detector's threshold and compare a90 at a matched
false-call rate (the NDT way to compare inspection procedures: POD at a fixed probability of false alarm).

  NDT half-max      frac 0.20 ... 0.45  (fraction of the way from vitreous to the darkest local tissue)
  v2 fixed-fraction frac 0.10 ... 0.35  (fraction of the way from vitreous to the retina median)

Same 120 healthy scans as mapod.py, lesion types cyst_px and srf, 10 sizes 10-120 um, 1 placement each.
  python mapod_sweep.py
Writes results/v3/mapod_sweep.csv, mapod_sweep.json, mapod_tradeoff.png
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from mapod import DATASETS, DET_COLOR, HIT_COVER, OUT, a_at, detect, fit_probit, insert, lesion_mask
from retina import plotstyle as ps
from retina.layers import flatten
from retina.preprocess import load_gray

warnings.filterwarnings("ignore")
FRACS = {"NDT half-max": [0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
         "v2 fixed-fraction": [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]}
DEFAULT = {"NDT half-max": 0.4, "v2 fixed-fraction": 0.2}
SIZES = np.geomspace(10, 120, 10)
KINDS = {"cyst_px": "intraretinal cyst (round on screen)", "srf": "subretinal fluid"}
TARGETS = (0.1, 0.2, 0.3)


def run_scan(row, seed):
    rng = np.random.default_rng(seed)
    um = float(row.axial_um)
    f = flatten(load_gray(row.path), um, None)
    clean = detect(f, FRACS)
    recs = [dict(dataset=row.dataset_key, patient=row.patient, kind="clean", detector=d, frac=fr, a_um=0.0,
                 hit=np.nan, false_calls=n) for (d, fr), (_, n) in clean.items()]
    for kind in KINDS:
        for a in SIZES:
            m = lesion_mask(f, kind, a, um, rng)
            if m is None:
                continue
            for (d, fr), (k, _) in detect(insert(f, m, rng), FRACS).items():
                cover = ((k & ~clean[(d, fr)][0]) & m).sum() / m.sum()
                recs.append(dict(dataset=row.dataset_key, patient=row.patient, kind=kind, detector=d, frac=fr,
                                 a_um=float(a), hit=float(cover >= HIT_COVER), false_calls=np.nan))
    return recs


def curves(h):
    rows = []
    for (ds, det, fr), g in h.groupby(["dataset", "detector", "frac"]):
        fcr = float((g[g.kind == "clean"].false_calls > 0).mean())
        for kind in KINDS:
            d = g[g.kind == kind]
            mu, sg = fit_probit(d.a_um.values, d.hit.values)
            ok = sg > 0
            rows.append(dict(dataset=ds, detector=det, frac=fr, kind=kind, false_call_rate=fcr,
                             a50=a_at(mu, sg, 0.5) if ok else np.nan, a90=a_at(mu, sg, 0.9) if ok else np.nan))
    return pd.DataFrame(rows)


def matched(r):
    """a90 linearly interpolated at a target false-call rate (only inside the swept range)."""
    out = {}
    for ds in DATASETS:
        for kind in KINDS:
            for t in TARGETS:
                for det in DET_COLOR:
                    d = r[(r.dataset == ds) & (r.kind == kind) & (r.detector == det)]
                    d = d.groupby("false_call_rate").a90.min().reset_index().sort_values("false_call_rate")
                    inside = d.false_call_rate.min() <= t <= d.false_call_rate.max()
                    out[f"{ds}|{kind}|fcr={t}|{det}"] = (float(np.interp(t, d.false_call_rate, d.a90))
                                                         if inside else None)
    return out


def figure(r):
    fig, axes = plt.subplots(len(KINDS), 2, figsize=(13, 9.5), sharex=True, sharey="row")
    for i, kind in enumerate(KINDS):
        for j, ds in enumerate(DATASETS):
            ax = axes[i, j]
            for det, col in DET_COLOR.items():
                d = r[(r.dataset == ds) & (r.kind == kind) & (r.detector == det)].sort_values("frac")
                ax.plot(d.false_call_rate, d.a90, color=col, lw=2, marker="o", ms=6, label=det,
                        markeredgecolor=ps.SURFACE, markeredgewidth=1.5)
                dd = d[np.isclose(d.frac, DEFAULT[det])]
                ax.scatter(dd.false_call_rate, dd.a90, s=170, facecolor="none", edgecolor=col, lw=1.8, zorder=4)
                for _, q in d.iterrows():
                    ax.annotate(f"{q.frac:g}", (q.false_call_rate, q.a90), xytext=(5, 4),
                                textcoords="offset points", fontsize=7, color=ps.INK2)
            ax.set_title(f"{DATASETS[ds]}: {KINDS[kind]}", loc="left")
            if i == len(KINDS) - 1:
                ax.set_xlabel("False-call rate (share of healthy scans with any fluid call)")
            if j == 0:
                ax.set_ylabel("a90: height detected 90% of the time (um)")
            ax.legend(frameon=False, fontsize=9, loc="upper right",
                      title="small labels = threshold fraction, ring = default", title_fontsize=8)
    fig.suptitle("Detection vs false calls as the threshold moves (lower-left is better)", weight="bold")
    plt.tight_layout(); plt.savefig(OUT / "mapod_tradeoff.png", dpi=140); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figs-only", action="store_true")
    args = ap.parse_args()
    ps.apply()
    if not args.figs_only:
        man = []
        for ds in DATASETS:
            m = pd.read_csv(f"manifests/{ds}.csv")
            m = m[m.label == "NORMAL"].groupby("patient").head(1).sample(frac=1, random_state=0)
            man += [row for _, row in m.head(60).assign(dataset_key=ds).iterrows()]
        recs = Parallel(n_jobs=6, verbose=5)(delayed(run_scan)(row, 5000 + i) for i, row in enumerate(man))
        pd.DataFrame([x for rr in recs for x in rr]).to_csv(OUT / "mapod_sweep.csv", index=False)
    h = pd.read_csv(OUT / "mapod_sweep.csv")
    r = curves(h)
    print(r.round(3).to_string(index=False))
    mt = matched(r)
    for k, v in mt.items():
        print(f"{k:58s} a90 {'  n/a' if v is None else f'{v:6.1f}'}")
    figure(r)
    json.dump({"curves": r.round(4).to_dict("records"), "a90_at_matched_fcr": mt},
              open(OUT / "mapod_sweep.json", "w"), indent=2)
    print("saved", OUT / "mapod_sweep.json")


if __name__ == "__main__":
    main()
