"""Evaluate the retinal-boundary fractal ("coastline") features.

  1. Per-class medians and class signal (Kruskal-Wallis epsilon^2) of each frac_* feature, plus the
     healthy-eye scanner shift |Cohen's d| between Kermany (Spectralis) and OCTDL (Optovue)
  2. Paired test: v2 band-summary + disease + NDT robust, with vs without the frac_* features
     (same model, same patient-grouped folds, patient bootstrap of the difference)

  python fractal_eval.py
Writes results/v4/fractal_signal.json, fractal_by_class.png, fractal_paired_test.json
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from ndt_checks import cohen_d, eps2
from ndt_paired_test import paired_boot
from retina import plotstyle as ps
from retina.evaluation import add_band_summaries, add_row_summaries, feature_sets, hgb, logreg, oof_predictions

warnings.filterwarnings("ignore")
OUT = Path("results/v4")
BASE = "v2 band-summary + disease + NDT robust"
ADD = "v2 band-summary + disease + NDT robust + fractal"
REPEATS = 3
TASKS = {
    "OCTID CSR/DR/MH (500x750)": ("octid", lambda d: (d.label != "NORMAL") & (d.orig_h == 500) & (d.orig_w == 750)),
    "OCTID 4-class": ("octid", lambda d: d.label.notna()),
    "OCTDL 7-class": ("octdl", lambda d: d.label.notna()),
    "OCTDL 6 diseases": ("octdl", lambda d: d.label != "NORMAL"),
    "Kermany 4-class": ("kermany", lambda d: d.label.notna()),
    "Kermany CNV/DME/DRUSEN": ("kermany", lambda d: d.label != "NORMAL"),
}
SHOW = [("frac_ilm_D", "ILM fractal dimension D = 2 - H"), ("frac_ilm_amp", "ILM roughness amplitude (um / column)"),
        ("frac_rpe_D", "RPE fractal dimension D = 2 - H"), ("frac_rpe_amp", "RPE roughness amplitude (um / column)")]


def load(ds):
    return pd.read_csv(OUT / f"{ds}_features.csv")


def signal():
    data = {ds: load(ds) for ds in ("octid", "octdl", "kermany")}
    data["octid"] = data["octid"][(data["octid"].orig_h == 500) & (data["octid"].orig_w == 750)
                                  | (data["octid"].label == "NORMAL")]
    cols = [c for c in data["kermany"].columns if c.startswith("frac_") and c != "frac_error"]
    kn = data["kermany"][data["kermany"].label == "NORMAL"]
    dn = data["octdl"][data["octdl"].label == "NORMAL"]
    res = {}
    for c in cols:
        r = {"scanner_shift_d": cohen_d(kn[c].values.astype(float), dn[c].values.astype(float))}
        for ds, d in data.items():
            dd = d[d.label != "NORMAL"] if ds == "octid" else d     # OCTID healthy = other scanner
            r[f"{ds}_eps2"] = eps2(dd[c].values.astype(float), dd.label.values)
            r[f"{ds}_median"] = d.groupby("label")[c].median().round(4).to_dict()
        res[c] = r
        print(f"{c:22s} shift |d| {r['scanner_shift_d']:.2f}   eps2 OCTID-dis {r['octid_eps2']:.3f}  "
              f"OCTDL {r['octdl_eps2']:.3f}  Kermany {r['kermany_eps2']:.3f}")
    json.dump(res, open(OUT / "fractal_signal.json", "w"), indent=2, default=float)

    fig, axes = plt.subplots(len(SHOW), 2, figsize=(14, 13))
    for j, ds in enumerate(("octdl", "kermany")):
        d = data[ds]
        order = ["NORMAL"] + sorted(set(d.label) - {"NORMAL"})
        for i, (c, title) in enumerate(SHOW):
            ax = axes[i, j]
            vals = [d.loc[d.label == lab, c].dropna().values for lab in order]
            bp = ax.boxplot(vals, tick_labels=order, showfliers=False, widths=0.55, patch_artist=True,
                            medianprops=dict(color=ps.INK, lw=1.5))
            for patch, lab in zip(bp["boxes"], order):
                patch.set_facecolor(ps.CLASS_COLOR.get(lab, "#b9b8b3")); patch.set_alpha(0.75)
                patch.set_edgecolor(ps.SURFACE)
            ax.set_title(f"{'OCTDL (Optovue)' if ds == 'octdl' else 'Kermany (Spectralis)'}: {title}  "
                         f"(eps2 {res[c][ds + '_eps2']:.3f})", loc="left", fontsize=10)
            ax.grid(axis="x", visible=False)
    fig.suptitle("Retinal boundaries as coastlines: fractal dimension and roughness by class", weight="bold")
    plt.tight_layout(); plt.savefig(OUT / "fractal_by_class.png", dpi=130); plt.close()


def paired():
    rng = np.random.default_rng(0)
    res = []
    for tname, (ds, sel) in TASKS.items():
        df = add_band_summaries(add_row_summaries(load(ds)))
        d = df[sel(df)].reset_index(drop=True)
        y, g = d.label.values, d.patient.values
        fs = feature_sets(d)

        def X(name):
            a = d[fs[name]].values.astype(float)
            a[~np.isfinite(a)] = np.nan
            return a

        for mname, fn in (("Logistic regression", logreg), ("LightGBM", hgb)):
            pa = [oof_predictions(fn, X(BASE), y, g, s, 5) for s in range(REPEATS)]
            pb = [oof_predictions(fn, X(ADD), y, g, s, 5) for s in range(REPEATS)]
            diffs = [balanced_accuracy_score(y, b) - balanced_accuracy_score(y, a) for a, b in zip(pa, pb)]
            lo, hi, p = paired_boot(y, pa[0], pb[0], g, rng)
            r = dict(task=tname, model=mname,
                     base_bacc=float(np.mean([balanced_accuracy_score(y, a) for a in pa])),
                     with_bacc=float(np.mean([balanced_accuracy_score(y, b) for b in pb])),
                     diff_mean=float(np.mean(diffs)), diff_per_repeat=[round(x, 4) for x in diffs],
                     diff_ci=[lo, hi], p_one_sided=p)
            res.append(r)
            print(f"{tname:28s} {mname:20s} {r['base_bacc']:.3f} -> {r['with_bacc']:.3f} diff {r['diff_mean']:+.3f} "
                  f"[{lo:+.3f}, {hi:+.3f}] p={p:.3f}  repeats {r['diff_per_repeat']}", flush=True)
    json.dump(res, open(OUT / "fractal_paired_test.json", "w"), indent=2)


if __name__ == "__main__":
    ps.apply()
    signal()
    paired()
