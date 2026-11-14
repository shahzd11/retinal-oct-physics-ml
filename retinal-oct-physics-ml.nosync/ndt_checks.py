"""Checks for the NDT feature family.

1. QC overlays: flattened scans with ILM/RPE, half-max fluid pockets and shadow columns.
2. Scanner robustness vs usefulness, per feature family:
     scanner shift  = median |Cohen's d| between HEALTHY Kermany (Spectralis) and HEALTHY OCTDL (Optovue)
                      scans (same anatomy, different scanner -> any difference is a scanner signature)
     class signal   = median epsilon^2 (Kruskal-Wallis effect size) across the 4 Kermany classes
   A good family has high class signal and low scanner shift.

  python ndt_checks.py
Writes results/v3/ndt_qc_examples.png, ndt_scanner_vs_signal.png, ndt_checks.json
"""
import json
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal

from retina import plotstyle as ps
from retina.features_ndt import ndt_masks
from retina.layers import flatten
from retina.preprocess import load_gray

warnings.filterwarnings("ignore")
OUT = Path("results/v3")
# label offsets (points) for families whose markers sit close together
OFFSETS = {"NDT: attenuation": (-6, -11, "right"), "NDT: interface contrast": (6, -11, "left"),
           "v2 texture: lac4": (-6, 5, "right"), "v2 texture: entropy": (-6, 6, "right"),
           "v2 disease: thickness": (6, 4, "left"), "v2 texture: std": (-4, 8, "right"),
           "v2 texture: coh": (6, -10, "left"), "v2 texture: H": (-6, 4, "right"),
           "v2 texture: nmsid_ratio": (-6, -11, "right"), "v2 disease: reflectivity/RPE": (-6, 6, "right"),
           "v2 texture: nmsid1": (6, -10, "left"), "NDT: SNR/CNR": (-6, 5, "right")}


def family(c):
    if re.fullmatch(r"grid_\d+_hurst", c):
        return "paper: grid Hurst"
    if re.fullmatch(r"grid_\d+_kt1", c):
        return "paper: grid KT-1"
    if c.startswith("tex_"):
        k = c.split("_")[-1]
        k = "H_r2" if c.endswith("_H_r2") else ("nmsid_ratio" if c.endswith("nmsid_ratio") else k)
        return f"v2 texture: {k}"
    if c.startswith(("thk_", "fovea_", "gap_", "thin_")):
        return "v2 disease: thickness"
    if c.startswith("fluid_"):
        return "v2 disease: fluid"
    if c.startswith(("hrf_", "retina_", "rpe_")):
        return "v2 disease: reflectivity/RPE"
    if c.startswith("ndtqc_"):
        return "NDT QC: raw reference levels"
    if c.startswith("ndt_"):
        k = c.split("_")[1]
        return {"att": "NDT: attenuation", "trans": "NDT: attenuation", "ic": "NDT: interface contrast",
                "ilm": "NDT: interface contrast", "rpe": "NDT: interface contrast",
                "hm": "NDT: half-max sizing", "sh": "NDT: shadowing", "hypertrans": "NDT: shadowing",
                "cnr": "NDT: SNR/CNR", "spk": "NDT: speckle"}.get(k)
    return None


def cohen_d(a, b):
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 10 or len(b) < 10:
        return np.nan
    s = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return abs(a.mean() - b.mean()) / s if s > 0 else np.nan


def eps2(x, y):
    ok = np.isfinite(x)
    x, y = x[ok], y[ok]
    groups = [x[y == g] for g in np.unique(y)]
    if min(len(g) for g in groups) < 10 or np.ptp(x) == 0:
        return np.nan
    h = kruskal(*groups).statistic
    return h / (len(x) - 1)


def qc_figure():
    man = {n: pd.read_csv(f"manifests/{n}.csv") for n in ("octid", "octdl", "kermany")}
    picks = [("octid", "CSR"), ("octid", "DR"), ("octid", "MH"), ("octid", "NORMAL"),
             ("kermany", "DME"), ("kermany", "CNV"), ("octdl", "AMD"), ("kermany", "NORMAL")]
    fig, axes = plt.subplots(2, 4, figsize=(16, 6.4))
    rng = np.random.default_rng(3)
    for ax, (ds, lab) in zip(axes.ravel(), picks):
        m = man[ds][man[ds].label == lab]
        row = m.iloc[int(rng.integers(len(m)))]
        icon = tuple(float(v) for v in row.icon_box.split(",")) if isinstance(row.icon_box, str) and row.icon_box else None
        ax_um = float(row.axial_um) if np.isfinite(row.axial_um) else None
        f = flatten(load_gray(row.path), ax_um, icon)
        mk = ndt_masks(f)
        ax.imshow(f.img, cmap="gray", vmin=0, vmax=1, aspect="auto")
        ov = np.zeros((*f.img.shape, 4))
        ov[mk["fluid"]] = (0.16, 0.47, 0.84, 0.55)                  # half-max fluid pockets
        ax.imshow(ov, aspect="auto")
        x = np.arange(f.img.shape[1])
        ax.plot(x, np.where(f.col_ok, f.ilm, np.nan), color=ps.SERIES[2], lw=1)
        ax.plot(x, np.where(f.col_ok, f.rpe, np.nan), color=ps.SERIES[3], lw=1)
        H = f.img.shape[0]
        for cols, color in ((mk["shadow_cols"], ps.SERIES[1]), (mk["hyper_cols"], "#ffffff")):
            ax.fill_between(x, H - 8, H - 1, where=cols, color=color, step="mid")
        ax.axvline(f.fovea, color="#ffffff", lw=0.6, ls=":")
        ax.set_title(f"{ds.upper()} {lab}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("NDT QC: blue = half-max fluid pockets, orange strip = shadowed columns, "
                 "white strip = hyper-transmission; green ILM, yellow RPE, dotted = fovea", fontsize=10)
    plt.tight_layout(); plt.savefig(OUT / "ndt_qc_examples.png", dpi=130); plt.close()


def main():
    ps.apply()
    qc_figure()
    ker = pd.read_csv(OUT / "kermany_features.csv")
    dl = pd.read_csv(OUT / "octdl_features.csv")
    cols = [c for c in ker.columns if c in dl.columns and family(c) and ker[c].dtype != object]
    kn, dn = ker[ker.label == "NORMAL"], dl[dl.label == "NORMAL"]
    rows = []
    for c in cols:
        rows.append(dict(feature=c, family=family(c),
                         shift=cohen_d(kn[c].values.astype(float), dn[c].values.astype(float)),
                         signal=eps2(ker[c].values.astype(float), ker.label.values)))
    r = pd.DataFrame(rows)
    fam = r.groupby("family").agg(shift=("shift", "median"), signal=("signal", "median"),
                                  top_signal=("signal", lambda s: s.quantile(0.9)), n=("feature", "size"))
    fam = fam.sort_values("signal", ascending=False)
    print(fam.round(3).to_string())
    best = r[r.family.str.startswith("NDT:")].sort_values("signal", ascending=False).head(12)
    print("\nTop NDT features by Kermany class signal (eps^2) and their healthy-eye scanner shift |d|:")
    print(best[["feature", "signal", "shift"]].round(3).to_string(index=False))

    # figure: family-level scatter
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    groups = {"paper": ps.SERIES[1], "v2": ps.SERIES[0], "NDT:": ps.SERIES[2], "NDT QC": "#8a8a8a"}
    for name, d in fam.iterrows():
        g = next(k for k in groups if name.startswith(k))
        ax.scatter(d["shift"], d["signal"], s=40 + 4 * d["n"] ** 0.5 * 4, color=groups[g],
                   edgecolor="white", linewidth=1.5, zorder=3)
        dx, dy, ha = OFFSETS.get(name, (6, 3, "left"))
        ax.annotate(name, (d["shift"], d["signal"]), xytext=(dx, dy), textcoords="offset points", fontsize=8,
                    color="#333333", ha=ha)
    for g, col in groups.items():
        ax.scatter([], [], color=col, label={"paper": "paper features", "v2": "v2 features",
                                             "NDT:": "NDT features", "NDT QC": "raw reference (QC only)"}[g])
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.62, 0.8), fontsize=9)
    ax.set_xlabel("Scanner shift: median |Cohen's d|, healthy Kermany vs healthy OCTDL  (lower = more robust)")
    ax.set_ylabel("Class signal: median epsilon² across Kermany classes")
    ax.set_title("Feature families: class signal vs scanner dependence", loc="left", weight="bold")
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    plt.tight_layout(); plt.savefig(OUT / "ndt_scanner_vs_signal.png", dpi=150); plt.close()

    json.dump({"families": fam.round(4).reset_index().to_dict("records"),
               "top_ndt": best.round(4).to_dict("records")}, open(OUT / "ndt_checks.json", "w"), indent=2)
    print("saved", OUT / "ndt_checks.json")


if __name__ == "__main__":
    main()
