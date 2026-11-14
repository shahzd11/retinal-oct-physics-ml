"""v5 / Part A: reduced-order modelling (POD) of the feature fields.

Questions
  1. How compressible are the paper's 10x15 H and S fields, and the v2 4x5 (band x sector) fields?
  2. Do the high-energy POD modes carry the disease information? ("energy vs information")
  3. What does POD mode 1 of the paper's H field actually encode?
  4. Accuracy vs number of POD coefficients, against the raw-feature baselines.

  python v5_rom.py
Writes results/v5/rom.json and rom_energy.png, rom_modes.png, rom_energy_vs_info.png, rom_accuracy.png
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
from scipy.stats import spearmanr
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ndt_checks import eps2
from retina import plotstyle as ps
from retina.ensemble import lr
from retina.evaluation import BANDS, SECTORS, TEX_KINDS, add_band_summaries, add_row_summaries, feature_sets, repeated_cv
from retina.rom import FieldPOD, paper_fields, texture_fields

warnings.filterwarnings("ignore")
OUT = Path("results/v5")
OUT.mkdir(parents=True, exist_ok=True)
SRC = Path("results/v4")
REPEATS = 3
DS_NAME = {"octid": "OCTID (Cirrus)", "octdl": "OCTDL (Optovue)", "kermany": "Kermany (Spectralis)"}
TASKS = {
    "OCTID CSR/DR/MH (500x750)": ("octid", lambda d: (d.label != "NORMAL") & (d.orig_h == 500) & (d.orig_w == 750)),
    "OCTDL 7-class": ("octdl", lambda d: d.label.notna()),
    "Kermany 4-class": ("kermany", lambda d: d.label.notna()),
}
PAPER_R = [2, 5, 10, 20, 50]
TEX_E = [0.80, 0.90, 0.95, 0.99]
KINDS = [k for k in TEX_KINDS if k != "entropy"]


def load(ds):
    return add_band_summaries(add_row_summaries(pd.read_csv(SRC / f"{ds}_features.csv")))


def mat(d, cols):
    X = d[cols].values.astype(float)
    X[~np.isfinite(X)] = np.nan
    return X


def paper_cols():
    return [f"grid_{g}_{k}" for k in ("hurst", "kt1") for g in range(150)]


def spectrum(X):
    """Cumulative POD energy E(r) of one field (median-imputed, mean-centred)."""
    pod = FieldPOD([list(range(X.shape[1]))], None, energy=X.shape[1], scale=False).fit(X)
    s2 = pod.sigma_[0] ** 2
    return np.cumsum(s2) / s2.sum(), pod


def r_at(E, e):
    return int(np.searchsorted(E, e) + 1)


# ----------------------------------------------------------------------------- 1. energy
def energy_part(data):
    res, curves = {}, {}
    for ds, d in data.items():
        H = mat(d, [f"grid_{g}_hurst" for g in range(150)])
        S = mat(d, [f"grid_{g}_kt1" for g in range(150)])
        T = mat(d, [f"tex_{b}_{s}_H" for b in BANDS for s in SECTORS])
        for name, X in (("paper H field (150 cells)", H), ("paper S field (150 cells)", S),
                        ("v2 band x sector H field (20 cells)", T)):
            E, _ = spectrum(X)
            curves[(ds, name)] = E
            res[f"{ds} | {name}"] = {"cells": X.shape[1], "r90": r_at(E, .90), "r95": r_at(E, .95),
                                     "r99": r_at(E, .99), "energy_mode1": float(E[0])}
            print(f"{ds:8s} {name:38s} r90 {r_at(E, .9):3d}  r95 {r_at(E, .95):3d}  r99 {r_at(E, .99):3d}  "
                  f"mode-1 energy {E[0]:.2f}", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, name in zip(axes, ("paper H field (150 cells)", "v2 band x sector H field (20 cells)")):
        for ds in data:
            E = curves[(ds, name)]
            ax.plot(np.arange(1, len(E) + 1) / len(E), E, lw=2, label=DS_NAME[ds])
        ax.axhline(0.95, color=ps.INK2, lw=1, ls="--")
        ax.set_xlabel("fraction of POD modes kept (r / m)")
        ax.set_ylabel("cumulative energy E(r)")
        ax.set_title(name, loc="left")
        ax.set_ylim(0, 1.01)
        ax.legend(frameon=False, loc="lower right")
    fig.suptitle("POD energy spectrum: how compressible is each feature field?", weight="bold")
    plt.tight_layout(); plt.savefig(OUT / "rom_energy.png", dpi=130); plt.close()
    return res


# ----------------------------------------------------------------------------- 2+3. modes, energy vs info
def modes_part(data):
    d = data["octid"]
    H = mat(d, [f"grid_{g}_hurst" for g in range(150)])
    E, pod = spectrum(H)
    V = pod.modes_[0]
    a = ((pod._impute(H) - pod.means_[0]) / pod.scales_[0]) @ V[:6].T
    # vertical tissue centroid of every scan: H-weighted mean grid row (H = 0 on black background)
    Hm = np.nan_to_num(H).reshape(len(d), 10, 15).clip(min=0).sum(2)
    centroid = (Hm * np.arange(10)).sum(1) / np.maximum(Hm.sum(1), 1e-9)
    rho = [float(spearmanr(a[:, j], centroid).statistic) for j in range(6)]
    mean_H = float(np.nanmean(H))
    print("OCTID paper-H POD: Spearman(mode coefficient, tissue centroid row) =",
          [round(x, 2) for x in rho], flush=True)

    fig, axes = plt.subplots(1, 5, figsize=(17, 3.4))
    axes[0].imshow(pod.means_[0].reshape(10, 15), cmap="viridis")
    axes[0].set_title("mean field  x̄", loc="left")
    for j in range(4):
        m = V[j].reshape(10, 15)
        v = np.abs(m).max()
        axes[j + 1].imshow(m, cmap="RdBu_r", vmin=-v, vmax=v)
        axes[j + 1].set_title(f"mode {j + 1}  ({100 * (E[j] - (E[j - 1] if j else 0)):.0f}% energy)\n"
                              f"ρ with tissue height {rho[j]:+.2f}", loc="left", fontsize=10)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.suptitle("Paper H field on OCTID: mean and first four POD modes ('eigen-retinas', 10 rows x 15 cols)",
                 weight="bold")
    plt.tight_layout(); plt.savefig(OUT / "rom_modes.png", dpi=130); plt.close()

    # energy vs information, per mode, disease-only tasks (no scanner confound)
    info = {}
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for (ds, sel), lab in ((("octid", lambda x: (x.label != "NORMAL") & (x.orig_h == 500) & (x.orig_w == 750)),
                            "OCTID CSR/DR/MH"),
                           (("kermany", lambda x: x.label != "NORMAL"), "Kermany CNV/DME/DRUSEN")):
        dd = data[ds][sel(data[ds])].reset_index(drop=True)
        Hd = mat(dd, [f"grid_{g}_hurst" for g in range(150)])
        Ed, pd_ = spectrum(Hd)
        ad = ((pd_._impute(Hd) - pd_.means_[0]) / pd_.scales_[0]) @ pd_.modes_[0][:40].T
        en = np.diff(np.r_[0, Ed])[:40]
        ef = np.array([eps2(ad[:, j], dd.label.values) for j in range(40)])
        info[lab] = {"spearman_energy_vs_eps2": float(spearmanr(en, ef, nan_policy="omit").statistic),
                     "eps2_top5_modes": [round(float(x), 4) for x in ef[:5]],
                     "best_mode": int(np.nanargmax(ef) + 1), "best_mode_eps2": float(np.nanmax(ef))}
        ax.scatter(en, ef, s=28, label=lab, alpha=.85)
    ax.set_xscale("log")
    ax.set_xlabel("energy of POD mode j  (σ_j² / Σσ²)")
    ax.set_ylabel("class signal of coefficient a_j  (Kruskal-Wallis ε²)")
    ax.set_title("Energy is not information: high-energy modes are not the disease modes", loc="left")
    ax.legend(frameon=False)
    plt.tight_layout(); plt.savefig(OUT / "rom_energy_vs_info.png", dpi=130); plt.close()
    for k, v in info.items():
        print(f"{k}: Spearman(energy, eps2) {v['spearman_energy_vs_eps2']:+.2f}, best mode #{v['best_mode']}", flush=True)
    return {"octid_spearman_mode_vs_tissue_height": rho, "octid_mean_H": mean_H, "energy_vs_info": info}


# ----------------------------------------------------------------------------- 4. accuracy vs r
def accuracy_part():
    res = []
    for tname, (ds, sel) in TASKS.items():
        df = load(ds)
        d = df[sel(df)].reset_index(drop=True)
        y, g = d.label.values, d.patient.values
        fs = feature_sets(d)
        pc = paper_cols()
        Xp = mat(d, pc)
        pf = paper_fields(pc)

        def run(label, family, X, fn, n_feat):
            r, _ = repeated_cv(fn, X, y, g, repeats=REPEATS)
            res.append(dict(task=tname, family=family, config=label, n_features=int(n_feat),
                            bacc=r["bacc_mean"], bacc_sd=r["bacc_sd"], ci=[r["bacc_ci_lo"], r["bacc_ci_hi"]],
                            acc=r["acc_mean"]))
            print(f"  {tname:26s} {label:34s} p={n_feat:4d}  bacc {r['bacc_mean']:.3f} ± {r['bacc_sd']:.3f}", flush=True)

        run("paper raw (300)", "paper", Xp, lr, 300)
        for r_ in PAPER_R:
            run(f"paper POD, r={r_} per field", "paper",
                Xp, lambda s=0, r_=r_: make_pipeline(FieldPOD(pf, None, r_), StandardScaler(), lr()), 2 * r_)

        tex = fs["v2 texture, no entropy"]
        scal = fs["v2 disease-specific"] + fs["NDT robust"] + fs["fractal boundary only"]
        cols = tex + scal
        Xt = mat(d, cols)
        pos = {c: i for i, c in enumerate(cols)}
        tf = texture_fields(cols, KINDS, BANDS, SECTORS)
        pt = [pos[c] for c in scal]
        run(f"texture raw + scalars ({len(cols)})", "texture", Xt, lr, len(cols))
        for e in TEX_E:
            n = FieldPOD(tf, pt, e).fit(Xt).transform(Xt[:2]).shape[1]
            run(f"texture POD E={e:.2f} + scalars", "texture",
                Xt, lambda s=0, e=e: make_pipeline(FieldPOD(tf, pt, e), StandardScaler(), lr()), n)
        best = fs["v2 band-summary + disease + NDT robust + fractal"]
        run(f"repo best set ({len(best)})", "reference", mat(d, best), lr, len(best))

    fig, axes = plt.subplots(1, len(TASKS), figsize=(16, 4.4))
    rdf = pd.DataFrame(res)
    for ax, t in zip(axes, TASKS):
        q = rdf[rdf.task == t]
        for fam, col, mk in (("paper", ps.SERIES[1], "o"),
                             ("texture", ps.SERIES[0], "s")):
            f = q[(q.family == fam) & q.config.str.contains("POD")].sort_values("n_features")
            raw = q[(q.family == fam) & q.config.str.contains("raw")].iloc[0]
            ax.plot(f.n_features, f.bacc, marker=mk, color=col, lw=2, label=f"{fam} POD")
            ax.axhline(raw.bacc, color=col, ls="--", lw=1.2, label=f"{fam} raw ({raw.n_features})")
        ref = q[q.family == "reference"].iloc[0]
        ax.axhline(ref.bacc, color=ps.INK, ls=":", lw=1.2, label=f"repo best ({ref.n_features})")
        ax.set_xscale("log"); ax.set_xlabel("number of features given to the classifier")
        ax.set_ylabel("balanced accuracy (LR)"); ax.set_title(t, loc="left", fontsize=10)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Accuracy vs number of POD coefficients", weight="bold")
    plt.tight_layout(); plt.savefig(OUT / "rom_accuracy.png", dpi=130); plt.close()
    return res


if __name__ == "__main__":
    ps.apply()
    data = {ds: load(ds) for ds in ("octid", "octdl", "kermany")}
    out = {"energy": energy_part(data)}
    out.update(modes_part(data))
    out["accuracy"] = accuracy_part()
    json.dump(out, open(OUT / "rom.json", "w"), indent=2, default=float)
    print("saved", OUT / "rom.json")
