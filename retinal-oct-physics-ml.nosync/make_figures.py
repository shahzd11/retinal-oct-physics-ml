"""Build every presentation figure from files in results/ only (no images, no training).

  python make_figures.py            -> presentation/figures/*.png (200 dpi, 16:9-friendly)
                                       presentation/figures/figure_numbers.json (every plotted headline value + source)
No number is typed by hand: values are read from results/ JSON/CSV files (paths recorded in
figure_numbers.json). A figure whose input does not exist yet is drawn as a labelled "not yet run" panel.
"""
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from scipy.stats import norm, spearmanr

from retina import plotstyle as ps
from retina.preprocess import GRID, H_TARGET, N_COLS, N_ROWS, W_TARGET

R = Path("results")
V6 = R / "v6"
FIG = Path("presentation/figures")
NUM = {}                                   # figure -> {label: (value, source)}
SEQ = LinearSegmentedColormap.from_list("seq", ["#fcfcfb"] + ps.SEQ_BLUE)
DIV = LinearSegmentedColormap.from_list("div", ["#c0392b", "#e88b7d", "#f0efec", "#86b6ef", "#184f95"])
TASKS = ["OCTID CSR/DR/MH (500x750)", "OCTDL 7-class", "OCTDL 6 diseases", "Kermany 4-class",
         "Kermany CNV/DME/DRUSEN", "Kermany -> OCTDL (NORMAL/DME/AMD)"]
SHORT = {"OCTID CSR/DR/MH (500x750)": "OCTID\nCSR/DR/MH", "OCTDL 7-class": "OCTDL\n7-class",
         "OCTDL 6 diseases": "OCTDL\n6 diseases", "Kermany 4-class": "Kermany\n4-class",
         "Kermany CNV/DME/DRUSEN": "Kermany\nCNV/DME/DRUSEN", "Kermany -> OCTDL (NORMAL/DME/AMD)": "Kermany→OCTDL\n(new scanner)"}
SLUG = {"OCTID CSR/DR/MH (500x750)": "octid_dis750", "OCTDL 7-class": "octdl_7", "OCTDL 6 diseases": "octdl_6dis",
        "Kermany 4-class": "kermany_4", "Kermany CNV/DME/DRUSEN": "kermany_3dis",
        "Kermany -> OCTDL (NORMAL/DME/AMD)": "kermany_to_octdl"}
GRID_TASK = {"OCTID CSR/DR/MH (500x750)": ("octid", "disease-only, 500x750 scans only (no size confound)"),
             "OCTDL 7-class": ("octdl", "all classes (AMD/DME/ERM/NORMAL/RAO/RVO/VID)"),
             "OCTDL 6 diseases": ("octdl", "disease-only (AMD/DME/ERM/RAO/RVO/VID)"),
             "Kermany 4-class": ("kermany", "all classes (CNV/DME/DRUSEN/NORMAL)"),
             "Kermany CNV/DME/DRUSEN": ("kermany", "disease-only (CNV/DME/DRUSEN)")}
HYB = "hybrid ResNet-18"          # headline v6 model (DINOv2-S hybrid was pre-registered; kept in all tables)
HYB2 = "hybrid DINOv2-S"


# ============================================================================ helpers
def style():
    ps.apply()
    plt.rcParams.update({"font.size": 15, "axes.titlesize": 17, "axes.labelsize": 15, "xtick.labelsize": 14,
                         "ytick.labelsize": 14, "legend.fontsize": 14, "figure.titlesize": 19})


def js(p):
    p = Path(p)
    return json.load(open(p)) if p.exists() else None


def record(fig, label, value, source):
    NUM.setdefault(fig, {})[label] = {"value": value, "source": str(source)}


def save(fig, name):
    fig.savefig(FIG / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", FIG / name)


def missing(ax, what):
    ax.axis("off")
    ax.text(0.5, 0.5, f"not yet run:\n{what}", ha="center", va="center", fontsize=15, color=ps.INK2,
            bbox=dict(boxstyle="round", fc="#f0efec", ec=ps.GRID))


def heat(ax, M, rows, cols, fmt="{:.2f}", cmap=SEQ, vmin=None, vmax=None, norm_=None, annot=True, fs=13):
    M = np.asarray(M, float)
    im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=vmin, vmax=vmax, norm=norm_, aspect="auto")
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    if annot:
        lo, hi = np.nanmin(M) if np.isfinite(M).any() else 0, np.nanmax(M) if np.isfinite(M).any() else 1
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                if np.isfinite(v):
                    frac = (v - (vmin if vmin is not None else lo)) / max((vmax if vmax is not None else hi) - (vmin if vmin is not None else lo), 1e-9)
                    dark = (frac > 0.6) if norm_ is None else (float(norm_(v)) > 0.82 or float(norm_(v)) < 0.12)
                    ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=fs,
                            color="white" if dark else ps.INK)
                else:
                    ax.text(j, i, "n/a", ha="center", va="center", fontsize=fs - 2, color=ps.INK2)
    return im


def oof(path):
    p = Path(path)
    return pd.read_csv(p) if p.exists() else None


# ============================================================================ v6 result readers
def v6_models(task):
    """{model name: summary} over every v6 result file for this task (within or cross)."""
    s = SLUG[task]
    out = {}
    ip = js(V6 / f"image_pod_{s}.json")
    if ip:
        out.update(ip["models"])
    rc = js(V6 / "residual_clf.json")
    if rc and task in rc:
        out["v5 best + e(x)"] = rc[task]["v5 best + e(x)"]
    hy = js(V6 / f"hybrid_{s}.json")
    if hy:
        out.update(hy["models"])
    return out


def v6_best(task):
    """The headline v6 model (ResNet-18 hybrid) for a task: (name, summary)."""
    m = v6_models(task)
    return (HYB, m[HYB]) if HYB in m else (None, None)


def model_slug(name):
    return "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_").replace("__", "_")


def cnn(task):
    return js(V6 / f"cnn_{SLUG[task]}.json")


# ============================================================================ figure 1: pipelines
def fig_pipeline():
    paper = ["Crop white\nwedges", f"Resize to\n{W_TARGET}×{H_TARGET}", f"{N_COLS}×{N_ROWS} grid of\n{GRID}×{GRID} px cells",
             "KT-1 slope S\n+ Hurst H\nper cell", "Random\nForest"]
    ours = ["Flatten at\nthe RPE", "4 bands ×\n5 sectors", "Texture +\ndisease + NDT\n+ fractal", "POD /\nsoft-vote\nensemble",
            "Conformal\nprediction\nsets"]

    def row(ax, steps, color, title):
        ax.set_xlim(0, len(steps) * 2.2); ax.set_ylim(0, 2); ax.axis("off")
        ax.set_title(title, loc="left", fontsize=18)
        for i, s in enumerate(steps):
            x = i * 2.2 + 0.1
            ax.add_patch(FancyBboxPatch((x, 0.25), 1.75, 1.5, boxstyle="round,pad=0.04,rounding_size=0.12",
                                        fc=color, ec="none", alpha=0.18))
            ax.add_patch(FancyBboxPatch((x, 0.25), 1.75, 1.5, boxstyle="round,pad=0.04,rounding_size=0.12",
                                        fc="none", ec=color, lw=2))
            ax.text(x + 0.875, 1.0, s, ha="center", va="center", fontsize=14.5, color=ps.INK)
            if i < len(steps) - 1:
                ax.add_patch(FancyArrowPatch((x + 1.85, 1.0), (x + 2.25, 1.0), arrowstyle="-|>", mutation_scale=22,
                                             color=ps.INK2, lw=2))

    for name, steps, col, title in (("fig01a_pipeline_paper.png", paper, ps.SERIES[1], "Paper: fixed pixel grid"),
                                    ("fig01b_pipeline_ours.png", ours, ps.SERIES[0], "Ours: measure inside the anatomy")):
        fig, ax = plt.subplots(figsize=(13, 3.2))
        row(ax, steps, col, title)
        save(fig, name)
    fig, axs = plt.subplots(2, 1, figsize=(13, 6.4))
    row(axs[0], paper, ps.SERIES[1], "Paper: fixed pixel grid")
    row(axs[1], ours, ps.SERIES[0], "Ours: measure inside the anatomy")
    save(fig, "fig01_pipelines.png")


# ============================================================================ figure 2: examples
def fig_examples():
    z = np.load(V6 / "figdata.npz", allow_pickle=True) if (V6 / "figdata.npz").exists() else None
    if z is None:
        fig, ax = plt.subplots(figsize=(13, 7)); missing(ax, "v6_figdata.py"); save(fig, "fig02_examples.png"); return
    classes = ["NORMAL", "CSR", "DR", "MH"]
    fig, axs = plt.subplots(4, 3, figsize=(15, 11), gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
    R0 = int(z["R0"]); sec = z["sectors"]
    for i, c in enumerate(classes):
        a0, a1, a2 = axs[i]
        a0.imshow(z[f"ex_{c}_raw"], cmap="gray"); a0.axis("off")
        a0.set_title(f"{c}: raw scan" if i == 0 else c, loc="left", fontsize=15)
        a1.imshow(z[f"ex_{c}_norm"], cmap="gray"); a1.axis("off")
        for x in range(0, W_TARGET + 1, GRID):
            a1.axvline(x - 0.5, color=ps.SERIES[3], lw=0.7)
        for y in range(0, H_TARGET + 1, GRID):
            a1.axhline(y - 0.5, color=ps.SERIES[3], lw=0.7)
        if i == 0:
            a1.set_title(f"paper: {W_TARGET}×{H_TARGET}, {N_COLS}×{N_ROWS} grid", loc="left", fontsize=15)
        a2.imshow(z[f"ex_{c}_flat"], cmap="gray"); a2.axis("off")
        ilm, rpe = z[f"ex_{c}_ilm"], z[f"ex_{c}_rpe"]
        xs = np.arange(len(ilm))
        mid = (ilm + rpe - 10) / 2
        for yv, col, lab in ((ilm, ps.SERIES[0], "ILM"), (mid, ps.SERIES[4], "inner | outer"),
                             (rpe - 6, ps.SERIES[1], "RPE band"), (rpe + 6, ps.SERIES[1], None),
                             (rpe + 60, ps.SERIES[2], "choroid limit")):
            a2.plot(xs, yv, color=col, lw=1.4, label=lab)
        fov = int(z[f"ex_{c}_fovea"])
        for a, b in sec:
            for e in (a, b):
                a2.axvline(fov + e, color="white", lw=0.8, ls="--")
        if i == 0:
            a2.set_title("ours: flattened at RPE, bands + sectors", loc="left", fontsize=15)
            a2.legend(loc="lower left", fontsize=10, ncol=2, framealpha=0.85)
    fig.tight_layout()
    save(fig, "fig02_examples.png")
    # gallery of the other two datasets
    keys = [k for k in z.files if k.startswith("gal_")]
    fig, axs = plt.subplots(2, 7, figsize=(17, 4.6))
    for r, ds in enumerate(("kermany", "octdl")):
        ks = [k for k in keys if k.startswith(f"gal_{ds}_")]
        for j in range(7):
            ax = axs[r, j]; ax.axis("off")
            if j < len(ks):
                ax.imshow(z[ks[j]], cmap="gray")
                ax.set_title(("Kermany " if ds == "kermany" else "OCTDL ") + ks[j].split("_")[-1], fontsize=13)
    fig.tight_layout()
    save(fig, "fig02b_gallery.png")


# ============================================================================ figure 3: KT-1 and Hurst concepts
def fig_concepts():
    z = np.load(V6 / "figdata.npz", allow_pickle=True) if (V6 / "figdata.npz").exists() else None
    if z is None:
        return
    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    for c, col, lab in (("NORMAL", ps.SERIES[0], "healthy cell (NORMAL)"), ("CSR", ps.SERIES[1], "diseased cell (CSR)")):
        v = z[f"kt1_{c}"]
        x, y, s, i0 = v[:4], v[4:8], v[8], v[9]
        xx = np.linspace(0, x.max() * 1.05, 50)
        ax.plot(xx, s * xx + i0, color=col, lw=2)
        ax.plot(x, y, "o", color=col, ms=10, mec=ps.SURFACE, mew=2, label=f"{lab}: slope S = {s:.4f}")
        record("fig03a_kt1", f"KT-1 slope {c}", float(s), V6 / "figdata.npz")
    ax.set_xlabel(r"filter curvature  $|W''(0)| \propto \sigma^2$")
    ax.set_ylabel(r"$1 / N_{\max}$")
    ax.set_title("KT-1: how fast the peak drops as we blur", loc="left")
    ax.legend(loc="upper left", fontsize=12.5)
    save(fig, "fig03a_kt1.png")

    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    k = z["hurst_k"]
    for j, (lab, col) in enumerate((("smooth cell", ps.SERIES[0]), ("rough cell", ps.SERIES[1]))):
        c = z["hurst_curves"][j]
        H = float(z["hurst_H"][j])
        p = np.polyfit(np.log(k), np.log(c), 1)
        ax.loglog(k, c, "o", color=col, ms=7, mec=ps.SURFACE, mew=1.5)
        ax.loglog(k, np.exp(np.polyval(p, np.log(k))), color=col, lw=2, label=f"{lab}: H = {H:.2f}, D = 3 − H = {3 - H:.2f}")
        record("fig03b_hurst", f"H {lab}", H, V6 / "figdata.npz")
    ax.set_xlabel("pixel distance k"); ax.set_ylabel("NMSID(k)")
    ax.set_title("Hurst exponent: slope on a log–log plot", loc="left")
    ax.legend(loc="lower right", fontsize=12.5)
    for j in range(2):
        ins = ax.inset_axes([0.04 + 0.2 * j, 0.62, 0.17, 0.3])
        ins.imshow(z["hurst_cells"][j], cmap="gray", vmin=0, vmax=1); ins.set_xticks([]); ins.set_yticks([])
        ins.set_title(["smooth", "rough"][j], fontsize=11)
    save(fig, "fig03b_hurst.png")


# ============================================================================ figure 4: confound
def fig_confound():
    f = pd.read_csv(R / "v4/octid_features.csv", usecols=["label", "orig_h", "orig_w"])
    f["size"] = f.orig_h.astype(int).astype(str) + "×" + f.orig_w.astype(int).astype(str)
    ct = pd.crosstab(f["size"], f.label)
    ct = ct.loc[ct.sum(axis=1).sort_values(ascending=False).index]
    fig, axs = plt.subplots(1, 2, figsize=(16, 6.2), gridspec_kw=dict(width_ratios=[1, 1.15]))
    M = ct.values.astype(float)
    heat(axs[0], M, list(ct.index), list(ct.columns), fmt="{:.0f}", vmin=0)
    axs[0].set_title("OCTID: original image size × class (counts)", loc="left")
    axs[0].set_xlabel("class"); axs[0].set_ylabel("original size (h × w px)")
    record("fig04_confound", "OCTID class x size counts", ct.to_dict(), R / "v4/octid_features.csv")
    # mean H vs upsampling factor (same formula as sanity_checks.py)
    d = pd.read_csv(R / "features.csv")
    hcols = [c for c in d.columns if c.endswith("_hurst")]
    d["meanH"] = d[hcols].mean(axis=1)
    d["scale"] = [750 / min(w, h * 1.5) for h, w in zip(d.orig_h, d.orig_w)]
    rho = spearmanr(d.scale, d.meanH)[0]
    sc = js(R / "sanity_checks.json")
    ax = axs[1]
    for c in ["NORMAL", "CSR", "DR", "MH"]:
        m = d.label == c
        ax.scatter(d.scale[m], d.meanH[m], s=34, color=ps.CLASS_COLOR[c], edgecolor=ps.SURFACE, lw=0.8, label=c, alpha=0.9)
    ax.set_xlabel("upsampling factor in the paper's resize step"); ax.set_ylabel("mean Hurst H of the scan")
    ax.set_title(f"Resampling drives H: Spearman ρ = {rho:.2f}", loc="left")
    ax.legend(loc="lower right", ncol=2)
    record("fig04_confound", "spearman upsampling vs mean H", float(sc["spearman_scale_meanH"]), R / "sanity_checks.json")
    record("fig04_confound", "disease scans called NORMAL after resize",
           f"{sc['disease_called_normal_after_resample']}/{sc['disease_test_n']}", R / "sanity_checks.json")
    record("fig04_confound", "size-only NORMAL vs disease accuracy", sc["size_only_normal_vs_disease_acc"], R / "sanity_checks.json")
    fig.tight_layout()
    save(fig, "fig04_confound.png")


# ============================================================================ figure 5: confusion matrices
def cm_panel(ax, src, title, cm=None, classes=None):
    if cm is None:
        o = oof(src)
        if o is None:
            missing(ax, Path(src).name); return None
        classes = sorted(set(o.true) | set(o.pred))
        cm = pd.crosstab(pd.Categorical(o.true, classes), pd.Categorical(o.pred, classes), dropna=False).values
    cm = np.asarray(cm, float)
    rn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    ax.imshow(rn, cmap=SEQ, vmin=0, vmax=1)
    n = len(classes)
    fs = 13 if n <= 4 else 10.5
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{rn[i, j]:.2f}\n({int(cm[i, j])})", ha="center", va="center", fontsize=fs,
                    color="white" if rn[i, j] > 0.6 else ps.INK)
    ax.set_xticks(range(n), classes, rotation=45 if n > 4 else 0, ha="right" if n > 4 else "center")
    ax.set_yticks(range(n), classes)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.grid(False)
    bacc = float(np.mean(np.diag(rn)))
    ax.set_title(title + f"\nbalanced acc {bacc:.3f}" + ("" if "80/20" in title or "new scanner" in title else " (seed-0 folds)"),
                 loc="left", fontsize=14.5)
    return bacc


def fig_confusions():
    panels = []
    panels.append(("a", "Paper method, paper's 80/20 split\n(OCTID 4-class, scanner-confounded)",
                   V6 / "oof_octid_4_confounded_paper_rf_8020.csv", None))
    panels.append(("b", "Paper method, OCTID CSR/DR/MH\n(500×750, 5-fold, patient-grouped)", V6 / "oof_octid_dis750_paper_rf.csv", None))
    for letter, task in (("c", "OCTID CSR/DR/MH (500x750)"), ("d", "OCTDL 7-class"), ("e", "Kermany 4-class")):
        name, _ = v6_best(task)
        src = V6 / f"oof_{SLUG[task]}_{model_slug(name)}.csv" if name else V6 / f"oof_{SLUG[task]}_?.csv"
        panels.append((letter, f"v6 {name or 'hybrid (n/a)'} (headline)\n{task.replace(' (500x750)', ' 500×750')}", src, None))
    cr = js(V6 / "cross.json")
    if cr:
        best = cr["best_config"]
        c = cr["configs"][best]
        cmx = c.get("confusion") or c.get("confusion_summed_over_draws")
        panels.append(("f", f"Kermany → OCTDL, new scanner\n{best.split(' | ')[0]} | {' | '.join(best.split(' | ')[1:])}",
                       V6 / "cross.json", (cmx, c["classes"])))
    else:
        name = "v5 best (soft vote)"
        panels.append(("f", "Kermany → OCTDL, new scanner\nv5 best (soft vote)", V6 / "oof_kermany_to_octdl_v5_best_soft_vote.csv", None))
    fig, axs = plt.subplots(2, 3, figsize=(20, 13.5))
    for ax, (letter, title, src, cmx) in zip(axs.ravel(), panels):
        b = cm_panel(ax, src, f"({letter}) {title}", *(cmx if cmx else (None, None)))
        if b is not None:
            record("fig05_confusions", f"({letter}) balanced acc", b, src)
        f1, a1 = plt.subplots(figsize=(7, 6.3) if letter not in "d" else (9, 8))
        cm_panel(a1, src, f"({letter}) {title}", *(cmx if cmx else (None, None)))
        f1.tight_layout(); save(f1, f"fig05{letter}_confusion.png")
    fig.tight_layout()
    save(fig, "fig05_confusions.png")


# ============================================================================ figure 6: where the model looks
def fig_where():
    d = pd.read_csv(R / "features.csv")
    res = js(R / "results.json")
    top = [int(k.split("_")[1]) for k in res["top15"]]
    imp = js(V6 / "importance_band_family.json")
    fig = plt.figure(figsize=(20, 8.4))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 0.05, 1.0], wspace=0.18, hspace=0.3)
    vals = []
    for c in ["NORMAL", "CSR", "DR", "MH"]:
        vals.append(d.loc[d.label == c, [f"grid_{g}_hurst" for g in range(150)]].mean().values.reshape(N_ROWS, N_COLS))
    vmin, vmax = np.min(vals), np.max(vals)
    for k, (c, v) in enumerate(zip(["NORMAL", "CSR", "DR", "MH"], vals)):
        ax = fig.add_subplot(gs[k // 2, k % 2])
        ax.set_anchor("W")
        im = ax.imshow(v, cmap=SEQ, vmin=vmin, vmax=vmax)
        ax.set_title(f"mean H, {c}", loc="left", fontsize=15); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for g in top:
            r, cc = divmod(g, N_COLS)
            ax.add_patch(Rectangle((cc - 0.5, r - 0.5), 1, 1, fill=False, ec=ps.SERIES[1], lw=2))
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[:, 2]))
    cb.ax.yaxis.set_ticks_position("left"); cb.ax.yaxis.set_label_position("left")
    cb.set_label("Hurst H")
    fig.text(0.02, 0.01, "orange boxes: the paper's top-15 Gini cells (rows 7–8 = where the retina's lower edge sits)", fontsize=13, color=ps.INK2)
    ax = fig.add_subplot(gs[:, 3])
    if imp:
        fams, bands = imp["families"], imp["bands"]
        M = np.array([[imp["importance"][f].get(b, np.nan) for b in bands] for f in fams])
        lim = np.nanmax(np.abs(M))
        heat(ax, M, fams, bands, fmt="{:.3f}", cmap=DIV, norm_=TwoSlopeNorm(0, -lim, lim), fs=13)
        ax.set_title("Ours: permutation importance\n(feature family × anatomical band)", loc="left", fontsize=15)
        ax.set_xlabel("anatomical band (flattened retina)")
        ax.yaxis.tick_right()
        i, j = np.unravel_index(np.nanargmax(M), M.shape)
        record("fig06_where", "top family x band", f"{fams[i]} / {bands[j]} = {M[i, j]:.4f}", V6 / "importance_band_family.json")
    else:
        missing(ax, "v6_baselines.py")
    record("fig06_where", "paper top15 Hurst fraction", res["top15_hurst_fraction"], R / "results.json")
    save(fig, "fig06_where_looks.png")


# ============================================================================ figure 7: fractal validation
def fig_fractal():
    fv = js(R / "v4/fractal_validation.json")
    fig, ax = plt.subplots(figsize=(8, 6.3))
    for key, lab, col, mk in (("profile_1d", "1-D profile estimator", ps.SERIES[2], "s"),
                              ("paper_grid", "paper's 50×50 grid estimator", ps.SERIES[1], "o"),
                              ("paper_grid_upsampled", "same, after 1.32× upsampling", ps.SERIES[4], "^")):
        h = np.array(sorted(float(k) for k in fv[key]))
        m = np.array([fv[key][f"{x:g}" if f"{x:g}" in fv[key] else str(x)][0] for x in h])
        s = np.array([fv[key][f"{x:g}" if f"{x:g}" in fv[key] else str(x)][1] for x in h])
        ax.errorbar(h, m, yerr=s, fmt=mk + "-", color=col, ms=8, lw=2, capsize=3, label=lab, mec=ps.SURFACE)
    ax.plot([0, 1], [0, 1], color=ps.INK2, lw=1.2, ls="--", label="y = x (perfect)")
    ax.set_xlabel("true H of synthetic fBm"); ax.set_ylabel("measured H")
    ax.set_title("The paper's estimator over-reads low H", loc="left")
    ax.legend(loc="lower right", fontsize=12.5)
    record("fig07_fractal", "paper grid at true H=0.1", fv["paper_grid"]["0.1"], R / "v4/fractal_validation.json")
    record("fig07_fractal", "upsampled at true H=0.1", fv["paper_grid_upsampled"]["0.1"], R / "v4/fractal_validation.json")
    save(fig, "fig07_fractal_validation.png")


# ============================================================================ figure 8: MAPOD
def fig_mapod():
    mp = js(R / "v3/mapod.json")
    fig, ax = plt.subplots(figsize=(9, 6))
    a = np.linspace(5, 220, 400)
    for key, lab, col in (("kermany|srf|NDT half-max", "subretinal fluid (CSR-like)", ps.SERIES[1]),
                          ("kermany|cyst|NDT half-max", "intraretinal cyst", ps.SERIES[0])):
        p = mp[key]
        ax.plot(a, norm.cdf((np.log(a) - p["mu"]) / p["sigma"]), color=col, lw=2.5, label=lab)
        ax.axvline(p["a90"], color=col, lw=1.2, ls="--")
        ax.plot([p["a90_95"]], [0.9], marker="D", color=col, ms=9, mec=ps.SURFACE, mew=1.5)
        ax.text(p["a90"] + 2, 0.08 if "srf" in key else 0.2, f"$a_{{90}}$ = {p['a90']:.0f} µm\n$a_{{90/95}}$ = {p['a90_95']:.0f} µm",
                color=ps.INK, fontsize=13)
        record("fig08_mapod", f"{key} a90", p["a90"], R / "v3/mapod.json")
        record("fig08_mapod", f"{key} a90_95", p["a90_95"], R / "v3/mapod.json")
    ax.axhline(0.9, color=ps.INK2, lw=0.8)
    ax.set_xlabel("defect height a (µm)"); ax.set_ylabel("probability of detection POD(a)")
    ax.set_title("MAPOD (Kermany, synthetic flaws): how small can we see?", loc="left")
    ax.legend(loc="center right")
    save(fig, "fig08_mapod.png")


# ============================================================================ figure 9: ROM (reuse v5 images)
def fig_rom():
    for src, dst in (("rom_energy.png", "fig09a_rom_energy.png"), ("rom_modes.png", "fig09b_rom_modes.png"),
                     ("rom_energy_vs_info.png", "fig09c_rom_energy_vs_info.png"),
                     ("../v4/fractal_by_class.png", "fig13b_fractal_by_class.png")):
        shutil.copy(R / "v5" / src, FIG / dst)
        print("  copied", FIG / dst)
    rom = js(R / "v5/rom.json")
    # accuracy vs number of POD coefficients, regenerated larger from rom.json
    acc = pd.DataFrame(rom["accuracy"])
    tasks = list(dict.fromkeys(acc.task))
    fig, axs = plt.subplots(1, len(tasks), figsize=(18, 5.8))
    for ax, t in zip(axs, tasks):
        a = acc[acc.task == t]
        for fam, col, lab in (("paper", ps.SERIES[1], "paper grid, POD"), ("texture", ps.SERIES[0], "anatomy texture, POD")):
            f = a[(a.family == fam) & a.config.str.contains("POD")].sort_values("n_features")
            raw = a[(a.family == fam) & a.config.str.contains("raw")].iloc[0]
            ax.plot(f.n_features, f.bacc, "o-", color=col, lw=2.2, ms=8, mec=ps.SURFACE, label=lab)
            ax.axhline(raw.bacc, color=col, lw=1.4, ls="--", label=f"{lab.split(',')[0]}, all raw features")
        ref = a[a.family == "reference"].iloc[0]
        ax.axhline(ref.bacc, color=ps.INK2, lw=1.2, ls=":", label="repo best set (171 features)")
        ax.set_xscale("log"); ax.set_title(t.replace(" (500x750)", ""), loc="left", fontsize=15)
        ax.set_xlabel("number of features given to the classifier")
        record("fig09d_rom_accuracy", t, {r.config: r.bacc for r in a.itertuples()}, R / "v5/rom.json")
    axs[0].set_ylabel("balanced accuracy (LR)")
    axs[0].legend(fontsize=11.5, loc="lower right")
    fig.tight_layout()
    save(fig, "fig09d_rom_accuracy.png")
    rho = rom["octid_spearman_mode_vs_tissue_height"]
    record("fig09_rom", "mode 2 vs tissue height rho (OCTID)", rho[1] if isinstance(rho, list) else rho, R / "v5/rom.json")
    for k in ("octid | paper H field (150 cells)", "octid | v2 band x sector H field (20 cells)"):
        record("fig09_rom", f"{k} modes for 95% energy", rom["energy"][k]["r95"], R / "v5/rom.json")


# ============================================================================ figure 10: model x task heat map
def stage_values():
    """rows -> {task: (bacc, ci or None, source)}"""
    rows = {k: {} for k in ["Paper RF", "v2 anatomy", "v3 + NDT", "v4 + fractal", "LR", "LightGBM", "Soft vote",
                            "CNN (fine-tuned)", "ResNet-18 emb. only", "DINOv2-S emb. only", "v6 hybrid DINOv2-S",
                            "v6 hybrid ResNet-18"]}
    for task, (ds, gt) in GRID_TASK.items():
        for v, row in (("v2", "v2 anatomy"), ("v3", "v3 + NDT")):
            g = pd.read_csv(R / v / f"{ds}_model_grid.csv")
            g = g[g.task == gt]
            if v == "v2":
                p = g[(g.features == "paper (300 grid H+S)") & (g.model == "RF (paper config)")].iloc[0]
                rows["Paper RF"][task] = (p.bacc_mean, (p.bacc_ci_lo, p.bacc_ci_hi), R / v / f"{ds}_model_grid.csv")
                o = g[~g.features.str.startswith("paper")]
            else:
                o = g[g.features.str.contains("NDT")]
            b = o.loc[o.bacc_mean.idxmax()]
            rows[row][task] = (b.bacc_mean, (b.bacc_ci_lo, b.bacc_ci_hi), R / v / f"{ds}_model_grid.csv")
    fp = js(R / "v4/fractal_paired_test.json")
    for t in GRID_TASK:
        c = [r["with_bacc"] for r in fp if r["task"] == t]
        if c:
            rows["v4 + fractal"][t] = (max(c), None, R / "v4/fractal_paired_test.json")
    ens = js(R / "v5/ensemble.json")
    for r in ens:
        if r["task"] in GRID_TASK:
            for m, row in (("Logistic regression", "LR"), ("LightGBM", "LightGBM"), ("Soft vote (LR+GBM+SVM)", "Soft vote")):
                rows[row][r["task"]] = (r["models"][m]["bacc"], tuple(r["models"][m]["bacc_ci"]), R / "v5/ensemble.json")
    X = "Kermany -> OCTDL (NORMAL/DME/AMD)"
    for v, row in (("v2", "v2 anatomy"), ("v3", "v3 + NDT")):
        cs = js(R / v / "cross_scanner.json")
        if v == "v2":
            p = [r for r in cs if r["features"].startswith("paper (300") and r["model"] == "RF (paper config)"][0]
            rows["Paper RF"][X] = (p["ext_bacc"], None, R / v / "cross_scanner.json")
            o = [r for r in cs if not r["features"].startswith("paper")]
        else:
            o = [r for r in cs if "NDT" in r["features"]]
        rows[row][X] = (max(r["ext_bacc"] for r in o), None, R / v / "cross_scanner.json")
    hz = js(R / "v5/harmonize.json")
    for r in hz:
        if r["features"] == "v2 band-summary + disease + NDT robust + fractal":
            row = {"Logistic regression": "LR", "LightGBM": "LightGBM", "Soft vote (LR+GBM+SVM)": "Soft vote"}[r["model"]]
            rows[row][X] = (r["none"], None, R / "v5/harmonize.json")
    for t in TASKS:
        c = cnn(t)
        if c:
            rows["CNN (fine-tuned)"][t] = (c["bacc"], tuple(c["bacc_ci"]) if "bacc_ci" in c else None, V6 / f"cnn_{SLUG[t]}.json")
        m = v6_models(t)
        for row, key in (("v6 hybrid ResNet-18", HYB), ("v6 hybrid DINOv2-S", HYB2),
                         ("ResNet-18 emb. only", "ResNet-18 embedding only"), ("DINOv2-S emb. only", "DINOv2-S embedding only")):
            if key in m:
                rows[row][t] = (m[key]["bacc"], tuple(m[key].get("bacc_ci", [np.nan, np.nan])), V6 / f"hybrid_{SLUG[t]}.json")
    return rows


def fig_model_task():
    rows = stage_values()
    names = list(rows)
    M = np.array([[rows[r].get(t, (np.nan,))[0] for t in TASKS] for r in names], float)
    fig, ax = plt.subplots(figsize=(17, 9))
    heat(ax, M, names, [SHORT[t] for t in TASKS], fmt="{:.3f}", vmin=0.3, vmax=1.0, fs=14)
    for j in range(M.shape[1]):
        if np.isfinite(M[:, j]).any():
            i = int(np.nanargmax(M[:, j]))
            ax.add_patch(Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, fill=False, ec=ps.SERIES[1], lw=3.5))
    ax.set_title("Balanced accuracy by model and task (orange box = best in column)", loc="left")
    ax.xaxis.tick_top()
    fig.tight_layout()
    for r in names:
        for t in TASKS:
            if t in rows[r]:
                record("fig10_model_task", f"{r} | {t}", float(rows[r][t][0]), rows[r][t][2])
    save(fig, "fig10_model_task_heatmap.png")
    return rows


# ============================================================================ figure 11: conformal
def fig_conformal():
    v6 = js(V6 / "conformal.json")
    v5 = js(R / "v5/conformal.json")
    src, data, tag = (V6 / "conformal.json", v6, "v6 hybrid") if v6 else (R / "v5/conformal.json", v5, "v5 soft vote")
    rowsl, M = [], []
    for task in dict.fromkeys(r["task"] for r in data):
        rs = {r["alpha"]: r for r in data if r["task"] == task}
        for c in rs[0.05]["coverage_per_class"]:
            rowsl.append(f"{task.replace(' NORMAL vs ABNORMAL', '').replace(' (500x750)', '')} · {c}")
            M.append([rs[a]["coverage_per_class"][c] for a in (0.05, 0.10)])
    fig, axs = plt.subplots(1, 2, figsize=(18, 8.5), gridspec_kw=dict(width_ratios=[1, 1.3]))
    heat(axs[0], M, rowsl, ["α = 0.05\n(target 0.95)", "α = 0.10\n(target 0.90)"], fmt="{:.3f}", vmin=0.8, vmax=1.0, fs=13)
    axs[0].set_title(f"Coverage per class ({tag})", loc="left")
    ax = axs[1]
    tasks = list(dict.fromkeys(r["task"] for r in data))
    y = np.arange(len(tasks))
    for k, (a, col) in enumerate(((0.05, ps.SERIES[0]), (0.10, ps.SERIES[2]))):
        vals = [[r for r in data if r["task"] == t and r["alpha"] == a][0]["mean_set_size"] for t in tasks]
        ax.barh(y + (k - 0.5) * 0.36, vals, height=0.34, color=col, label=f"{tag}, α = {a:.2f}", edgecolor=ps.SURFACE, lw=2)
        for yi, v in zip(y + (k - 0.5) * 0.36, vals):
            ax.text(v + 0.02, yi, f"{v:.2f}", va="center", fontsize=12.5, color=ps.INK2)
        if v6:
            ref = [[r for r in v5 if r["task"] == t and r["alpha"] == a][0]["mean_set_size"] for t in tasks]
            ax.scatter(ref, y + (k - 0.5) * 0.36, marker="|", s=420, color=ps.INK, zorder=5,
                       label="v5 soft vote (same α)" if k == 0 else None)
    ax.set_yticks(y, [t.replace(" NORMAL vs ABNORMAL", "\nNORMAL vs ABNORMAL") for t in tasks]); ax.invert_yaxis()
    ax.axvline(1, color=ps.INK2, lw=0.8, ls=":")
    ax.set_xlabel("mean prediction-set size (1 = always one label)")
    ax.set_title("How many labels the model needs", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=12)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    for r in data:
        record("fig11_conformal", f"{r['task']} alpha={r['alpha']} coverage", r["coverage_per_class"], src)
        record("fig11_conformal", f"{r['task']} alpha={r['alpha']} set size", r["mean_set_size"], src)
    save(fig, "fig11_conformal.png")


# ============================================================================ figure 12: harmonisation
def fig_harmonise():
    hz = [r for r in js(R / "v5/harmonize.json") if r["features"] == "v2 band-summary + disease + NDT robust + fractal"]
    cals = ["none", "z-all", "healthy-ref k=10", "healthy-ref k=30", "healthy-ref k=100"]
    models = [r["model"].replace(" (LR+GBM+SVM)", "") for r in hz]
    M = np.array([[r[c] for r in hz] for c in cals])
    cr = js(V6 / "cross.json")
    fig, axs = plt.subplots(1, 2, figsize=(18, 6.8), gridspec_kw=dict(width_ratios=[1, 1.15]))
    heat(axs[0], M, cals, models, fmt="{:.3f}", vmin=0.55, vmax=0.8)
    axs[0].set_title("v5 features: calibration × model", loc="left"); axs[0].set_ylabel("calibration")
    for r in hz:
        record("fig12_harmonise", r["model"], {c: r[c] for c in cals}, R / "v5/harmonize.json")
    if cr:
        keys = list(cr["configs"])
        mods = list(dict.fromkeys(k.split(" | ")[0] for k in keys))
        cc = list(dict.fromkeys(" | ".join(k.split(" | ")[1:]) for k in keys))
        N = np.array([[cr["configs"].get(f"{m} | {c}", {}).get("bacc", np.nan) for m in mods] for c in cc])
        heat(axs[1], N, cc, [m.replace(" (DINOv2-S + physics)", "\n(DINOv2-S + physics)").replace(" (soft vote)", "\n(soft vote)") for m in mods],
             fmt="{:.3f}", vmin=0.55, vmax=0.8)
        axs[1].set_title("v6: calibration / speckle TTA × model", loc="left")
        for k in keys:
            record("fig12_harmonise", k, cr["configs"][k]["bacc"], V6 / "cross.json")
    else:
        missing(axs[1], "v6_cross.py")
    fig.tight_layout()
    save(fig, "fig12_harmonisation.png")


# ============================================================================ figure 13: progress
def fig_progress(rows):
    order = [("Paper RF", "paper method"), ("v2 anatomy", "v2"), ("v3 + NDT", "v3"), ("v4 + fractal", "v4"),
             ("Soft vote", "v5 vote"), ("v6 hybrid ResNet-18", "v6 hybrid"), ("CNN (fine-tuned)", "CNN")]
    fig, axs = plt.subplots(2, 3, figsize=(19, 10), sharey=True)
    cols = [ps.SERIES[1], ps.SEQ_BLUE[1], ps.SEQ_BLUE[2], ps.SEQ_BLUE[3], ps.SEQ_BLUE[4], ps.SEQ_BLUE[6], ps.INK2]
    for ax, t in zip(axs.ravel(), TASKS):
        for i, ((r, lab), col) in enumerate(zip(order, cols)):
            v = rows[r].get(t)
            if v is None or not np.isfinite(v[0]):
                ax.text(i, 0.02, "n/a", ha="center", fontsize=11, color=ps.INK2, rotation=90)
                continue
            ci = v[1]
            err = None if ci is None or not np.all(np.isfinite(ci)) else [[v[0] - ci[0]], [ci[1] - v[0]]]
            ax.bar(i, v[0], color=col, width=0.72, edgecolor=ps.SURFACE, lw=2)
            if err:
                ax.errorbar(i, v[0], yerr=err, color=ps.INK, capsize=4, lw=1.4)
            ax.text(i, v[0] + (err[1][0] if err else 0) + 0.015, f"{v[0]:.2f}", ha="center", fontsize=11.5, color=ps.INK2)
        ax.set_xticks(range(len(order)), [o[1] for o in order], rotation=40, ha="right", fontsize=12.5)
        ax.set_title(SHORT[t].replace("\n", " "), loc="left", fontsize=15)
        ax.set_ylim(0, 1.08); ax.grid(axis="x", visible=False)
    axs[0, 0].set_ylabel("balanced accuracy"); axs[1, 0].set_ylabel("balanced accuracy")
    fig.suptitle("Progress per task (error bars: 95% patient cluster-bootstrap CI where available)", x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "fig13_progress.png")


# ============================================================================ figure 14: v6 image level
def fig_v6():
    # (a) image-POD modes + energy
    descs = [(s, V6 / f"image_pod_desc_{s}.npz") for s in ("octid_dis750", "kermany_4")]
    fig, axs = plt.subplots(2, 5, figsize=(20, 5.6))
    for r, (s, p) in enumerate(descs):
        if not p.exists():
            for a in axs[r]:
                missing(a, p.name)
            continue
        z = np.load(p, allow_pickle=True)
        sh = tuple(z["shape"])
        ims = [z["mean"]] + [z["modes"][i] for i in range(4)]
        for j, (a, im) in enumerate(zip(axs[r], ims)):
            v = im.reshape(sh)
            if j == 0:
                a.imshow(v, cmap="gray")
            else:
                lim = np.abs(v).max()
                a.imshow(v, cmap=DIV, vmin=-lim, vmax=lim)
            a.set_xticks([]); a.set_yticks([]); a.grid(False)
            a.set_title(("mean retina" if j == 0 else f"mode {j} ({100 * (z['energy'][j - 1] - (z['energy'][j - 2] if j > 1 else 0)):.1f}% energy)"),
                        fontsize=13, loc="left")
        axs[r, 0].set_ylabel({"octid_dis750": "OCTID", "kermany_4": "Kermany"}[s], fontsize=15)
    fig.suptitle("Eigen-retinas: POD of the RPE-flattened image (descriptive fit on all scans)", x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "fig14a_image_pod_modes.png")

    # (b) accuracy vs r
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, t in enumerate(TASKS):
        ip = js(V6 / f"image_pod_{SLUG[t]}.json")
        if not ip:
            continue
        rs = [int(r) for r in ip["pod_sweep"]["LR"]]
        best = [max(ip["pod_sweep"][m][str(r)]["bacc"] for m in ip["pod_sweep"]) for r in rs]
        ax.plot(rs, best, "o-", color=ps.SERIES[i], lw=2, ms=8, mec=ps.SURFACE, label=SHORT[t].replace("\n", " "))
        v5 = ip["models"].get("v5 best (soft vote)", {}).get("bacc")
        if v5:
            ax.axhline(v5, color=ps.SERIES[i], lw=1, ls=":")
        record("fig14b_pod_r", t, dict(zip(rs, best)), V6 / f"image_pod_{SLUG[t]}.json")
    ax.set_xscale("log"); ax.set_xticks([10, 25, 50, 100, 200], ["10", "25", "50", "100", "200"])
    ax.set_xlabel("number of image-POD modes r"); ax.set_ylabel("balanced accuracy (best of LR / LightGBM)")
    ax.set_title("Image-POD alone vs v5 features (dotted)", loc="left")
    ax.legend(fontsize=11.5, loc="lower right", ncol=2)
    save(fig, "fig14b_image_pod_accuracy.png")

    # (c) residual maps
    for s in ("kermany_4", "octdl_7"):
        p = V6 / f"residual_maps_{s}.npz"
        if not p.exists():
            continue
        z = np.load(p, allow_pickle=True)
        n = len(z["idx"])
        cols = int(np.ceil(n / 2))
        fig, axs = plt.subplots(4, cols, figsize=(3.6 * cols, 7.4))
        for i in range(n):
            c, rr = i % cols, (i // cols) * 2
            t, rs = z["tiles"][i].reshape(64, 256), z["resid"][i].reshape(64, 256)
            axs[rr, c].imshow(t, cmap="gray"); axs[rr, c].set_title(f"{z['labels'][i]}  e = {z['e'][i]:.2f}", fontsize=12.5, loc="left")
            lim = np.percentile(np.abs(z["resid"]), 99)
            axs[rr + 1, c].imshow(rs, cmap=DIV, vmin=-lim, vmax=lim)
        for a in axs.ravel():
            a.set_xticks([]); a.set_yticks([]); a.grid(False)
        fig.suptitle(f"POD residual maps ({'Kermany' if s.startswith('kermany') else 'OCTDL'}): scan (top) and where it departs from "
                     "a healthy retina (bottom; red = darker, blue = brighter than healthy)", x=0.01, ha="left", fontsize=15)
        fig.tight_layout()
        save(fig, f"fig14c_residual_maps_{s}.png")

    # (d) Grad-CAM vs physics
    g = js(V6 / "gradcam_octid_dis750.json")
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.6))
    if g:
        bands, classes = g["bands"], g["classes"]
        C = np.array([[g["cam_mass_fraction"][b][k] for k in classes] for b in bands])
        P = np.array([[g["physics_band_importance"][b][k] for k in classes] for b in bands])
        heat(axs[0], C, bands, classes, fmt="{:.2f}", vmin=0)
        axs[0].set_title("CNN Grad-CAM: share of attention per band", loc="left", fontsize=15)
        lim = np.abs(P).max()
        heat(axs[1], P, bands, classes, fmt="{:.3f}", cmap=DIV, norm_=TwoSlopeNorm(0, -lim, lim))
        axs[1].set_title("Physics model: recall drop when band is shuffled", loc="left", fontsize=15)
        fig.suptitle(f"Do they look at the same layers? Spearman ρ = {g['spearman_rho']:.2f} over {g['n_cells']} band×class cells "
                     f"(p = {g['spearman_p']:.2f})", x=0.01, ha="left", fontsize=16)
        record("fig14d_gradcam", "spearman rho", g["spearman_rho"], V6 / "gradcam_octid_dis750.json")
    else:
        for a in axs:
            missing(a, "v6_gradcam.py")
    fig.tight_layout()
    save(fig, "fig14d_gradcam_vs_physics.png")

    # (e) hybrid vs CNN vs v5
    fig, ax = plt.subplots(figsize=(15, 6.2))
    series = [("v5 best (soft vote)", ps.SEQ_BLUE[3]), (HYB, ps.SEQ_BLUE[6]), ("CNN", ps.SERIES[1])]  # HYB = ResNet-18 hybrid
    x = np.arange(len(TASKS))
    for k, (name, col) in enumerate(series):
        for i, t in enumerate(TASKS):
            if name == "CNN":
                c = cnn(t); v = (c["bacc"], c.get("bacc_ci")) if c else None; src = V6 / f"cnn_{SLUG[t]}.json"
            else:
                m = v6_models(t).get(name); v = (m["bacc"], m.get("bacc_ci")) if m else None
                src = V6 / (f"image_pod_{SLUG[t]}.json" if name.startswith("v5") else f"hybrid_{SLUG[t]}.json")
            xi = i + (k - 1) * 0.27
            if v is None:
                ax.text(xi, 0.03, "n/a", ha="center", rotation=90, fontsize=11, color=ps.INK2); continue
            ax.bar(xi, v[0], width=0.25, color=col, edgecolor=ps.SURFACE, lw=2, label=name if i == 0 else None)
            if v[1]:
                ax.errorbar(xi, v[0], yerr=[[v[0] - v[1][0]], [v[1][1] - v[0]]], color=ps.INK, capsize=3, lw=1.2)
            ax.text(xi, 0.03, f"{v[0]:.2f}", ha="center", rotation=90, fontsize=11, color="white")
            record("fig14e_hybrid_vs_cnn", f"{name} | {t}", v[0], src)
    ax.set_xticks(x, [SHORT[t] for t in TASKS], fontsize=12.5)
    ax.set_ylabel("balanced accuracy"); ax.set_ylim(0, 1.05); ax.grid(axis="x", visible=False)
    ax.set_title("v6 hybrid vs v5 and the fine-tuned CNN (95% CI; CNN run on 3 tasks)", loc="left")
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.17))
    save(fig, "fig14e_hybrid_vs_cnn.png")


# ============================================================================ figure 14f: ablation
def fig_ablation():
    ab = js(V6 / "ablation.json")
    fig, ax = plt.subplots(figsize=(15, 6.4))
    if not ab:
        missing(ax, "v6_ablation.py"); save(fig, "fig14f_ablation.png"); return
    series = [("ResNet-18 embedding only", ps.SEQ_BLUE[2], "ResNet-18 embedding only"),
              ("hybrid ResNet-18", ps.SEQ_BLUE[6], "ResNet-18 + physics + ROM (headline)"),
              ("DINOv2-S embedding only", "#f6c3a6", "DINOv2-S embedding only"),
              ("hybrid DINOv2-S", ps.SERIES[1], "DINOv2-S + physics + ROM")]
    w = 0.2
    for i, t in enumerate(TASKS):
        m = ab["tasks"][t]["models"]
        for k, (name, col, lab) in enumerate(series):
            v, ci = m[name]["bacc"], m[name].get("bacc_ci")
            x = i + (k - 1.5) * w
            ax.bar(x, v, width=w * 0.95, color=col, edgecolor=ps.SURFACE, lw=1.5, label=lab if i == 0 else None)
            if ci:
                ax.errorbar(x, v, yerr=[[v - ci[0]], [ci[1] - v]], color=ps.INK, capsize=3, lw=1.1)
            record("fig14f_ablation", f"{name} | {t}", v, V6 / "ablation.json")
        p = ab["tasks"][t]["paired"]["hybrid ResNet-18 vs ResNet-18 embedding only"]
        sig = p["diff_ci"][0] > 0 or p["diff_ci"][1] < 0
        top = max(m[n].get("bacc_ci", [0, m[n]["bacc"]])[1] for n, _, _ in series)
        ax.text(i, top + 0.03, f"{p['diff_bacc']:+.3f}" + (" *" if sig else ""), ha="center", fontsize=14,
                fontweight="bold" if sig else "normal", color=ps.INK)
        record("fig14f_ablation", f"paired ResNet-18 hybrid - embedding | {t}", p, V6 / "ablation.json")
    ax.set_xticks(range(len(TASKS)), [SHORT[t] for t in TASKS], fontsize=13)
    ax.set_ylim(0.4, 1.08); ax.set_ylabel("balanced accuracy (95% CI)"); ax.grid(axis="x", visible=False)
    ax.set_title("What physics + ROM add on top of a frozen embedding (number = paired ResNet-18 difference, * = CI excludes 0)",
                 loc="left", fontsize=15)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, fontsize=12.5)
    save(fig, "fig14f_ablation.png")


# ============================================================================ figure 15: paper vs ours
def triage_metrics(o):
    t = np.where(o.true == "NORMAL", "NORMAL", "ABNORMAL")
    p = np.where(o.pred == "NORMAL", "NORMAL", "ABNORMAL")
    rn = float(((p == "NORMAL") & (t == "NORMAL")).sum() / max((t == "NORMAL").sum(), 1))
    pn = float(((p == "NORMAL") & (t == "NORMAL")).sum() / max((p == "NORMAL").sum(), 1))
    ra = float(((p == "ABNORMAL") & (t == "ABNORMAL")).sum() / max((t == "ABNORMAL").sum(), 1))
    return {"NORMAL recall": rn, "NORMAL precision": pn, "ABNORMAL recall": ra, "balanced acc": (rn + ra) / 2}


def multiclass_metrics(o):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    return {"accuracy": accuracy_score(o.true, o.pred), "balanced acc": balanced_accuracy_score(o.true, o.pred),
            "macro-F1": f1_score(o.true, o.pred, average="macro")}


def fig_paper_vs_ours():
    pc = js(V6 / "paper_claims.json")
    bl = js(V6 / "baselines.json")
    metrics = ["accuracy", "balanced acc", "macro-F1", "NORMAL recall", "NORMAL precision", "ABNORMAL recall"]
    cols, data, srcs = [], [], []

    def add(name, d, src):
        cols.append(name); data.append([d.get(m, np.nan) for m in metrics]); srcs.append(str(src))

    add("OCTID\n4-class\npaper\nreported", {"accuracy": pc["multiclass_accuracy"]["value"]}, V6 / "paper_claims.json")
    add("OCTID\n3 dis.\npaper\nmethod", {k: bl["paper_cv_octid_dis750"][kk] for k, kk in
                                                         (("accuracy", "acc"), ("balanced acc", "bacc"), ("macro-F1", "macro_f1"))},
        V6 / "baselines.json")
    name, best = v6_best("OCTID CSR/DR/MH (500x750)")
    if best:
        add("OCTID\n3 dis.\nours", {"accuracy": best["acc"], "balanced acc": best["bacc"], "macro-F1": best["macro_f1"]},
            V6 / f"oof_octid_dis750_{model_slug(name)}.csv")
    v = pc["normal_triage_precision_recall"]["value"]
    add("Triage\npaper\nreported", {"NORMAL recall": v, "NORMAL precision": v}, V6 / "paper_claims.json")
    o = oof(V6 / "oof_octid_4_confounded_paper_rf_8020.csv")
    if o is not None:
        add("Triage\npaper\nmethod", triage_metrics(o), V6 / "oof_octid_4_confounded_paper_rf_8020.csv")
    for t, lab in (("OCTDL 7-class", "OCTDL"), ("Kermany 4-class", "Kermany")):
        name, _ = v6_best(t)
        o = oof(V6 / f"oof_{SLUG[t]}_{model_slug(name)}.csv") if name else None
        if o is not None:
            add(f"Triage\nours\n{lab}", triage_metrics(o), V6 / f"oof_{SLUG[t]}_{model_slug(name)}.csv")
    M = np.array(data, float).T
    fig, ax = plt.subplots(figsize=(13.2, 6.0))   # placed on a slide at ~0.86 scale: 17-19 pt here -> >= 14.5 pt there
    heat(ax, M, metrics, cols, fmt="{:.2f}", vmin=0.3, vmax=1.0, fs=19)
    ax.xaxis.tick_top()
    plt.setp(ax.get_xticklabels(), fontsize=17)
    plt.setp(ax.get_yticklabels(), fontsize=18)
    fig.tight_layout()
    for c, d, s in zip(cols, data, srcs):
        record("fig15a_paper_vs_ours", c.replace("\n", " "), dict(zip(metrics, [None if not np.isfinite(x) else x for x in d])), s)
    save(fig, "fig15a_paper_vs_ours.png")
    claims_table()


def claims_table():
    pc, sc, res = js(V6 / "paper_claims.json"), js(R / "sanity_checks.json"), js(R / "results.json")
    ctl, rom, bl = js(R / "v2/octid_controls.json"), js(R / "v5/rom.json"), js(V6 / "baselines.json")
    imp, conf, gc = js(V6 / "importance_band_family.json"), js(V6 / "conformal.json") or js(R / "v5/conformal.json"), js(V6 / "gradcam_octid_dis750.json")
    g = pd.read_csv(R / "v2/octid_model_grid.csv"); g4 = g[g.task == "all classes (CSR/DR/MH/NORMAL)"]
    ent0, ent1 = g4[g4.features == "paper (300 grid H+S)"].bacc_mean.max(), g4[g4.features == "paper + entropy"].bacc_mean.max()
    tri = [r for r in conf if r["task"] == "OCTDL triage NORMAL vs ABNORMAL" and r["alpha"] == 0.05][0]
    fam = max(((f, b, v) for f, d in imp["importance"].items() for b, v in d.items()), key=lambda x: x[2]) if imp else None
    rho = rom["octid_spearman_mode_vs_tissue_height"]
    rho2 = rho[1] if isinstance(rho, list) else rho
    cnn_o = cnn("OCTID CSR/DR/MH (500x750)")
    hy = v6_models("OCTID CSR/DR/MH (500x750)").get(HYB)
    rows = [
        {"claim": f"Multiclass accuracy {pc['multiclass_accuracy']['value']:.0%} on {pc['n_test_scans']['value']} test scans",
         "found": f"Reproduced: {res['accuracy']:.1%} on the same 80/20 split; 5-fold CV {sc['cv_accuracy_mean']:.1%}",
         "verdict": "holds", "evidence": "results/results.json, results/sanity_checks.json"},
        {"claim": f"NORMAL triage precision/recall {pc['normal_triage_precision_recall']['value']:.2f}, 'mathematically guaranteed'",
         "found": f"Scanner confound: image size alone separates NORMAL with {sc['size_only_normal_vs_disease_acc']:.0%}; resized disease "
                  f"scans called NORMAL {sc['disease_called_normal_after_resample']}/{sc['disease_test_n']}. A stated guarantee via conformal "
                  f"prediction: ABNORMAL coverage {tri['coverage_per_class']['ABNORMAL']:.3f} at target 0.95 (OCTDL)",
         "verdict": "does not hold as stated", "evidence": "results/sanity_checks.json, " + ("results/v6/conformal.json" if js(V6 / "conformal.json") else "results/v5/conformal.json")},
        {"claim": f"Top-15 features are {pc['top15_hurst_fraction']['value']:.0%} Hurst, in rows {pc['top_grid_rows']['value'][0]}–{pc['top_grid_rows']['value'][1]}: 'autonomous localisation'",
         "found": f"Reproduced ({res['top15_hurst_fraction']:.0%} Hurst), but rows 7–8 track retina position: POD mode 2 vs tissue height "
                  f"ρ = {rho2:.2f}; destroying location (sorting cells) raises disease-only accuracy "
                  f"{ctl['(b) original']['disease_only_acc']:.2f} → {ctl['(c) sorted values (no location)']['disease_only_acc']:.2f}",
         "verdict": "partly position", "evidence": "results/results.json, results/v5/rom.json, results/v2/octid_controls.json"},
        {"claim": "Shannon entropy discarded because collinear with H",
         "found": f"Adding grid entropy improves OCTID 4-class balanced accuracy {ent0:.3f} → {ent1:.3f} (best model)",
         "verdict": "does not hold", "evidence": "results/v2/octid_model_grid.csv"},
        {"claim": "KT-1 and Hurst carry diagnostic signal",
         "found": (f"Yes, once measured inside anatomical layers: top group is {fam[0]} in the {fam[1]} band; paper grid "
                   f"{bl['paper_cv_octid_dis750']['bacc']:.2f} vs anatomy-aware features on the honest OCTID task") if fam else "n/a",
         "verdict": "holds (in anatomy)", "evidence": "results/v6/importance_band_family.json, results/v6/baselines.json"},
        {"claim": "Physics engine is explainable where black boxes are not",
         "found": (f"Explainability holds; a fine-tuned CNN is more accurate ({cnn_o['bacc']:.2f} vs hybrid {hy['bacc']:.2f} on OCTID CSR/DR/MH); "
                   f"Grad-CAM vs physics band importance ρ = {gc['spearman_rho']:.2f}") if (cnn_o and hy and gc) else "pending v6 runs",
         "verdict": "holds, at a cost", "evidence": "results/v6/cnn_octid_dis750.json, results/v6/hybrid_octid_dis750.json, results/v6/gradcam_octid_dis750.json"},
    ]
    json.dump(rows, open(FIG / "claims.json", "w"), indent=2)
    fig, ax = plt.subplots(figsize=(20, 9.5)); ax.axis("off")
    tb = ax.table(cellText=[[r["claim"], r["found"], r["verdict"], r["evidence"].replace(", ", "\n")] for r in rows],
                  colLabels=["Paper's statement", "What we found", "Verdict", "Evidence file"],
                  colWidths=[0.24, 0.46, 0.1, 0.2], loc="center", cellLoc="left")
    tb.auto_set_font_size(False); tb.set_fontsize(11.5)
    import textwrap
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor(ps.GRID)
        if r == 0:
            cell.set_facecolor("#e8f0fb"); cell.get_text().set_weight("bold"); cell.set_height(0.05)
        else:
            w = [36, 74, 14, 34][c]
            cell.get_text().set_text("\n".join(textwrap.wrap(cell.get_text().get_text(), w)) if c != 3 else cell.get_text().get_text())
            cell.set_height(0.15)
    save(fig, "fig15b_claims_table.png")


# ============================================================================ main
def main():
    FIG.mkdir(parents=True, exist_ok=True)
    style()
    steps = [fig_pipeline, fig_examples, fig_concepts, fig_confound, fig_confusions, fig_where, fig_fractal, fig_mapod,
             fig_rom, None, fig_conformal, fig_harmonise, None, fig_v6, fig_ablation, fig_paper_vs_ours]
    rows = None
    for f in steps:
        if f is None:
            continue
        print(f.__name__)
        f()
    print("fig_model_task"); rows = fig_model_task()
    print("fig_progress"); fig_progress(rows)
    json.dump(NUM, open(FIG / "figure_numbers.json", "w"), indent=1, default=float)
    print("wrote", FIG / "figure_numbers.json")


if __name__ == "__main__":
    main()
