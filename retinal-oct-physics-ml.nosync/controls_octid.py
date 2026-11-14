"""Controls for the paper's 'autonomous choroid localization' claim (OCTID).

(a) Where does the RPE actually sit in the paper's 750x500 frame, per class?
(b) Mask the lower-left overlay icon (present on disease scans only) in *every* image, recompute the
    paper features, and see whether accuracy / top-grid locations hold.
(c) Grid-shuffle: permute grid positions within each scan (keeps the bag of H/S values, destroys
    location). If accuracy barely drops, 'localization' carries little information.
(d) Grouped permutation importance per grid row, with vs. without the NORMAL class.
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

from retina.preprocess import load_gray, normalize, micro_slice
from retina.features import hurst, kt1_slopes
from retina.layers import trace
from retina.evaluation import rf_paper, repeated_cv, grouped_permutation_importance
from retina import plotstyle as ps
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
OUT = Path("results/v2")
ICON_GRIDS = [r * 15 + c for r in (7, 8, 9) for c in (0, 1, 2)]   # x<150, y>=350 in 750x500


def per_scan(path):
    norm = normalize(load_gray(path))
    img = norm.astype(np.float32) / 255
    rpe, _ = trace(img, norm > 0)
    rpe_rows = rpe[150:600]                         # central columns
    masked = norm.copy()
    masked[350:, :150] = 0                          # icon corner, applied to every scan
    g = micro_slice(masked)[ICON_GRIDS].astype(np.float64) / 255
    return np.nanmedian(rpe_rows), hurst(g), kt1_slopes(g)


def main():
    ps.apply()
    df = pd.read_csv(OUT / "octid_features.csv")
    res = {}
    rows = Parallel(n_jobs=6)(delayed(per_scan)(p) for p in df.path)
    df["rpe_y"] = [r[0] for r in rows]
    df["rpe_grid_row"] = (df.rpe_y // 50).astype("Int64")

    # (a)
    tab = pd.crosstab(df.label, df.rpe_grid_row, normalize="index").round(3)
    print("(a) grid row containing the RPE (central columns), fraction of scans per class:\n", tab)
    res["rpe_grid_row_by_class"] = tab.to_dict(orient="index")
    res["rpe_y_median_by_class"] = df.groupby("label").rpe_y.median().round(1).to_dict()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, c in enumerate(["NORMAL", "CSR", "DR", "MH"]):
        v = df.loc[df.label == c, "rpe_y"].dropna()
        ax.hist(v, bins=np.arange(0, 501, 10), histtype="step", lw=2, color=ps.CLASS_COLOR[c], label=c)
    for r in (7, 8, 9):
        ax.axvspan(r * 50, r * 50 + 50, color=ps.GRID, alpha=0.35 if r in (7, 8) else 0.15, lw=0)
    ax.set_xlabel("RPE depth in the paper's 750x500 frame (px); shaded = grid rows 7-8 (and 9)")
    ax.set_ylabel("scans"); ax.legend(title="class"); ax.set_title("Where the RPE sits, per class")
    plt.tight_layout(); plt.savefig(OUT / "octid_rpe_position.png", dpi=150); plt.close()

    pcols = [f"grid_{g}_hurst" for g in range(150)] + [f"grid_{g}_kt1" for g in range(150)]
    y, groups = df.label.values, df.patient.values
    X = df[pcols].values.copy()

    # (b) icon-masked features
    Xm = X.copy()
    for j, g in enumerate(ICON_GRIDS):
        Xm[:, g] = [r[1][j] for r in rows]
        Xm[:, 150 + g] = [r[2][j] for r in rows]
    dis = y != "NORMAL"
    for tag, XX in [("original", X), ("icon-masked", Xm)]:
        r4, _ = repeated_cv(rf_paper, XX, y, groups, repeats=3)
        r3, _ = repeated_cv(rf_paper, XX[dis], y[dis], groups[dis], repeats=3)
        Xtr, Xte, ytr, yte = train_test_split(XX, y, test_size=0.2, stratify=y, random_state=42)
        rf = rf_paper(42).fit(Xtr, ytr)
        imp = pd.Series(rf[-1].feature_importances_, index=pcols).sort_values(ascending=False)
        top = [int(c.split("_")[1]) for c in imp.index[:15]]
        res[f"(b) {tag}"] = {"4class_acc": r4["acc_mean"], "disease_only_acc": r3["acc_mean"],
                             "paper_split_acc": float((rf.predict(Xte) == yte).mean()),
                             "top15_grids": top, "top15_in_icon_corner": sum(g in ICON_GRIDS for g in top)}
        print(f"(b) {tag:12s}: 4-class CV acc {r4['acc_mean']:.3f}, disease-only CV acc {r3['acc_mean']:.3f}, "
              f"top-15 grids {top} ({res[f'(b) {tag}']['top15_in_icon_corner']} in icon corner)")

    # (c) grid shuffle
    rng = np.random.default_rng(0)
    Xs = X.copy()
    for i in range(len(Xs)):
        p = rng.permutation(150)
        Xs[i, :150] = X[i, :150][p]
        Xs[i, 150:] = X[i, 150:][p]
    Xsorted = np.c_[np.sort(X[:, :150], 1), np.sort(X[:, 150:], 1)]   # location-free 'bag of values'
    for tag, XX in [("grid positions shuffled per scan", Xs), ("sorted values (no location)", Xsorted)]:
        r4, _ = repeated_cv(rf_paper, XX, y, groups, repeats=3)
        r3, _ = repeated_cv(rf_paper, XX[dis], y[dis], groups[dis], repeats=3)
        res[f"(c) {tag}"] = {"4class_acc": r4["acc_mean"], "disease_only_acc": r3["acc_mean"]}
        print(f"(c) {tag:34s}: 4-class CV acc {r4['acc_mean']:.3f}, disease-only CV acc {r3['acc_mean']:.3f}")

    # (d) row importance
    row_groups = {f"row {r}": [g for g in range(r * 15, r * 15 + 15)] + [150 + g for g in range(r * 15, r * 15 + 15)]
                  for r in range(10)}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, (tag, m) in zip(axes, [("all classes", np.ones(len(y), bool)), ("disease-only", dis)]):
        imp = grouped_permutation_importance(rf_paper, X[m], y[m], groups[m], row_groups, n_rep=3)
        imp = imp.reindex(list(row_groups))
        res[f"(d) row importance, {tag}"] = imp.round(4).to_dict()
        print(f"(d) {tag}: " + ", ".join(f"{k}={v:+.3f}" for k, v in imp.items()))
        ps.hbar(ax, list(imp.index), list(imp.values))
        ax.set_title(f"Grid-row importance ({tag})")
        ax.set_xlabel("Drop in balanced accuracy (held-out)")
    plt.tight_layout(); plt.savefig(OUT / "octid_row_importance.png", dpi=150); plt.close()

    json.dump(res, open(OUT / "octid_controls.json", "w"), indent=2, default=str)
    print("saved", OUT / "octid_controls.json")


if __name__ == "__main__":
    main()
