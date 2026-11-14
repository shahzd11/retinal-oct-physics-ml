"""Robustness / confound checks for the reproduced pipeline (not in the paper).

1. Split variance: repeated stratified 5-fold CV instead of one 80/20 split.
2. Scanner leak: can image *size alone* separate NORMAL from disease?
3. Non-physics baseline: RF on plain 15x10 grid mean/std intensities.
4. Resampling confound: NORMAL scans are cubic-upsampled ~1.46x by Sec. 2.1 while
   most disease scans are not. Simulate that on held-out disease scans
   (shrink to 512 px wide -> run the identical pipeline) and see what the model says.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, cross_val_predict, train_test_split
from sklearn.metrics import recall_score, precision_score, balanced_accuracy_score

from retina.preprocess import load_gray, normalize, micro_slice
from retina.features import extract
from train_eval import make_rf, feature_cols

DATA = Path("data/OCTID")
OUT = Path("results")
res = {}

df = pd.read_csv(OUT / "features.csv")
cols = feature_cols(df)
X, y = df[cols].values, df["label"].values
is_norm = (y == "NORMAL").astype(int)

# ---- 1. split variance --------------------------------------------------------
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)
acc = cross_val_score(make_rf(0), X, y, cv=cv, scoring="accuracy", n_jobs=1)
bacc = cross_val_score(make_rf(0), X, y, cv=cv, scoring="balanced_accuracy", n_jobs=1)
res["cv_accuracy_mean"], res["cv_accuracy_std"] = acc.mean(), acc.std()
res["cv_accuracy_min_max"] = [acc.min(), acc.max()]
res["cv_balanced_accuracy_mean"] = bacc.mean()
# disease-only (3-class) performance: the clinically hard part
m = y != "NORMAL"
acc3 = cross_val_score(make_rf(0), X[m], y[m], cv=cv, scoring="accuracy", n_jobs=1)
res["cv_disease_only_3class_accuracy"] = acc3.mean()
print(f"[1] 5x10 CV acc {acc.mean():.3f} ± {acc.std():.3f} (range {acc.min():.3f}-{acc.max():.3f}); "
      f"balanced {bacc.mean():.3f}; disease-only 3-class {acc3.mean():.3f} (chance 0.33)")

# ---- 2. scanner leak -----------------------------------------------------------
size = df[["orig_h", "orig_w"]].values
print("\n[2] original image sizes by class:")
tab = pd.crosstab(df["orig_h"].astype(str) + "x" + df["orig_w"].astype(str), df["label"])
print(tab)
sz_pred = cross_val_predict(RandomForestClassifier(100, random_state=0), size, is_norm, cv=5)
res["size_only_normal_vs_disease_acc"] = float((sz_pred == is_norm).mean())
sz4 = cross_val_score(RandomForestClassifier(100, random_state=0), size, y, cv=5).mean()
res["size_only_4class_acc"] = sz4
print(f"    image size ALONE -> NORMAL vs disease acc = {res['size_only_normal_vs_disease_acc']:.3f}; 4-class acc = {sz4:.3f}")

# ---- 3. non-physics baseline ----------------------------------------------------
def simple_feats(path):
    g = micro_slice(normalize(load_gray(path))).astype(np.float32) / 255
    return np.r_[g.mean((1, 2)), g.std((1, 2))]

paths = [DATA / l / f for l, f in zip(df["label"], df["file"])]
B = np.array(Parallel(n_jobs=-1)(delayed(simple_feats)(p) for p in paths))
bacc_simple = cross_val_score(make_rf(0), B, y, cv=cv, scoring="accuracy", n_jobs=1)
bp = cross_val_predict(make_rf(0), B, y, cv=5)
res["baseline_meanstd_cv_accuracy"] = bacc_simple.mean()
res["baseline_meanstd_normal_recall"] = recall_score(y, bp, labels=["NORMAL"], average=None)[0]
res["baseline_meanstd_normal_precision"] = precision_score(y, bp, labels=["NORMAL"], average=None)[0]
print(f"\n[3] baseline (grid mean+std intensity, no physics): CV acc {bacc_simple.mean():.3f}; "
      f"NORMAL P/R {res['baseline_meanstd_normal_precision']:.3f}/{res['baseline_meanstd_normal_recall']:.3f}")

# ---- 4. resampling confound -----------------------------------------------------
def crop_scale(h, w):
    r = 1.5
    return 750 / (min(w, h * r))      # effective upsampling factor after 3:2 crop
df["scale"] = [crop_scale(h, w) for h, w in zip(df.orig_h, df.orig_w)]
hcols = [c for c in cols if c.endswith("_hurst")]
df["meanH"] = df[hcols].mean(axis=1)
rho_all, _ = spearmanr(df.scale, df.meanH)
print(f"\n[4] effective upsampling factor by class:\n{df.groupby('label').scale.describe()[['mean','min','max']]}")
print(f"    Spearman(scale, mean H) over all scans = {rho_all:.3f}")
res["spearman_scale_meanH"] = rho_all

# train on the paper's split, then degrade held-out *disease* test scans to NORMAL-like geometry
idx = np.arange(len(df))
tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=42)
rf = make_rf(42).fit(X[tr], y[tr])
te_dis = [i for i in te if y[i] != "NORMAL"]

def degraded_feats(path, target_w=512):
    raw = load_gray(path)
    h, w = raw.shape
    small = cv2.resize(raw, (target_w, int(round(h * target_w / w))), interpolation=cv2.INTER_AREA)
    H, S = extract(normalize(small), micro_slice)
    return np.r_[H, S]            # same column order as features.csv (hurst block then kt1 block)

assert cols[:150] == [f"grid_{g}_hurst" for g in range(150)]
D = np.array(Parallel(n_jobs=-1)(delayed(degraded_feats)(paths[i]) for i in te_dis))
p_orig = rf.predict(X[te_dis])
p_deg = rf.predict(D)
res["disease_test_n"] = len(te_dis)
res["disease_called_normal_original"] = int((p_orig == "NORMAL").sum())
res["disease_called_normal_after_resample"] = int((p_deg == "NORMAL").sum())
res["meanH_disease_original"] = float(X[te_dis][:, :150].mean())
res["meanH_disease_after_resample"] = float(D[:, :150].mean())
res["meanH_normal"] = float(df.loc[y == "NORMAL", "meanH"].mean())
print(f"    held-out DISEASE scans called NORMAL: original {res['disease_called_normal_original']}/{len(te_dis)}"
      f" -> after shrinking to 512 px wide (NORMAL-like geometry) {res['disease_called_normal_after_resample']}/{len(te_dis)}")
print(f"    mean H: disease {res['meanH_disease_original']:.3f} -> resampled disease "
      f"{res['meanH_disease_after_resample']:.3f}  (NORMAL {res['meanH_normal']:.3f})")

json.dump({k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in res.items()},
          open(OUT / "sanity_checks.json", "w"), indent=2, default=float)
print("\nsaved results/sanity_checks.json")
