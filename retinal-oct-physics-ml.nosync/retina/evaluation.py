"""Shared evaluation helpers: feature sets, models, patient-grouped repeated CV with CIs,
grouped permutation importance."""
import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# NDT features whose healthy-eye distribution differs strongly between scanners (|Cohen's d| > 0.8,
# healthy Kermany vs healthy OCTDL; see ndt_checks.py). Chosen using OCTDL healthy scans, so the
# Kermany -> OCTDL score of the "robust" sets is mildly optimistic (transductive selection).
NDT_SCANNER_DEPENDENT = ("ndt_cnr_retina_vit", "ndt_ic_rpe_below_med", "ndt_trans_med", "ndt_cnr_rpe_choroid",
                         "ndt_ic_rpe_above_med", "ndt_ic_rpe_below_c_med", "ndt_ic_rpe_above_c_med",
                         "ndt_sh_under_foci")
DISEASE_PREFIX = ("thk_", "fovea_", "gap_", "thin_", "fluid_", "hrf_", "retina_", "rpe_")
TEX_KINDS = ["H", "H_r2", "nmsid1", "nmsid_ratio", "lac4", "lac8", "entropy", "mean", "std", "kt1", "coh"]
BANDS = ["inner", "outer", "rpe", "choroid"]
SECTORS = ["peri_L", "para_L", "central", "para_R", "peri_R"]


# ----------------------------------------------------------------------------- feature sets
def add_row_summaries(df):
    """Paper grid features summarized per grid row (10 rows) + left/right asymmetry: 40 features."""
    out = {}
    for kind in ("hurst", "kt1"):
        G = df[[f"grid_{g}_{kind}" for g in range(150)]].values.reshape(len(df), 10, 15)
        for r in range(10):
            out[f"row{r}_{kind}_mean"] = G[:, r].mean(1)
            out[f"row{r}_{kind}_lr_asym"] = np.abs(G[:, r, :7].mean(1) - G[:, r, 8:].mean(1))
    return pd.concat([df, pd.DataFrame(out, index=df.index)], axis=1)


def add_band_summaries(df):
    """v2 texture averaged over the 5 sectors per band (4 x 11 = 44) + central-vs-peripheral contrast."""
    out = {}
    for b in BANDS:
        for k in TEX_KINDS:
            cols = [f"tex_{b}_{s}_{k}" for s in SECTORS if f"tex_{b}_{s}_{k}" in df]
            v = df[cols]
            out[f"band_{b}_{k}"] = v.mean(axis=1)
            cen = f"tex_{b}_central_{k}"
            per = [f"tex_{b}_peri_L_{k}", f"tex_{b}_peri_R_{k}"]
            if cen in df and all(p in df for p in per):
                out[f"band_{b}_{k}_cen_minus_peri"] = df[cen] - df[per].mean(axis=1)
    return pd.concat([df, pd.DataFrame(out, index=df.index)], axis=1)


def feature_sets(df):
    c = list(df.columns)
    paper = [x for x in c if re.fullmatch(r"grid_\d+_(hurst|kt1)", x)]
    ent = [x for x in c if re.fullmatch(r"grid_\d+_entropy", x)]
    rows = [x for x in c if x.startswith("row") and ("_mean" in x or "_lr_asym" in x)]
    tex = [x for x in c if x.startswith("tex_")]
    dis = [x for x in c if x.startswith(DISEASE_PREFIX)]
    band = [x for x in c if x.startswith("band_")]
    # NDT features (ndtqc_* are scanner-dependent reference levels: QC only, never inputs)
    ndt = [x for x in c if x.startswith("ndt_") and x not in ("ndt_failed", "ndt_error")]
    ndt_r = [x for x in ndt if x not in NDT_SCANNER_DEPENDENT]
    # retinal-boundary fractal ("coastline") features, retina/fractal.py
    frac = [x for x in c if x.startswith("frac_") and x != "frac_error"]
    sets = {
        "paper (300 grid H+S)": paper,
        "paper + entropy": paper + ent,
        "paper row-summary (40)": rows,
        "v2 texture (band x sector)": tex,
        "v2 texture, no entropy": [x for x in tex if not x.endswith("_entropy")],
        "v2 disease-specific": dis,
        "v2 band-summary + disease": band + dis,
        "v2 all": tex + dis,
        "NDT only": ndt,
        "NDT robust": ndt_r,
        "v2 band-summary + disease + NDT": band + dis + ndt if ndt else [],
        "v2 band-summary + disease + NDT robust": band + dis + ndt_r if ndt else [],
        "v2 all + NDT": tex + dis + ndt if ndt else [],
        "fractal boundary only": frac,
        "v2 band-summary + disease + NDT robust + fractal": band + dis + ndt_r + frac if (ndt and frac) else [],
    }
    return {k: v for k, v in sets.items() if v}


# ----------------------------------------------------------------------------- models
class TwoStage(BaseEstimator, ClassifierMixin):
    """Stage 1: NORMAL vs abnormal. Stage 2: which disease (trained on abnormal scans only)."""

    def __init__(self, base=None, normal_label="NORMAL"):
        self.base = base
        self.normal_label = normal_label

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        b = self.base if self.base is not None else hgb()
        self.s1_ = clone(b).fit(X, (y == self.normal_label).astype(int))
        m = y != self.normal_label
        self.s2_ = clone(b).fit(X[m], y[m])
        return self

    def predict(self, X):
        p = self.s2_.predict(X).astype(object)
        p[self.s1_.predict(X) == 1] = self.normal_label
        return p


def rf_paper(seed=0):
    return make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=2,
                                                class_weight="balanced", random_state=seed, n_jobs=-1))


def hgb(seed=0):
    """Gradient boosting (LightGBM; sklearn's HistGradientBoosting has ~45 ms/tree overhead on Windows)."""
    return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=5,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0,
                          class_weight="balanced", random_state=seed, n_jobs=4, verbose=-1)


def logreg(seed=0):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000))


MODELS = {"RF (paper config)": rf_paper, "Gradient boosting": hgb, "Logistic regression": logreg,
          "Two-stage (GB)": lambda seed=0: TwoStage(hgb(seed))}


# ----------------------------------------------------------------------------- CV
def oof_predictions(model_fn, X, y, groups, seed, n_splits=5):
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pred = np.empty(len(y), dtype=object)
    for tr, te in cv.split(X, y, groups):
        m = model_fn(seed).fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


def scores(y, p):
    return {"acc": accuracy_score(y, p), "bacc": balanced_accuracy_score(y, p),
            "f1": f1_score(y, p, average="macro")}


def cluster_bootstrap_ci(y, p, groups, metric="bacc", n=1000, seed=0):
    """95% CI resampling patients (not images) with replacement."""
    rng = np.random.default_rng(seed)
    ug, inv = np.unique(groups, return_inverse=True)
    idx_by_g = [np.nonzero(inv == i)[0] for i in range(len(ug))]
    fn = {"acc": accuracy_score, "bacc": balanced_accuracy_score,
          "f1": lambda a, b: f1_score(a, b, average="macro")}[metric]
    vals = []
    for _ in range(n):
        pick = np.concatenate([idx_by_g[i] for i in rng.integers(0, len(ug), len(ug))])
        vals.append(fn(y[pick], p[pick]))
    return np.percentile(vals, [2.5, 97.5])


def repeated_cv(model_fn, X, y, groups, repeats=5, n_splits=5):
    preds = [oof_predictions(model_fn, X, y, groups, s, n_splits) for s in range(repeats)]
    per = [scores(y, p) for p in preds]
    p0 = preds[0]
    lo, hi = cluster_bootstrap_ci(y, p0, groups, "bacc")
    out = {f"{k}_mean": np.mean([d[k] for d in per]) for k in per[0]}
    out.update({f"{k}_sd": np.std([d[k] for d in per]) for k in per[0]})
    out.update(bacc_ci_lo=lo, bacc_ci_hi=hi)
    return out, p0


# ----------------------------------------------------------------------------- importance
def grouped_permutation_importance(model_fn, X, y, groups, col_groups, seed=0, n_splits=5, n_rep=5):
    """Drop in balanced accuracy when a whole group of columns is permuted jointly in held-out folds.
    col_groups: {name: [column indices]}. Unlike Gini importance this is measured on unseen data and
    is not biased toward high-cardinality / many-split features."""
    rng = np.random.default_rng(seed)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    drops = {g: [] for g in col_groups}
    for tr, te in cv.split(X, y, groups):
        m = model_fn(seed).fit(X[tr], y[tr])
        base = balanced_accuracy_score(y[te], m.predict(X[te]))
        for g, idx in col_groups.items():
            for _ in range(n_rep):
                Xp = X[te].copy()
                perm = rng.permutation(len(te))
                Xp[:, idx] = Xp[perm][:, idx]
                drops[g].append(base - balanced_accuracy_score(y[te], m.predict(Xp)))
    return pd.Series({g: np.mean(v) for g, v in drops.items()}).sort_values(ascending=False)
