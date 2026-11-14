"""v6 evaluation helpers: out-of-fold probabilities on the *same* folds as retina.evaluation,
metric summaries with patient cluster-bootstrap CIs, paired tests, and OOF CSV writing."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from ndt_paired_test import paired_boot
from retina.evaluation import cluster_bootstrap_ci


def folds(y, groups, seed, n_splits=5):
    """Exactly the split used by retina.evaluation.oof_predictions (depends only on y, groups, seed)."""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(np.zeros(len(y)), y, groups))


def oof_proba(model_fn, X, y, groups, seed, n_splits=5, return_models=False):
    """Out-of-fold class probabilities. Everything learned (POD, PCA, scalers, ...) lives inside
    model_fn(seed) and is fitted on the training fold only."""
    classes = np.unique(y)
    P = np.full((len(y), len(classes)), np.nan)
    models = []
    for tr, te in folds(y, groups, seed, n_splits):
        m = clone(model_fn(seed)).fit(X[tr], y[tr])
        order = [list(m.classes_).index(k) for k in classes]
        P[te] = m.predict_proba(X[te])[:, order]
        if return_models:
            models.append((tr, te, m))
    return (P, classes, models) if return_models else (P, classes)


def summarise(y, preds, groups):
    """preds: list (one per seed) of label arrays. CI from seed 0 (as in v5_ensemble.py)."""
    lo, hi = cluster_bootstrap_ci(y, preds[0], groups, "bacc")
    b = [balanced_accuracy_score(y, p) for p in preds]
    return {"bacc": float(np.mean(b)), "bacc_sd": float(np.std(b)), "bacc_ci": [float(lo), float(hi)],
            "acc": float(np.mean([accuracy_score(y, p) for p in preds])),
            "macro_f1": float(np.mean([f1_score(y, p, average="macro") for p in preds])),
            "bacc_per_seed": [float(v) for v in b]}


def paired(y, pa, pb, groups, seed=0):
    """B - A on identical folds, patient bootstrap (ndt_paired_test.paired_boot)."""
    lo, hi, p = paired_boot(y, pa, pb, groups, np.random.default_rng(seed))
    return {"diff_bacc": float(balanced_accuracy_score(y, pb) - balanced_accuracy_score(y, pa)),
            "diff_ci": [lo, hi], "p_one_sided": p}


def save_oof(path, paths, y, P, classes):
    """results/<stage>/oof_<task>_<model>.csv: path, true, pred, p_<class>..."""
    df = pd.DataFrame({"path": paths, "true": y, "pred": classes[np.nanargmax(P, 1)]})
    for j, k in enumerate(classes):
        df[f"p_{k}"] = P[:, j]
    df.to_csv(path, index=False)
    return df


def argmax_labels(P, classes):
    return classes[np.nanargmax(P, 1)]


# ---------------------------------------------------------------------------- fold loops with a fit log
def pod_sweep_oof(T, y, groups, seed, rs, model_fns, n_splits=5, log=None):
    """Image-POD coefficients -> classifier, for every r in rs. POD is fitted once per training fold
    with r_max = max(rs); the first r modes of that fit are the rank-r POD of the same fold.
    Returns {model: {r: P}}, classes. log (list) receives ('pod', fit_idx, test_idx) per fold."""
    from retina.image_rom import ImagePOD
    classes = np.unique(y)
    out = {m: {r: np.full((len(y), len(classes)), np.nan) for r in rs} for m in model_fns}
    for tr, te in folds(y, groups, seed, n_splits):
        pod = ImagePOD(max(rs), random_state=seed).fit(T[tr])
        if log is not None:
            log.append(("pod", tr, te))
        a_tr, a_te = pod.transform(T[tr]), pod.transform(T[te])
        for m, fn in model_fns.items():
            for r in rs:
                clf = fn(seed).fit(a_tr[:, :r], y[tr])
                order = [list(clf.classes_).index(k) for k in classes]
                out[m][r][te] = clf.predict_proba(a_te[:, :r])[:, order]
    return out, classes


def residual_oof(T, y, groups, seed, r, normal="NORMAL", n_splits=5, log=None, maps_for=None):
    """Out-of-fold POD residual e(x): POD fitted on the healthy eyes of the training fold only.
    maps_for: optional index array; returns their residual maps (from the fold that tests them)."""
    from retina.image_rom import ImagePOD
    e = np.full(len(y), np.nan)
    maps = {}
    for tr, te in folds(y, groups, seed, n_splits):
        fit_idx = tr[y[tr] == normal]
        pod = ImagePOD(r, random_state=seed).fit(T[fit_idx])
        if log is not None:
            log.append(("resid", fit_idx, te))
        e[te], R = pod.residual(T[te], maps=True)
        if maps_for is not None:
            for j, i in enumerate(te):
                if i in maps_for:
                    maps[int(i)] = R[j]
    return (e, maps) if maps_for is not None else e


def assert_no_leak(log):
    """Every fitted transformer must have seen training-fold rows only."""
    for name, fit_idx, test_idx in log:
        bad = np.intersect1d(np.asarray(list(fit_idx), dtype=int), np.asarray(list(test_idx), dtype=int))
        assert bad.size == 0, f"{name} was fitted on {bad.size} test rows"
