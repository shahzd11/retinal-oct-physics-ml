"""Class-conditional (Mondrian) split conformal prediction.

Turns any probabilistic classifier into one that outputs a *set* of labels with a coverage
guarantee. For a new scan with true label y and a user-chosen error level alpha:

    P( y ∈ C(x) | y = k ) >= 1 - alpha        for every class k

provided calibration and test scans are exchangeable (drawn the same way, same scanner).
This is the honest replacement for "100% triage guaranteed": the guarantee is stated, finite-sample,
distribution-free, and it holds per class (so it bounds the miss rate for diseased eyes).

Recipe (Vovk et al. 2005; Angelopoulos & Bates 2021):
    1. split the training patients into proper-train and calibration
    2. fit the model on proper-train
    3. nonconformity score on calibration:    s_i = 1 - p̂(y_i | x_i)
    4. per class k, with n_k calibration scans of class k:
           q_k = the ceil((n_k + 1)(1 - alpha))-th smallest score of class k
    5. prediction set for a new scan:         C(x) = { k : 1 - p̂(k | x) <= q_k }
"""
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold


def class_quantiles(scores, y, classes, alpha):
    q = {}
    for k in classes:
        s = np.sort(scores[y == k])
        n = len(s)
        r = int(np.ceil((n + 1) * (1 - alpha)))
        q[k] = np.inf if r > n else s[r - 1]          # too few calibration scans -> include always
    return q


def conformal_cv(model_fn, X, y, groups, alpha=0.05, seed=0, n_splits=5, cal_frac_splits=4):
    """Outer patient-grouped CV; inside each training fold a patient-grouped calibration split.
    Returns a boolean matrix S (n_scans x n_classes): S[i, k] = class k is in scan i's set."""
    classes = np.unique(y)
    S = np.zeros((len(y), len(classes)), bool)
    P = np.zeros((len(y), len(classes)))
    outer = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in outer.split(X, y, groups):
        inner = StratifiedGroupKFold(n_splits=cal_frac_splits, shuffle=True, random_state=seed + 1)
        fit_i, cal_i = next(inner.split(X[tr], y[tr], groups[tr]))
        fit_i, cal_i = tr[fit_i], tr[cal_i]
        m = clone(model_fn(seed)).fit(X[fit_i], y[fit_i])
        order = [list(m.classes_).index(k) for k in classes]
        pc = m.predict_proba(X[cal_i])[:, order]
        sc = 1 - pc[np.arange(len(cal_i)), np.searchsorted(classes, y[cal_i])]
        q = class_quantiles(sc, y[cal_i], classes, alpha)
        pt = m.predict_proba(X[te])[:, order]
        P[te] = pt
        S[te] = (1 - pt) <= np.array([q[k] for k in classes])[None, :]
    return S, P, classes


def summarize(S, y, classes):
    yi = np.searchsorted(classes, y)
    hit = S[np.arange(len(y)), yi]
    size = S.sum(1)
    single = size == 1
    pred_single = classes[S.argmax(1)]
    return {
        "coverage": float(hit.mean()),
        "coverage_per_class": {str(k): float(hit[y == k].mean()) for k in classes},
        "mean_set_size": float(size.mean()),
        "singleton_rate": float(single.mean()),
        "singleton_accuracy": float((pred_single[single] == y[single]).mean()) if single.any() else float("nan"),
        "empty_rate": float((size == 0).mean()),
    }
