"""v6 hybrid model: [v5 physics features | image-POD coefficients | POD residual e(x) | PCA of frozen
deep embeddings] -> soft vote (LR + LightGBM + RBF-SVM).

Everything learned from data is fitted inside fit(), i.e. on the training fold only:
  * ImagePOD of the tiles                       (pod)
  * healthy-only ImagePOD for the residual e(x) (resid). Training rows get *cross-fitted* e(x):
    each healthy training eye is scored by a POD that did not see its patient (inner grouped folds),
    so the classifier sees the same kind of value at train and test time
  * PCA(64) of each embedding                   (emb_resnet, emb_dino)
  * the classifier's own imputer / scaler

X is one numeric matrix with named column blocks (see assemble()). The 'group' block carries a
numeric patient code (used only for inner cross-fitting, never as a feature). An optional 'rowid'
block lets tests check which rows each transformer was fitted on (tests/test_no_leak.py).
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold

from retina.ensemble import gbm, lr, soft_vote
from retina.image_rom import ImagePOD

BASES = {"vote": soft_vote, "lr": lr, "gbm": gbm}


def assemble(**blocks):
    """assemble(group=g, tile=T, feat=F, ...) -> (X, layout {name: (start, stop)})."""
    cols, layout, s = [], {}, 0
    for name, a in blocks.items():
        a = np.asarray(a, dtype=float)
        a = a[:, None] if a.ndim == 1 else a
        layout[name] = (s, s + a.shape[1]); s += a.shape[1]
        cols.append(a)
    return np.hstack(cols), layout


class HybridClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, layout=None, use=("feat",), r_pod=50, r_resid=25, pca_dim=64, base="vote",
                 seed=0, normal_label="NORMAL", ref_tiles=None, inner_splits=5):
        self.layout = layout
        self.use = use
        self.r_pod = r_pod
        self.r_resid = r_resid
        self.pca_dim = pca_dim
        self.base = base
        self.seed = seed
        self.normal_label = normal_label
        self.ref_tiles = ref_tiles
        self.inner_splits = inner_splits

    # ------------------------------------------------------------------ helpers
    def _b(self, X, name):
        a, b = self.layout[name]
        return X[:, a:b]

    def _log(self, key, X, mask=None):
        if "rowid" in self.layout:
            ids = self._b(X, "rowid")[:, 0]
            self.seen_[key] = set((ids if mask is None else ids[mask]).astype(int).tolist())

    def _resid_feature(self, e):
        return np.log(np.clip(e, 1e-6, None))[:, None]

    # ------------------------------------------------------------------ fit / transform
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.seen_ = {}
        parts = []
        if "feat" in self.use:
            parts.append(self._b(X, "feat"))
        if "pod" in self.use:
            T = self._b(X, "tile")
            self.pod_ = ImagePOD(self.r_pod, random_state=self.seed).fit(T)
            self._log("pod", X)
            parts.append(self.pod_.transform(T))
        if "resid" in self.use:
            parts.append(self._resid_feature(self._fit_resid(X, y)))
        self.pca_ = {}
        for emb in ("emb_resnet", "emb_dino"):
            if emb in self.use:
                E = self._b(X, emb)
                k = int(min(self.pca_dim, E.shape[0] - 1, E.shape[1]))
                self.pca_[emb] = PCA(k, random_state=self.seed).fit(E)
                self._log(emb, X)
                parts.append(self.pca_[emb].transform(E))
        Z = np.hstack(parts)
        self.clf_ = BASES[self.base](self.seed).fit(Z, y)
        self._log("clf", X)
        self.classes_ = self.clf_.classes_
        self.n_features_z_ = Z.shape[1]
        return self

    def _fit_resid(self, X, y):
        """Fit the healthy-only residual POD; return cross-fitted e(x) for the training rows."""
        T = self._b(X, "tile")
        if self.ref_tiles is not None:                    # task without a healthy class: fixed
            self.rpod_ = ImagePOD(self.r_resid, random_state=self.seed).fit(self.ref_tiles)
            self.seen_["resid"] = set()                   # external reference, no task rows
            return self.rpod_.residual(T)
        healthy = y == self.normal_label
        if healthy.sum() < self.r_resid + 2:
            raise ValueError("residual POD needs healthy training eyes (or ref_tiles)")
        self.rpod_ = ImagePOD(self.r_resid, random_state=self.seed).fit(T[healthy])
        self._log("resid", X, healthy)
        e = self.rpod_.residual(T)                        # unhealthy training rows: not in the fit
        hi = np.nonzero(healthy)[0]
        g = self._b(X, "group")[hi, 0]
        k = int(min(self.inner_splits, len(np.unique(g))))
        for a, b in GroupKFold(k).split(hi, groups=g):    # healthy rows: scored by a POD without them
            m = ImagePOD(self.r_resid, random_state=self.seed).fit(T[hi[a]])
            e[hi[b]] = m.residual(T[hi[b]])
        return e

    def transform_z(self, X):
        X = np.asarray(X, dtype=float)
        parts = []
        if "feat" in self.use:
            parts.append(self._b(X, "feat"))
        if "pod" in self.use:
            parts.append(self.pod_.transform(self._b(X, "tile")))
        if "resid" in self.use:
            parts.append(self._resid_feature(self.rpod_.residual(self._b(X, "tile"))))
        for emb in ("emb_resnet", "emb_dino"):
            if emb in self.use:
                parts.append(self.pca_[emb].transform(self._b(X, emb)))
        return np.hstack(parts)

    def z_groups(self):
        """Column indices of each block inside Z (for grouped permutation importance)."""
        out, s = {}, 0
        sizes = {"feat": lambda: self.layout["feat"][1] - self.layout["feat"][0],
                 "pod": lambda: self.pod_.modes_.shape[0], "resid": lambda: 1}
        for name in ("feat", "pod", "resid", "emb_resnet", "emb_dino"):
            if name in self.use:
                n = sizes[name]() if name in sizes else self.pca_[name].n_components_
                out[name] = list(range(s, s + n)); s += n
        return out

    def predict_proba(self, X):
        return self.clf_.predict_proba(self.transform_z(X))

    def predict(self, X):
        return self.clf_.predict(self.transform_z(X))
