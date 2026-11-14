"""Reduced-order modelling (ROM) of the feature fields with Proper Orthogonal Decomposition (POD).

The paper's 300 features are not 300 independent numbers: they are two *spatial fields*
(H on a 10x15 grid, S on a 10x15 grid). The v2 texture features are 11 fields on a 4x5
(band x sector) grid. Neighbouring cells are strongly correlated, so each field lives close to a
low-dimensional subspace. POD finds that subspace from data, exactly as in fluid mechanics:

    snapshot matrix   A = [x_1 - x̄, ..., x_n - x̄]^T          (n scans x m cells)
    thin SVD          A = U Σ V^T
    POD modes         φ_j = j-th column of V                     (spatial patterns, "eigen-retinas")
    energy captured   E(r) = Σ_{j<=r} σ_j² / Σ_j σ_j²
    reduced coords    a = V_r^T (x - x̄)                          (r numbers instead of m)
    reconstruction    x ≈ x̄ + V_r a

The transformer is fitted inside every CV training fold, so the modes never see test scans.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FieldPOD(BaseEstimator, TransformerMixin):
    """Per-field POD + passthrough of scalar features.

    fields      : list of lists of column indices; each inner list is one spatial field
    passthrough : column indices kept unchanged (scalars such as thickness, fluid fraction)
    energy      : keep the smallest r with E(r) >= energy (float in (0,1]) or a fixed r (int)
    scale       : divide each field by its global std (one number per field) so that fields with
                  different units (H is ~0.1, S is ~10) get comparable weight; the *shape* of every
                  field is untouched
    """

    def __init__(self, fields=None, passthrough=None, energy=0.95, scale=True):
        self.fields = fields
        self.passthrough = passthrough
        self.energy = energy
        self.scale = scale

    def _impute(self, X):
        X = np.array(X, dtype=float, copy=True)
        bad = ~np.isfinite(X)
        if bad.any():
            X[bad] = np.take(self.median_, np.nonzero(bad)[1])
        return X

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        with np.errstate(all="ignore"):
            med = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0)
        self.median_ = np.nan_to_num(med)
        X = self._impute(X)
        self.means_, self.scales_, self.modes_, self.sigma_, self.r_ = [], [], [], [], []
        for idx in self.fields or []:
            F = X[:, idx]
            mu = F.mean(0)
            A = F - mu
            sc = A.std() if self.scale and A.std() > 0 else 1.0
            A = A / sc
            _, s, Vt = np.linalg.svd(A, full_matrices=False)
            if isinstance(self.energy, (int, np.integer)):
                r = int(min(self.energy, len(s)))
            else:
                e = np.cumsum(s ** 2) / max((s ** 2).sum(), 1e-12)
                r = int(np.searchsorted(e, self.energy) + 1)
            self.means_.append(mu); self.scales_.append(sc)
            self.modes_.append(Vt[:r]); self.sigma_.append(s); self.r_.append(r)
        return self

    def transform(self, X):
        X = self._impute(X)
        parts = [((X[:, idx] - mu) / sc) @ V.T
                 for idx, mu, sc, V in zip(self.fields or [], self.means_, self.scales_, self.modes_)]
        if self.passthrough:
            parts.append(X[:, self.passthrough])
        return np.hstack(parts)

    def reconstruct(self, X, field=0):
        """x̂ = x̄ + V_r V_r^T (x - x̄) for one field (used for the energy / error plots)."""
        X = self._impute(X)
        idx, mu, sc, V = self.fields[field], self.means_[field], self.scales_[field], self.modes_[field]
        a = ((X[:, idx] - mu) / sc) @ V.T
        return mu + sc * (a @ V)


def paper_fields(cols):
    """Index lists for the paper's H field and S field (150 cells each, grid id = row*15 + col)."""
    pos = {c: i for i, c in enumerate(cols)}
    h = [pos[f"grid_{g}_hurst"] for g in range(150) if f"grid_{g}_hurst" in pos]
    s = [pos[f"grid_{g}_kt1"] for g in range(150) if f"grid_{g}_kt1" in pos]
    return [f for f in (h, s) if f]


def texture_fields(cols, kinds, bands, sectors):
    """One 20-cell field (4 bands x 5 sectors) per texture kind."""
    pos = {c: i for i, c in enumerate(cols)}
    out = []
    for k in kinds:
        idx = [pos[f"tex_{b}_{s}_{k}"] for b in bands for s in sectors if f"tex_{b}_{s}_{k}" in pos]
        if len(idx) >= 3:
            out.append(idx)
    return out
