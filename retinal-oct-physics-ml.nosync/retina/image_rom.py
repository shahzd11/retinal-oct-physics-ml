"""v6: image-level reduced-order modelling ("eigen-retinas") on the RPE-flattened retina.

v5 showed that POD of the paper's fixed 10x15 grid is dominated by *where the retina sits*
(mode 2 vs tissue height, rho ~ -0.9). POD needs a consistent coordinate frame, like a fixed CFD
mesh. retina.layers.flatten provides one: every column is shifted so the RPE is a straight line.

Tile (one per scan)
    rows    from 40 px above the highest ILM point to 60 px below the RPE (flat canvas)
    columns fovea +/- 200 canvas px (the same extent as the v2 sectors)
    resampled to 128 x 512 (stored) and 64 x 256 (POD), with a validity mask

POD of the tiles (snapshot method, fitted inside the training fold only)
    A = [x_1 - x̄, ..., x_n - x̄]^T   (n x 16384),  masked pixels -> training-fold pixel mean
    A ≈ U_r Σ_r V_r^T                 (sklearn randomized_svd)
    a = V_r^T (x - x̄),  x̂ = x̄ + V_r a
    relative residual  e(x) = ||x - x̄ - V_r V_r^T (x - x̄)||² / ||x - x̄||²
"""
import cv2
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.extmath import randomized_svd

from retina.layers import R0, flatten

ABOVE_ILM, BELOW_RPE, HALF_WIDTH = 40, 60, 200
HI = (128, 512)          # stored tile (rows, cols): used for the deep embeddings
LO = (64, 256)           # POD tile: 16384 pixels


def flat_tile(gray, axial_um=None, icon_box=None, size=HI):
    """Return (tile float32 in [0,1], valid bool, info dict) for one raw grayscale scan."""
    f = flatten(gray, axial_um, icon_box)
    Hc, W = f.img.shape
    c0, c1 = f.fovea - HALF_WIDTH, f.fovea + HALF_WIDTH
    lo, hi = max(c0, 0), min(c1, W)
    ilm = f.ilm[lo:hi][f.col_ok[lo:hi]]
    top_ilm = np.nanpercentile(ilm, 2) if ilm.size and np.isfinite(ilm).any() else R0 - 120
    top = int(max(0, np.floor(top_ilm) - ABOVE_ILM))
    bot = int(min(Hc, R0 + BELOW_RPE))
    img = np.zeros((bot - top, c1 - c0), np.float32)
    val = np.zeros((bot - top, c1 - c0), np.float32)
    img[:, lo - c0:hi - c0] = f.img[top:bot, lo:hi]
    val[:, lo - c0:hi - c0] = f.valid[top:bot, lo:hi]
    rs = (size[1], size[0])
    t = cv2.resize(img * val, rs, interpolation=cv2.INTER_AREA)
    v = cv2.resize(val, rs, interpolation=cv2.INTER_AREA)
    tile = np.where(v > 1e-3, t / np.maximum(v, 1e-3), 0).astype(np.float32)
    info = {"top": top, "bot": bot, "c0": int(c0), "fovea": int(f.fovea),
            "tile_rows_px": bot - top, "valid_frac": float((v > 0.5).mean())}
    return tile, v > 0.5, info


def to_lo(tiles_hi, valid_hi):
    """128x512 -> 64x256 (2x2 mean), masked pixels -> NaN. Returns (n, 16384) float32."""
    n = len(tiles_hi)
    t = tiles_hi.reshape(n, LO[0], 2, LO[1], 2).astype(np.float32)
    v = valid_hi.reshape(n, LO[0], 2, LO[1], 2).astype(np.float32)
    num, den = (t * v).sum((2, 4)), v.sum((2, 4))
    out = np.where(den >= 2, num / np.maximum(den, 1), np.nan)
    return out.reshape(n, -1).astype(np.float32)


class ImagePOD(BaseEstimator, TransformerMixin):
    """POD of image tiles. Input rows may contain NaN (masked pixels): they are replaced by the
    *training* pixel mean, so masked pixels contribute zero deviation."""

    def __init__(self, r=50, random_state=0, n_iter=4):
        self.r = r
        self.random_state = random_state
        self.n_iter = n_iter

    def _fill(self, T):
        T = np.asarray(T, dtype=np.float64)
        bad = ~np.isfinite(T)
        if bad.any():
            T = T.copy()
            T[bad] = np.broadcast_to(self.mean_, T.shape)[bad]
        return T

    def fit(self, T, y=None):
        T = np.asarray(T, dtype=np.float64)
        with np.errstate(all="ignore"):
            mu = np.nanmean(T, axis=0)
        mu[~np.isfinite(mu)] = np.nanmean(mu) if np.isfinite(mu).any() else 0.0
        self.mean_ = mu
        A = self._fill(T) - mu
        r = int(max(1, min(self.r, A.shape[0] - 1, A.shape[1])))
        U, S, Vt = randomized_svd(A, n_components=r, n_iter=self.n_iter, random_state=self.random_state)
        self.modes_, self.sigma_ = Vt, S
        self.total_energy_ = float((A ** 2).sum())
        self.energy_ = np.cumsum(S ** 2) / max(self.total_energy_, 1e-12)
        self.n_fit_ = A.shape[0]
        return self

    def transform(self, T, r=None):
        V = self.modes_[: (r or len(self.modes_))]
        return (self._fill(T) - self.mean_) @ V.T

    def reconstruct(self, T, r=None):
        V = self.modes_[: (r or len(self.modes_))]
        d = self._fill(T) - self.mean_
        return self.mean_ + (d @ V.T) @ V

    def residual(self, T, r=None, maps=False):
        """Relative residual e(x) per row; with maps=True also the signed residual images."""
        V = self.modes_[: (r or len(self.modes_))]
        d = self._fill(T) - self.mean_
        res = d - (d @ V.T) @ V
        e = (res ** 2).sum(1) / np.maximum((d ** 2).sum(1), 1e-12)
        return (e, res) if maps else e
