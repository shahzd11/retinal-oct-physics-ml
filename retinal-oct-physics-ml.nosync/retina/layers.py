"""v2 preprocessing: physical scale normalization, artifact masking, RPE/ILM tracing,
retina flattening, anatomical bands and fovea-centred sectors.

All steps are classical image processing (no learning), so the pipeline stays white-box.
"""
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import median_filter, gaussian_filter1d

CANVAS_W = 512            # lateral samples after resampling
TARGET_AXIAL_UM = 3.87    # Heidelberg Spectralis axial pixel spacing; everything is resampled to this
UP, DOWN = 200, 70        # flattened canvas: rows above / below the RPE reference row
R0 = UP                   # row where the (smoothed) RPE lands after flattening
WHITE = 240

# bands: (name, top, bottom) resolved per column in flatten(); see band_masks()
BANDS = ("inner", "outer", "rpe", "choroid")
# fovea-centred sectors, in canvas columns relative to the fovea
SECTORS = {"peri_L": (-200, -110), "para_L": (-110, -40), "central": (-40, 40),
           "para_R": (40, 110), "peri_R": (110, 200)}


@dataclass
class Flat:
    img: np.ndarray        # (UP+DOWN, CANVAS_W) float32 in [0,1], flattened
    valid: np.ndarray      # bool, real tissue/background pixels (not padding/artifact)
    ilm: np.ndarray        # (W,) ILM row on the flat canvas (nan where not found)
    rpe: np.ndarray        # (W,) raw RPE row on the flat canvas (smooth RPE == R0)
    rpe_resid: np.ndarray  # (W,) raw minus smooth RPE in px (negative = elevated RPE)
    fovea: int             # fovea column
    col_ok: np.ndarray     # (W,) bool, column segmented reliably
    scale_known: bool


def resample(gray, axial_um=None, icon_box=None):
    """Return float image on a common grid and a validity mask.
    - near-white (>=240) acquisition wedges are invalid
    - icon_box=(x0,y0,x1,y1) as fractions of the raw frame marks overlays invalid (applied to every
      image of a dataset regardless of class, so the mask can't leak labels)
    - width -> 512; height -> physical axial spacing of 3.87 um when known, else keep aspect."""
    h, w = gray.shape
    valid = gray < WHITE
    if icon_box is not None:
        x0, y0, x1, y1 = icon_box
        valid[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = False
    new_h = int(round(h * axial_um / TARGET_AXIAL_UM)) if axial_um else int(round(h * CANVAS_W / w))
    img = cv2.resize(gray, (CANVAS_W, new_h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    valid = cv2.resize(valid.astype(np.uint8), (CANVAS_W, new_h), interpolation=cv2.INTER_NEAREST) > 0
    img[~valid] = 0
    return img, valid


def _robust_poly(x, y, deg=2, iters=5):
    keep = np.isfinite(y)
    coef = np.polyfit(x[keep], y[keep], deg)
    for _ in range(iters):
        r = y - np.polyval(coef, x)
        mad = np.nanmedian(np.abs(r[keep])) + 1e-6
        keep = np.isfinite(y) & (np.abs(r) < 3 * 1.4826 * mad + 2)
        if keep.sum() < deg + 5:
            break
        coef = np.polyfit(x[keep], y[keep], deg)
    return coef


def trace(img, valid):
    """Per-column RPE (deepest strong reflectivity peak) and ILM (first sustained rise from vitreous)."""
    H, W = img.shape
    med = cv2.medianBlur((img * 255).astype(np.uint8), 5).astype(np.float32) / 255
    sm = cv2.GaussianBlur(med, (0, 0), 2)
    # anisotropic smoothing along the (roughly horizontal) layers suppresses speckle
    an = cv2.GaussianBlur(med, (0, 0), sigmaX=8, sigmaY=2)
    sm[~valid] = 0
    an[~valid] = 0
    colmax = an.max(0)
    peak = an.argmax(0)
    rpe = np.full(W, np.nan)
    for c in range(W):              # RPE = deepest strong peak within 20 px below the brightest row
        p = peak[c]
        seg = an[p:min(H, p + 21), c]
        rpe[c] = p + np.nonzero(seg >= 0.85 * colmax[c])[0].max()
    good = colmax > np.percentile(colmax, 20) * 0.8
    rpe[~good] = np.nan
    rpe = median_filter(np.nan_to_num(rpe, nan=np.nanmedian(rpe)), size=15, mode="nearest")
    rpe[~good] = np.nan

    # ILM: background from the vitreous (top rows), threshold per column, need an 8-px sustained band
    bg = np.median(sm[: max(5, H // 10)][valid[: max(5, H // 10)]]) if valid[: H // 10].any() else 0.0
    thr = bg + 0.35 * (colmax - bg)
    above = (sm > thr[None, :]).astype(np.float32)
    # tissue must persist for 12 px below the candidate ILM (rejects thin floating membranes)
    run = cv2.filter2D(above, -1, np.ones((12, 1), np.float32) / 12, anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)
    ilm = np.full(W, np.nan)
    for c in range(W):
        if not np.isfinite(rpe[c]):
            continue
        cand = np.nonzero(run[: int(rpe[c]) - 3, c] >= 0.8)[0]
        if len(cand):
            ilm[c] = cand[0]
    ilm = median_filter(np.nan_to_num(ilm, nan=-1), size=5, mode="nearest").astype(float)
    ilm[ilm < 0] = np.nan
    return rpe, ilm


def flatten(gray, axial_um=None, icon_box=None):
    img, valid = resample(gray, axial_um, icon_box)
    H, W = img.shape
    rpe, ilm = trace(img, valid)
    x = np.arange(W)
    ok = np.isfinite(rpe)
    if ok.sum() < W // 4:                       # segmentation failed: fall back to flat reference
        rpe[:] = np.nanmedian(rpe) if ok.any() else H * 0.7
        ok[:] = True
    coef = _robust_poly(x[ok].astype(float), rpe[ok], 2)
    smooth = np.polyval(coef, x)
    shift = np.round(R0 - smooth).astype(int)   # dest_row = src_row + shift

    out = np.zeros((UP + DOWN, W), np.float32)
    ov = np.zeros((UP + DOWN, W), bool)
    for c in range(W):
        s = shift[c]
        d0, d1 = max(0, s), min(UP + DOWN, H + s)
        if d1 > d0:
            out[d0:d1, c] = img[d0 - s:d1 - s, c]
            ov[d0:d1, c] = valid[d0 - s:d1 - s, c]
    rpe_f = rpe + shift
    ilm_f = ilm + shift
    resid = rpe - smooth
    col_ok = np.isfinite(rpe_f) & np.isfinite(ilm_f) & (ilm_f < rpe_f - 3) & (ilm_f >= 0)

    th = np.where(col_ok, rpe_f - ilm_f, np.nan)
    # RPE present but no tissue above it in the central half -> full-thickness gap (macular hole)
    gap = np.isfinite(rpe_f) & ~np.isfinite(ilm_f)
    gap[: int(W * 0.25)] = False
    gap[int(W * 0.75):] = False
    th[gap] = 0.0
    th_s = gaussian_filter1d(np.nan_to_num(th, nan=np.nanmedian(th) if np.isfinite(th).any() else 0), 5)
    lo, hi = int(W * 0.25), int(W * 0.75)
    fovea = lo + int(np.argmin(th_s[lo:hi]))
    return Flat(out, ov, ilm_f, rpe_f, resid, fovea, col_ok, axial_um is not None)


def band_masks(f: Flat):
    """Boolean masks for each anatomical band on the flat canvas."""
    Hc, W = f.img.shape
    rows = np.arange(Hc)[:, None]
    ilm = np.where(f.col_ok, f.ilm, np.inf)[None, :]
    rpe = np.where(f.col_ok, f.rpe, -np.inf)[None, :]
    with np.errstate(invalid="ignore"):
        mid = (ilm + rpe - 10) / 2
    m = {
        "inner": (rows >= ilm) & (rows < mid),
        "outer": (rows >= mid) & (rows < rpe - 6),
        "rpe": (rows >= rpe - 6) & (rows <= rpe + 6),
        "choroid": (rows > rpe + 8) & (rows <= rpe + 60),
    }
    return {k: v & f.valid for k, v in m.items()}


def sector_masks(f: Flat):
    W = f.img.shape[1]
    cols = np.arange(W)
    return {k: ((cols - f.fovea) >= a) & ((cols - f.fovea) < b) for k, (a, b) in SECTORS.items()}
