"""v2 features on the flattened retina.

Texture, per anatomical band x fovea-centred sector (4 x 5 = 20 regions), 11 features each:
  H, H_r2         Hurst slope of the masked NMSID curve (k=1..8) and its log-log fit quality
  nmsid1, nmsid_ratio   roughness at k=1 and curve shape NMSID(8)/NMSID(1)
  lac4, lac8      gliding-box lacunarity at box sizes 4 and 8
  entropy         Shannon entropy (32-bin histogram), restored to test the collinearity claim
  mean, std       reflectivity statistics
  kt1             KT-1 slope (robust 99th-percentile N_max vs sigma^2), as in the paper
  coh             mean structure-tensor coherence (directional order, anisotropy)

Disease-specific / global (thickness profile, fluid, hyper-reflective foci, RPE elevation,
full-thickness gap, symmetry, segmentation QC).
"""
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, uniform_filter, label

from retina.layers import flatten, band_masks, sector_masks, BANDS, SECTORS

K = 8
SIGMAS = (3.0, 2.0, 1.0, 0.1)
NMAX_FLOOR = 0.05
MIN_PIX = 60


def _half_offsets(kmax):
    out = []
    for dy in range(0, kmax + 1):
        for dx in range(-kmax, kmax + 1):
            if dy == 0 and dx <= 0:
                continue
            k = int(np.floor(np.hypot(dx, dy)))
            if 1 <= k <= kmax:
                out.append((dx, dy, k))
    return out


_OFFS = _half_offsets(K)


def _pair(a, dx, dy):
    """Return (p1, p2) views of array a for offset (dx, dy)."""
    n0, n1 = a.shape[-2:]
    if dx >= 0:
        return a[..., dy:, dx:], a[..., :n0 - dy, :n1 - dx]
    return a[..., dy:, :n1 + dx], a[..., :n0 - dy, -dx:]


def masked_nmsid(img, masks):
    """NMSID(k) per region (pairs count only if both pixels lie in the region).
    masks: (R, H, W) bool. Returns (R, K). Each region is cropped to its bounding box first."""
    out = np.full((len(masks), K), np.nan)
    for r, m in enumerate(masks):
        ys, xs = np.nonzero(m)
        if len(ys) < MIN_PIX:
            continue
        sl = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
        im, M = img[sl], m[sl].astype(np.float32)
        sums = np.zeros(K + 1); cnts = np.zeros(K + 1)
        for dx, dy, k in _OFFS:
            if dy >= M.shape[0] or abs(dx) >= M.shape[1]:
                continue
            a, b = _pair(im, dx, dy)
            ma, mb = _pair(M, dx, dy)
            pm = ma * mb
            sums[k] += (pm * np.abs(a - b)).sum()
            cnts[k] += pm.sum()
        with np.errstate(invalid="ignore", divide="ignore"):
            out[r] = sums[1:] / cnts[1:]
    return out


def _hurst_fit(curve):
    logk = np.log(np.arange(1, K + 1))
    ok = np.isfinite(curve) & (curve > 0)
    if ok.sum() < 3:
        return np.nan, np.nan
    y = np.log(curve[ok]); x = logk[ok]
    p = np.polyfit(x, y, 1)
    yhat = np.polyval(p, x)
    ss = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ((y - yhat) ** 2).sum() / ss if ss > 0 else np.nan
    return p[0], r2


def _lacunarity(img, mask, r):
    m = uniform_filter(mask.astype(np.float32), r)
    s = uniform_filter(img * mask, r) * r * r
    full = m > 0.999
    if full.sum() < 5:
        return np.nan
    v = s[full]
    mu = v.mean()
    return float((v ** 2).mean() / (mu ** 2)) if mu > 0 else np.nan


def _coherence(img):
    g = gaussian_filter(img, 1.0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    jxx, jyy, jxy = (gaussian_filter(v, 3.0) for v in (gx * gx, gy * gy, gx * gy))
    tr = jxx + jyy
    disc = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(tr > 1e-8, (disc / tr) ** 2, 0.0)


def _kt1(img, mask):
    x = np.array([s * s for s in SIGMAS]); xc = x - x.mean()
    inv = []
    for s in SIGMAS:
        f = gaussian_filter(img, s)
        inv.append(1.0 / max(np.percentile(f[mask], 99), NMAX_FLOOR))
    inv = np.array(inv)
    return float((inv - inv.mean()) @ xc / (xc @ xc))


def texture_features(f):
    bm, sm = band_masks(f), sector_masks(f)
    names, masks = [], []
    for b in BANDS:
        for s in SECTORS:
            names.append(f"{b}_{s}")
            masks.append(bm[b] & sm[s][None, :])
    masks = np.array(masks)
    curves = masked_nmsid(f.img, masks)
    coh = _coherence(f.img)
    out = {}
    for i, n in enumerate(names):
        m = masks[i]
        feats = dict.fromkeys(["H", "H_r2", "nmsid1", "nmsid_ratio", "lac4", "lac8",
                               "entropy", "mean", "std", "kt1", "coh"], np.nan)
        if m.sum() >= MIN_PIX:
            v = f.img[m]
            feats["H"], feats["H_r2"] = _hurst_fit(curves[i])
            feats["nmsid1"] = curves[i][0]
            feats["nmsid_ratio"] = curves[i][-1] / curves[i][0] if curves[i][0] > 0 else np.nan
            feats["lac4"] = _lacunarity(f.img, m, 4)
            feats["lac8"] = _lacunarity(f.img, m, 8)
            h, _ = np.histogram(v, bins=32, range=(0, 1))
            p = h[h > 0] / h.sum()
            feats["entropy"] = float(-(p * np.log2(p)).sum())
            feats["mean"], feats["std"] = float(v.mean()), float(v.std())
            feats["kt1"] = _kt1(f.img, m)
            feats["coh"] = float(coh[m].mean())
        for k, v in feats.items():
            out[f"tex_{n}_{k}"] = v
    return out


def v2_fluid_mask(f, ret, sm, frac=0.2):
    """v2 fixed-fraction fluid rule: darker than vitreous + 20% of (retina median - vitreous)."""
    Hc = f.img.shape[0]
    rows = np.arange(Hc)[:, None]
    rmed = np.median(sm[ret])
    # fluid is about as dark as the vitreous, so threshold relative to this scan's vitreous level
    vit = (rows < np.where(f.col_ok, f.ilm - 6, -np.inf)[None]) & f.valid
    bg = np.median(sm[vit]) if vit.sum() > 200 else np.percentile(sm[ret], 1)
    dark = ret & (sm < bg + frac * (rmed - bg))
    # opening removes thin dark normal layers (e.g. outer nuclear layer); fluid pockets survive
    return cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).astype(bool)


def disease_features(f):
    """Thickness profile, fluid, hyper-reflective foci, RPE elevation, gaps, symmetry, QC."""
    W = f.img.shape[1]
    um = 3.87 if f.scale_known else 1.0       # px -> um when the axial spacing is known
    cols = np.arange(W)
    th = np.where(f.col_ok, f.rpe - f.ilm, np.nan) * um
    gapcols = np.isfinite(f.rpe) & ~f.col_ok & (np.abs(cols - f.fovea) < 60)
    th_g = th.copy(); th_g[gapcols] = 0.0
    out = {}
    med = np.nanmedian(th_g) if np.isfinite(th_g).any() else np.nan

    def band_mean(a, b):
        sel = (cols - f.fovea >= a) & (cols - f.fovea < b)
        v = th_g[sel]
        return np.nanmean(v) if np.isfinite(v).any() else np.nan

    out["thk_central"] = band_mean(-15, 15)
    out["thk_mean"] = np.nanmean(th_g)
    out["thk_std"] = np.nanstd(th_g)
    out["thk_max"] = np.nanmax(th_g) if np.isfinite(th_g).any() else np.nan
    out["thk_min"] = np.nanmin(th_g) if np.isfinite(th_g).any() else np.nan
    para = np.nanmean([band_mean(-110, -40), band_mean(40, 110)])
    out["fovea_dip_depth"] = para - out["thk_central"]
    out["fovea_dip_ratio"] = out["thk_central"] / para if para and np.isfinite(para) else np.nan
    out["thk_asym_para"] = abs(band_mean(-110, -40) - band_mean(40, 110))
    out["thk_asym_peri"] = abs(band_mean(-200, -110) - band_mean(110, 200))
    out["gap_cols"] = float(gapcols.sum())                       # full-thickness defect width (MH)
    out["thin_cols"] = float(np.nansum(th_g < 0.25 * med)) if np.isfinite(med) else np.nan

    # retina mask between ILM and RPE
    Hc = f.img.shape[0]
    rows = np.arange(Hc)[:, None]
    ret = (rows > np.where(f.col_ok, f.ilm, np.inf)[None]) & (rows < np.where(f.col_ok, f.rpe - 4, -np.inf)[None]) & f.valid
    sm = gaussian_filter(f.img, 2)
    if ret.sum() > 200:
        dark = v2_fluid_mask(f, ret, sm)
        lab, n = label(dark)
        sizes = np.bincount(lab.ravel())[1:] if n else np.array([0])
        sizes = sizes[sizes >= 20] if n else sizes
        out["fluid_frac"] = float(dark.sum() / ret.sum())
        out["fluid_largest"] = float(sizes.max() * um * um) if len(sizes) else 0.0
        out["fluid_n"] = float(len(sizes))
        # fluid in the outer third (subretinal, CSR) vs inner (cystoid, DME/DR)
        outer = ret & (rows > (f.ilm + 2 * (f.rpe - f.ilm) / 3)[None])
        out["fluid_outer_frac"] = float((dark & outer).sum() / max(outer.sum(), 1))
        # hyper-reflective foci: small bright spots inside the retina (white top-hat)
        u8 = (np.clip(f.img, 0, 1) * 255).astype(np.uint8)
        th_img = cv2.morphologyEx(u8, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        hr = ret & (th_img > np.percentile(th_img[ret], 99.5)) & (f.img > np.percentile(f.img[ret], 97))
        lab2, n2 = label(hr)
        out["hrf_n"] = float(n2)
        out["hrf_area_frac"] = float(hr.sum() / ret.sum())
        out["retina_mean"] = float(sm[ret].mean())
        out["retina_cv"] = float(sm[ret].std() / (sm[ret].mean() + 1e-6))
    else:
        for k in ["fluid_frac", "fluid_largest", "fluid_n", "fluid_outer_frac", "hrf_n",
                  "hrf_area_frac", "retina_mean", "retina_cv"]:
            out[k] = np.nan

    # RPE elevation (drusen / PED / CSR serous detachment): negative residual = RPE raised
    r = f.rpe_resid
    ok = np.isfinite(r)
    out["rpe_resid_std"] = float(np.nanstd(r) * um) if ok.any() else np.nan
    out["rpe_elev_max"] = float(-np.nanmin(r) * um) if ok.any() else np.nan
    elev = ok & (r < -4)
    out["rpe_elev_cols"] = float(elev.sum())
    out["rpe_bumps"] = float(label(elev)[1])

    out["qc_col_ok_frac"] = float(f.col_ok.mean())
    out["qc_resid_bad_frac"] = float((np.abs(np.nan_to_num(r)) > 15).mean())
    return out


def extract_v2(gray, axial_um=None, icon_box=None):
    f = flatten(gray, axial_um, icon_box)
    d = texture_features(f)
    d.update(disease_features(f))
    return d
