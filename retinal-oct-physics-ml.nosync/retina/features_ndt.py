"""NDT-inspired features (pulse-echo view of OCT), computed on the v2 flattened B-scan.

OCT is optical pulse-echo testing: each column is an A-scan, the RPE is the "back-wall echo",
fluid / holes / exudates are "defect echoes". The datasets are display images (log-compressed,
contrast-stretched JPEGs), not raw interferometric signal, so every quantity below is
*relative* and normalised to references inside the same image:

  vit  = vitreous level (clear gel above the retina, the image's own noise floor)
  rpe  = RPE echo level (strongest stable reflector, the "back wall")
  D    = rpe - vit, the image's dynamic range

Feature families (all prefixed ``ndt_``):
  att_*     attenuation index: depth slope of the (log-display) A-scan inside a band, /D.
            In a log display, Beer-Lambert I(z)=I0 exp(-2 mu z) becomes a straight line whose
            slope is proportional to mu, so this is a relative attenuation index, not mu in 1/mm.
  ic_*      interface contrast at the ILM and RPE, Michelson form (a-b)/(a+b) on vitreous-
            subtracted levels: the image analogue of the reflection coefficient ((n1-n2)/(n1+n2))^2
            (refractive indices themselves are not recoverable from display images).
  hm_*      half-maximum ("6 dB drop") sizing: defect / fluid edges where the echo falls halfway
            between the local tissue level and the vitreous, instead of a fixed threshold.
  sh_*      shadowing behind strong reflectors: loss of signal in the choroid under a column,
            relative to a lateral running baseline; also hyper-transmission (the opposite).
  cnr_*     SNR / CNR / detectability (retina vs vitreous, fluid pocket vs surrounding tissue).
  spk_*     speckle statistics per band after removing layer structure: Nakagami m and speckle
            contrast. Log compression + JPEG distort true Rayleigh/K statistics, so treat as
            empirical shape descriptors.
"""
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, label, median_filter, uniform_filter1d

EPS = 1e-6
CENTRAL = 110          # columns either side of the fovea counted as "macular"


def _refs(f, sm):
    """vitreous level, RPE echo level, dynamic range, per-column RPE peak."""
    Hc, W = f.img.shape
    rows = np.arange(Hc)[:, None]
    vit_m = (rows < np.where(f.col_ok, f.ilm - 6, -np.inf)[None]) & f.valid
    vit_v = sm[vit_m] if vit_m.sum() > 200 else sm[f.valid & (sm <= np.percentile(sm[f.valid], 5))]
    vit, vit_sd = float(np.median(vit_v)), float(vit_v.std())
    rpe_col = np.full(W, np.nan)
    for c in np.nonzero(f.col_ok)[0]:
        r = int(f.rpe[c])
        seg = sm[max(0, r - 4):r + 5, c]
        if seg.size:
            rpe_col[c] = seg.max()
    rpe = float(np.nanmedian(rpe_col)) if np.isfinite(rpe_col).any() else float(np.percentile(sm[f.valid], 99))
    return vit, vit_sd, rpe, max(rpe - vit, 0.05), rpe_col


def _col_slope(sm, top, bot, cols):
    """Least-squares slope of intensity vs depth per column over rows [top, bot)."""
    out = np.full(sm.shape[1], np.nan)
    for c in cols:
        t, b = max(0, int(top[c])), min(sm.shape[0], int(bot[c]))
        if b - t >= 8:
            y = sm[t:b, c]
            z = np.arange(b - t, dtype=np.float32)
            z -= z.mean()
            out[c] = float((z * (y - y.mean())).sum() / ((z * z).sum() + EPS))
    return out


def _stats(prefix, a, sel=None):
    v = a if sel is None else a[sel]
    v = v[np.isfinite(v)]
    if v.size < 5:
        return {f"{prefix}_med": np.nan, f"{prefix}_iqr": np.nan}
    q1, q2, q3 = np.percentile(v, [25, 50, 75])
    return {f"{prefix}_med": float(q2), f"{prefix}_iqr": float(q3 - q1)}


def _runs(mask):
    """(start, stop) of contiguous True runs."""
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))


def _halfmax_defect(p, ok, fovea, half, W):
    """Width of the below-half-max run nearest the fovea (within +-60 cols) and its edge sharpness."""
    below = ok & (p < half)
    best, width, sharp = None, 0.0, np.nan
    for a, b in _runs(below):
        if b - a < 2:
            continue
        dist = 0 if a <= fovea < b else min(abs(a - fovea), abs(b - 1 - fovea))
        if dist <= 60 and (best is None or dist < best):
            best, width = dist, float(b - a)
            g = np.abs(np.gradient(np.nan_to_num(p, nan=half)))
            edges = [e for e in (a - 1, a, b - 1, b) if 0 <= e < W]
            sharp = float(np.mean(g[edges]))
    return width, sharp


def fluid_mask(f, sm, vit, q=25, frac=0.4, win=151):
    """Anechoic (fluid-like) pockets inside the retina.
    Half-max rule between the vitreous level and the local *darkest tissue* level (q-th percentile of
    the retina in each column, smoothed laterally over `win` columns). Normal dark layers such as the
    outer nuclear layer still carry speckle above the vitreous and stay above the threshold."""
    Hc, W = f.img.shape
    rows = np.arange(Hc)[:, None]
    ilm = np.where(f.col_ok, f.ilm, np.nan)
    rpe = np.where(f.col_ok, f.rpe, np.nan)
    ret = (rows > ilm[None] + 2) & (rows < rpe[None] - 6) & f.valid & np.isfinite(ilm)[None] & np.isfinite(rpe)[None]
    if ret.sum() <= 300:
        return ret, np.zeros_like(ret)
    lo = np.array([np.percentile(sm[ret[:, c], c], q) if ret[:, c].sum() > 5 else np.nan for c in range(W)])
    fill = np.nanmedian(lo) if np.isfinite(lo).any() else vit
    lo = median_filter(np.nan_to_num(lo, nan=fill), size=win, mode="nearest")
    dark = ret & (sm < (vit + frac * (lo - vit))[None])
    dark = cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).astype(bool)
    return ret, dark


def ndt_features(f):
    Hc, W = f.img.shape
    rows = np.arange(Hc)[:, None]
    cols = np.arange(W)
    um = 3.87 if f.scale_known else 1.0
    sm = cv2.GaussianBlur(f.img, (0, 0), sigmaX=3, sigmaY=1)      # smooth along layers, keep depth detail
    sm[~f.valid] = np.nan
    vit, vit_sd, rpe_lvl, D, rpe_col = _refs(f, np.nan_to_num(sm, nan=0.0))
    ok = f.col_ok.copy()
    okc = np.nonzero(ok)[0]
    cen = ok & (np.abs(cols - f.fovea) < CENTRAL)
    # raw reference levels depend on the scanner -> QC only (ndtqc_*), never model inputs
    out = {"ndtqc_vit": vit, "ndtqc_dyn": D}

    ilm = np.where(ok, f.ilm, np.nan)
    rpe = np.where(ok, f.rpe, np.nan)
    smz = np.nan_to_num(sm, nan=vit)

    # ------------------------------------------------------------------ attenuation index
    # choroid below the RPE (classic depth-attenuation region), whole retina, and inner retina
    att_ch = -_col_slope(smz, rpe + 8, np.minimum(rpe + 60, Hc), okc) / D * 100
    att_rt = -_col_slope(smz, ilm + 3, rpe - 8, okc) / D * 100
    att_in = -_col_slope(smz, ilm + 3, (ilm + rpe) / 2, okc) / D * 100
    for name, a in (("choroid", att_ch), ("retina", att_rt), ("inner", att_in)):
        out.update(_stats(f"ndt_att_{name}", a))
        out.update(_stats(f"ndt_att_{name}_c", a, cen))
    # light reaching the choroid relative to the back-wall echo (transmission index)
    ch_mean = np.array([np.nanmean(smz[int(rpe[c]) + 8:int(rpe[c]) + 40, c]) if ok[c] and rpe[c] + 40 < Hc
                        else np.nan for c in cols])
    trans = (ch_mean - vit) / D
    out.update(_stats("ndt_trans", trans))

    # ------------------------------------------------------------------ interface contrast
    def lvl(r0, r1, fn):
        o = np.full(W, np.nan)
        for c in okc:
            a, b = int(max(0, r0[c])), int(min(Hc, r1[c]))
            if b > a:
                o[c] = fn(smz[a:b, c])
        return o - vit

    ilm_peak = lvl(ilm, ilm + 8, np.max)             # ILM / RNFL echo
    above_ilm = lvl(ilm - 12, ilm - 4, np.mean)      # vitreous just above
    rpe_peak = rpe_col - vit
    pre_rpe = lvl(rpe - 22, rpe - 8, np.mean)        # outer nuclear / photoreceptor layer above RPE
    post_rpe = lvl(rpe + 10, rpe + 30, np.mean)      # choroid just below

    def mich(a, b):
        a, b = np.clip(a, 0, None), np.clip(b, 0, None)
        return (a - b) / (a + b + EPS * D + 0.01)

    for name, a in (("ilm", mich(ilm_peak, above_ilm)), ("rpe_above", mich(rpe_peak, pre_rpe)),
                    ("rpe_below", mich(rpe_peak, post_rpe))):
        out.update(_stats(f"ndt_ic_{name}", a))
        out.update(_stats(f"ndt_ic_{name}_c", a, cen))
    out["ndt_ilm_echo"] = float(np.nanmedian(ilm_peak) / D) if np.isfinite(ilm_peak).any() else np.nan
    # RPE echo continuity: fraction of macular columns whose RPE echo drops below half of median
    rp = rpe_peak[cen]
    out["ndt_rpe_echo_loss"] = float(np.mean(rp < 0.5 * np.nanmedian(rpe_peak))) if rp.size else np.nan

    # ------------------------------------------------------------------ half-maximum sizing
    # Is there any tissue echo in the column? 90th percentile of the A-scan segment at two height
    # ranges above the RPE. A full-thickness macular hole is dark over the full height; subretinal
    # fluid (CSR) only close to the RPE; the normal foveal pit keeps the bright photoreceptor bands.
    gap_ok = np.isfinite(f.rpe)
    rpe_any = np.where(gap_ok, f.rpe, np.nan)
    for name, (d0, d1) in (("low", (8, 40)), ("full", (8, 120))):
        p = np.full(W, np.nan)
        for c in np.nonzero(gap_ok)[0]:
            a, b = max(0, int(rpe_any[c]) - d1), min(Hc, max(0, int(rpe_any[c]) - d0))
            if b - a >= 5:
                p[c] = np.percentile(smz[a:b, c], 90)
        far = gap_ok & (np.abs(cols - f.fovea) > 60) & np.isfinite(p)
        base = np.nanmedian(p[far]) if far.sum() > 10 else np.nanmedian(p)
        half = vit + 0.5 * (base - vit)
        w, s_ = _halfmax_defect(p, gap_ok & np.isfinite(p), f.fovea, half, W)
        out[f"ndt_hm_defect_{name}_w"] = w            # lateral columns (lateral spacing unknown)
        out[f"ndt_hm_defect_{name}_edge"] = s_ / D if np.isfinite(s_) else 0.0

    # fluid pockets: anechoic regions (half-max between vitreous and the darkest local tissue)
    ret, dark = fluid_mask(f, smz, vit)
    fluid_keys = ["ndt_hm_fluid_frac", "ndt_hm_fluid_n", "ndt_hm_fluid_w", "ndt_hm_fluid_h",
                  "ndt_hm_fluid_area", "ndt_cnr_fluid", "ndt_hm_fluid_outer"]
    if ret.sum() > 300:
        lab, n = label(dark)
        if n:
            sizes = np.bincount(lab.ravel())[1:]
            keep = np.nonzero(sizes >= 20)[0] + 1
        else:
            keep = []
        out["ndt_hm_fluid_frac"] = float(np.isin(lab, keep).sum() / ret.sum()) if len(keep) else 0.0
        out["ndt_hm_fluid_n"] = float(len(keep))
        if len(keep):
            big = keep[np.argmax(sizes[keep - 1])]
            m = lab == big
            rr, cc = np.nonzero(m)
            out["ndt_hm_fluid_w"] = float(cc.max() - cc.min() + 1)          # columns
            out["ndt_hm_fluid_h"] = float((rr.max() - rr.min() + 1) * um)  # um when axial spacing known
            out["ndt_hm_fluid_area"] = float(m.sum() / max(ret.sum(), 1))  # fraction of retina
            ring = cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & ~m & ret
            if ring.sum() > 10:
                out["ndt_cnr_fluid"] = float((smz[ring].mean() - smz[m].mean()) / (smz[ring].std() + smz[m].std() + EPS))
            else:
                out["ndt_cnr_fluid"] = 0.0
            outer = rows > (ilm + 2 * (rpe - ilm) / 3)[None]
            out["ndt_hm_fluid_outer"] = float((m & outer).sum() / m.sum())
        else:
            for k in fluid_keys[2:]:
                out[k] = 0.0
    else:
        for k in fluid_keys:
            out[k] = np.nan

    # ------------------------------------------------------------------ shadowing
    base = median_filter(np.nan_to_num(ch_mean, nan=np.nanmedian(ch_mean) if np.isfinite(ch_mean).any() else 0),
                         size=61, mode="nearest")
    rel = (ch_mean - vit) / (base - vit + 0.02)       # 1 = normal transmission under this column
    shadow = ok & (rel < 0.6)
    hyper = ok & (rel > 1.4)
    out["ndt_sh_frac"] = float(shadow.sum() / max(ok.sum(), 1))
    out["ndt_sh_frac_c"] = float((shadow & cen).sum() / max(cen.sum(), 1))
    out["ndt_sh_depth_max"] = float(np.nanmax(1 - rel[ok])) if ok.any() else np.nan
    out["ndt_sh_n"] = float(len([r for r in _runs(shadow) if r[1] - r[0] >= 3]))
    out["ndt_hypertrans_frac"] = float(hyper.sum() / max(ok.sum(), 1))
    # shadow specifically behind bright intraretinal foci (exudates / HRF)
    if ret.sum() > 300:
        u8 = (np.clip(np.nan_to_num(f.img), 0, 1) * 255).astype(np.uint8)
        th = cv2.morphologyEx(u8, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        bright = ret & (th > max(np.percentile(th[ret], 99.5), 20))
        bc = bright.any(0) & ok
        out["ndt_sh_under_foci"] = float(np.nanmean(1 - rel[bc])) if bc.sum() >= 2 else 0.0
        out["ndt_sh_foci_cols"] = float(bc.sum())
    else:
        out["ndt_sh_under_foci"] = out["ndt_sh_foci_cols"] = np.nan

    # ------------------------------------------------------------------ SNR / CNR
    rv = smz[ret] if ret.sum() > 300 else np.array([np.nan])
    vit_m = (rows < np.where(ok, f.ilm - 6, -np.inf)[None]) & f.valid
    vv = smz[vit_m] if vit_m.sum() > 200 else np.array([vit])
    out["ndt_cnr_retina_vit"] = float((np.nanmean(rv) - vv.mean()) / np.sqrt(np.nanvar(rv) + vv.var() + EPS))
    out["ndtqc_snr_rpe"] = float(D / (vit_sd + 0.005))
    out["ndt_cnr_rpe_choroid"] = float(np.nanmedian((rpe_peak - post_rpe) / D)) if np.isfinite(post_rpe).any() else np.nan

    # ------------------------------------------------------------------ speckle statistics
    raw = np.nan_to_num(f.img, nan=0.0).astype(np.float32)
    local = gaussian_filter(raw, 4) + 0.02
    norm = raw / local                                   # removes layer structure, keeps speckle
    bands = {
        "inner": (rows >= ilm[None] + 2) & (rows < ((ilm + rpe) / 2)[None]),
        "outer": (rows >= ((ilm + rpe) / 2)[None]) & (rows < rpe[None] - 8),
        "choroid": (rows > rpe[None] + 8) & (rows <= rpe[None] + 60),
    }
    for name, m in bands.items():
        m = m & f.valid & (raw > vit + 0.1 * D)
        if m.sum() > 200:
            x = norm[m].astype(np.float64)
            x2 = x * x
            out[f"ndt_spk_{name}_m"] = float(x2.mean() ** 2 / (x2.var() + EPS))       # Nakagami m
            out[f"ndt_spk_{name}_contrast"] = float(x.std() / (x.mean() + EPS))
        else:
            out[f"ndt_spk_{name}_m"] = out[f"ndt_spk_{name}_contrast"] = np.nan
    return out


def ndt_masks(f):
    """Masks used by ndt_features, for QC figures: shadow columns, fluid pockets, defect columns."""
    feats_img = {}
    Hc, W = f.img.shape
    rows = np.arange(Hc)[:, None]
    sm = cv2.GaussianBlur(f.img, (0, 0), sigmaX=3, sigmaY=1)
    vit, _, _, D, _ = _refs(f, sm)
    ok = f.col_ok
    ilm = np.where(ok, f.ilm, np.nan)
    rpe = np.where(ok, f.rpe, np.nan)
    _, dark = fluid_mask(f, sm, vit)
    ch = np.array([np.mean(sm[int(rpe[c]) + 8:int(rpe[c]) + 40, c]) if ok[c] and rpe[c] + 40 < Hc else np.nan
                   for c in range(W)])
    base = median_filter(np.nan_to_num(ch, nan=np.nanmedian(ch) if np.isfinite(ch).any() else 0), size=61, mode="nearest")
    rel = (ch - vit) / (base - vit + 0.02)
    feats_img["fluid"] = dark
    feats_img["shadow_cols"] = ok & (rel < 0.6)
    feats_img["hyper_cols"] = ok & (rel > 1.4)
    return feats_img
