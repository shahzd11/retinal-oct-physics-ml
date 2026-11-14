"""Fractal-dimension tools and retinal-boundary ("coastline") fractal features.

Background (Mandelbrot; Athe, Shakya, Munshi, Flow Meas. Instrum. 33 (2013) 122-137, the source of
the KT-1 + NMSID method used by the paper this repo reproduces):

  box counting     N(eps) ~ eps^-D                       (Hausdorff/box dimension of a set)
  fBm surface      E|I(p2) - I(p1)| ~ |p2 - p1|^H,  D = 3 - H   (image as a 3-D landscape)
  fBm profile      E|z(x+d) - z(x)| ~ d^H,          D = 2 - H   (a curve, e.g. a coastline)
  divider/ruler    L(r) ~ r^(1-D)                         (Richardson's coastline method)

Note: D = 3 - H of the *intensity surface* is a relabelling of the Hurst exponent the paper
already uses, so it carries no new information. The new use here is the fractal dimension of the
retina's layer boundaries (ILM = inner surface, RPE = outer boundary): a "coastline" whose
roughness changes with disease (epiretinal membrane wrinkles the ILM; drusen / CNV / PED make the
RPE bumpy), measured scale-free instead of as one amplitude number.

Known-answer test shapes (for validating the estimators): Sierpinski carpet (D = log 8 / log 3 =
1.8928), Koch curve (log 4 / log 3 = 1.2619), plane (D = 2), exact fBm profiles (D = 2 - H) and
spectral fBm surfaces (D = 3 - H).
"""
import numpy as np
from scipy.ndimage import gaussian_filter

LAT_UM = 11.7      # assumed lateral spacing (6 mm / 512 columns); the images don't store it
LAGS = np.arange(1, 17)       # boundary lags in columns (11.7 ... 187 um)


# ----------------------------------------------------------------------------- known-answer shapes
def sierpinski_carpet(level):
    a = np.ones((1, 1), bool)
    for _ in range(level):
        z = np.zeros_like(a)
        a = np.block([[a, a, a], [a, z, a], [a, a, a]])
    return a


def koch_curve(level):
    """(N, 2) vertices of the Koch curve on [0, 1]."""
    p = np.array([[0.0, 0.0], [1.0, 0.0]])
    rot = np.array([[0.5, -np.sqrt(3) / 2], [np.sqrt(3) / 2, 0.5]])
    for _ in range(level):
        a, b = p[:-1], p[1:]
        d = (b - a) / 3
        q1, q3 = a + d, a + 2 * d
        q2 = q1 + d @ rot.T
        p = np.vstack([np.stack([a, q1, q2, q3], 1).reshape(-1, 2), p[-1:]])
    return p


def fbm_profile(n, H, rng):
    """Exact fractional Brownian motion of length n (Davies-Harte circulant embedding)."""
    k = np.arange(n + 1)
    g = 0.5 * ((k + 1.0) ** (2 * H) - 2 * k ** (2.0 * H) + np.abs(k - 1.0) ** (2 * H))
    c = np.r_[g, g[-2:0:-1]]
    lam = np.clip(np.fft.fft(c).real, 0, None)
    m = len(c)
    w = rng.normal(size=m) + 1j * rng.normal(size=m)
    x = np.fft.fft(np.sqrt(lam / m) * w).real[:n]          # fractional Gaussian noise
    return np.cumsum(x)


def fbm_surface(n, H, rng):
    """Spectral-synthesis fBm surface (power ~ f^-(2H+2)); approximate, as in Voss / Saupe."""
    fx = np.fft.fftfreq(n)[:, None]
    fy = np.fft.fftfreq(n)[None, :]
    f = np.hypot(fx, fy)
    f[0, 0] = 1
    amp = f ** (-(H + 1))
    amp[0, 0] = 0
    ph = np.exp(2j * np.pi * rng.random((n, n)))
    z = np.fft.ifft2(amp * ph).real
    return (z - z.min()) / (np.ptp(z) + 1e-12)


# ----------------------------------------------------------------------------- estimators
def _fit(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return np.nan, np.nan
    p = np.polyfit(x[ok], y[ok], 1)
    ss = ((y[ok] - y[ok].mean()) ** 2).sum()
    r2 = 1 - ((y[ok] - np.polyval(p, x[ok])) ** 2).sum() / ss if ss > 0 else np.nan
    return p[0], r2


def box_dimension(mask, sizes=None):
    """Box-counting dimension of a binary set. Returns D, r2."""
    n = min(mask.shape)
    sizes = sizes if sizes is not None else 2 ** np.arange(1, int(np.log2(n)) - 1)
    counts = []
    for s in sizes:
        h, w = (mask.shape[0] // s) * s, (mask.shape[1] // s) * s
        b = mask[:h, :w].reshape(h // s, s, w // s, s).any(axis=(1, 3))
        counts.append(b.sum())
    slope, r2 = _fit(np.log(sizes), np.log(np.maximum(counts, 1)))
    return -slope, r2


def divider_dimension(xy, rulers):
    """Richardson ruler (coastline) method on a polyline. L(r) ~ r^(1-D). Returns D, r2."""
    seg = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    dense = np.linspace(0, seg[-1], max(len(xy) * 4, 2000))
    pts = np.c_[np.interp(dense, seg, xy[:, 0]), np.interp(dense, seg, xy[:, 1])]
    lengths = []
    for r in rulers:
        i, n = 0, 0
        while True:
            d = np.hypot(*(pts[i + 1:] - pts[i]).T)
            j = np.nonzero(d >= r)[0]
            if not len(j):
                break
            i += j[0] + 1
            n += 1
        lengths.append(n * r if n >= 3 else np.nan)
    slope, r2 = _fit(np.log(rulers), np.log(lengths))
    return 1 - slope, r2


def profile_hurst(z, lags=LAGS, segments=None):
    """H from mean |z(x+d) - z(x)| ~ d^H (1-D NMSID). segments: bool mask of valid samples; pairs
    only count when both ends are valid. Returns H, r2, M(1) (roughness amplitude at 1 lag)."""
    z = np.asarray(z, float)
    ok = np.isfinite(z) if segments is None else (segments & np.isfinite(z))
    m = []
    for d in lags:
        both = ok[d:] & ok[:-d]
        m.append(np.mean(np.abs(z[d:] - z[:-d])[both]) if both.sum() >= 10 else np.nan)
    m = np.array(m)
    H, r2 = _fit(np.log(lags), np.log(np.where(m > 0, m, np.nan)))
    return H, r2, m[0]


def surface_hurst(img, kmax=16):
    """2-D NMSID Hurst of an intensity surface (the paper's estimator, D = 3 - H)."""
    sums, cnts = np.zeros(kmax + 1), np.zeros(kmax + 1)
    for dy in range(0, kmax + 1):
        for dx in range(-kmax, kmax + 1):
            if dy == 0 and dx <= 0:
                continue
            k = int(np.floor(np.hypot(dx, dy)))
            if not 1 <= k <= kmax:
                continue
            a = img[dy:, max(dx, 0):img.shape[1] + min(dx, 0)]
            b = img[:img.shape[0] - dy, max(-dx, 0):img.shape[1] - max(dx, 0)]
            sums[k] += np.abs(a - b).sum()
            cnts[k] += a.size
    nmsid = sums[1:] / cnts[1:]
    H, r2 = _fit(np.log(np.arange(1, kmax + 1)), np.log(nmsid))
    return H, r2


# ----------------------------------------------------------------------------- retinal boundaries
def _subpixel(prof, i):
    if 0 < i < len(prof) - 1:
        a, b, c = prof[i - 1], prof[i], prof[i + 1]
        den = a - 2 * b + c
        if den != 0:
            return i + float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))
    return float(i)


def refine_boundaries(f, search=4):
    """Re-locate ILM and RPE per column without the tracer's lateral median filters (which would
    erase the small-scale roughness we want to measure).
      ILM = strongest dark->bright vertical step within +-search rows of the traced ILM
      RPE = brightest row within +-search rows of the traced RPE
    Both on an image smoothed only 1 px laterally / 1 px axially. Sub-pixel by parabola fit."""
    Hc, W = f.img.shape
    sm = gaussian_filter(np.nan_to_num(f.img), (1.0, 1.0))
    grad = np.zeros_like(sm)
    grad[1:-1] = sm[2:] - sm[:-2]
    ilm = np.full(W, np.nan)
    rpe = np.full(W, np.nan)
    for c in np.nonzero(f.col_ok)[0]:
        i0, r0 = int(round(f.ilm[c])), int(round(f.rpe[c]))
        a, b = max(1, i0 - search), min(Hc - 1, i0 + search + 1)
        if b - a >= 3:
            g = grad[a:b, c]
            ilm[c] = a + _subpixel(g, int(np.argmax(g)))
        a, b = max(0, r0 - search), min(Hc, r0 + search + 1)
        if b - a >= 3:
            p = sm[a:b, c]
            rpe[c] = a + _subpixel(p, int(np.argmax(p)))
    return ilm, rpe


def boundary_features(f, central=110):
    """Fractal ("coastline") features of the ILM and RPE boundaries.

    For each boundary b in {ilm, rpe}, heights in um (axial spacing when known, else px):
      frac_<b>_H, _D (= 2 - H), _r2      1-D NMSID / variogram over lags 1-16 columns, all columns
      frac_<b>_D_c                        same, central +-110 columns around the fovea
      frac_<b>_amp                        mean |step| between neighbouring columns (roughness amplitude)
      frac_<b>_Ddiv, _Ddiv_r2             Richardson divider dimension of the (x um, y um) polyline,
                                          rulers 2-16 columns, largest valid run of columns
    The big anatomical shape (foveal pit, residual curvature) lives at scales above the fitted
    lags; the increment-based estimator ignores constant offsets and is insensitive to slow trends.
    """
    um = 3.87 if f.scale_known else 1.0
    W = f.img.shape[1]
    cols = np.arange(W)
    cen = np.abs(cols - f.fovea) <= central
    ilm, rpe = refine_boundaries(f)
    out = {}
    for name, z in (("ilm", ilm), ("rpe", rpe)):
        zz = z * um
        ok = np.isfinite(zz)
        H, r2, amp = profile_hurst(zz, segments=ok)
        Hc, _, _ = profile_hurst(zz, segments=ok & cen)
        out[f"frac_{name}_H"] = H
        out[f"frac_{name}_D"] = 2 - H if np.isfinite(H) else np.nan
        out[f"frac_{name}_r2"] = r2
        out[f"frac_{name}_D_c"] = 2 - Hc if np.isfinite(Hc) else np.nan
        out[f"frac_{name}_amp"] = amp
        # longest run of valid columns for the divider method
        d = np.diff(np.r_[0, ok.astype(np.int8), 0])
        starts, stops = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
        if len(starts) and (stops - starts).max() >= 64:
            k = int(np.argmax(stops - starts))
            s = slice(starts[k], stops[k])
            xy = np.c_[cols[s] * LAT_UM, zz[s]]
            D, r2d = divider_dimension(xy, np.arange(2, 17) * LAT_UM)
            out[f"frac_{name}_Ddiv"], out[f"frac_{name}_Ddiv_r2"] = D, r2d
        else:
            out[f"frac_{name}_Ddiv"] = out[f"frac_{name}_Ddiv_r2"] = np.nan
    return out
