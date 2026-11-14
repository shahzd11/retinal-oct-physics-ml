"""Section 3: KT-1 tissue density (S) and NMSID Hurst coefficient (H) per grid."""
import numpy as np
from scipy.ndimage import gaussian_filter

SIGMAS = (3.0, 2.0, 1.0, 0.1)      # simulating tomographic filters h54, h75, h91, h99
K_MAX = 25                         # NMSID scales k = 1..K_MAX (lower limit = pixel, upper ~ half grid)
NMAX_FLOOR = 0.05                  # ~13/255: grids darker than this are background, not tissue


def _w2(sigma):
    """|W''(0)| for a Gaussian window W(xi) = exp(-2 pi^2 sigma^2 xi^2): 4 pi^2 sigma^2.
    Only proportionality matters (Eq. 2), so we regress on sigma^2."""
    return sigma ** 2


def kt1_slopes(grids):
    """grids: (G, 50, 50) float in [0,1]. Returns KT-1 slope per grid (Eq. 2):
    linear regression of 1/Nmax (robust 99th percentile) against W''(0)."""
    x = np.array([_w2(s) for s in SIGMAS])
    inv = np.empty((len(grids), len(SIGMAS)))
    for j, s in enumerate(SIGMAS):
        filt = gaussian_filter(grids, sigma=(0, s, s), mode="reflect")
        nmax = np.percentile(filt.reshape(len(grids), -1), 99, axis=1)
        # floor Nmax: near-black background grids would otherwise make 1/Nmax explode
        inv[:, j] = 1.0 / np.maximum(nmax, NMAX_FLOOR)
    xc = x - x.mean()
    slope = (inv - inv.mean(1, keepdims=True)) @ xc / (xc @ xc)
    # background grids hit the floor at every sigma -> slope 0
    slope[~np.isfinite(slope)] = 0.0
    return slope


def _offsets(k_max):
    """Half-plane integer offsets (dx, dy) with 1 <= floor(dist) <= k_max."""
    out = []
    for dy in range(0, k_max + 1):
        for dx in range(-k_max, k_max + 1):
            if dy == 0 and dx <= 0:
                continue
            k = int(np.floor(np.hypot(dx, dy)))
            if 1 <= k <= k_max:
                out.append((dx, dy, k))
    return out


_OFFS = _offsets(K_MAX)


def nmsid_curves(grids, k_max=K_MAX):
    """Eq. 3: mean |I(p2) - I(p1)| over all pixel pairs with floor(dist) = k."""
    G, n, _ = grids.shape
    sums = np.zeros((G, k_max + 1))
    cnts = np.zeros(k_max + 1)
    for dx, dy, k in (_OFFS if k_max == K_MAX else _offsets(k_max)):
        if dx >= 0:
            a = grids[:, dy:, dx:]
            b = grids[:, :n - dy, :n - dx]
        else:
            a = grids[:, dy:, :n + dx]
            b = grids[:, :n - dy, -dx:]
        d = np.abs(a - b)
        sums[:, k] += d.sum(axis=(1, 2))
        cnts[k] += d[0].size
    return sums[:, 1:] / cnts[1:]          # (G, k_max) for k = 1..k_max


def hurst(grids, k_max=K_MAX):
    """Eq. 4: H = slope of log NMSID(k) vs log k."""
    curves = nmsid_curves(grids, k_max)
    logk = np.log(np.arange(1, k_max + 1))
    H = np.zeros(len(grids))
    for g, c in enumerate(curves):
        ok = c > 0
        if ok.sum() >= 2:
            H[g] = np.polyfit(logk[ok], np.log(c[ok]), 1)[0]
    return H


def extract(img_u8, slicer):
    grids = slicer(img_u8).astype(np.float64) / 255.0
    return hurst(grids), kt1_slopes(grids)
