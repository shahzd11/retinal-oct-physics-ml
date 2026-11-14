"""Validate the fractal estimators on shapes with known dimension (as Athe et al. 2013, Sec. 6.3, do
for a plane and for fractional Brownian surfaces), and test the paper's own grid Hurst estimator.

  1. Sierpinski carpet, box counting             expect D = log8/log3 = 1.893
  2. Koch curve, divider (coastline) + box        expect D = log4/log3 = 1.262
  3. Plane, 2-D NMSID                            expect H = 1, D = 2
  4. fBm profiles (exact), 1-D NMSID             expect H, D = 2 - H
  5. fBm surfaces, 2-D NMSID on 256 px            expect H, D = 3 - H
  6. Paper estimator: 50 x 50 grids, k = 1..25, and after the paper's ~1.32x cubic upsampling

  python fractal_validation.py
Writes results/v4/fractal_validation.json and fractal_validation.png
"""
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from retina import plotstyle as ps
from retina.features import hurst as paper_hurst
from retina.fractal import (box_dimension, divider_dimension, fbm_profile, fbm_surface, koch_curve,
                            profile_hurst, sierpinski_carpet, surface_hurst)

OUT = Path("results/v4")
HS = np.round(np.arange(0.1, 0.95, 0.1), 2)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ps.apply()
    rng = np.random.default_rng(0)
    res = {}

    car = sierpinski_carpet(6)                                  # 729 x 729
    res["sierpinski_box_D"] = box_dimension(car, sizes=3 ** np.arange(0, 6))[0]
    res["sierpinski_true"] = np.log(8) / np.log(3)

    k = koch_curve(7)
    res["koch_divider_D"] = divider_dimension(k, np.geomspace(0.002, 0.05, 10))[0]
    n = 2048
    img = np.zeros((n // 2, n), bool)
    p = np.clip((k * (n - 1)).astype(int), 0, n - 1)
    img[np.clip(n // 2 - 1 - p[:, 1], 0, n // 2 - 1), p[:, 0]] = True
    res["koch_box_D"] = box_dimension(img, sizes=2 ** np.arange(1, 8))[0]
    res["koch_true"] = np.log(4) / np.log(3)

    plane = np.add.outer(np.linspace(0, 0.5, 256), np.linspace(0, 0.5, 256))
    res["plane_H"] = surface_hurst(plane)[0]

    prof, surf, grid, grid_up = {}, {}, {}, {}
    for H in HS:
        prof[H] = [profile_hurst(fbm_profile(4096, H, rng))[0] for _ in range(20)]
        s = [fbm_surface(256, H, rng) for _ in range(4)]
        surf[H] = [surface_hurst(x)[0] for x in s]
        g = np.concatenate([x[:250, :250].reshape(5, 50, 5, 50).swapaxes(1, 2).reshape(-1, 50, 50) for x in s])
        grid[H] = paper_hurst(g).tolist()
        up = [cv2.resize(x.astype(np.float32), None, fx=1.32, fy=1.32, interpolation=cv2.INTER_CUBIC) for x in s]
        gu = np.concatenate([u[:300, :300].reshape(6, 50, 6, 50).swapaxes(1, 2).reshape(-1, 50, 50) for u in up])
        grid_up[H] = paper_hurst(gu.astype(np.float64)).tolist()
    for name, d in (("profile_1d", prof), ("surface_2d", surf), ("paper_grid", grid), ("paper_grid_upsampled", grid_up)):
        res[name] = {str(h): [float(np.mean(v)), float(np.std(v))] for h, v in d.items()}

    for k_, v in res.items():
        print(k_, v if not isinstance(v, dict) else {h: round(m[0], 3) for h, m in v.items()})

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
    ax = axes[0]
    ax.imshow(car, cmap="gray_r", interpolation="nearest")
    ax.set_title(f"Sierpinski carpet: box-count D = {res['sierpinski_box_D']:.3f} (true 1.893)", loc="left", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax = axes[1]
    ax.plot(k[:, 0], k[:, 1], color=ps.SERIES[0], lw=0.6)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_title(f"Koch curve: divider D = {res['koch_divider_D']:.3f}, box D = {res['koch_box_D']:.3f} (true 1.262)",
                 loc="left", fontsize=10)
    ax = axes[2]
    ax.plot([0, 1], [0, 1], color=ps.INK2, lw=1, ls="--", label="perfect")
    for (name, d), col in zip((("1-D NMSID, exact fBm profile", prof), ("2-D NMSID, fBm surface 256 px", surf),
                               ("paper estimator, 50x50 grid, k 1-25", grid),
                               ("paper estimator after 1.32x upsampling", grid_up)), ps.SERIES):
        m = np.array([np.mean(d[h]) for h in HS]); s = np.array([np.std(d[h]) for h in HS])
        ax.errorbar(HS, m, yerr=s, color=col, lw=2, marker="o", ms=5, capsize=2, label=name)
    ax.set_xlabel("True Hurst exponent H"); ax.set_ylabel("Estimated H")
    ax.set_title(f"Hurst estimators on synthetic fBm (plane: H = {res['plane_H']:.3f})", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Known-answer validation of the fractal estimators", weight="bold")
    plt.tight_layout(); plt.savefig(OUT / "fractal_validation.png", dpi=140); plt.close()
    json.dump(res, open(OUT / "fractal_validation.json", "w"), indent=2)


if __name__ == "__main__":
    main()
