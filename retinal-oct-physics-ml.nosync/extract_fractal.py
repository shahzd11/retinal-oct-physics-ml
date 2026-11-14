"""Extract retinal-boundary fractal ("coastline") features (retina/fractal.py) and merge them onto
the v2 + NDT features.

  python extract_fractal.py manifests/octid.csv
    reads  results/v3/octid_features.csv
    writes results/v4/octid_fractal.csv and results/v4/octid_features.csv (v3 + frac_* columns)
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from retina.fractal import boundary_features
from retina.layers import flatten
from retina.preprocess import load_gray


def process(row):
    warnings.filterwarnings("ignore")
    icon = row.get("icon_box")
    icon = tuple(float(v) for v in icon.split(",")) if isinstance(icon, str) and icon else None
    ax = row.get("axial_um")
    ax = float(ax) if ax is not None and np.isfinite(ax) else None
    d = {"path": row["path"]}
    try:
        d.update(boundary_features(flatten(load_gray(row["path"]), ax, icon)))
    except Exception as e:                      # keep the row; features stay NaN
        d["frac_error"] = repr(e)[:200]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    name = Path(a.manifest).stem
    out = Path("results/v4"); out.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(a.manifest)
    rows = Parallel(n_jobs=a.jobs, verbose=2)(delayed(process)(r) for r in man.to_dict("records"))
    fr = pd.DataFrame(rows)
    fr.to_csv(out / f"{name}_fractal.csv", index=False)
    base = pd.read_csv(Path("results/v3") / f"{name}_features.csv")
    merged = base.merge(fr, on="path", how="left")
    merged.to_csv(out / f"{name}_features.csv", index=False)
    print(name, "scans", len(fr), "errors", int(fr.get("frac_error", pd.Series(dtype=str)).notna().sum()),
          "->", out / f"{name}_features.csv", merged.shape)


if __name__ == "__main__":
    main()
