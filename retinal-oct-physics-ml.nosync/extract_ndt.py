"""Extract NDT features (retina/features_ndt.py) for a manifest and merge them onto the v2 features.

  python extract_ndt.py manifests/octid.csv
    reads  results/v2/octid_features.csv
    writes results/v3/octid_ndt.csv and results/v3/octid_features.csv (v2 + ndt_* columns)
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
from tqdm import tqdm

from retina.preprocess import load_gray
from retina.layers import flatten
from retina.features_ndt import ndt_features


def process(row):
    warnings.filterwarnings("ignore")
    icon = row.get("icon_box")
    icon = tuple(float(v) for v in icon.split(",")) if isinstance(icon, str) and icon else None
    ax = row.get("axial_um")
    ax = float(ax) if ax is not None and np.isfinite(ax) else None
    d = {"path": row["path"]}
    try:
        d.update(ndt_features(flatten(load_gray(row["path"]), ax, icon)))
        d["ndt_failed"] = 0
    except Exception as e:
        d["ndt_failed"] = 1
        d["ndt_error"] = repr(e)[:200]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N scans")
    a = ap.parse_args()
    man = pd.read_csv(a.manifest)
    man["icon_box"] = man["icon_box"].astype(object).where(man["icon_box"].notna(), "")
    if a.limit:
        man = man.groupby("label").head(a.limit)
    rows = Parallel(n_jobs=a.jobs)(delayed(process)(r) for r in tqdm(man.to_dict("records")))
    ndt = pd.DataFrame(rows)
    name = Path(a.manifest).stem
    out = Path("results/v3"); out.mkdir(parents=True, exist_ok=True)
    ndt.to_csv(out / f"{name}_ndt.csv", index=False)
    print("saved", out / f"{name}_ndt.csv", ndt.shape, "failures:", int(ndt.ndt_failed.sum()))
    if ndt.ndt_failed.any():
        print(ndt.loc[ndt.ndt_failed == 1, "ndt_error"].value_counts().head())
    v2 = Path("results/v2") / f"{name}_features.csv"
    if v2.exists() and not a.limit:
        df = pd.read_csv(v2).merge(ndt.drop(columns=[c for c in ("ndt_error",) if c in ndt]), on="path", how="left")
        df.to_csv(out / f"{name}_features.csv", index=False)
        print("saved", out / f"{name}_features.csv", df.shape)


if __name__ == "__main__":
    main()
