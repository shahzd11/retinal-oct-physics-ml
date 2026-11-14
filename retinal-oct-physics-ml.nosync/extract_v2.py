"""Extract paper features (300 grid H/S + grid entropy) and v2 features from a manifest.

  python extract_v2.py manifests/octid.csv          -> results/v2/octid_features.csv
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # one BLAS thread per worker (avoids memory blow-up)
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from retina.preprocess import load_gray, normalize, micro_slice
from retina.features import extract as extract_paper
from retina.features_v2 import extract_v2


def grid_entropy(grids):
    out = np.empty(len(grids))
    for i, g in enumerate(grids):
        h, _ = np.histogram(g, bins=32, range=(0, 256))
        p = h[h > 0] / h.sum()
        out[i] = -(p * np.log2(p)).sum()
    return out


def process(row):
    raw = load_gray(row["path"])
    norm = normalize(raw)
    H, S = extract_paper(norm, micro_slice)
    E = grid_entropy(micro_slice(norm))
    d = {"path": row["path"], "label": row["label"], "patient": row["patient"],
         "dataset": row["dataset"], "orig_h": raw.shape[0], "orig_w": raw.shape[1]}
    d.update({f"grid_{g}_hurst": H[g] for g in range(150)})
    d.update({f"grid_{g}_kt1": S[g] for g in range(150)})
    d.update({f"grid_{g}_entropy": E[g] for g in range(150)})
    icon = row.get("icon_box")
    icon = tuple(float(v) for v in icon.split(",")) if isinstance(icon, str) and icon else None
    ax = row.get("axial_um")
    ax = float(ax) if ax is not None and np.isfinite(ax) else None
    try:
        d.update(extract_v2(raw, ax, icon))
        d["v2_failed"] = 0
    except Exception as e:          # keep the row; v2 columns stay NaN
        d["v2_failed"] = 1
        d["v2_error"] = repr(e)[:200]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    man = pd.read_csv(a.manifest, keep_default_na=True)
    man["icon_box"] = man["icon_box"].fillna("")
    rows = Parallel(n_jobs=a.jobs)(delayed(process)(r) for r in tqdm(man.to_dict("records")))
    df = pd.DataFrame(rows)
    out = Path("results/v2") / (Path(a.manifest).stem + "_features.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print("saved", out, df.shape, "v2 failures:", int(df.v2_failed.sum()))


if __name__ == "__main__":
    main()
