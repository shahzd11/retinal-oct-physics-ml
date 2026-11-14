"""Run preprocessing + feature extraction on OCTID and cache to results/features.csv.

300 features per scan: grid_{g}_hurst and grid_{g}_kt1 for g in 0..149.
Also stores original image width/height (used only by sanity_checks.py, never by the model).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from retina.preprocess import load_gray, normalize, micro_slice
from retina.features import extract

CLASSES = ["CSR", "DR", "MH", "NORMAL"]


def process(path, label):
    raw = load_gray(path)
    H, S = extract(normalize(raw), micro_slice)
    row = {"file": path.name, "label": label, "orig_h": raw.shape[0], "orig_w": raw.shape[1]}
    row.update({f"grid_{g}_hurst": H[g] for g in range(150)})
    row.update({f"grid_{g}_kt1": S[g] for g in range(150)})
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/OCTID")
    ap.add_argument("--out", default="results/features.csv")
    ap.add_argument("--jobs", type=int, default=-1)
    a = ap.parse_args()

    items = [(p, c) for c in CLASSES for p in sorted(Path(a.data, c).iterdir())
             if p.suffix.lower() in (".jpg", ".jpeg")]
    print({c: sum(1 for _, l in items if l == c) for c in CLASSES}, "total", len(items))
    rows = Parallel(n_jobs=a.jobs)(delayed(process)(p, c) for p, c in tqdm(items))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print("saved", a.out)


if __name__ == "__main__":
    main()
