"""v6 shared data access: tasks, row order, feature matrices and the image caches.

Row order is always the order of results/v4/<ds>_features.csv (= committed manifests), so the
patient-grouped folds StratifiedGroupKFold(5, shuffle=True, random_state=seed) are *identical*
to those of retina.evaluation.oof_predictions / v5_ensemble.py. Every v6 number is therefore
paired with the v2-v5 numbers.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from retina.evaluation import add_band_summaries, add_row_summaries, feature_sets

FSET = "v2 band-summary + disease + NDT robust + fractal"      # the v5 best feature set (171)
CACHE = Path("data/cache/v6")
HARMONISE = {"NORMAL": "NORMAL", "NO": "NORMAL", "DME": "DME", "CNV": "AMD", "DRUSEN": "AMD", "AMD": "AMD"}

TASKS = {
    "OCTID CSR/DR/MH (500x750)": ("octid", lambda d: (d.label != "NORMAL") & (d.orig_h == 500) & (d.orig_w == 750)),
    "OCTDL 7-class": ("octdl", lambda d: d.label.notna()),
    "OCTDL 6 diseases": ("octdl", lambda d: d.label != "NORMAL"),
    "Kermany 4-class": ("kermany", lambda d: d.label.notna()),
    "Kermany CNV/DME/DRUSEN": ("kermany", lambda d: d.label != "NORMAL"),
}
CROSS = "Kermany -> OCTDL (NORMAL/DME/AMD)"
SLUG = {"OCTID CSR/DR/MH (500x750)": "octid_dis750", "OCTDL 7-class": "octdl_7",
        "OCTDL 6 diseases": "octdl_6dis", "Kermany 4-class": "kermany_4",
        "Kermany CNV/DME/DRUSEN": "kermany_3dis", CROSS: "kermany_to_octdl",
        "OCTID 4-class (scanner-confounded)": "octid_4_confounded"}


def posix(p):
    return str(p).replace("\\", "/")


def load_features(ds):
    df = add_band_summaries(add_row_summaries(pd.read_csv(f"results/v4/{ds}_features.csv")))
    df["path"] = df.path.map(posix)
    return df


def manifest(ds):
    m = pd.read_csv(f"manifests/{ds}.csv")
    m["path"] = m.path.map(posix)
    m["icon_box"] = m["icon_box"].astype(object).where(m["icon_box"].notna(), "")
    return m


def feat_matrix(d, cols=None):
    cols = cols or feature_sets(d)[FSET]
    X = d.reindex(columns=cols).values.astype(float)
    X[~np.isfinite(X)] = np.nan
    return X, cols


def cache_path(ds, blur=0.0):
    return CACHE / (f"tiles_{ds}.npz" if not blur else f"tiles_{ds}_blur{blur:g}.npz")


def load_tiles(ds, blur=0.0):
    """Returns dict(paths, hi uint8 (n,128,512), valid bool (n,128,512), info DataFrame)."""
    z = np.load(cache_path(ds, blur), allow_pickle=True)
    n = len(z["paths"])
    valid = np.unpackbits(z["valid_bits"], axis=1)[:, : 128 * 512].reshape(n, 128, 512).astype(bool)
    return {"paths": z["paths"], "hi": z["hi"], "valid": valid, "info": pd.DataFrame(list(z["info"]))}


def load_emb(ds, net, blur=0.0):
    f = CACHE / (f"emb_{net}_{ds}.npy" if not blur else f"emb_{net}_{ds}_blur{blur:g}.npy")
    return np.load(f)


def task_frame(task, df=None):
    ds, sel = TASKS[task]
    df = load_features(ds) if df is None else df
    m = sel(df).values
    return df[m].reset_index(drop=True), np.nonzero(m)[0], ds
