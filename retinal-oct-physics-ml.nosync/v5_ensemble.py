"""v5 / Part B: soft-voting ensemble vs the best single model, with a paired patient bootstrap.

Feature set: "v2 band-summary + disease + NDT robust + fractal" (the repo's strongest set).
Same patient-grouped folds (identical seeds) for every model, so differences are paired.

  python v5_ensemble.py
Writes results/v5/ensemble.json and ensemble.md
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from ndt_paired_test import paired_boot
from retina.ensemble import V5_MODELS
from retina.evaluation import add_band_summaries, add_row_summaries, cluster_bootstrap_ci, feature_sets, oof_predictions

warnings.filterwarnings("ignore")
OUT = Path("results/v5"); OUT.mkdir(parents=True, exist_ok=True)
FSET = "v2 band-summary + disease + NDT robust + fractal"
REPEATS = 3
TASKS = {
    "OCTID CSR/DR/MH (500x750)": ("octid", lambda d: (d.label != "NORMAL") & (d.orig_h == 500) & (d.orig_w == 750)),
    "OCTID 4-class (scanner-confounded)": ("octid", lambda d: d.label.notna()),
    "OCTDL 7-class": ("octdl", lambda d: d.label.notna()),
    "OCTDL 6 diseases": ("octdl", lambda d: d.label != "NORMAL"),
    "Kermany 4-class": ("kermany", lambda d: d.label.notna()),
    "Kermany CNV/DME/DRUSEN": ("kermany", lambda d: d.label != "NORMAL"),
}


def main():
    rng = np.random.default_rng(0)
    res = []
    for tname, (ds, sel) in TASKS.items():
        df = add_band_summaries(add_row_summaries(pd.read_csv(f"results/v4/{ds}_features.csv")))
        d = df[sel(df)].reset_index(drop=True)
        y, g = d.label.values, d.patient.values
        X = d[feature_sets(d)[FSET]].values.astype(float)
        X[~np.isfinite(X)] = np.nan
        preds = {m: [oof_predictions(fn, X, y, g, s, 5) for s in range(REPEATS)] for m, fn in V5_MODELS.items()}
        row = {"task": tname, "n": len(d), "n_features": X.shape[1], "models": {}}
        for m, ps_ in preds.items():
            lo, hi = cluster_bootstrap_ci(y, ps_[0], g, "bacc")
            row["models"][m] = {
                "bacc": float(np.mean([balanced_accuracy_score(y, p) for p in ps_])),
                "bacc_sd": float(np.std([balanced_accuracy_score(y, p) for p in ps_])),
                "bacc_ci": [float(lo), float(hi)],
                "acc": float(np.mean([accuracy_score(y, p) for p in ps_])),
                "macro_f1": float(np.mean([f1_score(y, p, average="macro") for p in ps_])),
            }
        single = max(("Logistic regression", "LightGBM"), key=lambda m: row["models"][m]["bacc"])
        lo, hi, p = paired_boot(y, preds[single][0], preds["Soft vote (LR+GBM+SVM)"][0], g, rng)
        row["vote_vs_best_single"] = {"best_single": single,
                                      "diff_bacc": row["models"]["Soft vote (LR+GBM+SVM)"]["bacc"] - row["models"][single]["bacc"],
                                      "diff_ci": [lo, hi], "p_one_sided": p}
        res.append(row)
        print(f"\n{tname} (n={len(d)})")
        for m, r in row["models"].items():
            print(f"  {m:24s} bacc {r['bacc']:.3f} ± {r['bacc_sd']:.3f} [{r['bacc_ci'][0]:.3f}, {r['bacc_ci'][1]:.3f}]"
                  f"  acc {r['acc']:.3f}  macro-F1 {r['macro_f1']:.3f}")
        v = row["vote_vs_best_single"]
        print(f"  vote - {single}: {v['diff_bacc']:+.3f} [{lo:+.3f}, {hi:+.3f}] p={p:.3f}", flush=True)
    json.dump(res, open(OUT / "ensemble.json", "w"), indent=2)
    with open(OUT / "ensemble.md", "w", encoding="utf-8") as fh:
        fh.write("| Task | Model | Balanced acc [95% CI] | Accuracy | Macro-F1 |\n|---|---|---|---|---|\n")
        for row in res:
            for m, r in row["models"].items():
                fh.write(f"| {row['task']} | {m} | {r['bacc']:.3f} [{r['bacc_ci'][0]:.3f}, {r['bacc_ci'][1]:.3f}] | "
                         f"{r['acc']:.3f} | {r['macro_f1']:.3f} |\n")


if __name__ == "__main__":
    main()
