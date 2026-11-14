"""Progress figure: best balanced accuracy at every stage of the project, read from the result files.

  python v5_summary.py        (run after v5_ensemble.py)
Writes results/v5/progress.json and progress.png
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from retina import plotstyle as ps

OUT = Path("results/v5")
TASKS = {  # display name: (v2/v3 grid file stem, grid task name, v4 task, v5 task, cnn json or None)
    "OCTID CSR/DR/MH\n(500x750, no confound)": ("octid", "disease-only, 500x750 scans only (no size confound)",
                                               "OCTID CSR/DR/MH (500x750)", "OCTID CSR/DR/MH (500x750)",
                                               "results/v2/cnn_octid_disease-only-750.json"),
    "OCTDL 7-class": ("octdl", "all classes (AMD/DME/ERM/NORMAL/RAO/RVO/VID)", "OCTDL 7-class", "OCTDL 7-class", None),
    "Kermany 4-class": ("kermany", "all classes (CNV/DME/DRUSEN/NORMAL)", "Kermany 4-class", "Kermany 4-class",
                        "results/v2/cnn_kermany_all.json"),
}


def grid_best(ver, stem, task, paper_only=False):
    f = Path(f"results/{ver}/{stem}_model_grid.csv")
    if not f.exists():
        return None
    d = pd.read_csv(f)
    d = d[d.task == task]
    if paper_only:
        d = d[(d.features == "paper (300 grid H+S)") & (d.model == "RF (paper config)")]
    return float(d.bacc_mean.max()) if len(d) else None


def main():
    v4 = json.load(open("results/v4/fractal_paired_test.json"))
    v5 = {r["task"]: r for r in json.load(open(OUT / "ensemble.json"))}
    rows = {}
    for name, (stem, gtask, t4, t5, cnn) in TASKS.items():
        r = {
            "paper method (300 H+S, RF)": grid_best("v2", stem, gtask, paper_only=True),
            "v2 anatomy-aware features": grid_best("v2", stem, gtask),
            "v3 + NDT features": grid_best("v3", stem, gtask),
            "v4 + boundary fractal": max(x["with_bacc"] for x in v4 if x["task"] == t4),
            "v5 soft-vote ensemble": v5[t5]["models"]["Soft vote (LR+GBM+SVM)"]["bacc"],
            "CNN (ResNet-18, black box)": json.load(open(cnn))["bacc"] if cnn else None,
        }
        rows[name.replace("\n", " ")] = r
    json.dump(rows, open(OUT / "progress.json", "w"), indent=2)

    stages = list(next(iter(rows.values())).keys())
    fig, ax = plt.subplots(figsize=(13, 4.8))
    w = 0.13
    for i, s in enumerate(stages):
        vals = [rows[t][s] for t in rows]
        xs = np.arange(len(rows)) + (i - (len(stages) - 1) / 2) * w
        col = ps.INK2 if s.startswith("CNN") else ps.SERIES[i % len(ps.SERIES)]
        for x, v in zip(xs, vals):
            if v is None:
                continue
            ax.bar(x, v, w * 0.92, color=col, edgecolor=ps.SURFACE, hatch="//" if s.startswith("CNN") else None)
            ax.text(x, v + 0.01, f"{v:.2f}", ha="center", fontsize=7, color=ps.INK2)
        ax.bar(0, 0, color=col, label=s, hatch="//" if s.startswith("CNN") else None)
    ax.set_xticks(range(len(rows)), list(rows))
    ax.set_ylabel("balanced accuracy (patient-grouped 5-fold CV)")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, ncol=3, fontsize=8, loc="upper right")
    ax.set_title("From the paper's method to v5: best balanced accuracy at each stage", loc="left")
    plt.tight_layout(); plt.savefig(OUT / "progress.png", dpi=130); plt.close()
    for t, r in rows.items():
        print(t, {k: (round(v, 3) if v is not None else None) for k, v in r.items()})


if __name__ == "__main__":
    ps.apply()
    main()
