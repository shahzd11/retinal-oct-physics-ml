"""v6 ablation: what does the physics add on top of a frozen embedding (and vice versa)?

Reads the out-of-fold prediction files already written by v6_image_pod.py, v6_hybrid.py and v6_cnn.py
(seed-0 folds, identical across models) and runs paired patient-bootstrap tests:

  hybrid  vs  embedding only        (same backbone)  -> what physics + ROM add to deep features
  hybrid  vs  v5 best (soft vote)                    -> what deep features add to physics
  hybrid  vs  fine-tuned CNN        (where run)

for both backbones (ResNet-18 = headline, DINOv2-S) on all six tasks. Point estimates and CIs of each
model are copied from results/v6/hybrid_<task>.json (3 seeds).

  python v6_ablation.py
Writes results/v6/ablation.json
"""
import json
import warnings

import numpy as np
import pandas as pd

from retina.v6data import CROSS, SLUG, TASKS
from retina.v6eval import paired
from retina.v6run import cross_data, dump, model_slug, task_data

warnings.filterwarnings("ignore")
V6 = "results/v6"
BACK = {"ResNet-18": ("hybrid ResNet-18", "ResNet-18 embedding only"),
        "DINOv2-S": ("hybrid DINOv2-S", "DINOv2-S embedding only")}
V5 = "v5 best (soft vote)"


def oof(slug, name):
    try:
        return pd.read_csv(f"{V6}/oof_{slug}_{model_slug(name)}.csv")
    except FileNotFoundError:
        return None


def main():
    out = {"headline": "hybrid ResNet-18", "tasks": {}}
    for task in list(TASKS) + [CROSS]:
        slug = SLUG[task]
        if task == CROSS:
            d = cross_data(())
            paths, y, g = d["paths"], d["yte"], d["gte"]
        else:
            d = task_data(task, ())
            paths, y, g = d["paths"], d["y"], d["groups"]
        hj = json.load(open(f"{V6}/hybrid_{slug}.json"))["models"]
        P = {n: oof(slug, n) for n in [V5, "cnn"] + [m for pair in BACK.values() for m in pair]}
        for n, o in P.items():
            if o is not None:
                assert (o.path.values == paths).all(), (task, n)
        r = {"models": {n: {k: hj[n][k] for k in ("bacc", "bacc_ci") if k in hj[n]} for n in hj}, "paired": {}}
        for bb, (hyb, emb) in BACK.items():
            ph = P[hyb].pred.values
            r["paired"][f"{hyb} vs {emb}"] = paired(y, P[emb].pred.values, ph, g)
            r["paired"][f"{hyb} vs {V5}"] = paired(y, P[V5].pred.values, ph, g)
            if P["cnn"] is not None:
                r["paired"][f"{hyb} vs CNN"] = paired(y, P["cnn"].pred.values, ph, g)
        out["tasks"][task] = r
        print(task)
        for k, v in r["paired"].items():
            print(f"   {k:48s} {v['diff_bacc']:+.3f} [{v['diff_ci'][0]:+.3f}, {v['diff_ci'][1]:+.3f}] p={v['p_one_sided']:.3f}",
                  flush=True)
    dump(out, f"{V6}/ablation.json")


if __name__ == "__main__":
    main()
