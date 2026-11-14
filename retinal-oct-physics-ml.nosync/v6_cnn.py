"""v6: fine-tuned ResNet-18 on the *same* patient-grouped folds as every other v6 model, saving
out-of-fold probabilities (paired tests, confusion matrices) and the fold models (Grad-CAM, Step 4).

Recipe identical to cnn_baseline.py (ImageNet ResNet-18, raw scan -> 224x224 grey x3, flips +
small affine, AdamW 3e-4, cosine, 8 epochs, class-weighted CE). cnn_baseline.py is not modified;
it used folds of seed 0 only, as here.

  python v6_cnn.py --task octid_dis750            # one task (slug), or 'all', or 'cross'
  python v6_cnn.py --task octid_dis750 --quick    # 2 folds, 1 epoch
Writes results/v6/cnn_<task>.json, oof_<task>_cnn.csv; fold weights in data/cache/v6/cnn/
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from retina.v6data import CACHE, CROSS, SLUG, TASKS
from retina.v6eval import argmax_labels, folds, save_oof, summarise
from retina.v6run import cross_summary, dump, task_data, _harm
from retina.v6data import load_features

OUT = "results/v6"
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class Scans(Dataset):
    def __init__(self, paths, labels, classes, train):
        self.p, self.y = list(paths), [classes.index(l) for l in labels]
        aug = [transforms.RandomHorizontalFlip(), transforms.RandomAffine(5, translate=(0.05, 0.05), scale=(0.9, 1.1))]
        self.t = transforms.Compose([transforms.Grayscale(3), transforms.Resize((224, 224))]
                                    + (aug if train else []) + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        return self.t(Image.open(self.p[i]).convert("L")), self.y[i]


def device(name="auto"):
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def net_for(k):
    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Linear(net.fc.in_features, k)
    return net


def train(paths, labels, classes, epochs, dev, seed=0, bs=32, lr=3e-4):
    torch.manual_seed(seed); np.random.seed(seed)
    net = net_for(len(classes)).to(dev)
    counts = np.bincount([classes.index(l) for l in labels], minlength=len(classes))
    w = torch.tensor(len(labels) / (len(classes) * np.maximum(counts, 1)), dtype=torch.float32, device=dev)
    lossf = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    dl = DataLoader(Scans(paths, labels, classes, True), batch_size=bs, shuffle=True, num_workers=2,
                    persistent_workers=True)
    for ep in range(epochs):
        net.train(); tot = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(); loss = lossf(net(xb), yb); loss.backward(); opt.step(); tot += loss.item() * len(yb)
        sched.step()
        print(f"    epoch {ep + 1}/{epochs} loss {tot / len(labels):.3f}", flush=True)
    return net.eval()


@torch.no_grad()
def proba(net, paths, classes, dev, bs=64):
    out = []
    for xb, _ in DataLoader(Scans(paths, [classes[0]] * len(paths), classes, False), batch_size=bs, num_workers=0):
        out.append(torch.softmax(net(xb.to(dev)), 1).float().cpu().numpy())
    return np.concatenate(out)


def run_task(task, epochs, n_splits, dev):
    data = task_data(task, (), quick=n_splits == 2)
    paths, y, g = data["paths"], data["y"], data["groups"]
    classes = list(np.unique(y))
    P = np.full((len(y), len(classes)), np.nan)
    wdir = CACHE / "cnn" / data["slug"]; wdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for k, (tr, te) in enumerate(folds(y, g, 0, n_splits)):
        print(f"  {task} fold {k + 1}/{n_splits}", flush=True)
        net = train(paths[tr], y[tr], classes, epochs, dev)
        P[te] = proba(net, paths[te], classes, dev)
        torch.save({"state_dict": net.state_dict(), "classes": classes, "test_idx": te}, wdir / f"fold{k}.pt")
    pred = argmax_labels(P, np.array(classes))
    res = {"task": task, "n": len(y), "classes": classes, "epochs": epochs, "n_splits": n_splits, "seed_folds": 0,
           "device": str(dev), "minutes": (time.time() - t0) / 60, **summarise(y, [pred], g)}
    save_oof(f"{OUT}/oof_{data['slug']}_cnn.csv", paths, y, P, np.array(classes))
    dump(res, f"{OUT}/cnn_{data['slug']}.json")
    print(f"  {task}: bacc {res['bacc']:.3f} {res['bacc_ci']} ({res['minutes']:.1f} min)", flush=True)


def run_cross(epochs, dev):
    tr, _ = _harm(load_features("kermany")); te, _ = _harm(load_features("octdl"))
    classes = sorted(tr.label.unique())
    t0 = time.time()
    net = train(tr.path.values, tr.label.values, classes, epochs, dev)
    P = proba(net, te.path.values, classes, dev)
    pred = argmax_labels(P, np.array(classes))
    res = {"task": CROSS, "n_train": len(tr), "n_test": len(te), "classes": classes, "epochs": epochs,
           "device": str(dev), "minutes": (time.time() - t0) / 60,
           **cross_summary(te.label.values, pred, te.patient.values)}
    save_oof(f"{OUT}/oof_{SLUG[CROSS]}_cnn.csv", te.path.values, te.label.values, P, np.array(classes))
    dump(res, f"{OUT}/cnn_{SLUG[CROSS]}.json")
    print(f"  {CROSS}: bacc {res['bacc']:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="all")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--quick", action="store_true", help="2 folds, 1 epoch")
    a = ap.parse_args()
    dev = device(a.device)
    print("device", dev, flush=True)
    epochs, n_splits = (1, 2) if a.quick else (a.epochs, 5)
    os.makedirs(OUT, exist_ok=True)
    for t in TASKS:
        if a.task in ("all", SLUG[t]):
            run_task(t, epochs, n_splits, dev)
    if a.task in ("all", "cross"):
        run_cross(epochs, dev)


if __name__ == "__main__":
    main()
