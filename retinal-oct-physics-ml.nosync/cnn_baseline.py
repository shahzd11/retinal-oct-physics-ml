"""CNN baseline on the same patient-grouped splits as the physics models.

ImageNet-pretrained ResNet-18, fine-tuned on CPU. Input: grayscale scan replicated to 3 channels,
resized to 224x224 (the CNN is *allowed* to see everything, including scanner artefacts, which is
exactly why it must be evaluated on the same de-confounded tasks / external scanners).

  python cnn_baseline.py manifests/octid.csv --task disease-only --folds 5 --epochs 8
  python cnn_baseline.py manifests/kermany.csv --external manifests/octdl.csv --harmonise
Requires: pip install torch torchvision  (CPU wheels)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

HARMONISE = {"NORMAL": "NORMAL", "NO": "NORMAL", "DME": "DME", "CNV": "AMD", "DRUSEN": "AMD", "AMD": "AMD"}
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


def train_eval(tr, te_sets, classes, epochs, bs=32, lr=3e-4, seed=0):
    torch.manual_seed(seed)
    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Linear(net.fc.in_features, len(classes))
    counts = np.bincount([classes.index(l) for l in tr.label], minlength=len(classes))
    w = torch.tensor(len(tr) / (len(classes) * np.maximum(counts, 1)), dtype=torch.float32)
    lossf = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    dl = DataLoader(Scans(tr.path, tr.label, classes, True), batch_size=bs, shuffle=True, num_workers=0)
    for ep in range(epochs):
        net.train(); tot = 0
        for xb, yb in dl:
            opt.zero_grad(); loss = lossf(net(xb), yb); loss.backward(); opt.step(); tot += loss.item() * len(yb)
        sched.step()
        print(f"    epoch {ep + 1}/{epochs} loss {tot / len(tr):.3f}", flush=True)
    net.eval(); outs = {}
    for name, te in te_sets.items():
        preds = []
        with torch.no_grad():
            for xb, _ in DataLoader(Scans(te.path, te.label, classes, False), batch_size=64):
                preds += net(xb).argmax(1).tolist()
        outs[name] = np.array([classes[i] for i in preds])
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--task", default="all", choices=["all", "disease-only", "disease-only-750"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--external", help="manifest of another scanner to test on (trained on all of manifest)")
    ap.add_argument("--harmonise", action="store_true", help="map labels to NORMAL/DME/AMD")
    a = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    df = pd.read_csv(a.manifest)
    if a.harmonise:
        df["label"] = df.label.map(HARMONISE); df = df.dropna(subset=["label"])
    if a.task.startswith("disease-only"):
        df = df[df.label != "NORMAL"]
    if a.task == "disease-only-750":
        sz = [Image.open(p).size for p in df.path]
        df = df[[s == (750, 500) for s in sz]]
    df = df.reset_index(drop=True)
    classes = sorted(df.label.unique())
    res = {"manifest": a.manifest, "task": a.task, "classes": classes, "n": len(df)}
    if a.external:
        ext = pd.read_csv(a.external)
        if a.harmonise:
            ext["label"] = ext.label.map(HARMONISE)
        ext = ext[ext.label.isin(classes)].reset_index(drop=True)
        p = train_eval(df, {"ext": ext}, classes, a.epochs)["ext"]
        res["external"] = {"manifest": a.external, "bacc": balanced_accuracy_score(ext.label, p),
                           "acc": accuracy_score(ext.label, p)}
        print("external:", res["external"])
    else:
        cv = StratifiedGroupKFold(a.folds, shuffle=True, random_state=0)
        oof = np.empty(len(df), dtype=object)
        for k, (tr, te) in enumerate(cv.split(df, df.label, df.patient)):
            print(f"  fold {k + 1}/{a.folds}", flush=True)
            oof[te] = train_eval(df.iloc[tr], {"te": df.iloc[te]}, classes, a.epochs)["te"]
        res.update(acc=accuracy_score(df.label, oof), bacc=balanced_accuracy_score(df.label, oof),
                   f1=f1_score(df.label, oof, average="macro"))
        print({k: res[k] for k in ("acc", "bacc", "f1")})
    out = Path("results/v2") / f"cnn_{Path(a.manifest).stem}_{a.task}{'_ext' if a.external else ''}.json"
    json.dump(res, open(out, "w"), indent=2, default=float)
    print("saved", out)


if __name__ == "__main__":
    main()
