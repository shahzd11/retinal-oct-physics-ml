"""Improvement experiment: remove the resampling confound by giving every scan the same
resampling history (shrink to 512 px wide, INTER_AREA) before the paper's Sec. 2 pipeline.
Then re-run the paper's RF (5x10 CV + the paper's 80/20 split)."""
from pathlib import Path
import cv2, numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, cross_val_predict, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from retina.preprocess import load_gray, normalize, micro_slice
from retina.features import extract
from train_eval import make_rf, CLASSES

df = pd.read_csv("results/features.csv")[["file", "label"]]

def feats(path, w=512):
    raw = load_gray(path); h0, w0 = raw.shape
    raw = cv2.resize(raw, (w, int(round(h0 * w / w0))), interpolation=cv2.INTER_AREA)
    H, S = extract(normalize(raw), micro_slice)
    return np.r_[H, S]

X = np.array(Parallel(n_jobs=-1)(delayed(feats)(Path("data/OCTID", l, f)) for l, f in zip(df.label, df.file)))
y = df.label.values
cols = [f"grid_{g}_hurst" for g in range(150)] + [f"grid_{g}_kt1" for g in range(150)]
pd.concat([df, pd.DataFrame(X, columns=cols)], axis=1).to_csv("results/features_harmonized.csv", index=False)

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)
acc = cross_val_score(make_rf(0), X, y, cv=cv, n_jobs=1)
print(f"harmonized 5x10 CV acc {acc.mean():.3f} ± {acc.std():.3f}")
p = cross_val_predict(make_rf(0), X, y, cv=5)
print(classification_report(y, p, labels=CLASSES, digits=3))
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
pt = make_rf(42).fit(Xtr, ytr).predict(Xte)
print("paper split confusion matrix:\n", confusion_matrix(yte, pt, labels=CLASSES), "acc", (pt == yte).mean())
