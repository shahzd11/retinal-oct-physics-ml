"""Section 4-5: armored Random Forest, stratified 80/20 split, confusion matrix,
phase space, violin, Gini importance, anatomical localization (Figs. 3-7)."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

CLASSES = ["CSR", "DR", "MH", "NORMAL"]
COLORS = {"NORMAL": "#2ecc71", "CSR": "#f1c40f", "DR": "#e67e22", "MH": "#e74c3c"}


def make_rf(seed):
    return RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=2,
                                  class_weight="balanced", random_state=seed, n_jobs=-1)


def feature_cols(df):
    return [c for c in df.columns if c.endswith(("_hurst", "_kt1"))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="results/features.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)

    df = pd.read_csv(a.features)
    cols = feature_cols(df)
    X, y = df[cols].values, df["label"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=a.seed)
    print(f"features={len(cols)} train={len(ytr)} test={len(yte)}")

    rf = make_rf(a.seed).fit(Xtr, ytr)
    pred = rf.predict(Xte)
    acc = accuracy_score(yte, pred)
    rep = classification_report(yte, pred, labels=CLASSES, digits=3)
    cm = confusion_matrix(yte, pred, labels=CLASSES)
    print(rep); print(cm); print(f"accuracy={acc:.4f}")

    # Fig 3: confusion matrix
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel("Diagnosed Pathology (Predicted)"); plt.ylabel("Actual Pathology (Ground Truth)")
    plt.title(f"Physics Engine Confusion Matrix (acc={acc:.2%})", weight="bold")
    plt.tight_layout(); plt.savefig(out / "fig3_confusion_matrix.png", dpi=150); plt.close()

    # Fig 4: phase space (per-scan mean H vs mean S)
    hcols = [c for c in cols if c.endswith("_hurst")]
    scols = [c for c in cols if c.endswith("_kt1")]
    ph = pd.DataFrame({"S": df[scols].mean(axis=1), "H": df[hcols].mean(axis=1), "label": df["label"]})
    plt.figure(figsize=(9, 6))
    for c in ["NORMAL", "CSR", "DR", "MH"]:
        d = ph[ph.label == c]
        plt.scatter(d.S, d.H, c=COLORS[c], label=c, edgecolor="k", lw=0.4, alpha=0.8, s=30)
    plt.xlabel("Tissue Density & Sharpness (mean KT-1 slope)")
    plt.ylabel("Fractal Structural Chaos (mean Hurst Coefficient)")
    plt.title("Tomographic Phase Space: KT-1 Density vs. Hurst Roughness", weight="bold")
    plt.legend(title="Retinal Pathology"); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(out / "fig4_phase_space.png", dpi=150); plt.close()

    # Fig 6: Gini importances
    imp = pd.Series(rf.feature_importances_, index=cols).sort_values(ascending=False)
    top = imp.head(15)
    plt.figure(figsize=(8, 6))
    sns.barplot(x=top.values, y=top.index, hue=top.index, palette="viridis", legend=False)
    plt.xlabel("Gini Importance"); plt.title("Top 15 Tomographic Biomarkers", weight="bold")
    plt.tight_layout(); plt.savefig(out / "fig6_feature_importance.png", dpi=150); plt.close()

    # Fig 5: violin of the top feature
    best = top.index[0]
    plt.figure(figsize=(8, 5))
    sns.violinplot(data=df, x="label", y=best, order=["NORMAL", "CSR", "DR", "MH"],
                   hue="label", palette=COLORS, inner="quartile", legend=False)
    plt.title(f"Pathological Variance of the Primary Biomarker ({best})", weight="bold")
    plt.tight_layout(); plt.savefig(out / "fig5_violin_top_feature.png", dpi=150); plt.close()

    # Fig 7: localization of top-15 grids (sum importance per grid for heatmap)
    grid_imp = np.zeros(150)
    for name, v in imp.items():
        grid_imp[int(name.split("_")[1])] += v
    top_grids = sorted({int(n.split("_")[1]) for n in top.index})
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.heatmap(grid_imp.reshape(10, 15), cmap="Blues", cbar_kws={"label": "summed Gini importance"},
                annot=np.arange(150).reshape(10, 15), fmt="d", annot_kws={"size": 7}, ax=ax,
                linewidths=0.5, linecolor="grey")
    for g in top_grids:
        ax.add_patch(plt.Rectangle((g % 15, g // 15), 1, 1, fill=False, ec="red", lw=2))
    ax.set_xlabel("Scan Width (15 grids)"); ax.set_ylabel("Scan Depth (10 grids)")
    ax.set_title("Anatomical Localization (red = top-15 feature grids)", weight="bold")
    plt.tight_layout(); plt.savefig(out / "fig7_localization.png", dpi=150); plt.close()

    n_h = sum(n.endswith("_hurst") for n in top.index)
    summary = {
        "accuracy": acc,
        "confusion_matrix": cm.tolist(),
        "report": classification_report(yte, pred, labels=CLASSES, output_dict=True),
        "top15": top.to_dict(),
        "top15_hurst_fraction": n_h / 15,
        "top15_grid_rows": sorted({g // 15 for g in top_grids}),
        "total_importance_hurst": float(imp[[c for c in imp.index if c.endswith('_hurst')]].sum()),
    }
    json.dump(summary, open(out / "results.json", "w"), indent=2)
    print("top15:", list(top.index))
    print(f"Hurst share of top-15: {n_h}/15, rows of top grids: {summary['top15_grid_rows']}")
    print(f"total Gini mass on Hurst features: {summary['total_importance_hurst']:.3f}")


if __name__ == "__main__":
    main()
