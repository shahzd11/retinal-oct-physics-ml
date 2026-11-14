### Step 1: image-level POD

| Task | Image-POD alone, best r (model) | v5 best | v5 best + image-POD r=50 | Paired Δ [95% CI], one-sided p |
|---|---|---|---|---|
| OCTID CSR/DR/MH (500x750) | 0.696 [0.646, 0.755] (r=100, LR) | 0.872 [0.833, 0.911] | 0.881 [0.855, 0.924] | +0.015 [-0.010, +0.041], p = 0.141 |
| OCTDL 7-class | 0.439 [0.377, 0.479] (r=25, LR) | 0.669 [0.605, 0.723] | 0.664 [0.606, 0.720] | -0.002 [-0.015, +0.010], p = 0.637 |
| OCTDL 6 diseases | 0.423 [0.357, 0.469] (r=10, LR) | 0.653 [0.589, 0.722] | 0.634 [0.558, 0.690] | -0.029 [-0.061, -0.005], p = 0.992 |
| Kermany 4-class | 0.537 [0.509, 0.558] (r=50, LightGBM) | 0.695 [0.671, 0.718] | 0.698 [0.678, 0.724] | +0.006 [-0.004, +0.015], p = 0.123 |
| Kermany CNV/DME/DRUSEN | 0.611 [0.578, 0.635] (r=25, LightGBM) | 0.731 [0.706, 0.765] | 0.731 [0.708, 0.765] | +0.001 [-0.013, +0.015], p = 0.456 |
| Kermany -> OCTDL (NORMAL/DME/AMD) | 0.546 [0.508, 0.584] (r=200, LR) | 0.709 [0.673, 0.743] | 0.693 [0.657, 0.730] | -0.017 [-0.036, +0.002], p = 0.963 |

*Source: `results/v6/image_pod_<task>.json`.*

### Step 2: POD residual anomaly score

| NORMAL vs ABNORMAL | AUROC r=10 | AUROC r=25 [95% CI] | AUROC r=50 |
|---|---|---|---|
| OCTDL 7-class | 0.665 | 0.696 [0.659, 0.731] | 0.717 |
| Kermany 4-class | 0.660 | 0.700 [0.673, 0.732] | 0.720 |
| Kermany healthy -> OCTDL (NORMAL vs ABNORMAL) | 0.659 | 0.691 [0.656, 0.724] | 0.708 |

*Source: `results/v6/residual_auroc.json`.*

| Task | v5 best + e(x) | Paired Δ vs v5 best |
|---|---|---|
| OCTID CSR/DR/MH (500x750) | 0.875 [0.846, 0.920] | +0.011 [+0.000, +0.027], p = 0.049 |
| OCTDL 7-class | 0.666 [0.607, 0.724] | +0.000 [-0.007, +0.008], p = 0.476 |
| OCTDL 6 diseases | 0.649 [0.584, 0.716] | -0.005 [-0.014, +0.004], p = 0.859 |
| Kermany 4-class | 0.699 [0.673, 0.721] | +0.003 [-0.004, +0.010], p = 0.202 |
| Kermany CNV/DME/DRUSEN | 0.729 [0.704, 0.762] | -0.003 [-0.010, +0.005], p = 0.746 |
| Kermany -> OCTDL (NORMAL/DME/AMD) | 0.704 [0.667, 0.739] | -0.005 [-0.018, +0.006], p = 0.821 |

*Source: `results/v6/residual_clf.json`.*

### Step 3: hybrid (frozen deep features + physics) vs fine-tuned CNN

Balanced accuracy [95% CI], mean of 3 seeds (CNN: seed-0 folds only). Embedding-only rows are the ablation.

| Model | OCTID CSR/DR/MH | OCTDL 7-class | OCTDL 6 dis. | Kermany 4-class | Kermany 3 dis. | Kermany → OCTDL |
|---|---|---|---|---|---|---|
| v5 best (soft vote) | 0.872 [0.833, 0.911] | 0.669 [0.605, 0.723] | 0.653 [0.589, 0.722] | 0.695 [0.671, 0.718] | 0.731 [0.706, 0.765] | 0.709 [0.673, 0.743] |
| ResNet-18 embedding only | 0.908 [0.884, 0.948] | 0.671 [0.603, 0.709] | 0.649 [0.585, 0.704] | 0.695 [0.670, 0.715] | 0.718 [0.701, 0.751] | 0.737 [0.702, 0.770] |
| DINOv2-S embedding only | 0.865 [0.831, 0.908] | 0.600 [0.541, 0.649] | 0.581 [0.520, 0.655] | 0.683 [0.668, 0.711] | 0.722 [0.691, 0.742] | 0.683 [0.647, 0.717] |
| **hybrid ResNet-18 (headline)** | 0.904 [0.865, 0.932] | 0.708 [0.657, 0.764] | 0.682 [0.616, 0.741] | 0.730 [0.704, 0.750] | 0.759 [0.735, 0.789] | 0.753 [0.719, 0.789] |
| hybrid DINOv2-S | 0.899 [0.864, 0.933] | 0.688 [0.632, 0.748] | 0.669 [0.602, 0.736] | 0.734 [0.715, 0.759] | 0.763 [0.740, 0.791] | 0.729 [0.694, 0.766] |
| CNN (fine-tuned ResNet-18) | 0.935 [0.905, 0.961] | n/a | n/a | 0.867 [0.851, 0.885] | n/a | 0.925 [0.908, 0.940] |

Paired differences on identical folds (seed 0), 95% patient-bootstrap CI and one-sided p (source `results/v6/ablation.json`):

| Task | ResNet-18 hybrid − ResNet-18 embedding | ResNet-18 hybrid − v5 | ResNet-18 hybrid − CNN | DINOv2-S hybrid − DINOv2-S embedding | DINOv2-S hybrid − v5 |
|---|---|---|---|---|---|
| OCTID CSR/DR/MH | -0.018 [-0.062, +0.024], p = 0.779 | +0.025 [+0.001, +0.050], p = 0.020 | -0.036 [-0.076, +0.003], p = 0.966 | +0.028 [-0.020, +0.075], p = 0.116 | +0.025 [-0.004, +0.054], p = 0.045 |
| OCTDL 7-class | +0.051 [+0.018, +0.096], p = 0.002 | +0.045 [+0.022, +0.073], p = 0.000 | n/a | +0.098 [+0.066, +0.130], p = 0.000 | +0.024 [+0.006, +0.044], p = 0.005 |
| OCTDL 6 dis. | +0.035 [-0.004, +0.077], p = 0.040 | +0.028 [-0.007, +0.059], p = 0.051 | n/a | +0.080 [+0.047, +0.119], p = 0.000 | +0.014 [-0.006, +0.035], p = 0.085 |
| Kermany 4-class | +0.034 [+0.014, +0.053], p = 0.001 | +0.032 [+0.018, +0.045], p = 0.000 | -0.141 [-0.162, -0.120], p = 1.000 | +0.047 [+0.026, +0.069], p = 0.000 | +0.043 [+0.028, +0.057], p = 0.000 |
| Kermany 3 dis. | +0.036 [+0.013, +0.059], p = 0.002 | +0.027 [+0.011, +0.043], p = 0.001 | n/a | +0.050 [+0.028, +0.071], p = 0.000 | +0.031 [+0.014, +0.048], p = 0.001 |
| Kermany → OCTDL | +0.016 [-0.024, +0.057], p = 0.221 | +0.044 [+0.021, +0.067], p = 0.000 | -0.172 [-0.206, -0.138], p = 1.000 | +0.046 [+0.011, +0.082], p = 0.006 | +0.020 [-0.004, +0.046], p = 0.061 |

Grouped permutation importance of the headline hybrid ResNet-18 hybrid, **measured on OCTID CSR/DR/MH only** (drop in balanced accuracy when a group is permuted in held-out folds). Groups are labelled by family: *physics* = hand-crafted v5 features, *ROM* = the v6 image-level reduced-order model, *deep* = the frozen embedding. Importance shows how much the fitted model **relies** on a group, not how much accuracy the group **adds**; the ablation above answers that.

| Rank | Group | Importance |
|---|---|---|
| 1 | deep: ResNet-18 embedding (PCA 64) | 0.0558 |
| 2 | physics: NDT (pulse-echo) | 0.0456 |
| 3 | physics: disease-specific (thickness, fluid, gap, RPE) | 0.0390 |
| 4 | physics: texture: rpe band | 0.0316 |
| 5 | physics: texture: choroid band | 0.0078 |
| 6 | physics: texture: inner band | 0.0067 |
| 7 | physics: texture: outer band | 0.0052 |
| 8 | ROM: image-POD (50 modes) | 0.0022 |
| 9 | ROM: POD residual e(x) | 0.0000 |
| 10 | physics: boundary fractal | -0.0019 |

*Source: `results/v6/hybrid_<task>.json`, `cnn_<task>.json`, `ablation.json`, `hybrid_importance.json` (DINOv2-S hybrid: `hybrid_importance_dinov2s.json`).*

### Step 4: Grad-CAM vs physics band importance (OCTID CSR/DR/MH)

| Band | CAM share CSR | CAM share DR | CAM share MH | physics recall drop CSR | physics recall drop DR | physics recall drop MH |
|---|---|---|---|---|---|---|
| inner | 0.120 | 0.122 | 0.137 | 0.116 | 0.029 | -0.010 |
| outer | 0.148 | 0.169 | 0.168 | 0.020 | 0.121 | 0.008 |
| rpe | 0.081 | 0.082 | 0.081 | 0.104 | 0.072 | 0.010 |
| choroid | 0.251 | 0.275 | 0.233 | 0.016 | 0.066 | -0.002 |

Spearman ρ over the 12 band × class cells: -0.22 (p = 0.48). *Source: `results/v6/gradcam_octid_dis750.json`.*

### Step 5: cross-scanner robustness (Kermany → OCTDL)

| Model | Calibration | TTA | Balanced acc | Recall NORMAL / DME / AMD |
|---|---|---|---|---|
| v5 best (soft vote) | none | no TTA | 0.709 | 0.79 / 0.55 / 0.78 |
| v5 best (soft vote) | none | TTA | 0.670 | 0.66 / 0.50 / 0.85 |
| v5 best (soft vote) | healthy-ref k=30 | no TTA | 0.682 ± 0.015 | 0.72 / 0.75 / 0.57 |
| v5 best (soft vote) | healthy-ref k=30 | TTA | 0.705 ± 0.018 | 0.75 / 0.74 / 0.62 |
| v6 hybrid (ResNet-18 + physics) | none | no TTA | 0.753 | 0.88 / 0.62 / 0.76 |
| v6 hybrid (ResNet-18 + physics) | none | TTA | 0.753 | 0.80 / 0.64 / 0.82 |
| v6 hybrid (ResNet-18 + physics) | healthy-ref k=30 | no TTA | 0.730 ± 0.018 | 0.79 / 0.80 / 0.60 |
| v6 hybrid (ResNet-18 + physics) | healthy-ref k=30 | TTA | 0.752 ± 0.018 | 0.82 / 0.80 / 0.64 |
| v6 hybrid (DINOv2-S + physics) | none | no TTA | 0.729 | 0.81 / 0.58 / 0.79 |
| v6 hybrid (DINOv2-S + physics) | none | TTA | 0.729 | 0.73 / 0.63 / 0.83 |
| v6 hybrid (DINOv2-S + physics) | healthy-ref k=30 | no TTA | 0.701 ± 0.015 | 0.77 / 0.70 / 0.63 |
| v6 hybrid (DINOv2-S + physics) | healthy-ref k=30 | TTA | 0.719 ± 0.015 | 0.77 / 0.73 / 0.66 |

Paired tests: v6 hybrid (ResNet-18 + physics) | none | no TTA vs v5 best (soft vote) | none | no TTA: +0.044 [+0.021, +0.067], p = 0.000; v6 hybrid (DINOv2-S + physics) | none | no TTA vs v5 best (soft vote) | none | no TTA: +0.020 [-0.004, +0.046], p = 0.061; v5 best (soft vote) | none | TTA vs no TTA: -0.039 [-0.060, -0.021], p = 1.000; v6 hybrid (ResNet-18 + physics) | none | TTA vs no TTA: -0.000 [-0.020, +0.019], p = 0.524; v6 hybrid (DINOv2-S + physics) | none | TTA vs no TTA: +0.000 [-0.017, +0.019], p = 0.501; v6 hybrid (ResNet-18 + physics) | none | no TTA vs CNN: -0.172 [-0.206, -0.138], p = 1.000; v6 hybrid (DINOv2-S + physics) | none | no TTA vs CNN: -0.197 [-0.232, -0.159], p = 1.000

*Source: `results/v6/cross.json`.*

### Step 6: conformal triage on the headline v6 hybrid (ResNet-18)

| Task | Target | Coverage per class | Mean set size (v5) | Singleton rate (v5) | Singleton accuracy (v5) |
|---|---|---|---|---|---|
| OCTDL triage NORMAL vs ABNORMAL | 0.95 | ABNORMAL 0.945, NORMAL 0.961 | 1.04 (1.08) | 0.95 (0.92) | 0.947 (0.952) |
| OCTDL triage NORMAL vs ABNORMAL | 0.90 | ABNORMAL 0.892, NORMAL 0.919 | 0.96 (0.98) | 0.95 (0.96) | 0.940 (0.931) |
| Kermany triage NORMAL vs ABNORMAL | 0.95 | ABNORMAL 0.941, NORMAL 0.943 | 1.17 (1.27) | 0.83 (0.73) | 0.929 (0.921) |
| Kermany triage NORMAL vs ABNORMAL | 0.90 | ABNORMAL 0.889, NORMAL 0.894 | 1.04 (1.09) | 0.96 (0.91) | 0.886 (0.878) |
| OCTID CSR/DR/MH (500x750) | 0.95 | CSR 1.000, DR 0.975, MH 0.936 | 2.01 (2.10) | 0.19 (0.18) | 0.951 (0.948) |
| OCTID CSR/DR/MH (500x750) | 0.90 | CSR 0.897, DR 0.951, MH 0.887 | 1.14 (1.25) | 0.85 (0.75) | 0.910 (0.910) |
| Kermany CNV/DME/DRUSEN | 0.95 | CNV 0.952, DME 0.952, DRUSEN 0.952 | 1.78 (1.82) | 0.37 (0.33) | 0.910 (0.899) |
| Kermany CNV/DME/DRUSEN | 0.90 | CNV 0.903, DME 0.899, DRUSEN 0.897 | 1.44 (1.53) | 0.60 (0.52) | 0.860 (0.841) |

*Source: `results/v6/conformal.json` (ResNet-18 hybrid; DINOv2-S hybrid in `conformal_dinov2s.json`; v5 values copied from `results/v5/conformal.json`).*
