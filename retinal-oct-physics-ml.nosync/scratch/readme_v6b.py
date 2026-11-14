import json
from pathlib import Path
J = lambda p: json.load(open(p))
V6 = Path("results/v6")
r = Path("README.md").read_text(encoding="utf-8")
T = Path("results/v6/readme_tables.md").read_text(encoding="utf-8")
blocks = {}
cur = None
for line in T.split("\n"):
    if line.startswith("### Step"):
        cur = line; blocks[cur] = []
    elif cur:
        blocks[cur].append(line)
B = lambda k: "\n".join(next(v for kk, v in blocks.items() if kk.startswith(f"### Step {k}"))).strip()

def sub(a, b):
    global r
    assert a in r, a[:70]
    r = r.replace(a, b, 1)

sub("- v6 adds frozen deep features to the physics and gains 2–4 points everywhere,", "- v6 adds frozen deep features to the physics and gains 2–4 points over v5 on every task,")
ra, cr, hi, g = J(V6 / "residual_auroc.json"), J(V6 / "cross.json"), J(V6 / "hybrid_importance.json"), J(V6 / "gradcam_octid_dis750.json")
cf = J(V6 / "conformal.json")
tri = [x for x in cf if x["task"].startswith("OCTDL triage") and x["alpha"] == 0.05][0]
hyo = J(V6 / "hybrid_octid_dis750.json")
pc = hyo["paired"]["hybrid DINOv2-S vs CNN (fine-tuned ResNet-18)"]
best = cr["best_config"]

sec = f"""## 11. v6: image-level ROM and a hybrid model

New code: `retina/image_rom.py`, `retina/hybrid.py`, `retina/v6data.py`, `retina/v6eval.py`, `retina/v6run.py`, scripts `v6_*.py`, test `tests/test_no_leak.py`. All results in `results/v6/`, summary in [`results/v6/SUMMARY.md`](results/v6/SUMMARY.md).

**Same rules as before, made stricter.** Every v6 model uses exactly the folds of `retina.evaluation.oof_predictions` (`StratifiedGroupKFold(5, shuffle=True, random_state=seed)`, seeds 0, 1, 2), so each new number is *paired* with the v5 number on the same patients. Everything learned from data (POD modes, the healthy-only residual basis, PCA of the embeddings, scalers, conformal thresholds) lives inside the model's `fit()` and therefore sees training patients only; `tests/test_no_leak.py` fails if any of them is fitted on a test row. Every script saves its out-of-fold predictions and probabilities (`results/v6/oof_<task>_<model>.csv`).

### 11.1 Eigen-retinas: POD of the flattened image

v5 showed that POD needs a consistent coordinate frame. The flattened retina is one. For every scan we cut a tile from 40 px above the ILM to 60 px below the RPE, ±200 px around the fovea, and resample it to 64 × 256 = 16 384 pixels (masked pixels → training-fold mean):

$$
A = \\begin{{bmatrix}} (x_1 - \\bar x)^T \\\\ \\vdots \\\\ (x_n - \\bar x)^T \\end{{bmatrix}} \\in \\mathbb{{R}}^{{n \\times 16384}}, \\qquad
A \\approx U_r \\Sigma_r V_r^T \;\;(\\text{{randomized SVD}}), \\qquad a = V_r^T (x - \\bar x)
$$

![eigen-retinas](presentation/figures/fig14a_image_pod_modes.png)

The modes now look like anatomy (layer brightness, thickness, the foveal dip) instead of retina position.

{B(1)}

**Reading.** The image modes alone are much weaker than the hand-crafted physics features, and adding them to v5 changes nothing significant (best case OCTID, p = 0.14; worst case OCTDL 6 diseases, −0.029). An honest negative: *a good coordinate frame makes POD interpretable, but the energetic image patterns are still not the diagnostic ones* (the v5 "energy ≠ information" lesson again).

### 11.2 POD residual: ROM-based defect detection

In NDT, a part is flagged when it cannot be explained by a model of good parts. Here: fit POD on **healthy training eyes only**, then for a new scan

$$
e(x) = \\frac{{\\lVert x - \\bar x - V_r V_r^T (x - \\bar x) \\rVert^2}}{{\\lVert x - \\bar x \\rVert^2}}
$$

is the fraction of its deviation from the average healthy retina that the healthy modes cannot rebuild. The residual image $x - \\hat x$ shows *where* it departs.

![residual maps](presentation/figures/fig14c_residual_maps_kermany_4.png)

{B(2)}

**Reading.** With no disease labels at all, e(x) separates healthy from diseased eyes with AUROC ≈ 0.70, and the same healthy basis transfers to the other scanner almost unchanged ({ra['auroc_cross']['by_r']['25']['auroc']:.3f}). It is a useful, explainable screening signal, but as one more feature next to 171 physics features it is redundant.

### 11.3 Hybrid: frozen deep features + physics

A frozen network (never trained on our labels) turns each tile into an embedding: ResNet-18 (ImageNet, 512 numbers) or DINOv2-S (self-supervised, 384 numbers). Inside each training fold we reduce it with PCA to 64 numbers and concatenate

$$
z = [\\,\\text{{PCA}}_{{64}}(\\text{{embedding}}),\; \\text{{v5 features (171)}},\; \\text{{image-POD }} a_{{1..50}},\; \\log e(x)\\,] \;\\to\; \\text{{soft vote (LR + LightGBM + SVM)}}
$$

The DINOv2-S hybrid was chosen as "the v6 model" before seeing any result. The fine-tuned CNN (`v6_cnn.py`, same recipe as `cnn_baseline.py`) was re-run on the identical folds so the comparison is paired.

![hybrid vs CNN](presentation/figures/fig14e_hybrid_vs_cnn.png)

{B(3)}

**Acceptance check (stated plainly).** The target was to match the fine-tuned CNN within the CI on OCTID CSR/DR/MH while keeping physics features in the top-10 importances. The paired difference is {pc['diff_bacc']:+.3f} with 95% CI [{pc['diff_ci'][0]:+.3f}, {pc['diff_ci'][1]:+.3f}]: the interval *just* includes zero, so it is not significantly worse, but the point estimate is 3.6 points lower, and on Kermany and on the new scanner the CNN is clearly better. The physics part is not decorative: the NDT pulse-echo group is the single most important group, ahead of the embedding. (With only 10 groups, "in the top 10" is automatic, so we report the top-5 share instead.) **The hybrid does not close the gap to the CNN; it narrows it while staying explainable.**

### 11.4 Does the CNN look where the physics looks?

Grad-CAM weights the last convolutional feature maps $A_c$ by the average gradient of the class score, $\\text{{CAM}} = \\text{{ReLU}}(\\sum_c w_c A_c)$, $w_c = \\overline{{\\partial y / \\partial A_c}}$. We map each CAM onto the flattened retina with the same column shifts as `retina.layers.flatten`, sum it inside each band, and compare with the physics model's per-band permutation importance.

![gradcam](presentation/figures/fig14d_gradcam_vs_physics.png)

{B(4)}

**Reading.** No agreement (ρ = {g['spearman_rho']:.2f}). The CNN spends about a quarter of its attention in the choroid band, where the physics model finds little; the physics model uses the inner retina for CSR and the outer retina for DR. Either the CNN uses choroidal/shadowing cues the hand-crafted features miss, or it uses context we would not call pathology. This is a question, not an answer, and a good one for future work.

### 11.5 Cross-scanner robustness (Kermany → OCTDL)

Two NDT-style corrections: a **reference block** (z-score every input block with k = 30 healthy eyes of the target scanner, removed from the test set, 10 draws) and **speckle-matched test-time augmentation** (blur the test scan with σ ∈ {{0, 0.5, 1}} px, recompute *every* feature, average the probabilities).

![harmonisation](presentation/figures/fig12_harmonisation.png)

{B(5)}

**Reading.** The hybrid is the best non-CNN model on the new scanner ({cr['configs'][best]['bacc']:.3f}), but the CNN reaches {J(V6 / 'cnn_kermany_to_octdl.json')['bacc']:.3f}. Calibration with healthy eyes trades AMD recall for DME recall and lowers balanced accuracy overall; TTA leaves the hybrid unchanged and hurts v5. **DME recall is still the weak point.** Affine calibration cannot fix texture differences between scanners, the same conclusion as v5.

### 11.6 Conformal triage on the hybrid

{B(6)}

**Reading.** The guarantee holds for the hybrid as it did for v5 (ABNORMAL coverage {tri['coverage_per_class']['ABNORMAL']:.3f} at target 0.95 on OCTDL), and because the hybrid is more accurate the prediction sets are smaller on every task: fewer scans need a second opinion for the same guarantee.

---

## 12. Presentation

[![deck preview](presentation/preview.png)](presentation/Retinal_OCT_Physics_ML.pptx)

- Deck: [`presentation/Retinal_OCT_Physics_ML.pptx`](presentation/Retinal_OCT_Physics_ML.pptx) (26 slides, 16:9, speaker notes on every slide), PDF: [`presentation/Retinal_OCT_Physics_ML.pdf`](presentation/Retinal_OCT_Physics_ML.pdf).
- Rebuild: `python make_figures.py && python make_deck.py --pdf`. `make_figures.py` reads only `results/` and writes `presentation/figures/*.png`; `make_deck.py` builds the deck with python-pptx.
- Every number on a slide is read from `results/` at build time and listed with its source file in [`presentation/numbers.json`](presentation/numbers.json) (figure values in `presentation/figures/figure_numbers.json`).

---

"""
sub("## 13. How we evaluate (and why)", sec + "## 13. How we evaluate (and why)")
# code map
sub("v5_rom.py  v5_ensemble.py  v5_conformal.py  v5_harmonize.py  v5_summary.py",
    "  image_rom.py      v6: flattened-retina tiles, ImagePOD (randomized SVD, residual e(x))\n"
    "  hybrid.py         v6: HybridClassifier (POD / residual / PCA fitted inside fit())\n"
    "  v6data.py v6eval.py v6run.py   v6: tasks, identical folds, OOF CSVs, paired tests\n"
    "v5_rom.py  v5_ensemble.py  v5_conformal.py  v5_harmonize.py  v5_summary.py\n"
    "v6_tiles.py v6_embed.py v6_image_pod.py v6_residual.py v6_cnn.py v6_hybrid.py v6_gradcam.py\n"
    "v6_cross.py v6_conformal.py v6_baselines.py v6_figdata.py v6_paper_claims.py v6_summary.py\n"
    "make_figures.py make_deck.py   presentation (figures from results/ only, python-pptx deck)\n"
    "tests/test_no_leak.py          fails if any transformer is fitted on test rows")
sub("results/v2 ... v5       per-stage outputs (CSV/JSON/PNG/MD)", "results/v2 ... v6       per-stage outputs (CSV/JSON/PNG/MD/NPZ)")
# limitations / next steps
a = r.index("**Next (v6, see [`CLAUDE_CODE_PROMPT.md`]")
b = r.index("---", a)
r = r[:a] + ("**Next steps after v6**\n"
             "1. Find out what the CNN sees in the choroid band (Section 11.4): occlusion tests per band, or choroid-specific NDT features (shadowing, vessel lumen texture).\n"
             "2. Scanner robustness needs more than affine calibration: train on two scanners and test on the third, or learn a scanner-invariant tile representation.\n"
             "3. Fine-tune the hybrid's backbone (with physics features kept as a parallel branch) and re-check the importance ranking.\n"
             "4. 3-D volumes (the paper's \"3D anisotropic\" idea) would let POD and the NDT features use the full scan, not one B-scan.\n\n") + r[b:]
sub("- The v5 ensemble gains are small and not significant on OCTID or OCTDL. The robust win is on Kermany and across scanners.",
    "- The v5 ensemble gains are small and not significant on OCTID or OCTDL. The robust win is on Kermany and across scanners.\n"
    "- The CNN was run on 3 of the 6 tasks (OCTID CSR/DR/MH, Kermany 4-class, Kermany → OCTDL) to stay inside a 2-hour GPU budget, with one seed (seed-0 folds).\n"
    "- v6 adds 3 ImageNet / self-supervised backbones' worth of assumptions: the embeddings are frozen, so their behaviour on OCT is not tuned.")
sub("| MAPOD, $a_{90/95}$ |", "| Eigen-retina | A POD mode of the flattened retina image |\n| POD residual e(x) | Fraction of a scan's deviation that the healthy-eye POD modes cannot rebuild |\n| Grad-CAM | Heat map of where a CNN's class score depends on the image |\n| MAPOD, $a_{90/95}$ |")
sub("- MIL-HDBK-1823A. *Nondestructive Evaluation System Reliability Assessment.* US DoD, 2009.",
    "- MIL-HDBK-1823A. *Nondestructive Evaluation System Reliability Assessment.* US DoD, 2009.\n"
    "- Halko N., Martinsson P.G., Tropp J.A. *Finding structure with randomness: probabilistic algorithms for constructing approximate matrix decompositions.* SIAM Review 53(2), 2011.\n"
    "- Oquab M. et al. *DINOv2: Learning robust visual features without supervision.* TMLR, 2024.\n"
    "- He K. et al. *Deep residual learning for image recognition.* CVPR 2016.\n"
    "- Selvaraju R.R. et al. *Grad-CAM: Visual explanations from deep networks via gradient-based localization.* ICCV 2017.")
Path("README.md").write_text(r, encoding="utf-8")
print("ok")
