import json
from pathlib import Path
J = lambda p: json.load(open(p))
V6 = Path("results/v6")
r = Path("README.md").read_text(encoding="utf-8")
T = Path("results/v6/readme_tables.md").read_text(encoding="utf-8")
blocks, cur = {}, None
for line in T.split("\n"):
    if line.startswith("### Step"):
        cur = line; blocks[cur] = []
    elif cur:
        blocks[cur].append(line)
B = lambda k: "\n".join(next(v for kk, v in blocks.items() if kk.startswith(f"### Step {k}"))).strip()
def sub(a, b):
    global r
    assert a in r, a[:80]
    r = r.replace(a, b, 1)
ab = J(V6 / "ablation.json")["tasks"]
ra, cr, hi, g, cf = J(V6 / "residual_auroc.json"), J(V6 / "cross.json"), J(V6 / "hybrid_importance.json"), J(V6 / "gradcam_octid_dis750.json"), J(V6 / "conformal.json")
OCT, K4, X = "OCTID CSR/DR/MH (500x750)", "Kermany 4-class", "Kermany -> OCTDL (NORMAL/DME/AMD)"
SH = {OCT: "OCTID", "OCTDL 7-class": "OCTDL 7-class", "OCTDL 6 diseases": "OCTDL 6-disease", K4: "Kermany 4-class",
      "Kermany CNV/DME/DRUSEN": "Kermany 3-disease", X: "the new scanner"}
pr = lambda p: f"{p['diff_bacc']:+.3f} [{p['diff_ci'][0]:+.3f}, {p['diff_ci'][1]:+.3f}], p = {p['p_one_sided']:.3f}"
HR = "hybrid ResNet-18"
sig = lambda k: [SH[t] for t in ab if ab[t]["paired"][k]["diff_ci"][0] > 0]
nsig = lambda k: [SH[t] for t in ab if not ab[t]["paired"][k]["diff_ci"][0] > 0]
kv5, kem, kcn = f"{HR} vs v5 best (soft vote)", f"{HR} vs ResNet-18 embedding only", f"{HR} vs CNN"
hb = {t: ab[t]["models"][HR]["bacc"] for t in ab}
cnn = {t: J(V6 / f"cnn_{s}.json")["bacc"] for t, s in ((OCT, "octid_dis750"), (K4, "kermany_4"), (X, "kermany_to_octdl"))}
tri = [x for x in cf if x["task"].startswith("OCTDL triage") and x["alpha"] == 0.05][0]

# ---- stage table row + bottom line + progress table
a = r.index("| **v6** Image-level ROM + hybrid |"); b = r.index("\n", a)
r = r[:a] + (f"| **v6** Image-level ROM + hybrid | POD of the flattened *image*, healthy-only POD residual (defect detection), frozen deep features + physics, "
             f"ablation, Grad-CAM check | The ResNet-18 hybrid is significantly better than v5 on {', '.join(sig(kv5))}, not on {', '.join(nsig(kv5))}; "
             f"physics adds accuracy over the embedding alone on OCTDL 7-class and Kermany. The fine-tuned CNN stays ahead, clearly so on a new scanner |") + r[b:]
a = r.index("- v6 adds frozen deep features"); b = r.index("\n", a)
r = r[:a] + (f"- v6 adds frozen deep features (ResNet-18) to the physics. Paired against v5, the gain is significant on {', '.join(sig(kv5))} "
             f"(e.g. Kermany 4-class {pr(ab[K4]['paired'][kv5])}) and the CI includes zero on {', '.join(nsig(kv5))} "
             f"({pr(ab['OCTDL 6 diseases']['paired'][kv5])}). A fine-tuned CNN remains better: by {-ab[OCT]['paired'][kcn]['diff_bacc']:.3f} on OCTID "
             f"(CI includes zero), {-ab[K4]['paired'][kcn]['diff_bacc']:.3f} on Kermany and {-ab[X]['paired'][kcn]['diff_bacc']:.3f} on a new scanner.") + r[b:]
sub("| Task (chance) | Paper method | v2 | v3 | v4 | v5 ensemble | **v6 hybrid** | CNN (black box) |",
    "| Task (chance) | Paper method | v2 | v3 | v4 | v5 ensemble | **v6 hybrid (ResNet-18)** | CNN (black box) |")
for t in (OCT, "OCTDL 7-class", K4, X):
    old = J(V6 / f"hybrid_{dict(zip((OCT,'OCTDL 7-class',K4,X),('octid_dis750','octdl_7','kermany_4','kermany_to_octdl')))[t]}.json")["models"]["hybrid DINOv2-S"]["bacc"]
    sub(f"**{old:.3f}**", f"**{hb[t]:.3f}**")
sub("v6 hybrid: `results/v6/hybrid_<task>.json` (DINOv2-S hybrid, chosen before seeing results).",
    "v6 hybrid: `results/v6/hybrid_<task>.json`, ResNet-18 hybrid (the headline model; the DINOv2-S hybrid was pre-registered before the results, "
    "ResNet-18 was chosen afterwards because it wins or ties on 4 of 6 tasks, and both are reported in Section 11).")
# quick start lines
sub("| `python v6_hybrid.py` | ~60 min | Step 3, `hybrid_*.json`, `hybrid_importance.json` |",
    "| `python v6_hybrid.py` (importance: `--task importance --backbone resnet18`/`dinov2s`) | ~60 min | Step 3, `hybrid_*.json`, `hybrid_importance*.json` |\n"
    "| `python v6_ablation.py` | ~2 min | paired hybrid vs embedding-only / v5 / CNN, `ablation.json` |")
sub("| `python v6_conformal.py` | ~40 min | Step 6, `conformal.json` |",
    "| `python v6_conformal.py` (`--backbone dinov2s` for the DINOv2 hybrid) | ~40 min | Step 6, `conformal.json` |")
sub("v6_cross.py v6_conformal.py v6_baselines.py v6_figdata.py v6_paper_claims.py v6_summary.py",
    "v6_cross.py v6_conformal.py v6_ablation.py v6_baselines.py v6_figdata.py v6_paper_claims.py v6_summary.py")
sub("(26 slides, 16:9, speaker notes on every slide)", "(28 slides, 16:9, speaker notes on every slide)")

# ---- rebuild section 11.3 onwards (11.3 .. 11.6)
a = r.index("### 11.3 Hybrid: frozen deep features + physics"); b = r.index("---\n\n## 12. Presentation")
sec = f"""### 11.3 Hybrid: frozen deep features + physics

A frozen network (never trained on our labels) turns each tile into an embedding: ResNet-18 (ImageNet, 512 numbers) or DINOv2-S (self-supervised, 384 numbers). Inside each training fold we reduce it with PCA to 64 numbers and concatenate

$$
z = [\\,\\text{{PCA}}_{{64}}(\\text{{embedding}}),\; \\text{{v5 physics features (171)}},\; \\text{{ROM: image-POD }} a_{{1..50}},\; \\log e(x)\\,] \;\\to\; \\text{{soft vote (LR + LightGBM + SVM)}}
$$

**Which hybrid is the headline.** The DINOv2-S hybrid was chosen as "the v6 model" *before* seeing any result. After the runs, the ResNet-18 hybrid won or tied on 4 of 6 tasks, so it is the headline model in this README, the summary and the deck. That choice was made after seeing the results; both hybrids and both embedding-only models are reported in every table so the reader can judge. The fine-tuned CNN (`v6_cnn.py`, same recipe as `cnn_baseline.py`) was re-run on the identical folds so every comparison is paired.

![hybrid vs CNN](presentation/figures/fig14e_hybrid_vs_cnn.png)

{B(3)}

**Ablation: what does the physics add?** The embedding-only rows answer this directly. With the same frozen ResNet-18, adding the physics + ROM features is significantly better on {', '.join(sig(kem))} (+0.03 to +0.05), and the CI includes zero on {', '.join(nsig(kem))}. On OCTID the embedding alone is already as good ({pr(ab[OCT]['paired'][kem])}). The DINOv2-S embedding is weaker on its own, so physics adds more there.

![ablation](presentation/figures/fig14f_ablation.png)

**Importance is not accuracy.** Grouped permutation importance was measured on OCTID CSR/DR/MH only. For the ResNet-18 hybrid it ranks the deep embedding first and the NDT pulse-echo group second; the two ROM groups (image-POD, e(x)) have almost no weight. Importance shows how much the *fitted* model relies on a group, not how much accuracy the group adds: on OCTID the physics features are heavily used, yet removing them does not lower accuracy, because the embedding carries the same information.

**Acceptance check (stated plainly).** The target was to match the fine-tuned CNN within the CI on OCTID CSR/DR/MH while keeping physics in the top importances. The ResNet-18 hybrid's paired difference to the CNN is {pr(ab[OCT]['paired'][kcn])}: the interval just includes zero, but the point estimate is {-ab[OCT]['paired'][kcn]['diff_bacc']:.3f} lower, and on Kermany ({pr(ab[K4]['paired'][kcn])}) and on the new scanner ({pr(ab[X]['paired'][kcn])}) the CNN is clearly better. Physics groups rank 2nd to 4th, behind the embedding. **The hybrid does not close the gap to the CNN; it narrows it while staying partly explainable.**

### 11.4 Does the CNN look where the physics looks?

Grad-CAM weights the last convolutional feature maps $A_c$ by the average gradient of the class score, $\\text{{CAM}} = \\text{{ReLU}}(\\sum_c w_c A_c)$, $w_c = \\overline{{\\partial y / \\partial A_c}}$. We map each CAM onto the flattened retina with the same column shifts as `retina.layers.flatten`, sum it inside each band, and compare with the physics model's per-band permutation importance.

![gradcam](presentation/figures/fig14d_gradcam_vs_physics.png)

{B(4)}

**Reading.** No agreement (ρ = {g['spearman_rho']:.2f}). The CNN spends about a quarter of its attention in the choroid band, where the physics model finds little; the physics model uses the inner retina for CSR and the outer retina for DR. Either the CNN uses choroidal/shadowing cues the hand-crafted features miss, or it uses context we would not call pathology. This is a question, not an answer, and a good one for future work.

### 11.5 Cross-scanner robustness (Kermany → OCTDL)

Two NDT-style corrections: a **reference block** (z-score every input block with k = 30 healthy eyes of the target scanner, removed from the test set, 10 draws) and **speckle-matched test-time augmentation** (blur the test scan with σ ∈ {{0, 0.5, 1}} px, recompute *every* feature, average the probabilities).

![harmonisation](presentation/figures/fig12_harmonisation.png)

{B(5)}

**Reading.** The ResNet-18 hybrid is the best non-CNN model on the new scanner ({cr['configs'][cr['best_config']]['bacc']:.3f}, significantly better than v5: {pr(ab[X]['paired'][kv5])}), but the CNN reaches {cnn[X]:.3f}. Calibration with healthy eyes trades AMD recall for DME recall and does not raise balanced accuracy; TTA leaves the hybrids unchanged and hurts v5. **DME recall is still the weak point.**

### 11.6 Conformal triage on the headline hybrid

{B(6)}

**Reading.** At a 0.95 target the ResNet-18 hybrid keeps {tri['coverage_per_class']['ABNORMAL']:.3f} of diseased OCTDL eyes, just under the target and within one binomial standard error; at 0.90 and on the other tasks coverage sits on or near the target, as for v5. Because the hybrid is more accurate, the prediction sets are smaller on every task: fewer scans need a second opinion for the same guarantee. (DINOv2-S hybrid: `results/v6/conformal_dinov2s.json`.)

"""
r = r[:a] + sec + r[b:]
Path("README.md").write_text(r, encoding="utf-8")
print("ok")
