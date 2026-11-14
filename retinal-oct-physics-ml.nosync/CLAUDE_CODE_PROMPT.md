# Claude Code prompt: v6, image-level improvements

Paste everything below the line into Claude Code, run from the repo root on a machine that has the datasets
(`python download_data.py`, `python download_data.py octdl`, `python download_data.py kermany-subset <zip>`).
v1–v5 run from the committed feature CSVs. v6 needs the images.

---

You are working in the `Multi-Parametric-Retinal-Diagnostics` repo. It reproduces and extends Banerjee, Sarkar & Munshi,
*"Multi-Parametric Retinal Diagnostics: A Deterministic Physics-Based Engine for Tomographic Classification"* (IIT Gandhinagar).
Read `README.md` fully before touching code, especially "What the data can and cannot show" and the v5 section.

## Where things stand (do not redo)
- Paper pipeline reproduced (`retina/preprocess.py`, `retina/features.py`, `train_eval.py`).
- The paper's 100% NORMAL triage on OCTID is a scanner confound. Never report OCTID 4-class as a headline number.
- v2–v4 hand-crafted features (layer-flattened texture, disease-specific, NDT, boundary fractal) reach
  balanced accuracy ≈ 0.87 on OCTID CSR/DR/MH, ≈ 0.69 on OCTDL 7-class, ≈ 0.68–0.70 on Kermany 4-class.
- v5 (feature-level): POD of the feature fields, a soft-voting ensemble, conformal triage and scanner harmonisation.
  Results are in `results/v5/`. POD of the paper's *unaligned* 10×15 grid is dominated by retina position (mode 2 vs
  tissue height ρ ≈ −0.9), so POD must be done on the **RPE-flattened** image.
- `cnn_baseline.py` (fine-tuned ResNet-18) reaches 0.94 on OCTID CSR/DR/MH and 0.86 on Kermany 4-class.
  That gap is what v6 has to close or explain.

## Non-negotiable evaluation rules
1. Patient-grouped `StratifiedGroupKFold(5, shuffle=True, random_state=seed)`, seeds 0,1,2. Use the same folds as
   `retina/evaluation.oof_predictions`, so every new number is paired with the existing ones.
2. Everything learned from data (POD modes, scalers, PCA, thresholds, conformal quantiles) is fitted **inside** the
   training fold only. Add a unit test that fails if a transformer sees test indices.
3. Report balanced accuracy with the patient cluster-bootstrap CI (`cluster_bootstrap_ci`), plus accuracy and macro-F1.
   Any claim of "A beats B" uses `ndt_paired_test.paired_boot`.
4. Tasks: OCTID CSR/DR/MH (500×750 only), OCTDL 7-class, OCTDL 6 diseases, Kermany 4-class, Kermany CNV/DME/DRUSEN,
   and cross-scanner Kermany→OCTDL (NORMAL/DME/AMD).
5. Keep CPU-friendly defaults (`n_jobs=1` for LightGBM on ≤4 cores) and add `--quick` flags (1 seed, 2 folds) for smoke tests.
6. New code goes in `retina/` (library) and `v6_*.py` (scripts); outputs go to `results/v6/`. Do not edit v1–v5 outputs.

## Work plan (one commit per step; run and record results before moving on)

### Step 1: Image-level POD ("eigen-retinas") on the flattened retina
- New `retina/image_rom.py`. For each scan call `retina.layers.flatten`, take the canvas band from 40 px above the ILM
  to 60 px below the RPE, centre it on the fovea column, and resample to a fixed 64×256 tile with a validity mask
  (masked pixels → training-fold mean).
- Snapshot matrix `A` (n × 16384), fold-wise mean subtraction, `sklearn.utils.extmath.randomized_svd`, r ∈ {10, 25, 50, 100, 200}.
- Classifier on the POD coefficients: LR and LightGBM. Then concatenate with the v5 best feature set and refit the soft vote.
- Save: energy spectrum, first 8 modes as images, and per-class mean reconstructions.
- Acceptance: a table of balanced accuracy vs r for every task, and a paired test of "v5 best + image-POD" vs "v5 best".

### Step 2: POD residual anomaly score (ROM-based defect detection, NDT style)
- Fit POD on **healthy training eyes only**. For each test scan compute
  `e(x) = ‖x − x̄ − V_r V_rᵀ(x − x̄)‖² / ‖x − x̄‖²`.
  Also compute a spatial residual map, which shows *where* the scan departs from a healthy retina.
- Report NORMAL-vs-ABNORMAL AUROC on OCTDL and Kermany (within-scanner) and Kermany-healthy→OCTDL (cross-scanner).
  Add `e(x)` as one extra feature to the soft vote.
- Figure: residual maps for 2 scans per class (these are the explainability figures for the paper revision).

### Step 3: Hybrid model (frozen deep features + physics)
- `pip install timm torch torchvision` (CPU). Extract frozen embeddings once per scan and cache them to `.npy`:
  ResNet-18 (ImageNet) and `vit_small_patch14_dinov2` (DINOv2-S), on the flattened tile replicated to 3 channels.
- Fold-wise PCA to 64 dimensions, then concatenate [PCA embedding, v5 best features, image-POD coefficients, e(x)] and
  feed the soft vote. Compare against `cnn_baseline.py` on identical folds.
- Acceptance: the hybrid either matches the fine-tuned CNN within the CI on OCTID CSR/DR/MH while keeping physics
  features in the top-10 grouped permutation importance (`grouped_permutation_importance`), or the README says
  plainly that it doesn't.

### Step 4: Does the deep model look where the physics looks?
- Grad-CAM on the fine-tuned ResNet-18 (fold models). Average the CAM mass inside each anatomical band
  (`retina.layers.band_masks`) and compare it with the per-band grouped permutation importance of the physics model
  (Spearman across bands × classes). Show it as a band × class heatmap pair.

### Step 5: Cross-scanner robustness
- Kermany→OCTDL with the hybrid, with and without the v5 healthy-reference harmonisation (k = 30 healthy OCTDL eyes)
  and with speckle-matched test-time augmentation (Gaussian blur σ ∈ {0, 0.5, 1}; average the probabilities).
- Report a per-class confusion matrix. DME recall is the known weak point.

### Step 6: Conformal triage on the best model
- Reuse `retina/conformal.py` (class-conditional split conformal). α ∈ {0.05, 0.10}. Report per-class coverage, mean set size,
  singleton rate and singleton accuracy, and the abnormal-class coverage for triage.

### Step 7: README and paper material
- Add a "v6: image-level" section to `README.md` in the same style (undergraduate level, short formula boxes, one
  results table per step, honest negatives). Update the "Progress at a glance" table.
- Write `results/v6/SUMMARY.md` with 5 bullet findings and every number traced to a JSON file.

## Stop and ask me before
- downloading more than 15 GB, or training anything that needs a GPU for more than 2 h;
- changing any v1–v5 code path or result file;
- reporting any number that is not produced by a script in the repo.
