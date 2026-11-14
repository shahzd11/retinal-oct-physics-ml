import json, re
from pathlib import Path
J = lambda p: json.load(open(p))
V6 = Path("results/v6")
S = {"OCTID CSR/DR/MH (500x750)": "octid_dis750", "OCTDL 7-class": "octdl_7", "Kermany 4-class": "kermany_4",
     "Kermany -> OCTDL (NORMAL/DME/AMD)": "kermany_to_octdl"}
hy = {t: J(V6 / f"hybrid_{s}.json") for t, s in S.items()}
cn = {t: (J(V6 / f"cnn_{s}.json") if (V6 / f"cnn_{s}.json").exists() else None) for t, s in S.items()}
H = lambda t: f"{hy[t]['models']['hybrid DINOv2-S']['bacc']:.3f}"
C = lambda t: f"{cn[t]['bacc']:.3f}" if cn[t] else "n/a"
r = Path("README.md").read_text(encoding="utf-8")

def sub(a, b):
    global r
    assert a in r, a[:70]
    r = r.replace(a, b, 1)

# --- stage table + pointer to next step
sub("| **v5** ROM + ML | Reduced-order modelling (POD), ensembles, conformal guarantees, scanner calibration | POD explains *why* the paper's grid fails. The ensemble is the best model on Kermany and on the unseen scanner. Triage gets a real, stated guarantee |",
    "| **v5** ROM + ML | Reduced-order modelling (POD), ensembles, conformal guarantees, scanner calibration | POD explains *why* the paper's grid fails. The ensemble is the best model on Kermany and on the unseen scanner. Triage gets a real, stated guarantee |\n"
    f"| **v6** Image-level ROM + hybrid | POD of the flattened *image*, healthy-only POD residual (defect detection), frozen deep features + physics, Grad-CAM check | The hybrid beats v5 on every task (e.g. Kermany 4-class {hy['Kermany 4-class']['models']['hybrid DINOv2-S']['bacc']:.3f}) with physics still ranked first, but the fine-tuned CNN stays ahead, clearly so on a new scanner |")
sub("The next step (image-level ROM, hybrid deep + physics model) is written up as a ready-to-run prompt in [`CLAUDE_CODE_PROMPT.md`](CLAUDE_CODE_PROMPT.md).",
    "v6 (image-level ROM, hybrid deep + physics model) is in [Section 11](#11-v6-image-level-rom-and-a-hybrid-model), summarised in [`results/v6/SUMMARY.md`](results/v6/SUMMARY.md). "
    "A presentation deck that explains the whole project is in [Section 12](#12-presentation).")

# --- contents (renumber)
old_toc = r[r.index("11. [How we evaluate"):r.index("15. [References](#15-references)") + len("15. [References](#15-references)")]
new_toc = ("11. [v6: image-level ROM and a hybrid model](#11-v6-image-level-rom-and-a-hybrid-model)\n"
           "12. [Presentation](#12-presentation)\n"
           "13. [How we evaluate (and why)](#13-how-we-evaluate-and-why)\n"
           "14. [Code map](#14-code-map)\n"
           "15. [Limitations and next steps](#15-limitations-and-next-steps)\n"
           "16. [Glossary](#16-glossary)\n"
           "17. [References](#17-references)")
sub(old_toc, new_toc)
for a, b in (("## 11. How we evaluate", "## 13. How we evaluate"), ("## 12. Code map", "## 14. Code map"),
             ("## 13. Limitations and next steps", "## 15. Limitations and next steps"), ("## 14. Glossary", "## 16. Glossary"),
             ("## 15. References", "## 17. References")):
    sub(a, b)

# --- progress at a glance
sub("![progress](results/v5/progress.png)", "![progress](presentation/figures/fig13_progress.png)")
tbl_old = r[r.index("| Task (chance) | Paper method |"):r.index("*Every cell is the best row")]
rows = tbl_old.strip().split("\n")
hdr = "| Task (chance) | Paper method | v2 | v3 | v4 | v5 ensemble | **v6 hybrid** | CNN (black box) |"
sep = "|---|---|---|---|---|---|---|---|"
keys = ["OCTID CSR/DR/MH (500x750)", "OCTDL 7-class", "Kermany 4-class", "Kermany -> OCTDL (NORMAL/DME/AMD)"]
new_rows = [hdr, sep]
for row, t in zip(rows[2:], keys):
    cells = [c.strip() for c in row.strip("|").split("|")]
    cells[5] = cells[5].replace("**", "")
    new_rows.append("| " + " | ".join(cells[:6] + [f"**{H(t)}**", C(t)]) + " |")
sub(tbl_old, "\n".join(new_rows) + "\n\n")
sub("*Every cell is the best row of that stage's result file (`results/v5/progress.json` for the first three rows; `results/v2|v3/cross_scanner.json` and `results/v5/harmonize.json` for the last). v4 adds fractal features on top of v3, so a v4 cell can be slightly below v3 when the fractal features add nothing.*",
    "*Every cell is the best row of that stage's result file (`results/v5/progress.json` for the first three rows; `results/v2|v3/cross_scanner.json` and `results/v5/harmonize.json` for the last). v4 adds fractal features on top of v3, so a v4 cell can be slightly below v3 when the fractal features add nothing. "
    "v6 hybrid: `results/v6/hybrid_<task>.json` (DINOv2-S hybrid, chosen before seeing results). CNN: `results/v6/cnn_<task>.json`, fine-tuned ResNet-18 re-run on the *same* folds as every other model so the comparison is paired "
    "(the earlier `cnn_baseline.py` runs gave 0.940 / 0.864 / 0.932); not run on OCTDL to stay inside the GPU budget.*")
sub("- A fine-tuned CNN is still about 7 points better within one scanner. Closing that gap while staying explainable is the v6 plan.",
    f"- v6 adds frozen deep features to the physics and gains 2–4 points everywhere, with a physics group still the most important. "
    f"A fine-tuned CNN remains better: by {float(C('OCTID CSR/DR/MH (500x750)')) - float(H('OCTID CSR/DR/MH (500x750)')):.3f} on OCTID (not significant), "
    f"{float(C('Kermany 4-class')) - float(H('Kermany 4-class')):.3f} on Kermany and {float(C('Kermany -> OCTDL (NORMAL/DME/AMD)')) - float(H('Kermany -> OCTDL (NORMAL/DME/AMD)')):.3f} on a new scanner.")

# --- quick start
sub("**To recompute features from images (v1–v4)** you need the datasets:",
    "**v6 needs the images** (it works on the flattened image, not on the feature tables). After downloading (below):\n\n"
    "| Command | Time (8-core laptop, 8 GB) | Output |\n|---|---|---|\n"
    "| `python v6_tiles.py octid octdl kermany` | ~2 min | flattened 128×512 tiles, `data/cache/v6/` |\n"
    "| `python v6_embed.py octid octdl kermany` | ~5 min (Apple GPU) | frozen ResNet-18 / DINOv2-S embeddings |\n"
    "| `python v6_image_pod.py` | ~40 min | Step 1, `results/v6/image_pod_*.json` |\n"
    "| `python v6_residual.py` | ~60 min | Step 2, `residual_*.json` |\n"
    "| `python v6_cnn.py --task octid_dis750` (also `kermany_4`, `cross`) | 6 / 50 / 7 min (GPU) | CNN on the same folds, fold models for Grad-CAM |\n"
    "| `python v6_hybrid.py` | ~60 min | Step 3, `hybrid_*.json`, `hybrid_importance.json` |\n"
    "| `python v6_gradcam.py` | ~5 min | Step 4, `gradcam_octid_dis750.json` |\n"
    "| `python v6_tiles.py octdl --blur 0.5 1 --features` then `python v6_embed.py octdl --blur 0.5 1` and `python v6_cross.py` | ~25 min | Step 5, `cross.json` |\n"
    "| `python v6_conformal.py` | ~40 min | Step 6, `conformal.json` |\n"
    "| `python v6_baselines.py`, `python v6_figdata.py`, `python v6_paper_claims.py`, `python v6_summary.py` | minutes | figure inputs, `SUMMARY.md` |\n\n"
    "Every v6 script takes `--quick` (1 seed, 2 folds) for a smoke test. `pytest tests/` checks that no transformer is ever fitted on test rows.\n\n"
    "**To recompute features from images (v1–v4)** you need the datasets:")
Path("README.md").write_text(r, encoding="utf-8")
print("ok")
