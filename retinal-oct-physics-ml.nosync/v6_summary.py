"""v6 / Step 7: write results/v6/SUMMARY.md (5 findings, every number traced to a JSON file) and
results/v6/readme_tables.md (the per-step tables pasted into README.md). Reads results/ only.

  python v6_summary.py
"""
import json
from pathlib import Path

V6 = Path("results/v6")
TASKS = ["OCTID CSR/DR/MH (500x750)", "OCTDL 7-class", "OCTDL 6 diseases", "Kermany 4-class",
         "Kermany CNV/DME/DRUSEN", "Kermany -> OCTDL (NORMAL/DME/AMD)"]
SLUG = {"OCTID CSR/DR/MH (500x750)": "octid_dis750", "OCTDL 7-class": "octdl_7", "OCTDL 6 diseases": "octdl_6dis",
        "Kermany 4-class": "kermany_4", "Kermany CNV/DME/DRUSEN": "kermany_3dis",
        "Kermany -> OCTDL (NORMAL/DME/AMD)": "kermany_to_octdl"}


def js(name):
    p = V6 / name
    return json.load(open(p)) if p.exists() else None


def ci(d):
    c = d.get("bacc_ci")
    return f"{d['bacc']:.3f} [{c[0]:.3f}, {c[1]:.3f}]" if c else f"{d['bacc']:.3f}"


def pr(p):
    return f"{p['diff_bacc']:+.3f} [{p['diff_ci'][0]:+.3f}, {p['diff_ci'][1]:+.3f}], p = {p['p_one_sided']:.3f}"


def main():
    out = []
    # ---- Step 1
    out += ["### Step 1: image-level POD", "",
            "| Task | Image-POD alone, best r (model) | v5 best | v5 best + image-POD r=50 | Paired Δ [95% CI], one-sided p |",
            "|---|---|---|---|---|"]
    for t in TASKS:
        d = js(f"image_pod_{SLUG[t]}.json")
        if not d:
            continue
        best = max(((m, r, v) for m in d["pod_sweep"] for r, v in d["pod_sweep"][m].items()), key=lambda x: x[2]["bacc"])
        m = d["models"]
        out.append(f"| {t} | {ci(best[2])} (r={best[1]}, {best[0]}) | {ci(m['v5 best (soft vote)'])} | "
                   f"{ci(m['v5 best + image-POD r=50'])} | {pr(d['paired_pod_vs_v5'])} |")
    out += ["", "*Source: `results/v6/image_pod_<task>.json`.*", ""]
    # ---- Step 2
    ra, rc = js("residual_auroc.json"), js("residual_clf.json")
    if ra:
        out += ["### Step 2: POD residual anomaly score", "",
                "| NORMAL vs ABNORMAL | AUROC r=10 | AUROC r=25 [95% CI] | AUROC r=50 |", "|---|---|---|---|"]
        for t, d in list(ra["auroc_within"].items()) + [(ra["auroc_cross"]["task"], ra["auroc_cross"])]:
            b = d["by_r"]
            out.append(f"| {t} | {b['10']['auroc']:.3f} | {b['25']['auroc']:.3f} [{b['25']['auroc_ci'][0]:.3f}, "
                       f"{b['25']['auroc_ci'][1]:.3f}] | {b['50']['auroc']:.3f} |")
        out += ["", "*Source: `results/v6/residual_auroc.json`.*", ""]
    if rc:
        out += ["| Task | v5 best + e(x) | Paired Δ vs v5 best |", "|---|---|---|"]
        for t in TASKS:
            if t in rc:
                out.append(f"| {t} | {ci(rc[t]['v5 best + e(x)'])} | {pr(rc[t]['paired_vs_v5'])} |")
        out += ["", "*Source: `results/v6/residual_clf.json`.*", ""]
    # ---- Step 3
    ab = js("ablation.json")
    SH = {"OCTID CSR/DR/MH (500x750)": "OCTID CSR/DR/MH", "OCTDL 7-class": "OCTDL 7-class", "OCTDL 6 diseases": "OCTDL 6 dis.",
          "Kermany 4-class": "Kermany 4-class", "Kermany CNV/DME/DRUSEN": "Kermany 3 dis.", "Kermany -> OCTDL (NORMAL/DME/AMD)": "Kermany → OCTDL"}
    rows3 = [("v5 best (soft vote)", "v5 best (soft vote)"), ("ResNet-18 embedding only", "ResNet-18 embedding only"),
             ("DINOv2-S embedding only", "DINOv2-S embedding only"), ("**hybrid ResNet-18 (headline)**", "hybrid ResNet-18"),
             ("hybrid DINOv2-S", "hybrid DINOv2-S"), ("CNN (fine-tuned ResNet-18)", "CNN")]
    out += ["### Step 3: hybrid (frozen deep features + physics) vs fine-tuned CNN", "",
            "Balanced accuracy [95% CI], mean of 3 seeds (CNN: seed-0 folds only). Embedding-only rows are the ablation.", "",
            "| Model | " + " | ".join(SH[t] for t in TASKS) + " |", "|---|" + "---|" * len(TASKS)]
    for lab, key in rows3:
        cells = []
        for t in TASKS:
            if key == "CNN":
                c = js(f"cnn_{SLUG[t]}.json"); cells.append(ci(c) if c else "n/a")
            elif key == "v5 best (soft vote)":
                d = js(f"image_pod_{SLUG[t]}.json"); cells.append(ci(d["models"][key]) if d else "n/a")
            else:
                cells.append(ci(js(f"hybrid_{SLUG[t]}.json")["models"][key]))
        out.append(f"| {lab} | " + " | ".join(cells) + " |")
    if ab:
        out += ["", "Paired differences on identical folds (seed 0), 95% patient-bootstrap CI and one-sided p "
                "(source `results/v6/ablation.json`):", "",
                "| Task | ResNet-18 hybrid − ResNet-18 embedding | ResNet-18 hybrid − v5 | ResNet-18 hybrid − CNN | "
                "DINOv2-S hybrid − DINOv2-S embedding | DINOv2-S hybrid − v5 |", "|---|---|---|---|---|---|"]
        for t in TASKS:
            p_ = ab["tasks"][t]["paired"]
            g_ = lambda k: pr(p_[k]) if k in p_ else "n/a"
            out.append(f"| {SH[t]} | {g_('hybrid ResNet-18 vs ResNet-18 embedding only')} | {g_('hybrid ResNet-18 vs v5 best (soft vote)')} | "
                       f"{g_('hybrid ResNet-18 vs CNN')} | {g_('hybrid DINOv2-S vs DINOv2-S embedding only')} | "
                       f"{g_('hybrid DINOv2-S vs v5 best (soft vote)')} |")
    hi = js("hybrid_importance.json")
    if hi:
        out += ["", f"Grouped permutation importance of the headline {hi['model']} hybrid, **measured on OCTID CSR/DR/MH only** "
                "(drop in balanced accuracy when a group is permuted in held-out folds). Groups are labelled by family: "
                "*physics* = hand-crafted v5 features, *ROM* = the v6 image-level reduced-order model, *deep* = the frozen "
                "embedding. Importance shows how much the fitted model **relies** on a group, not how much accuracy the group "
                "**adds**; the ablation above answers that.", "",
                "| Rank | Group | Importance |", "|---|---|---|"]
        out += [f"| {i + 1} | {k} | {hi['importance'][k]:.4f} |" for i, k in enumerate(hi["rank"])]
    out += ["", "*Source: `results/v6/hybrid_<task>.json`, `cnn_<task>.json`, `ablation.json`, `hybrid_importance.json` "
            "(DINOv2-S hybrid: `hybrid_importance_dinov2s.json`).*", ""]
    # ---- Step 4
    g = js("gradcam_octid_dis750.json")
    if g:
        out += ["### Step 4: Grad-CAM vs physics band importance (OCTID CSR/DR/MH)", "",
                "| Band | " + " | ".join(f"CAM share {k}" for k in g["classes"]) + " | " +
                " | ".join(f"physics recall drop {k}" for k in g["classes"]) + " |",
                "|---|" + "---|" * (2 * len(g["classes"]))]
        for b in g["bands"]:
            out.append(f"| {b} | " + " | ".join(f"{g['cam_mass_fraction'][b][k]:.3f}" for k in g["classes"]) + " | " +
                       " | ".join(f"{g['physics_band_importance'][b][k]:.3f}" for k in g["classes"]) + " |")
        out += ["", f"Spearman ρ over the {g['n_cells']} band × class cells: {g['spearman_rho']:.2f} (p = {g['spearman_p']:.2f}). "
                f"*Source: `results/v6/gradcam_octid_dis750.json`.*", ""]
    # ---- Step 5
    cr = js("cross.json")
    if cr:
        out += ["### Step 5: cross-scanner robustness (Kermany → OCTDL)", "",
                "| Model | Calibration | TTA | Balanced acc | Recall NORMAL / DME / AMD |", "|---|---|---|---|---|"]
        for k, d in cr["configs"].items():
            m, cal, tta = k.split(" | ")
            rc_ = d["recall_per_class"]
            sd = f" ± {d['bacc_sd']:.3f}" if "bacc_sd" in d else ""
            out.append(f"| {m} | {cal} | {tta} | {d['bacc']:.3f}{sd} | {rc_.get('NORMAL', float('nan')):.2f} / "
                       f"{rc_.get('DME', float('nan')):.2f} / {rc_.get('AMD', float('nan')):.2f} |")
        out += ["", "Paired tests: " + "; ".join(f"{k}: {pr(v)}" for k, v in cr["paired"].items()),
                "", "*Source: `results/v6/cross.json`.*", ""]
    # ---- Step 6
    cf = js("conformal.json")
    if cf:
        out += ["### Step 6: conformal triage on the headline v6 hybrid (ResNet-18)", "",
                "| Task | Target | Coverage per class | Mean set size (v5) | Singleton rate (v5) | Singleton accuracy (v5) |",
                "|---|---|---|---|---|---|"]
        for r in cf:
            v5 = r.get("v5", {})
            out.append(f"| {r['task']} | {r['target']:.2f} | " + ", ".join(f"{k} {v:.3f}" for k, v in r["coverage_per_class"].items())
                       + f" | {r['mean_set_size']:.2f} ({v5.get('mean_set_size', float('nan')):.2f}) | {r['singleton_rate']:.2f} "
                       f"({v5.get('singleton_rate', float('nan')):.2f}) | {r['singleton_accuracy']:.3f} ({v5.get('singleton_accuracy', float('nan')):.3f}) |")
        out += ["", "*Source: `results/v6/conformal.json` (ResNet-18 hybrid; DINOv2-S hybrid in `conformal_dinov2s.json`; v5 values copied from `results/v5/conformal.json`).*", ""]
    (V6 / "readme_tables.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))
    summary(ra, rc, cr, cf, g, hi)


def summary(ra, rc, cr, cf, g, hi):
    ip = js("image_pod_octid_dis750.json")
    h = js("hybrid_octid_dis750.json")
    c = js("cnn_octid_dis750.json")
    lines = ["# v6 summary: image-level ROM and the hybrid model", "",
             "Five findings (3b is the ablation behind finding 3). Every number is read from the JSON file named after it (all in `results/v6/`).", ""]
    if ip:
        best = max(max(v["bacc"] for v in ip["pod_sweep"][m].values()) for m in ip["pod_sweep"])
        lines.append(f"1. **Eigen-retinas carry less signal than the physics features.** On OCTID CSR/DR/MH, POD coefficients of the "
                     f"flattened image alone reach {best:.3f} balanced accuracy vs {ip['models']['v5 best (soft vote)']['bacc']:.3f} for v5; "
                     f"adding them to v5 changes it by {pr(ip['paired_pod_vs_v5'])} (`image_pod_octid_dis750.json`). "
                     f"No other task gains (`image_pod_<task>.json`).")
    if ra:
        k, o, x = (ra["auroc_within"]["Kermany 4-class"]["by_r"]["25"], ra["auroc_within"]["OCTDL 7-class"]["by_r"]["25"],
                   ra["auroc_cross"]["by_r"]["25"])
        lines.append(f"2. **A healthy-only POD flags disease without labels.** The residual e(x) (r = 25) separates NORMAL from "
                     f"ABNORMAL with AUROC {k['auroc']:.3f} on Kermany, {o['auroc']:.3f} on OCTDL and {x['auroc']:.3f} from Kermany-healthy "
                     f"to OCTDL (`residual_auroc.json`), and its residual maps localise the departure. As a feature it adds "
                     + (f"{pr(rc['OCTID CSR/DR/MH (500x750)']['paired_vs_v5'])} on OCTID (`residual_clf.json`)." if rc and "OCTID CSR/DR/MH (500x750)" in rc else "little (`residual_clf.json`)."))
    ab = js("ablation.json")
    if ab and c:
        T = ab["tasks"]
        SH = {"OCTID CSR/DR/MH (500x750)": "OCTID", "OCTDL 7-class": "OCTDL 7-class", "OCTDL 6 diseases": "OCTDL 6-disease",
              "Kermany 4-class": "Kermany 4-class", "Kermany CNV/DME/DRUSEN": "Kermany 3-disease",
              "Kermany -> OCTDL (NORMAL/DME/AMD)": "the new scanner"}
        sigv = lambda k: ([SH[t] for t in TASKS if T[t]["paired"][k]["diff_ci"][0] > 0],
                          [SH[t] for t in TASKS if not T[t]["paired"][k]["diff_ci"][0] > 0])
        s5, n5 = sigv("hybrid ResNet-18 vs v5 best (soft vote)")
        se, ne = sigv("hybrid ResNet-18 vs ResNet-18 embedding only")
        po = T["OCTID CSR/DR/MH (500x750)"]["paired"]
        lines.append(f"3. **The headline ResNet-18 hybrid narrows but does not close the gap to the CNN.** On OCTID CSR/DR/MH it "
                     f"reaches {ci(h['models']['hybrid ResNet-18'])} vs {ci(c)} for the fine-tuned ResNet-18 on the same folds; "
                     f"paired difference {pr(po['hybrid ResNet-18 vs CNN'])}; on Kermany 4-class "
                     f"{pr(T['Kermany 4-class']['paired']['hybrid ResNet-18 vs CNN'])} and on the new scanner "
                     f"{pr(T['Kermany -> OCTDL (NORMAL/DME/AMD)']['paired']['hybrid ResNet-18 vs CNN'])} (`ablation.json`). "
                     f"Against v5 it is significantly better on {', '.join(s5)} (CI excludes zero), and the CI includes zero on "
                     f"{', '.join(n5) or 'no task'}; OCTID: {pr(po['hybrid ResNet-18 vs v5 best (soft vote)'])}. The DINOv2-S hybrid, "
                     f"pre-registered before the results, is reported in every table; ResNet-18 became the headline afterwards.")
        lines.append(f"3b. **Ablation: physics adds accuracy on OCTDL and Kermany, not on OCTID.** Same frozen ResNet-18 with and "
                     f"without the physics + ROM features: significant gain on {', '.join(se)}; CI includes zero on {', '.join(ne)} "
                     f"(OCTID {pr(po['hybrid ResNet-18 vs ResNet-18 embedding only'])}, `ablation.json`)."
                     + (f" Permutation importance (OCTID only) ranks \"{hi['rank'][0]}\" first, but importance shows reliance, "
                        f"not added accuracy (`hybrid_importance.json`)." if hi else ""))
    if g:
        lines.append(f"4. **The CNN does not look where the physics looks.** Grad-CAM mass by anatomical band vs the physics model's "
                     f"per-band importance: Spearman ρ = {g['spearman_rho']:.2f} (p = {g['spearman_p']:.2f}) over {g['n_cells']} band × class "
                     f"cells (`gradcam_octid_dis750.json`); the CNN puts most attention in the choroid band.")
    if cr:
        b = cr["configs"][cr["best_config"]]
        lines.append(f"5. **A new scanner is still the hard case.** Kermany → OCTDL best configuration: {cr['best_config']} = "
                     f"{b['bacc']:.3f}, DME recall {b['recall_per_class']['DME']:.2f} (`cross.json`), far below the fine-tuned CNN's "
                     f"{js('cnn_kermany_to_octdl.json')['bacc']:.3f} (`cnn_kermany_to_octdl.json`). Healthy-reference calibration "
                     f"raises DME recall but costs AMD recall and lowers balanced accuracy."
                     + (f" Conformal triage on the ResNet-18 hybrid gives ABNORMAL coverage "
                        f"{[r for r in cf if r['task'].startswith('OCTDL triage') and r['alpha'] == 0.05][0]['coverage_per_class']['ABNORMAL']:.3f} "
                        f"for a 0.95 target on OCTDL, just under target (`conformal.json`)." if cf else ""))
    (V6 / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
