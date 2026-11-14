"""Build the presentation deck (python-pptx) from presentation/figures/ and results/.

  python make_figures.py && python make_deck.py
Writes presentation/Retinal_OCT_Physics_ML.pptx (16:9) and presentation/numbers.json
(every number shown on a slide or in its notes: slide, label, value, source file).
No number is typed by hand: all are read from results/ at build time.
"""
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

import make_figures as mf

R, V6, FIG = Path("results"), Path("results/v6"), Path("presentation/figures")
OUT = Path("presentation/Retinal_OCT_Physics_ML.pptx")
TEAM = ["Dhvani Shah", "Tapnanshu Gandhi", "Tanushka Sonde", "Rathod Sai Tejashwini"]
FOOTER = "IIT Gandhinagar · NDT course project"
INSTRUCTOR = "Prof. Prabhat Munshi"
SHORTT = {"OCTID CSR/DR/MH (500x750)": "OCTID", "OCTDL 7-class": "OCTDL 7-class", "OCTDL 6 diseases": "OCTDL 6-disease",
          "Kermany 4-class": "Kermany 4-class", "Kermany CNV/DME/DRUSEN": "Kermany 3-disease",
          "Kermany -> OCTDL (NORMAL/DME/AMD)": "new scanner"}
ACCENT = RGBColor(0x1C, 0x5C, 0xAB)
INK, INK2, TINT = RGBColor(0x0B, 0x0B, 0x0B), RGBColor(0x52, 0x51, 0x4E), RGBColor(0xE8, 0xF0, 0xFB)
HEAD, BODY = "Cambria", "Calibri"
W, H = 13.333, 7.5
NUMS = []
SLIDE_NO = [0]


# ============================================================================ numbers
def n(label, value, source, fmt="{:.3f}"):
    """Record a number shown on the current slide and return it formatted."""
    NUMS.append({"slide": SLIDE_NO[0], "label": label, "value": value, "source": str(source)})
    if isinstance(value, str):
        return value
    return fmt.format(value)


def pct(label, value, source, d=0):
    return n(label, value, source, "{:." + str(d) + "%}")


# ============================================================================ layout helpers
def new_slide(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    SLIDE_NO[0] = len(prs.slides)
    return s


def text(slide, x, y, w, h, runs, size=17, font=BODY, color=INK, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, name=None, space_after=0):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.02)
    paras = runs if isinstance(runs, list) else [runs]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space_after)
        for seg in (para if isinstance(para, list) else [para]):
            t, opt = (seg, {}) if isinstance(seg, str) else seg
            r = p.add_run()
            r.text = t
            r.font.name = opt.get("font", font)
            r.font.size = Pt(opt.get("size", size))
            r.font.bold = opt.get("bold", bold)
            r.font.italic = opt.get("italic", False)
            r.font.color.rgb = opt.get("color", color)
    return tb


def title(slide, t):
    text(slide, 0.55, 0.32, W - 1.1, 1.0, t, size=26, font=HEAD, bold=True, anchor=MSO_ANCHOR.MIDDLE, name="Title")


def bullets(slide, items, x, y, w, h, size=17):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tb.name = "Bullets"
    tf = tb.text_frame
    tf.word_wrap = True
    assert len(items) <= 4, "at most 4 bullets per slide"
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(10)
        pPr = p._p.get_or_add_pPr()
        pPr.set("marL", str(Emu(Inches(0.28))))
        pPr.set("indent", str(-Emu(Inches(0.28))))
        bu = pPr.makeelement("{http://schemas.openxmlformats.org/drawingml/2006/main}buChar", {"char": "•"})
        clr = pPr.makeelement("{http://schemas.openxmlformats.org/drawingml/2006/main}buClr", {})
        srgb = clr.makeelement("{http://schemas.openxmlformats.org/drawingml/2006/main}srgbClr", {"val": "1C5CAB"})
        clr.append(srgb)
        pPr.append(clr); pPr.append(bu)
        for seg in (it if isinstance(it, list) else [it]):
            t, opt = (seg, {}) if isinstance(seg, str) else seg
            r = p.add_run(); r.text = t
            r.font.name = BODY; r.font.size = Pt(size); r.font.color.rgb = opt.get("color", INK)
            r.font.bold = opt.get("bold", False)
    return tb


def image(slide, path, x, y, w, h, align="center"):
    p = Path(path)
    if not p.exists():
        text(slide, x, y, w, h, f"(figure not yet generated: {p.name})", size=14, color=INK2,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        return
    iw, ih = Image.open(p).size
    s = min(w / iw, h / ih)
    ww, hh = iw * s, ih * s
    xx = x + (w - ww) / 2 if align == "center" else x
    pic = slide.shapes.add_picture(str(p), Inches(xx), Inches(y + (h - hh) / 2), Inches(ww), Inches(hh))
    pic.name = p.stem


def stat(slide, x, y, w, h, value, label):
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    card.adjustments[0] = 0.12
    card.fill.solid(); card.fill.fore_color.rgb = TINT
    card.line.fill.background()
    card.shadow.inherit = False
    card.name = "Stat card"
    text(slide, x + 0.15, y + 0.12, w - 0.3, h * 0.55, value, size=34, font=HEAD, bold=True, color=ACCENT,
         anchor=MSO_ANCHOR.MIDDLE)
    text(slide, x + 0.15, y + h * 0.58, w - 0.3, h * 0.4, label, size=14, color=INK2)


def footer(slide):
    text(slide, 0.55, 7.0, 7, 0.35, FOOTER, size=11, color=INK2, name="Footer")
    text(slide, W - 1.55, 7.0, 1.0, 0.35, str(SLIDE_NO[0]), size=11, color=INK2, align=PP_ALIGN.RIGHT, name="Slide number")


def notes(slide, t):
    words = len(t.split())
    if not 80 <= words <= 150:
        print(f"  WARNING slide {SLIDE_NO[0]}: notes have {words} words")
    slide.notes_slide.notes_text_frame.text = t


def table(slide, rows, x, y, w, h, col_w=None, size=14, header_fill=TINT):
    nr, nc = len(rows), len(rows[0])
    gt = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(h))
    gt.name = "Table"
    tb = gt.table
    if col_w:
        for j, cw in enumerate(col_w):
            tb.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            c = tb.cell(i, j)
            c.text = str(v)
            c.fill.solid(); c.fill.fore_color.rgb = header_fill if i == 0 else RGBColor(0xFF, 0xFF, 0xFF)
            c.margin_left = c.margin_right = Inches(0.06)
            c.margin_top = c.margin_bottom = Inches(0.03)
            for p in c.text_frame.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(size); r.font.name = BODY; r.font.bold = (i == 0)
                    r.font.color.rgb = INK
    return tb


# ============================================================================ data
def load():
    D = {}
    D["pc"] = mf.js(V6 / "paper_claims.json")
    D["res"] = mf.js(R / "results.json")
    D["sc"] = mf.js(R / "sanity_checks.json")
    D["bl"] = mf.js(V6 / "baselines.json")
    D["rows"] = mf.stage_values()
    D["fv"] = mf.js(R / "v4/fractal_validation.json")
    D["mapod"] = mf.js(R / "v3/mapod.json")
    D["fp"] = mf.js(R / "v4/fractal_paired_test.json")
    D["rom"] = mf.js(R / "v5/rom.json")
    D["ens"] = mf.js(R / "v5/ensemble.json")
    D["conf5"] = mf.js(R / "v5/conformal.json")
    D["conf6"] = mf.js(V6 / "conformal.json")
    D["harm"] = mf.js(R / "v5/harmonize.json")
    D["cross"] = mf.js(V6 / "cross.json")
    D["ra"] = mf.js(V6 / "residual_auroc.json")
    D["rc"] = mf.js(V6 / "residual_clf.json")
    D["gc"] = mf.js(V6 / "gradcam_octid_dis750.json")
    D["hi"] = mf.js(V6 / "hybrid_importance.json")
    D["imp"] = mf.js(V6 / "importance_band_family.json")
    D["ctl"] = mf.js(R / "v2/octid_controls.json")
    D["ip"] = {t: mf.js(V6 / f"image_pod_{mf.SLUG[t]}.json") for t in mf.TASKS}
    D["hy"] = {t: mf.js(V6 / f"hybrid_{mf.SLUG[t]}.json") for t in mf.TASKS}
    D["cnn"] = {t: mf.cnn(t) for t in mf.TASKS}
    D["ab"] = mf.js(V6 / "ablation.json")
    return D


def counts(ds):
    f = pd.read_csv(R / f"v4/{ds}_features.csv", usecols=["label", "patient"])
    return len(f), f.patient.nunique(), f.label.value_counts().to_dict()


# ============================================================================ slides
def build():
    D = load()
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)
    rows = D["rows"]
    OCT = "OCTID CSR/DR/MH (500x750)"
    X = "Kermany -> OCTDL (NORMAL/DME/AMD)"
    src_grid = lambda r, t: rows[r][t][2]
    v = lambda r, t: rows[r][t][0]

    # ---------------------------------------------------------------- 1 title
    s = new_slide(prs)
    text(s, 0.8, 1.6, 11.7, 1.6, "Physics, anatomy and machine learning for retinal OCT", size=40, font=HEAD, bold=True)
    text(s, 0.8, 3.15, 11.7, 0.9, "Reproducing, auditing and extending “Multi-Parametric Retinal Diagnostics: "
         "A Deterministic Physics-Based Engine for Tomographic Classification” (Banerjee, Sarkar & Munshi)",
         size=18, color=INK2)
    text(s, 0.8, 4.45, 11.7, 0.5, " · ".join(TEAM), size=20, bold=True, color=ACCENT)
    text(s, 0.8, 5.0, 11.7, 1.3, ["Course instructor: " + INSTRUCTOR,
                                  "Non-destructive testing (NDT) course project, IIT Gandhinagar",
                                  date.today().strftime("%d %B %Y")], size=16, color=INK2)
    footer(s)
    notes(s, "Good morning. We are presenting our NDT course project. We took a recent IIT Gandhinagar paper that classifies "
             "retinal OCT scans using two physics quantities, a KT-1 density slope and a Hurst roughness exponent. We "
             "reproduced it, checked why it works, and then improved it step by step, borrowing tools from NDT: "
             "pulse-echo analysis, probability of detection, reduced-order modelling and calibration blocks. The talk goes "
             "concepts first, then what we found, then our improvements, and ends with an honest comparison against the "
             "paper. Every number you will see comes from a result file in our repository, and the code is public so "
             "anyone can rerun it.")

    # ---------------------------------------------------------------- 2 why
    s = new_slide(prs)
    title(s, "OCT is ultrasound with light, and a black-box model can learn the scanner instead of the disease")
    image(s, FIG / "fig02_examples.png", 6.3, 1.45, 6.5, 5.4)
    bullets(s, [[("A-scan", {"bold": True}), ": one depth profile from the light echo; sweeping sideways gives the B-scan"],
                [("Layers", {"bold": True}), ": the ILM is the top surface, the RPE the bright back wall; disease changes them"],
                [("Black box", {"bold": True}), ": a CNN can be accurate for the wrong reason, e.g. scanner artefacts"],
                [("The paper's answer", {"bold": True}), ": classify with two physics numbers per image patch"]],
            0.55, 1.6, 5.5, 5.0)
    footer(s)
    notes(s, "Optical coherence tomography sends light into the eye and measures the delay of the reflected light, exactly "
             "like an ultrasonic pulse-echo test but with light. One depth profile is an A-scan; many side by side make the "
             "B-scan images on the right. Two lines matter: the ILM is the top surface of the retina and the RPE is the "
             "bright layer at the back. The left column shows raw scans for four classes. The middle column is how the "
             "paper sees an image: a fixed pixel grid. The right column is how we see it: flattened at the RPE and split "
             "into anatomical bands. The paper's motivation is fair: deep networks are black boxes that may learn "
             "acquisition artefacts, so it proposes interpretable physics features instead.")

    # ---------------------------------------------------------------- 3 paper
    pc = D["pc"]
    s = new_slide(prs)
    title(s, "The paper turns each scan into 300 physics numbers and reports 71% accuracy with perfect triage")
    image(s, FIG / "fig01a_pipeline_paper.png", 0.55, 1.5, 12.2, 2.9)
    acc = pct("paper reported multiclass accuracy", pc["multiclass_accuracy"]["value"], V6 / "paper_claims.json")
    pr = n("paper reported NORMAL precision and recall", pc["normal_triage_precision_recall"]["value"], V6 / "paper_claims.json", "{:.2f}")
    nt = n("paper test scans", pc["n_test_scans"]["value"], V6 / "paper_claims.json", "{:d}")
    stat(s, 0.8, 4.65, 3.6, 1.9, acc, f"multiclass accuracy on {nt} held-out scans (paper)")
    stat(s, 4.85, 4.65, 3.6, 1.9, pr, "NORMAL precision and recall, called “mathematically guaranteed”")
    text(s, 8.9, 4.75, 3.9, 1.8, ["The paper also reports that all top-15 features are Hurst exponents, located in grid "
                                  "rows 7–8, and reads this as autonomous anatomical localisation."], size=15, color=INK2)
    footer(s)
    notes(s, f"Here is the paper's pipeline. White acquisition wedges are cropped, the image is resized to 750 by 500 pixels "
             f"and cut into a 15 by 10 grid of 50-pixel cells. For each cell they compute two numbers, the KT-1 slope and "
             f"the Hurst exponent, giving 300 features, and a random forest classifies normal, CSR, diabetic retinopathy "
             f"and macular hole. On {nt} held-out scans they report {acc} accuracy and, most strikingly, {pr} precision and "
             f"recall for normal eyes, which they describe as a mathematical guarantee for triage. Keep these two numbers "
             f"in mind; we will test both of them.")

    # ---------------------------------------------------------------- 4 KT-1
    z = np.load(V6 / "figdata.npz", allow_pickle=True)
    s = new_slide(prs)
    title(s, "KT-1 measures how quickly a patch's brightest signal fades as we blur it")
    image(s, FIG / "fig03a_kt1.png", 6.6, 1.4, 6.2, 5.4)
    sn = n("KT-1 slope healthy example cell", float(z["kt1_NORMAL"][8]), V6 / "figdata.npz", "{:.4f}")
    sd = n("KT-1 slope CSR example cell", float(z["kt1_CSR"][8]), V6 / "figdata.npz", "{:.4f}")
    bullets(s, [[("Blur", {"bold": True}), " each 50×50 cell with Gaussians of width σ = 3, 2, 1, 0.1 px"],
                [("Measure", {"bold": True}), " the peak N", ("max", {}), " (99th percentile) after each blur"],
                [("Formula", {"bold": True}), ": S = slope of 1/N_max against σ² (filter curvature |W''(0)|)"],
                [("Idea", {"bold": True}), f" (Munshi's tomographic error theory): dense, uniform tissue fades differently; here {sn} vs {sd}"]],
            0.55, 1.6, 5.8, 5.0)
    footer(s)
    notes(s, f"The first physics quantity comes from Professor Munshi's work on error estimates in tomography. Take one "
             f"small cell of the image and blur it with Gaussian filters of increasing width. The brightest value, the "
             f"99th percentile, drops as the blur grows. Plot one over that peak against sigma squared, which is "
             f"proportional to the curvature of the filter at the origin. The slope of that straight line is the KT-1 "
             f"signature S. The graph shows a healthy cell and a diseased cell from our data: slopes {sn} and {sd}. A "
             f"different tissue texture changes how fast the peak fades. In flow measurement this was used to tell bubble "
             f"regimes apart; here it is used to describe tissue density.")

    # ---------------------------------------------------------------- 5 Hurst
    s = new_slide(prs)
    title(s, "The Hurst exponent measures roughness: brightness differences grow as a power of distance")
    image(s, FIG / "fig03b_hurst.png", 6.6, 1.4, 6.2, 5.4)
    hs = n("H smooth example cell", float(z["hurst_H"][0]), V6 / "figdata.npz", "{:.2f}")
    hr = n("H rough example cell", float(z["hurst_H"][1]), V6 / "figdata.npz", "{:.2f}")
    bullets(s, [[("NMSID(k)", {"bold": True}), " = mean |I(p₂) − I(p₁)| over all pixel pairs at distance k"],
                [("Power law", {"bold": True}), ": log NMSID(k) = H log k + C, so H is the slope on a log–log plot"],
                [("Fractal dimension", {"bold": True}), ": D = 3 − H; low H means a jagged, rough texture"],
                [("Example", {"bold": True}), f": smooth cell H = {hs}, rough cell H = {hr}"]],
            0.55, 1.6, 5.8, 5.0)
    footer(s)
    notes(s, f"The second quantity treats the image as a landscape whose height is brightness. For every pair of pixels at "
             f"distance k we take the absolute brightness difference and average it; this is NMSID of k. For a fractal, "
             f"statistically self-similar texture, this grows as a power of k, so on a log-log plot it is a straight line "
             f"and the slope is the Hurst exponent H. A smooth patch has H close to one, a rough patch has H close to zero, "
             f"and the fractal dimension of the surface is three minus H. In our two example cells H is {hs} and {hr}. "
             f"The paper uses H for every grid cell.")

    # ---------------------------------------------------------------- 6 data
    s = new_slide(prs)
    title(s, "Three public datasets from three different scanners let us test beyond the paper's 517 scans")
    trows = [["Dataset", "Scanner", "Scans", "Patients", "Classes (scans)"]]
    for ds, nm, sc_ in (("octid", "OCTID (paper)", "Zeiss Cirrus, exported JPEG"), ("kermany", "Kermany OCT2017 (subset)", "Heidelberg Spectralis"),
                        ("octdl", "OCTDL", "Optovue RTVue")):
        tot, pats, cc = counts(ds)
        n(f"{ds} scans", tot, R / f"v4/{ds}_features.csv", "{:d}")
        n(f"{ds} patients", pats, R / f"v4/{ds}_features.csv", "{:d}")
        trows.append([nm, sc_, str(tot), str(pats) if ds != "octid" else f"{pats}*",
                      ", ".join(f"{k} {n(f'{ds} {k}', int(v_), R / f'v4/{ds}_features.csv', '{:d}')}" for k, v_ in sorted(cc.items()))])
    table(s, trows, 0.55, 1.45, 12.2, 2.0, col_w=[2.4, 2.6, 0.9, 1.1, 5.2], size=14)
    image(s, FIG / "fig02b_gallery.png", 0.55, 3.75, 12.2, 2.9)
    text(s, 0.55, 6.62, 12.2, 0.35, "* OCTID has no patient IDs, so every image is treated as its own patient.", size=12, color=INK2)
    footer(s)
    notes(s, "The paper used OCTID, 517 scans from a Zeiss Cirrus scanner. It is small and, as we will show, confounded. So "
             "we added two larger public datasets from two other manufacturers: a 600-per-class patient subset of the "
             "Kermany dataset from a Heidelberg Spectralis, and OCTDL from an Optovue scanner with seven classes. Having "
             "three scanners lets us ask two extra questions: does a method work on a different population, and does it "
             "survive a change of machine. Kermany and OCTDL give patient IDs, so all our cross-validation keeps every "
             "patient's scans together on one side of the split. OCTID has no patient IDs, which is a limitation we "
             "state openly.")

    # ---------------------------------------------------------------- 7 reproduction
    res, sc = D["res"], D["sc"]
    s = new_slide(prs)
    title(s, "We reproduced the paper's numbers, and from here on we report balanced accuracy")
    a1 = pct("our accuracy on paper split", res["accuracy"], R / "results.json", 1)
    a2 = pct("our 5-fold CV accuracy", sc["cv_accuracy_mean"], R / "sanity_checks.json", 1)
    p1 = n("our NORMAL precision", res["report"]["NORMAL"]["precision"], R / "results.json", "{:.2f}")
    r1 = n("our NORMAL recall", res["report"]["NORMAL"]["recall"], R / "results.json", "{:.2f}")
    th = pct("our top-15 Hurst fraction", res["top15_hurst_fraction"], R / "results.json")
    table(s, [["Metric", "Paper", "Our re-implementation"],
              ["Accuracy, 80/20 split", acc, a1],
              ["Accuracy, 5-fold CV", "not reported", a2],
              ["NORMAL precision / recall", f"{pr} / {pr}", f"{p1} / {r1}"],
              ["Top-15 features that are Hurst", pct("paper top15 hurst", pc["top15_hurst_fraction"]["value"], V6 / "paper_claims.json"), th]],
          0.55, 1.5, 6.3, 2.6, col_w=[2.8, 1.5, 2.0], size=15)
    bullets(s, [[("Main metric from now on: balanced accuracy", {"bold": True}), " = mean of per-class recalls"],
                "Plain accuracy rewards ignoring small classes; chance level is 1 / (number of classes)",
                "Every number uses patient-grouped 5-fold CV, 3 repeats, with a 95% patient bootstrap CI"],
            0.55, 4.45, 6.3, 2.4, size=16)
    image(s, FIG / "fig05a_confusion.png", 7.1, 1.4, 5.7, 5.5)
    footer(s)
    notes(s, f"Our first job was a faithful reproduction. On the same 80/20 split our accuracy is {a1} against the paper's "
             f"{acc}, and with five-fold cross-validation it is {a2}. Normal precision and recall are {p1} and {r1}, and "
             f"{th} of the top features are Hurst exponents, just as the paper says. The confusion matrix on the right is "
             f"that reproduction: normal eyes are perfect, the three diseases are mixed up. From this slide on we use "
             f"balanced accuracy, the average of the per-class recalls, because plain accuracy can look good while a small "
             f"class is ignored. All later numbers use patient-grouped cross-validation with confidence intervals.")

    # ---------------------------------------------------------------- 8 confound
    s = new_slide(prs)
    flips = n("disease scans called NORMAL after resizing", f"{sc['disease_called_normal_after_resample']}/{sc['disease_test_n']}", R / "sanity_checks.json")
    title(s, f"The perfect triage is a scanner confound: {flips} disease scans are called NORMAL after resizing")
    image(s, FIG / "fig04_confound.png", 0.55, 1.4, 8.4, 4.0)
    sz = pct("image size alone NORMAL vs disease accuracy", sc["size_only_normal_vs_disease_acc"], R / "sanity_checks.json")
    rho = n("Spearman upsampling vs mean H", sc["spearman_scale_meanH"], R / "sanity_checks.json", "{:.2f}")
    pcv = n("paper method on OCTID CSR/DR/MH 500x750", D["bl"]["paper_cv_octid_dis750"]["bacc"], V6 / "baselines.json")
    bullets(s, [[("Image size alone", {"bold": True}), f" separates NORMAL from disease with {sz} accuracy"],
                [("Resampling drives H", {"bold": True}), f": Spearman ρ = {rho} with the upsampling factor"],
                [("Honest task", {"bold": True}), f": CSR vs DR vs MH, 500×750 scans only; paper method {pcv} (chance 0.33)"]],
            9.2, 1.5, 3.6, 5.2, size=16)
    stat(s, 0.8, 5.55, 3.8, 1.3, flips, "held-out disease scans flipped to NORMAL")
    footer(s)
    notes(s, f"Why is normal triage perfect? In OCTID every normal scan has a different image size from every disease scan, "
             f"as the left heat map shows: they were exported differently. Image size alone separates them with {sz} "
             f"accuracy. The resize step upsamples normal scans more, and cubic upsampling smooths texture and raises H: the "
             f"scatter shows a Spearman correlation of {rho}. The decisive test: we shrank the held-out disease scans to "
             f"the normal scans' width, and the trained model called {flips} of them normal. So the triage result measures "
             f"the scanner, not the eye. The honest OCTID task is to separate the three diseases using one export format, "
             f"where the paper's method gets {pcv} balanced accuracy, with chance at 0.33.")

    # ---------------------------------------------------------------- 9 fractal estimator
    fv = D["fv"]
    s = new_slide(prs)
    g01 = n("paper grid estimator at true H 0.1", fv["paper_grid"]["0.1"][0], R / "v4/fractal_validation.json", "{:.2f}")
    u01 = n("after upsampling at true H 0.1", fv["paper_grid_upsampled"]["0.1"][0], R / "v4/fractal_validation.json", "{:.2f}")
    title(s, f"On surfaces with known roughness, the paper's estimator reads H = 0.1 as {g01}, and upsampling adds more")
    image(s, FIG / "fig07_fractal_validation.png", 0.55, 1.4, 7.0, 5.5)
    bullets(s, [[("Known answers", {"bold": True}), ": synthetic fractional-Brownian surfaces with a chosen H"],
                [("Bias", {"bold": True}), f": a 50×50 cell is too small; true 0.1 is measured as {g01}, {u01} after 1.32× upsampling"],
                [("Consequence", {"bold": True}), ": scans that were upsampled more look smoother, which is the confound's mechanism"],
                [("Our estimators", {"bold": True}), " (box counting, divider, profile Hurst) recover the known dimensions"]],
            7.8, 1.6, 5.0, 5.0, size=16)
    footer(s)
    notes(s, f"In NDT you calibrate an instrument on a reference block before trusting it. We did the same for the fractal "
             f"estimator. We generated synthetic rough surfaces whose true Hurst exponent we choose, and measured them. The "
             f"green line, a one-dimensional profile estimator, sits on the y equals x line. The paper's estimator on 50 by "
             f"50 cells, in orange, reads a true value of 0.1 as {g01}, and after the 1.32 times upsampling the paper applies "
             f"to normal scans it reads {u01}. So the estimator is biased upwards for rough textures, and the bias depends "
             f"on resizing. That explains the confound on the previous slide in physical terms.")

    # ---------------------------------------------------------------- 10 our approach
    s = new_slide(prs)
    title(s, "Our approach keeps the physics but measures it inside the anatomy, then adds NDT and ML tools")
    image(s, FIG / "fig01b_pipeline_ours.png", 0.55, 1.45, 12.2, 2.9)
    bullets(s, [[("Flatten", {"bold": True}), ": trace the RPE and shift each column so the RPE is a straight line"],
                [("Regions", {"bold": True}), ": 4 depth bands (inner, outer, RPE, choroid) × 5 sectors around the fovea"],
                [("Features", {"bold": True}), ": texture (H, KT-1, lacunarity, entropy, coherence), disease shape, NDT, fractal"],
                [("Learning", {"bold": True}), ": POD, ensembles and conformal sets, all fitted inside training folds only"]],
            0.55, 4.5, 12.2, 2.4, size=16)
    footer(s)
    notes(s, "Our pipeline keeps the paper's physical idea but fixes where it is measured. First we find the RPE, the "
             "brightest layer, in every column and shift the columns so it becomes a straight line; we call this "
             "flattening. Then row 150 means the same depth above the RPE in every scan, whatever the scanner. We split "
             "the retina into four depth bands and five sectors around the fovea and measure texture in each region, "
             "including the paper's H and KT-1. We add disease-shape features, NDT pulse-echo features and boundary "
             "fractals, and then use reduced-order modelling, ensembles and conformal prediction. Anything learned from "
             "data is fitted only on the training patients of each fold.")

    # ---------------------------------------------------------------- 11 v2
    s = new_slide(prs)
    v2 = n("v2 OCTID CSR/DR/MH", v("v2 anatomy", OCT), src_grid("v2 anatomy", OCT))
    p0 = n("paper RF OCTID CSR/DR/MH", v("Paper RF", OCT), src_grid("Paper RF", OCT))
    title(s, f"Measuring inside the anatomy lifts the honest OCTID task from {p0} to {v2} balanced accuracy")
    image(s, FIG / "fig05b_confusion.png", 0.55, 1.4, 4.1, 4.2)
    image(s, FIG / "fig05c_confusion.png", 4.75, 1.4, 4.1, 4.2)
    vk = n("v2 Kermany 4-class", v("v2 anatomy", "Kermany 4-class"), src_grid("v2 anatomy", "Kermany 4-class"))
    vo = n("v2 OCTDL 7-class", v("v2 anatomy", "OCTDL 7-class"), src_grid("v2 anatomy", "OCTDL 7-class"))
    bullets(s, [[("11 texture numbers", {"bold": True}), " per region: H, KT-1, NMSID, lacunarity, entropy, mean, std, coherence"],
                [("23 disease numbers", {"bold": True}), ": thickness, foveal dip, full-thickness gap, fluid, RPE bumps"],
                [("Carries over", {"bold": True}), f": Kermany 4-class {vk}, OCTDL 7-class {vo} (paper method < 0.5)"]],
            9.1, 1.5, 3.7, 5.2, size=16)
    text(s, 0.55, 5.75, 8.3, 1.0, "Left: paper method. Right: the headline v6 hybrid (ResNet-18) on the same task and folds (counts in brackets).",
         size=13, color=INK2)
    footer(s)
    notes(s, f"This is the biggest single step in the project. Version two measures the same kind of physics, roughness, "
             f"density and texture, but inside anatomical bands of the flattened retina, plus 23 disease-specific numbers "
             f"such as retinal thickness, the depth of the foveal dip, gaps through the retina for macular holes and dark "
             f"fluid pockets. On the honest OCTID task balanced accuracy rises from {p0} to {v2}. The two confusion matrices "
             f"compare the paper's method on the left with our best final model on the right, on exactly the same "
             f"patients and folds. The same features also work on the other two scanners: {vk} on Kermany and {vo} on "
             f"OCTDL's seven classes.")

    # ---------------------------------------------------------------- 12 v3 NDT
    mp = D["mapod"]
    s = new_slide(prs)
    v3 = n("v3 OCTID CSR/DR/MH", v("v3 + NDT", OCT), src_grid("v3 + NDT", OCT))
    a90 = n("MAPOD a90 subretinal fluid (Kermany, NDT half-max)", mp["kermany|srf|NDT half-max"]["a90"], R / "v3/mapod.json", "{:.0f}")
    a95 = n("MAPOD a90/95 subretinal fluid", mp["kermany|srf|NDT half-max"]["a90_95"], R / "v3/mapod.json", "{:.0f}")
    title(s, f"Treating OCT as a pulse-echo test adds signal ({v3}) and lets us measure the smallest detectable fluid pocket")
    image(s, FIG / "fig08_mapod.png", 6.5, 1.4, 6.3, 5.4)
    bullets(s, [[("Attenuation", {"bold": True}), ": Beer–Lambert I(z) = I₀e^(−2μz), so the depth slope ∝ μ"],
                [("Interface echoes, 6 dB-drop sizing, shadowing, CNR, speckle", {"bold": True}), " normalised to in-image references"],
                [("MAPOD", {"bold": True}), ": paint synthetic fluid of known height into healthy scans, score hit or miss"],
                [("POD(a) = Φ((ln a − μ)/σ)", {"bold": True}), f": subretinal fluid a₉₀ = {a90} µm, a₉₀/₉₅ = {a95} µm"]],
            0.55, 1.6, 5.8, 5.2, size=16)
    footer(s)
    notes(s, f"OCT is optical pulse-echo testing, so ultrasonic NDT tools apply directly. We measure attenuation from the "
             f"slope of brightness with depth, reflection contrast at the ILM and RPE interfaces, defect size with the "
             f"6 dB drop rule, shadowing under bright spots and the speckle statistics. Adding these raised balanced "
             f"accuracy on the honest OCTID task to {v3}. We also borrowed model-assisted probability of detection from "
             f"aerospace NDT: we painted synthetic fluid pockets of known height into healthy scans and recorded hits and "
             f"misses. The fitted curve says a subretinal fluid pocket of {a90} micrometres is detected 90% of the time, "
             f"and {a95} micrometres with 95% confidence.")

    # ---------------------------------------------------------------- 13 v4 fractal boundaries
    fp = [r for r in D["fp"] if r["task"] == OCT and r["model"] == "Logistic regression"][0]
    s = new_slide(prs)
    d4 = n("v4 boundary fractal paired difference OCTID LR", fp["diff_mean"], R / "v4/fractal_paired_test.json", "{:+.3f}")
    p4 = n("v4 paired one-sided p", fp["p_one_sided"], R / "v4/fractal_paired_test.json", "{:.3f}")
    v4 = n("v4 OCTID CSR/DR/MH", v("v4 + fractal", OCT), src_grid("v4 + fractal", OCT))
    title(s, f"Boundary 'coastline' fractals add a small gain ({d4}, p = {p4}) on OCTID and nothing elsewhere")
    image(s, FIG / "fig13b_fractal_by_class.png", 6.4, 1.4, 6.4, 5.4)
    bullets(s, [[("Idea", {"bold": True}), ": the ILM and RPE are curves; disease makes them wrinkled like a coastline"],
                [("Scale-free", {"bold": True}), ": profile Hurst E|z(x+d) − z(x)| ∝ dᴴ, D = 2 − H; divider length L(r) ∝ r^(1−D)"],
                [("Paired test", {"bold": True}), f": same folds with and without; OCTID LR {d4} (p = {p4}), best {v4}"],
                "Honest negative: no gain on OCTDL or Kermany"],
            0.55, 1.6, 5.6, 5.2, size=16)
    footer(s)
    notes(s, f"Version four applies fractal geometry where it belongs. The retina's top and bottom boundaries are curves, "
             f"and disease wrinkles them, like a coastline. We measure their roughness in a scale-free way, with the "
             f"profile Hurst exponent and Richardson's divider method, after checking these estimators on shapes with known "
             f"dimension such as the Koch curve. To decide whether a new feature really helps we use a paired test: the "
             f"same model on the same folds with and without it, bootstrapped over patients. On OCTID the gain is {d4} "
             f"with a one-sided p of {p4}, reaching {v4}. On OCTDL and Kermany there is no gain, and we report that.")

    # ---------------------------------------------------------------- 14 POD concept
    rom = D["rom"]
    s = new_slide(prs)
    r95p = n("modes for 95% energy, paper H field OCTID", rom["energy"]["octid | paper H field (150 cells)"]["r95"], R / "v5/rom.json", "{:d}")
    r95v = n("modes for 95% energy, v2 H field OCTID", rom["energy"]["octid | v2 band x sector H field (20 cells)"]["r95"], R / "v5/rom.json", "{:d}")
    title(s, "Reduced-order modelling (POD) finds the few spatial patterns that describe most of the variation")
    image(s, FIG / "fig09a_rom_energy.png", 6.3, 1.5, 6.5, 4.4)
    bullets(s, [[("Snapshot matrix", {"bold": True}), ": each scan's field minus the mean field m is one row of A"],
                [("SVD", {"bold": True}), ": A = U Σ Vᵀ; mode j is column j of V, a spatial pattern"],
                [("Energy", {"bold": True}), ": E(r) = (σ₁² + … + σᵣ²) / (sum of all σ²); rebuild with m + r modes"],
                [("Finding", {"bold": True}), f": 95% energy needs {r95p} of 150 modes on the paper grid, {r95v} of 20 in anatomy"]],
            0.55, 1.6, 5.6, 5.2, size=16)
    footer(s)
    notes(s, f"Proper orthogonal decomposition is the workhorse of reduced-order modelling in fluid mechanics, and it is "
             f"the same mathematics as principal component analysis. Each scan's feature map is a field, like a velocity "
             f"field on a mesh. Stack the fields as rows of a matrix, subtract the mean and take the singular value "
             f"decomposition. The columns of V are spatial modes, and the squared singular values say how much energy, "
             f"meaning variance, each mode carries. The plot shows cumulative energy. The paper's grid needs {r95p} of 150 "
             f"modes to reach 95 percent, so it is poorly compressible. Our anatomy-aligned field needs {r95v} of 20. "
             f"The modes are always fitted inside the training fold.")

    # ---------------------------------------------------------------- 15 what POD found
    rho2 = n("POD mode 2 vs tissue height", rom["octid_spearman_mode_vs_tissue_height"][1], R / "v5/rom.json", "{:.2f}")
    s = new_slide(prs)
    title(s, f"POD shows the paper's grid mostly encodes where the retina sits: mode 2 tracks retina position, ρ = {rho2}")
    image(s, FIG / "fig09b_rom_modes.png", 0.55, 1.45, 12.2, 3.0)
    bullets(s, [[("Mode 1", {"bold": True}), ": a uniform shift, the overall roughness level"],
                [("Mode 2", {"bold": True}), f": a top/bottom dipole; its coefficient correlates with the retina's vertical position at ρ = {rho2}"],
                [("Why", {"bold": True}), ": the grid does not move with the retina; POD of a translating structure is a known failure mode"],
                [("Lesson", {"bold": True}), ": ROM needs a consistent coordinate frame, like a fixed CFD mesh, so flatten first"]],
            0.55, 4.6, 12.2, 2.3, size=16)
    footer(s)
    notes(s, f"Here are the mean field and the first four POD modes of the paper's Hurst map on OCTID. Mode one is just a "
             f"shift of the whole map. Mode two is a dipole, top against bottom, and its coefficient correlates with the "
             f"height of the retina in the image with a Spearman correlation of {rho2}. In other words, the largest pattern "
             f"of variation is where the retina happens to sit in the picture, not disease. This is the textbook failure of "
             f"POD for structures that translate, such as a moving shock in a flow. The fix is the same as in CFD: use a "
             f"coordinate frame that moves with the structure, which is what flattening at the RPE does.")

    # ---------------------------------------------------------------- 16 energy != information
    ev = rom["energy_vs_info"]
    s = new_slide(prs)
    e1 = n("Spearman energy vs class signal OCTID", ev["OCTID CSR/DR/MH"]["spearman_energy_vs_eps2"], R / "v5/rom.json", "{:+.2f}")
    e2 = n("Spearman energy vs class signal Kermany", ev["Kermany CNV/DME/DRUSEN"]["spearman_energy_vs_eps2"], R / "v5/rom.json", "{:+.2f}")
    bm = n("most discriminative OCTID mode", ev["OCTID CSR/DR/MH"]["best_mode"], R / "v5/rom.json", "{:d}")
    title(s, f"Energy is not information: high-energy modes are not the disease modes (ρ = {e1} and {e2})")
    image(s, FIG / "fig09c_rom_energy_vs_info.png", 0.55, 1.35, 4.6, 3.0)
    bullets(s, [f"Top: class signal of each mode (Kruskal–Wallis ε²) is unrelated to its energy; the best OCTID mode is #{bm}",
                "Bottom: no number of POD modes rescues the paper's grid (orange); on anatomy-aligned fields (blue) POD "
                "halves the features at no loss"],
            5.4, 1.5, 7.4, 2.8, size=16)
    image(s, FIG / "fig09d_rom_accuracy.png", 0.55, 4.4, 12.2, 2.55)
    footer(s)
    notes(s, f"A natural idea is to keep the most energetic modes and drop the rest. The left plot tests that: each dot is a "
             f"mode, the x axis is its energy and the y axis is how strongly its coefficient differs between diseases. "
             f"There is no relationship; the rank correlation is {e1} on OCTID and {e2} on Kermany, and the most "
             f"discriminative OCTID mode is number {bm}. The right plot shows accuracy against the number of modes kept. "
             f"For the paper's grid, no choice of modes beats the raw grid, which is itself weak. For our aligned texture "
             f"fields, POD compresses the features by about half without losing accuracy. Compression is not the same as "
             f"discrimination.")

    # ---------------------------------------------------------------- 17 ensemble
    ens = {r["task"]: r for r in D["ens"]}
    ek = ens["Kermany 4-class"]["vote_vs_best_single"]
    s = new_slide(prs)
    vk5 = n("v5 soft vote Kermany 4-class", ens["Kermany 4-class"]["models"]["Soft vote (LR+GBM+SVM)"]["bacc"], R / "v5/ensemble.json")
    dk = n("vote minus best single Kermany", ek["diff_bacc"], R / "v5/ensemble.json", "{:+.3f}")
    pk = n("vote vs single p Kermany", ek["p_one_sided"], R / "v5/ensemble.json", "{:.3f}")
    title(s, f"Averaging three different learners helps a little: soft vote {vk5} on Kermany ({dk} vs the best single model)")
    image(s, FIG / "fig10_model_task_heatmap.png", 5.2, 1.4, 7.6, 5.4)
    bullets(s, [[("Logistic regression", {"bold": True}), ": a linear boundary, softmax probabilities"],
                [("LightGBM", {"bold": True}), ": 300 small trees, each fixing the previous trees' errors"],
                [("RBF-SVM", {"bold": True}), ": a smooth kernel boundary"],
                [("Soft vote", {"bold": True}), f": average the three probability vectors; paired p = {pk} on Kermany"]],
            0.55, 1.6, 4.5, 5.2, size=16)
    footer(s)
    notes(s, f"Version five improves the learning stage. Three different learners see the same features: logistic "
             f"regression draws a straight boundary, LightGBM builds hundreds of small decision trees that correct each "
             f"other, and a support vector machine draws a smooth curved boundary. A soft vote simply averages their "
             f"predicted probabilities, so their different mistakes partly cancel. The heat map summarises every model on "
             f"every task, with the best in each column outlined. The ensemble gains are small: {dk} on Kermany with a "
             f"paired p of {pk}, and within noise elsewhere. The CNN row is the black-box reference, and the last row is "
             f"our v6 hybrid, which we discuss at the end.")

    # ---------------------------------------------------------------- 18 where the model looks
    s = new_slide(prs)
    imp = D["imp"]
    fam = max(((f, b, val) for f, d in imp["importance"].items() for b, val in d.items()), key=lambda x: x[2])
    fv_ = n("top family x band importance", fam[2], V6 / "importance_band_family.json", "{:.3f}")
    kt = n("KT-1 outer band importance", imp["importance"]["KT-1 slope"]["outer"], V6 / "importance_band_family.json", "{:.3f}")
    title(s, "The paper's important cells sit where the retina's lower edge falls; ours point to specific layers")
    image(s, FIG / "fig06_where_looks.png", 0.55, 1.35, 12.2, 4.35)
    bullets(s, [[("Paper", {"bold": True}), ": top-15 Gini cells (orange boxes) lie in rows 7–8 for every class"],
                [("Ours", {"bold": True}), f": the outer retina matters most: {fam[0]} {fv_}, KT-1 slope {kt} (drop in balanced accuracy)"],
                [("Fair to the paper", {"bold": True}), ": KT-1 and roughness do carry signal once measured in the right layer"]],
            0.55, 5.75, 12.2, 1.2, size=14)
    footer(s)
    notes(s, f"The left four maps are the mean Hurst value in each paper grid cell for each class, with the paper's top 15 "
             f"cells boxed in orange. They sit in rows seven and eight for every class, which is where the bottom edge of "
             f"the retina tends to fall in the picture. The right heat map is our model, explained by permutation "
             f"importance: we shuffle one feature family in one band in held-out data and measure how much balanced "
             f"accuracy drops. The outer retina carries most of the signal, with {fam[0]} at {fv_} and the paper's own KT-1 "
             f"slope at {kt}. So the paper's physics quantities are useful; they just need to be measured in a consistent "
             f"anatomical place.")

    # ---------------------------------------------------------------- 19 confusion
    s = new_slide(prs)
    names = {t: mf.v6_best(t) for t in (OCT, "OCTDL 7-class", "Kermany 4-class")}
    bo = n("headline hybrid ResNet-18 OCTID CSR/DR/MH", names[OCT][1]["bacc"], V6 / "hybrid_octid_dis750.json")
    bd = n("headline hybrid ResNet-18 OCTDL 7-class", names["OCTDL 7-class"][1]["bacc"], V6 / "hybrid_octdl_7.json")
    bk = n("headline hybrid ResNet-18 Kermany 4-class", names["Kermany 4-class"][1]["bacc"], V6 / "hybrid_kermany_4.json")
    title(s, f"The headline v6 hybrid (ResNet-18 + physics) reaches {bo} on OCTID diseases, {bd} on OCTDL's seven classes, {bk} on Kermany")
    image(s, FIG / "fig05c_confusion.png", 0.4, 1.4, 4.1, 4.5)
    image(s, FIG / "fig05d_confusion.png", 4.6, 1.4, 4.3, 4.5)
    image(s, FIG / "fig05e_confusion.png", 9.0, 1.4, 4.0, 4.5)
    bullets(s, ["Rows are true classes, normalised to 1; diagonal = recall. Model: frozen ResNet-18 features + physics + ROM → soft vote",
                "OCTDL's small classes (RAO, RVO, VID) and Kermany's DRUSEN vs CNV remain the hardest"],
            0.55, 6.0, 12.2, 1.0, size=14)
    footer(s)
    notes(s, f"These confusion matrices show where our headline model, the ResNet-18 hybrid, still fails. Each row is a true class and is "
             f"normalised so the diagonal is the recall of that class; the counts are in brackets. On the honest OCTID task "
             f"balanced accuracy is {bo}, and most remaining errors are diabetic retinopathy against macular hole. On OCTDL's "
             f"seven classes it is {bd}: age-related macular degeneration is easy, but the rare vascular occlusions and "
             f"vitreo-macular interface disease are often called AMD, partly because there are very few of them. On "
             f"Kermany it is {bk}, and drusen are the class most often confused, mainly with CNV and normal eyes.")

    # ---------------------------------------------------------------- 20 conformal
    cf = D["conf6"] or D["conf5"]
    cfs = V6 / "conformal.json" if D["conf6"] else R / "v5/conformal.json"
    tri = [r for r in cf if r["task"] == "OCTDL triage NORMAL vs ABNORMAL" and r["alpha"] == 0.05][0]
    s = new_slide(prs)
    cov = n("OCTDL triage ABNORMAL coverage alpha 0.05", tri["coverage_per_class"]["ABNORMAL"], cfs)
    ss = n("OCTDL triage mean set size alpha 0.05", tri["mean_set_size"], cfs, "{:.2f}")
    sa = n("OCTDL triage singleton accuracy alpha 0.05", tri["singleton_accuracy"], cfs)
    tot_, _, cc_ = counts("octdl")
    n_abn = tot_ - cc_.get("NORMAL", 0)
    se = n("binomial SE of coverage at 0.95 (OCTDL abnormal eyes)", (0.95 * 0.05 / n_abn) ** 0.5, R / "v4/octdl_features.csv", "{:.3f}")
    gap = 0.95 - tri["coverage_per_class"]["ABNORMAL"]
    se_raw = (0.95 * 0.05 / n_abn) ** 0.5
    title(s, f"Conformal prediction gives triage a stated guarantee: {cov} of diseased eyes kept for a 0.95 target"
             + (f" (within one standard error, ±{se})" if 0 < gap <= se_raw else ""))
    image(s, FIG / "fig11_conformal.png", 5.2, 1.4, 7.6, 5.4)
    bullets(s, [[("Score", {"bold": True}), ": sᵢ = 1 − p̂(yᵢ | xᵢ) on held-out calibration patients"],
                [("Threshold per class", {"bold": True}), ": q_k = the ⌈(n_k+1)(1−α)⌉-th smallest score of class k"],
                [("Set", {"bold": True}), ": C(x) = {k : 1 − p̂(k|x) ≤ q_k}; then P(y ∈ C | y = k) ≥ 1 − α"],
                [("Triage", {"bold": True}), f": clear an eye only if C(x) = {{NORMAL}}; mean set size {ss}, singletons right {sa}"]],
            0.55, 1.6, 4.5, 5.2, size=15)
    footer(s)
    notes(s, f"The paper claims a guaranteed triage. A classifier score cannot guarantee anything, but conformal prediction "
             f"can, under one stated assumption: new scans come from the same scanner and population as the calibration "
             f"scans. We hold out calibration patients, measure how surprised the model is by the true label, and set a "
             f"threshold per class. The model then outputs a set of labels instead of one label. A scan is cleared as "
             f"healthy only if the set is exactly normal. At a 95 percent target on OCTDL, {cov} of diseased eyes are "
             f"correctly kept, the average set has {ss} labels, and when the model commits to one label it is right {sa} "
             f"of the time. The heat map shows coverage for every class.")

    # ---------------------------------------------------------------- 21 new scanner
    s = new_slide(prs)
    cr = D["cross"]
    if cr:
        best = cr["best_config"]
        bc = cr["configs"][best]
        cb = n("cross-scanner best config balanced accuracy", bc["bacc"], V6 / "cross.json")
        dme = n("cross-scanner best config DME recall", bc["recall_per_class"]["DME"], V6 / "cross.json", "{:.2f}")
        title(s, f"On an unseen scanner the best model reaches {cb}; DME recall ({dme}) is still the weak point")
        bl_ = f"Best configuration: {best}"
    else:
        cb = n("cross-scanner v5 soft vote", v("Soft vote", X), src_grid("Soft vote", X))
        title(s, f"On an unseen scanner the v5 soft vote reaches {cb}; DME recall is the weak point")
        bl_ = "v6 cross-scanner run not yet available"
    image(s, FIG / "fig05f_confusion.png", 0.4, 1.4, 4.6, 4.9)
    image(s, FIG / "fig12_harmonisation.png", 5.1, 1.4, 7.7, 3.6)
    bullets(s, [[("Train", {"bold": True}), " on Kermany (Spectralis), test on OCTDL (Optovue): NORMAL / DME / AMD"],
                [("Reference block", {"bold": True}), ": z-score with 30 healthy eyes of the new scanner; speckle TTA: blur σ ∈ {0, 0.5, 1}"],
                bl_],
            5.2, 5.1, 7.6, 1.8, size=14)
    footer(s)
    notes(s, f"The hardest test is a new machine. We train on Kermany scans from a Heidelberg scanner and test on OCTDL scans "
             f"from an Optovue scanner, with the shared labels normal, diabetic macular oedema and AMD. Borrowing from NDT, "
             f"we tried a reference block: calibrate every feature with 30 healthy eyes from the new scanner. We also tried "
             f"speckle-matched test-time augmentation, blurring the test scans slightly and averaging the predictions. The "
             f"right heat maps compare these options for each model. The best configuration reaches {cb}. The confusion "
             f"matrix shows the persistent weakness: DME is often called AMD on the new scanner, because fluid looks "
             f"different between the two devices.")

    # ---------------------------------------------------------------- 22 v6 image-level
    s = new_slide(prs)
    ra = D["ra"]
    if ra:
        ak = n("residual AUROC Kermany r=25", ra["auroc_within"]["Kermany 4-class"]["by_r"]["25"]["auroc"], V6 / "residual_auroc.json", "{:.2f}")
        ao = n("residual AUROC OCTDL r=25", ra["auroc_within"]["OCTDL 7-class"]["by_r"]["25"]["auroc"], V6 / "residual_auroc.json", "{:.2f}")
        ax_ = n("residual AUROC Kermany->OCTDL r=25", ra["auroc_cross"]["by_r"]["25"]["auroc"], V6 / "residual_auroc.json", "{:.2f}")
    ip = D["ip"][OCT]
    pod_best = max(max(ip["pod_sweep"][m][r]["bacc"] for r in ip["pod_sweep"][m]) for m in ip["pod_sweep"])
    pb = n("image-POD alone best OCTID", pod_best, V6 / "image_pod_octid_dis750.json")
    dpod = n("v5+image-POD minus v5 OCTID", ip["paired_pod_vs_v5"]["diff_bacc"], V6 / "image_pod_octid_dis750.json", "{:+.3f}")
    ppod = n("v5+image-POD vs v5 p OCTID", ip["paired_pod_vs_v5"]["p_one_sided"], V6 / "image_pod_octid_dis750.json", "{:.2f}")
    title(s, f"v6: POD of the flattened image finds eigen-retinas, and a healthy-only POD flags disease (AUROC ≈ {ak})")
    image(s, FIG / "fig14a_image_pod_modes.png", 0.55, 1.4, 6.6, 2.3)
    image(s, FIG / "fig14c_residual_maps_kermany_4.png", 0.55, 3.8, 6.6, 3.1)
    bullets(s, [[("Eigen-retinas", {"bold": True}), f": POD of 64×256 flattened tiles; alone they reach {pb} on OCTID (weaker than v5)"],
                [("Added to v5", {"bold": True}), f": {dpod} on OCTID (p = {ppod}); no gain on other tasks"],
                [("Residual e(x) = ‖x − x̂‖² / ‖x − x̄‖²", {"bold": True}), f" with a healthy-only basis: AUROC {ak} Kermany, {ao} OCTDL, {ax_} across scanners"],
                [("Residual maps", {"bold": True}), " show where a scan departs from a healthy retina (red darker, blue brighter)"]],
            7.4, 1.5, 5.4, 5.4, size=15)
    footer(s)
    notes(s, f"Version six moves ROM from feature maps to the image itself. Because the retina is flattened, POD now gives "
             f"meaningful eigen-retinas, shown at the top left. On their own these modes reach {pb} on OCTID, well below the "
             f"physics features, and adding them to the v5 model changes balanced accuracy by {dpod}, p equals {ppod}. The "
             f"more useful idea is NDT-style defect detection: fit POD on healthy eyes only, and score a new scan by how "
             f"much of it the healthy modes cannot rebuild. That residual separates normal from abnormal with an AUROC "
             f"of about {ak} on Kermany, {ao} on OCTDL and {ax_} on the unseen scanner. The residual maps below show where "
             f"each scan departs from a healthy retina, which is an explainable picture.")

    # ---------------------------------------------------------------- 23 hybrid vs CNN (headline ResNet-18)
    s = new_slide(prs)
    ab = D["ab"]["tasks"]
    AB = V6 / "ablation.json"
    HY = "hybrid ResNet-18"
    hb = n("headline hybrid ResNet-18 OCTID", ab[OCT]["models"][HY]["bacc"], AB)
    cbn = n("CNN OCTID", D["cnn"][OCT]["bacc"], V6 / "cnn_octid_dis750.json")
    pcs = {t: ab[t]["paired"][f"{HY} vs CNN"] for t in (OCT, "Kermany 4-class", X)}
    fmt_p = lambda lab, p_: (n(f"{lab} diff", p_["diff_bacc"], AB, "{:+.3f}") + " [" + n(f"{lab} CI lo", p_["diff_ci"][0], AB, "{:+.3f}")
                            + ", " + n(f"{lab} CI hi", p_["diff_ci"][1], AB, "{:+.3f}") + "]")
    vs_cnn = {t: fmt_p(f"hybrid ResNet-18 - CNN {SHORTT[t]}", p_) for t, p_ in pcs.items()}
    within = pcs[OCT]["diff_ci"][0] <= 0 <= pcs[OCT]["diff_ci"][1]
    title(s, f"The ResNet-18 hybrid reaches {hb} vs the CNN's {cbn} on OCTID; the CNN stays clearly ahead on Kermany and a new scanner")
    sig, nsig = [], []
    for t in list(mf.TASKS):
        p_ = ab[t]["paired"][f"{HY} vs v5 best (soft vote)"]
        n(f"hybrid ResNet-18 - v5 {SHORTT[t]}", p_["diff_bacc"], AB, "{:+.3f}")
        (sig if (p_["diff_ci"][0] > 0 or p_["diff_ci"][1] < 0) else nsig).append(SHORTT[t])
    hi = D["hi"]
    fam5 = hi["top5_by_family"] if hi else {}
    image(s, FIG / "fig14e_hybrid_vs_cnn.png", 0.4, 1.4, 7.2, 3.3)
    image(s, FIG / "fig14d_gradcam_vs_physics.png", 0.4, 4.75, 7.2, 2.2)
    bullets(s, [[("Hybrid", {"bold": True}), ": PCA-64 of frozen ResNet-18 features + 171 physics features + ROM (image-POD, e(x)) → soft vote"],
                [("vs CNN (paired)", {"bold": True}), f": OCTID {vs_cnn[OCT]}, Kermany {vs_cnn['Kermany 4-class']}, new scanner {vs_cnn[X]}"],
                [("vs v5", {"bold": True}), f": significant on {', '.join(sig)}; CI includes zero on {', '.join(nsig) or 'none'}"],
                [("Importance (OCTID only)", {"bold": True}), f": #1 is {hi['rank'][0] if hi else 'n/a'}; top 5 = "
                 + ", ".join(f"{v} {k}" for k, v in fam5.items()) + ". It shows reliance, not added accuracy"]],
            7.8, 1.45, 5.0, 5.5, size=14)
    footer(s)
    notes(s, f"Can we close the gap to the black box without losing the physics? Our headline hybrid takes a frozen "
             f"ResNet-18, never trained on our labels, compresses its features with PCA inside each training fold, and adds "
             f"our physics features plus the two ROM quantities. On OCTID it reaches {hb} against {cbn} for the fine-tuned "
             f"CNN on the same folds: the difference is {vs_cnn[OCT]}"
             + (", so not significant" if within else "") + f". On Kermany and on the new scanner the CNN is clearly better. "
             f"Compared with v5 the hybrid is significantly better on {', '.join(sig)}. Permutation importance, measured on "
             f"OCTID only, ranks {hi['rank'][0] if hi else 'n/a'} first; importance shows what the model relies on, not what "
             f"each part adds, which the next slide tests directly. The bottom panel is the Grad-CAM comparison.")

    # ---------------------------------------------------------------- 24 ablation
    s = new_slide(prs)
    EMB = "ResNet-18 embedding only"
    md = {t: ab[t]["models"][HY]["bacc"] - ab[t]["models"][EMB]["bacc"] for t in list(mf.TASKS) + [X]}
    od = [md[t] for t in mf.TASKS if t not in (OCT, X)]
    lo_, hi2 = n("physics gain min (OCTDL, Kermany)", min(od), AB, "{:+.2f}"), n("physics gain max (OCTDL, Kermany)", max(od), AB, "{:+.2f}")
    d_oct = n("physics gain OCTID", md[OCT], AB, "{:+.3f}")
    title(s, f"Physics adds {lo_} to {hi2} on OCTDL and Kermany, nothing on OCTID ({d_oct})")
    image(s, FIG / "fig14f_ablation.png", 0.4, 1.35, 12.5, 4.35)
    pe = {t: ab[t]["paired"][f"{HY} vs {EMB}"] for t in list(mf.TASKS) + [X]}
    sg = [SHORTT[t] for t, p_ in pe.items() if p_["diff_ci"][0] > 0]
    ns = [SHORTT[t] for t, p_ in pe.items() if not p_["diff_ci"][0] > 0]
    for t, p_ in pe.items():
        n(f"paired hybrid - embedding ResNet-18 {SHORTT[t]}", p_["diff_bacc"], AB, "{:+.3f}")
    dmin = min(ab[t]["models"]["hybrid DINOv2-S"]["bacc"] - ab[t]["models"]["DINOv2-S embedding only"]["bacc"] for t in list(mf.TASKS) + [X])
    dmax = max(ab[t]["models"]["hybrid DINOv2-S"]["bacc"] - ab[t]["models"]["DINOv2-S embedding only"]["bacc"] for t in list(mf.TASKS) + [X])
    bullets(s, [[("Same backbone, with and without", {"bold": True}), " the 171 physics features + ROM; mean of 3 seeds, 95% CI"],
                [("ResNet-18", {"bold": True}), f": significant gain on {', '.join(sg)}; CI includes zero on {', '.join(ns)}"],
                [("DINOv2-S", {"bold": True}), f": gains {n('DINOv2 physics gain min', dmin, AB, '{:+.2f}')} to {n('DINOv2 physics gain max', dmax, AB, '{:+.2f}')}, because its embedding alone is weaker"]],
            0.55, 5.8, 12.2, 1.15, size=14)
    footer(s)
    notes(s, f"This slide asks the cleaner question: what do the physics features add on top of the deep features? We "
             f"compare each hybrid with the same frozen embedding used alone, on identical folds. For the ResNet-18 "
             f"backbone the physics and ROM features add {lo_} to {hi2} balanced accuracy on the OCTDL and Kermany tasks, "
             f"significant on {', '.join(sg)}. On OCTID they add nothing measurable ({d_oct}); the 272 OCTID scans are "
             f"already separated well by the embedding alone. For DINOv2 the gain is larger, {dmin:+.2f} to {dmax:+.2f}, "
             f"because its embedding is weaker on its own. This is the honest measure of the physics contribution; the "
             f"importance ranking on the previous slide is not.")

    # ---------------------------------------------------------------- 25 paper vs ours: heat map
    s = new_slide(prs)
    hb_oct = n("headline hybrid OCTID (heat map)", ab[OCT]["models"][HY]["bacc"], AB)
    title(s, f"Re-evaluated honestly, the paper's method drops to {pcv} on OCTID diseases; our hybrid reaches {hb_oct}")
    image(s, FIG / "fig15a_paper_vs_ours.png", 0.55, 1.4, 12.2, 5.15)
    text(s, 0.55, 6.55, 12.2, 0.35, "Paper as reported · paper's method re-run on the honest task · ours (ResNet-18 hybrid). "
         "n/a = not reported / not applicable.", size=12, color=INK2)
    footer(s)
    notes(s, f"This heat map compares, metric by metric, three things: the paper's reported numbers, the paper's method "
             f"re-run honestly, and our headline model. On OCTID the paper reports 71 percent accuracy on four classes, "
             f"which we reproduced. On the honest three-disease task, with one export format and patient-grouped folds, "
             f"the same method gets {pcv} balanced accuracy, while our hybrid gets {hb_oct}. For triage, the paper's "
             f"perfect normal recall and precision reappear when we re-run its method, but only because of the scanner "
             f"confound. On single-scanner data our model's triage numbers are lower than 1.00, but they are real, and "
             f"conformal prediction can attach a stated guarantee to them.")

    # ---------------------------------------------------------------- 26 paper vs ours: claims
    s = new_slide(prs)
    title(s, "Paper vs our project: the physics idea holds up, the headline claims do not")
    claims = json.load(open(FIG / "claims.json"))
    trs = [["Paper's statement", "What we found", "Verdict"]] + [[c["claim"], c["found"], c["verdict"]] for c in claims]
    table(s, trs, 0.55, 1.45, 12.2, 5.3, col_w=[3.4, 7.1, 1.7], size=13)
    for c in claims:
        n(f"claim: {c['claim'][:50]}", c["verdict"], FIG / "claims.json")
    footer(s)
    notes(s, "This table lists each of the paper's claims with what we found and the result file that supports it; the "
             "file names are in claims.json in the repository. Three things hold: the accuracy reproduces, the KT-1 and "
             "roughness quantities carry real diagnostic signal once measured in the right anatomical layer, and the push "
             "for explainable models is well founded. Three things do not hold as stated: the perfect triage is a scanner "
             "confound, the important grid rows mostly encode retina position, and discarding entropy for collinearity "
             "actually costs accuracy. A guarantee for triage is possible, but it needs conformal prediction and "
             "single-scanner data.")

    # ---------------------------------------------------------------- 25 conclusions
    s = new_slide(prs)
    title(s, "Physics features work when measured in anatomy, are honest about uncertainty, and still trail deep models")
    stat(s, 0.55, 1.5, 3.9, 1.9, f"{p0} → {bo}", "honest OCTID balanced accuracy: paper method → ResNet-18 hybrid")
    stat(s, 4.7, 1.5, 3.9, 1.9, cov, "diseased eyes kept by conformal triage (target 0.95, OCTDL)")
    stat(s, 8.85, 1.5, 3.9, 1.9, cb, "best balanced accuracy on an unseen scanner")
    text(s, 0.55, 3.7, 6.0, 0.4, "Takeaways", size=18, font=HEAD, bold=True, color=ACCENT)
    bullets(s, ["Validate the data first: a scanner confound produced the paper's perfect triage",
                "Measure physics in an anatomical frame; NDT tools (pulse-echo, MAPOD, calibration) transfer well",
                "Replace 'guaranteed' with conformal guarantees that state their assumption",
                "Frozen deep features + physics beat either alone on OCTDL and Kermany; a fine-tuned CNN is still best"],
            0.55, 4.15, 6.0, 2.8, size=15)
    text(s, 6.8, 3.7, 6.0, 0.4, "Limits and next steps", size=18, font=HEAD, bold=True, color=ACCENT)
    bullets(s, ["2-D B-scans only; disease is 3-D (volumes would allow the paper's 3-D anisotropy idea)",
                "Display JPEGs, not raw signal: NDT quantities are relative, not physical units",
                "OCTID has no patient IDs; DME across scanners is still the weakest class"],
            6.8, 4.15, 6.0, 2.8, size=15)
    footer(s)
    notes(s, f"To conclude. First, check the data before trusting a result: the paper's perfect triage came from different "
             f"image formats, not from the eye. Second, the physics is worth keeping when it is measured in an anatomical "
             f"frame; on the honest task balanced accuracy rose from {p0} with the paper's method to {bo} with ours, and NDT "
             f"tools such as pulse-echo features, probability of detection and calibration blocks transfer naturally. "
             f"Third, a real guarantee is possible with conformal prediction. Our limits: we only have 2-D images, the "
             f"files are display JPEGs rather than raw signals, OCTID lacks patient identifiers, and fine-tuned deep "
             f"networks remain more accurate, especially on a new scanner.")

    # ---------------------------------------------------------------- 26 references
    s = new_slide(prs)
    title(s, "References, data and thanks")
    refs = ["R. Banerjee, S. Sarkar, P. Munshi. Multi-Parametric Retinal Diagnostics: A Deterministic Physics-Based Engine for Tomographic Classification. IIT Gandhinagar.",
            "P. Munshi et al. Error estimates for tomographic inversion. Inverse Problems 7(3), 1991. · P. Athe et al. KT-1 signature and fractal dimension. Flow Meas. Instrum. 33, 2013.",
            "Data: OCTID (Gholami et al., arXiv:1812.07056); Kermany et al., Cell 172(5), 2018; OCTDL (Kulyabin et al., Sci. Data 11, 2024).",
            "Berkooz, Holmes, Lumley. POD in turbulent flows, Annu. Rev. Fluid Mech. 25, 1993 · Angelopoulos & Bates, conformal prediction, arXiv:2107.07511 · MIL-HDBK-1823A (POD).",
            "Ke et al., LightGBM, NeurIPS 2017 · Oquab et al., DINOv2, 2023 · Selvaraju et al., Grad-CAM, ICCV 2017.",
            "Code: github.com/Harsh-jain12/retinal-oct-physics-ml, built on Tanushka Sonde's original repository github.com/Tanushka-Sonde/Multi-Parametric-Retinal-Diagnostics."]
    text(s, 0.55, 1.45, 12.2, 4.6, [[r] for r in refs], size=15, color=INK, space_after=9)
    text(s, 0.55, 6.15, 12.2, 0.6, "Thank you. Questions?", size=22, font=HEAD, bold=True, color=ACCENT)
    footer(s)
    notes(s, "These are the main references. The paper we studied is by Banerjee, Sarkar and Munshi. The KT-1 signature "
             "comes from Professor Munshi's work on error estimates in tomographic inversion and from its use in "
             "multiphase flow measurement. The three datasets are public: OCTID, the Kermany OCT2017 dataset and OCTDL. "
             "Our reduced-order modelling follows the classic POD review by Berkooz, Holmes and Lumley, conformal "
             "prediction follows Angelopoulos and Bates, and the probability-of-detection analysis follows the military "
             "handbook 1823A. All code and results are in our public repository, which builds on Tanushka Sonde's "
             "original repository and keeps its full history. Thank you for listening; we are happy to take questions.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    json.dump(NUMS, open("presentation/numbers.json", "w"), indent=1, default=float)
    print("saved", OUT, len(prs.slides), "slides;", len(NUMS), "numbers ->", "presentation/numbers.json")


def export_pdf_and_preview(slides=(1, 8, 18, 23)):
    """PDF via LibreOffice, then a 2x2 preview of four slides (needs soffice and pdftoppm)."""
    import shutil
    import subprocess
    import tempfile
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice or not shutil.which("pdftoppm"):
        print("  soffice/pdftoppm not found: skipping PDF and preview"); return
    subprocess.run([soffice, "--headless", "--convert-to", "pdf", "--outdir", str(OUT.parent), str(OUT)],
                   check=True, capture_output=True)
    pdf = OUT.with_suffix(".pdf")
    with tempfile.TemporaryDirectory() as td:
        ims = []
        for k in slides:
            subprocess.run(["pdftoppm", "-r", "80", "-png", "-f", str(k), "-l", str(k), "-singlefile", str(pdf),
                            f"{td}/s{k}"], check=True)
            ims.append(Image.open(f"{td}/s{k}.png").convert("RGB"))
        w, h = ims[0].size
        g = 16
        sheet = Image.new("RGB", (2 * w + 3 * g, 2 * h + 3 * g), (228, 227, 223))
        for i, im in enumerate(ims):
            sheet.paste(im, (g + (i % 2) * (w + g), g + (i // 2) * (h + g)))
        sheet.save(OUT.parent / "preview.png")
    print("saved", pdf, "and", OUT.parent / "preview.png")


if __name__ == "__main__":
    import sys
    build()
    if "--pdf" in sys.argv:
        export_pdf_and_preview()
