"""Extract the paper's own headline claims (with the quoted sentence) from the PDF, so that every
"paper reported" number on a slide traces to a file produced by a script.

  python v6_paper_claims.py            (needs pdftotext from poppler)
Writes results/v6/paper_claims.json
"""
import json
import re
import subprocess

PDF = "paper/Banerjee_Sarkar_Munshi_2025.pdf"
CLAIMS = {  # key: (regex on whitespace-normalised text, value parser)
    "multiclass_accuracy": (r"accuracy across all four conditions stabilized at (\d+)%", lambda m: int(m.group(1)) / 100),
    "normal_triage_precision_recall": (r"achieved a (1\.00) Recall and (1\.00) Precision rate for NORMAL", lambda m: float(m.group(1))),
    "n_test_scans": (r"holding back (\d+) blind scans", lambda m: int(m.group(1))),
    "n_scans_total": (r"dataset of (\d+) OCT scans", lambda m: int(m.group(1))),
    "top15_hurst_fraction": (r"(\d+)% of the top 15 most critical", lambda m: int(m.group(1)) / 100),
    "top_grid_rows": (r"map to Rows (\d+) and (\d+)", lambda m: [int(m.group(1)), int(m.group(2))]),
    "entropy_discarded_for_collinearity": (r"(Shannon\s+Entropy\s+was\s+explicitly\s+discarded)", lambda m: True),
    "triage_guarantee_wording": (r"(mathematically guaranteeing[^.]*)", lambda m: True),
}


def main():
    txt = subprocess.run(["pdftotext", PDF, "-"], capture_output=True, text=True, check=True).stdout
    flat = re.sub(r"-\s*\n\s*", "", txt)               # undo hyphenation
    flat = re.sub(r"\s+", " ", flat)
    out = {"source": PDF}
    for k, (pat, fn) in CLAIMS.items():
        m = re.search(pat, flat, flags=re.I)
        if not m:
            out[k] = None
            continue
        a, b = max(0, m.start() - 80), min(len(flat), m.end() + 80)
        out[k] = {"value": fn(m), "quote": flat[a:b].strip()}
    json.dump(out, open("results/v6/paper_claims.json", "w"), indent=2)
    for k, v in out.items():
        print(k, "->", v if not isinstance(v, dict) else v["value"])


if __name__ == "__main__":
    main()
