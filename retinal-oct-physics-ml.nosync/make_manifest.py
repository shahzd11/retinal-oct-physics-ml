"""Build dataset manifests: one CSV row per scan with
path, label, patient, dataset, device, axial_um, icon_box.

  python make_manifest.py octid
  python make_manifest.py kermany --per-class 600
  python make_manifest.py octdl
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

IMG = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
# OCTID disease scans carry a scan-direction icon in the lower-left corner; mask that corner
# in *every* OCTID scan (all classes) so the mask cannot leak the label.
OCTID_ICON = "0.0,0.72,0.18,1.0"


def octid(root="data/OCTID"):
    rows = []
    for c in ["CSR", "DR", "MH", "NORMAL"]:
        for p in sorted(Path(root, c).iterdir()):
            if p.suffix.lower() in IMG:
                # no patient IDs in OCTID -> each image is its own group (documented limitation)
                rows.append(dict(path=str(p), label=c, patient=f"octid_{p.stem}", dataset="OCTID",
                                 device="Cirrus HD-OCT (exported)", axial_um=np.nan, icon_box=OCTID_ICON))
    return pd.DataFrame(rows)


def kermany(root="data/OCT2017", per_class=600, seed=0):
    """Kermany OCT2017 (Spectralis, 3.87 um axial). Filenames: CLASS-PATIENTID-N.jpeg.
    Subsample whole patients per class until per_class images are reached."""
    rows = []
    for split in ["train", "test"]:
        for cdir in Path(root, split).iterdir():
            for p in cdir.iterdir():
                if p.suffix.lower() in IMG:
                    m = re.match(r"([A-Z]+)-(\d+)-(\d+)", p.stem)
                    if m:
                        rows.append(dict(path=str(p), label=m.group(1), patient=f"kermany_{m.group(2)}",
                                         dataset="Kermany", device="Heidelberg Spectralis",
                                         axial_um=3.87, icon_box=""))
    df = pd.DataFrame(rows).drop_duplicates("path")
    if per_class:
        rng = np.random.default_rng(seed)
        keep = []
        for c, d in df.groupby("label"):
            pats = np.array(d.patient.unique()); rng.shuffle(pats)
            n = 0
            for pid in pats:
                g = d[d.patient == pid].head(3)      # at most 3 scans per patient -> more patients
                keep.append(g); n += len(g)
                if n >= per_class:
                    break
        df = pd.concat(keep)
    return df.reset_index(drop=True)


def octdl(root="data/OCTDL"):
    """OCTDL (Optovue Avanti RTVue XR). Uses the labels CSV shipped with the dataset if present."""
    root = Path(root)
    csvs = list(root.rglob("*.csv"))
    rows = []
    meta = pd.read_csv(csvs[0]) if csvs else None
    lut = {}
    if meta is not None and "file_name" in meta.columns:
        for r in meta.to_dict("records"):
            lut[Path(str(r["file_name"])).stem] = r
    for p in root.rglob("*"):
        if p.suffix.lower() in IMG:
            label = p.parent.name
            pid = p.stem
            r = lut.get(p.stem)
            if r is not None:
                pid = str(r.get("patient_id", pid))
                label = str(r.get("disease", label))
            label = {"NO": "NORMAL"}.get(label, label)   # same name for healthy across datasets
            rows.append(dict(path=str(p), label=label, patient=f"octdl_{pid}", dataset="OCTDL",
                             device="Optovue RTVue XR Avanti", axial_um=3.9, icon_box=""))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["octid", "kermany", "octdl"])
    ap.add_argument("--per-class", type=int, default=600)
    a = ap.parse_args()
    df = {"octid": octid, "kermany": lambda: kermany(per_class=a.per_class), "octdl": octdl}[a.dataset]()
    Path("manifests").mkdir(exist_ok=True)
    out = f"manifests/{a.dataset}.csv"
    df.to_csv(out, index=False)
    print(out, len(df), "scans,", df.patient.nunique(), "patients"); print(df.label.value_counts())
