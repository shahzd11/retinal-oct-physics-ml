"""Download datasets into data/.

  python download_data.py            # OCTID only (~35 MB)  - the paper's dataset
  python download_data.py kermany    # Kermany OCT2017, Heidelberg Spectralis (~6 GB, ~12 GB peak while extracting)
  python download_data.py kermany-subset <zip>   # low disk: extract a 600/class patient subset only
  python download_data.py octdl      # OCTDL, Optovue RTVue (~400 MB) from Zenodo
"""
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

import kagglehub


def need(gb):
    free = shutil.disk_usage(".").free / 1e9
    if free < gb:
        sys.exit(f"Need ~{gb} GB free, only {free:.1f} GB available. Free space and retry.")


def octid():
    src = Path(kagglehub.dataset_download("kawtarnaim/octid-dataset")) / "OCTID"
    dst = Path("data/OCTID")
    if not dst.exists():
        shutil.copytree(src, dst)
    for c in ["CSR", "DR", "MH", "NORMAL"]:
        print(c, sum(p.suffix.lower() in (".jpg", ".jpeg") for p in (dst / c).iterdir()))


def kermany():
    need(13)
    src = Path(kagglehub.dataset_download("paultimothymooney/kermany2018"))
    cand = [p for p in src.rglob("OCT2017*") if p.is_dir() and (p / "train").exists()]
    dst = Path("data/OCT2017")
    if not dst.exists():
        shutil.move(str(cand[0]), dst)          # move, not copy: disk is precious
    print("Kermany ->", dst, sum(1 for _ in dst.rglob("*.jpeg")), "images")


def kermany_subset(archive, per_class=600, seed=0):
    """Extract only a patient-level subset from the Kermany zip (full extraction needs ~12 GB).
    Keeps whole patients (<=3 scans each) per class until per_class scans, same rule as make_manifest."""
    import re
    import numpy as np
    dst = Path("data/OCT2017")
    with zipfile.ZipFile(archive) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".jpeg") and "__MACOSX" not in n
                 and len(Path(n).parts) >= 3 and Path(n).parts[-3] in ("train", "test", "val")]
        by = {}
        for n in names:
            m = re.match(r"([A-Z]+)-(\d+)-(\d+)", Path(n).name)
            if m:
                by.setdefault(m.group(1), {}).setdefault(m.group(2), {}).setdefault(Path(n).name, n)
        rng = np.random.default_rng(seed)
        picked = []
        for c, pats in sorted(by.items()):
            ids = sorted(pats); rng.shuffle(ids); k = 0
            for pid in ids:
                files = sorted(pats[pid].values())[:3]
                picked += files; k += len(files)
                if k >= per_class:
                    break
            print(c, k, "scans")
        for n in picked:
            parts = Path(n).parts
            out = dst / parts[-3] / parts[-2] / parts[-1]          # split/CLASS/file
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(z.read(n))
    print("Kermany subset ->", dst, len(picked), "images")


def octdl(record="10839556"):
    need(1)
    dst = Path("data/OCTDL"); dst.mkdir(parents=True, exist_ok=True)
    meta = json.load(urllib.request.urlopen(f"https://zenodo.org/api/records/{record}"))
    for f in meta["files"]:
        out = dst / f["key"]
        if not out.exists():
            print("downloading", f["key"], f"{f['size'] / 1e6:.0f} MB")
            urllib.request.urlretrieve(f["links"]["self"], out)
        if out.suffix == ".zip":
            with zipfile.ZipFile(out) as z:
                z.extractall(dst)
            out.unlink()                          # drop the archive after extraction
    print("OCTDL ->", dst, sum(1 for p in dst.rglob("*") if p.suffix.lower() in (".jpg", ".png")), "images")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "octid"
    if cmd == "kermany-subset":       # python download_data.py kermany-subset <path to kagglehub archive>
        kermany_subset(sys.argv[2])
    else:
        {"octid": octid, "kermany": kermany, "octdl": octdl}[cmd]()
