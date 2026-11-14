"""Download the Kermany archive with kagglehub but keep the zip instead of extracting it (low disk)."""
import os, shutil, kagglehub, kagglehub.http_resolver as hr
def keep(archive_path, out_path):
    os.makedirs(out_path, exist_ok=True)
    shutil.move(archive_path, "data/kermany2018.zip")
    open(archive_path, "w").close()          # so kagglehub's os.remove() succeeds
hr._extract_archive = keep
print(kagglehub.dataset_download("paultimothymooney/kermany2018"))
