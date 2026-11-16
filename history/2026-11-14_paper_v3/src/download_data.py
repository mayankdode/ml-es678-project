"""Download the official UCI HAR archive (dataset 240) and extract it into $HAR_DATA_DIR."""
import hashlib
import io
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone

from common import DATA_DIR, DATASET_DIR, save_json

URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
DOI = "https://doi.org/10.24432/C54S4K"


def safe_extract(zf, dest):
    root = dest.resolve()
    for member in zf.namelist():
        target = (dest / member).resolve()
        if root not in target.parents and target != root:
            sys.exit(f"refusing to extract unsafe path: {member}")
    zf.extractall(dest)


def main():
    if (DATASET_DIR / "train" / "X_train.txt").exists():
        print(f"dataset already present at {DATASET_DIR}")
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"downloading {URL}", flush=True)
    raw = urllib.request.urlopen(URL, timeout=300).read()
    outer = zipfile.ZipFile(io.BytesIO(raw))
    inner_name = "UCI HAR Dataset.zip"
    if inner_name not in outer.namelist():
        sys.exit(f"unexpected archive layout: {outer.namelist()}")
    inner = outer.read(inner_name)
    safe_extract(zipfile.ZipFile(io.BytesIO(inner)), DATA_DIR)
    save_json({"source_url": URL, "doi": DOI, "license": "CC BY 4.0",
               "downloaded_utc": datetime.now(timezone.utc).isoformat(),
               "outer_zip_sha256": hashlib.sha256(raw).hexdigest(),
               "inner_zip_sha256": hashlib.sha256(inner).hexdigest(),
               "bytes": len(raw)}, DATA_DIR / "provenance.json")
    print(f"extracted to {DATASET_DIR}")


if __name__ == "__main__":
    main()
