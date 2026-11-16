"""Load the official UCI HAR train/test partitions with alignment checks."""
import sys

import numpy as np
import pandas as pd

from common import CONFIG, DATASET_DIR, LABELS


def _read(path, **kw):
    if not path.exists():
        sys.exit(f"missing data file: {path}\nRun `python src/download_data.py` first.")
    return pd.read_csv(path, sep=r"\s+", header=None, **kw)


def feature_names():
    f = _read(DATASET_DIR / "features.txt", names=["idx", "name"])
    # The original names contain duplicates; prefix with the 1-based column index to make them unique.
    f["sanitized"] = [f"f{i:03d}_{n}" for i, n in zip(f["idx"], f["name"])]
    return f


def activity_map():
    a = _read(DATASET_DIR / "activity_labels.txt", names=["id", "name"])
    return dict(zip(a["id"], a["name"]))


def load_split(split):
    """Return X (DataFrame, 561 cols), y (activity names), subject IDs for 'train' or 'test'."""
    d = DATASET_DIR / split
    names = feature_names()["sanitized"].tolist()
    X = _read(d / f"X_{split}.txt", dtype=np.float64)
    y_id = _read(d / f"y_{split}.txt")[0].to_numpy()
    subj = _read(d / f"subject_{split}.txt")[0].to_numpy()
    if not (len(X) == len(y_id) == len(subj)):
        sys.exit(f"{split}: row mismatch X={len(X)} y={len(y_id)} subject={len(subj)}")
    if X.shape[1] != CONFIG["expected"]["n_features"]:
        sys.exit(f"{split}: expected {CONFIG['expected']['n_features']} features, got {X.shape[1]}")
    X.columns = names
    amap = activity_map()
    y = np.array([amap[i] for i in y_id])
    if set(y) - set(LABELS):
        sys.exit(f"{split}: unexpected labels {set(y) - set(LABELS)}")
    return X, y, subj
