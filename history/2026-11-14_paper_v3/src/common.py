"""Shared paths, configuration, model factory and metric helpers."""
import json
import os
import time
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("HAR_DATA_DIR", ROOT / "data" / "raw"))
DATASET_DIR = DATA_DIR / "UCI HAR Dataset"
RESULTS = Path(os.environ.get("HAR_RESULTS_DIR", ROOT / "results"))
SMOKE = os.environ.get("HAR_SMOKE", "0") == "1"
STAGE = os.environ.get("HAR_STAGE", "dev")  # names the timings/metadata files so parallel shards can be merged
N_JOBS = int(os.environ.get("HAR_JOBS", "-1"))

CONFIG = json.loads((ROOT / "config" / "experiment_config.json").read_text())
LABELS = CONFIG["labels"]
SEED = CONFIG["seed"]
ZERO_DIV = CONFIG["zero_division"]


def out_dir(sub):
    d = RESULTS / sub
    d.mkdir(parents=True, exist_ok=True)
    return d


def fixed_params(name):
    params = dict(CONFIG["fixed_models"][name])
    if SMOKE:
        params.update(CONFIG["smoke"]["fixed_models"].get(name, {}))
    return params


def search_space(name):
    return (CONFIG["smoke"] if SMOKE else CONFIG)["search_spaces"][name]


def search_space_extension(key):
    """Grid (list of dicts) of a bounded follow-up search, e.g. RandomForest_round2."""
    return (CONFIG["smoke"] if SMOKE else CONFIG)["search_space_extensions"][key]["grid"]


def repeat_seeds():
    return (CONFIG["smoke"] if SMOKE else CONFIG)["repeat_seeds"]


def permutation_repeats():
    return (CONFIG["smoke"] if SMOKE else CONFIG)["permutation_repeats"]


def gravity_columns(columns):
    """The gravity feature family named in features_info.txt: tGravityAcc*, tGravityAccMag* and angle(*, gravity*)."""
    return [c for c in columns if "gravity" in c.lower()]


def feature_columns(feature_set, columns):
    if feature_set == "full":
        return list(columns)
    if feature_set == "no_gravity":
        drop = set(gravity_columns(columns))
        return [c for c in columns if c not in drop]
    raise ValueError(feature_set)


def make_pipeline(name, **params):
    """Leakage-safe pipelines: every learned transform is fitted inside each training fold."""
    if name == "Dummy":
        return Pipeline([("clf", DummyClassifier(**params))])
    if name == "LogReg":
        return Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(random_state=SEED, **params))])
    if name == "SVM_RBF":
        return Pipeline([("scale", StandardScaler()), ("clf", SVC(**params))])
    if name == "RandomForest":
        return Pipeline([("clf", RandomForestClassifier(**params))])
    raise ValueError(name)


def scores(y_true, y_pred):
    """Fixed six-label order; zero_division=0 for classes that are never predicted."""
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=ZERO_DIV)
    out = {"macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=ZERO_DIV),
           "accuracy": accuracy_score(y_true, y_pred),
           "balanced_accuracy": balanced_accuracy_score(y_true, y_pred)}
    for i, lab in enumerate(LABELS):
        out[f"precision_{lab}"], out[f"recall_{lab}"], out[f"f1_{lab}"] = p[i], r[i], f[i]
        out[f"support_{lab}"] = int(s[i])
    return out


def confusion(y_true, y_pred):
    return confusion_matrix(y_true, y_pred, labels=LABELS)


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, default=lambda o: o.item() if isinstance(o, np.generic) else str(o)))


class Timer:
    """Appends wall-clock seconds per step to results/logs/timings_<stage>.json."""

    def __init__(self, step):
        self.step = step

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        path = out_dir("logs") / f"timings_{STAGE}.json"
        t = json.loads(path.read_text()) if path.exists() else {}
        t[self.step] = round(time.time() - self.t0, 1)
        save_json(t, path)
        print(f"[timing] {self.step}: {t[self.step]:.1f} s", flush=True)
