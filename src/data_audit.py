"""Data-quality audit of the official partitions. Reports findings; never drops rows."""
import sys

import numpy as np
import pandas as pd

from common import CONFIG, DATA_DIR, LABELS, Timer, gravity_columns, out_dir, save_json
from load_data import feature_names, load_split


def main():
    exp = CONFIG["expected"]
    Xtr, ytr, str_ = load_split("train")
    Xte, yte, ste = load_split("test")
    tables, meta = out_dir("tables"), out_dir("metadata")

    fn = feature_names()
    fn.to_csv(meta / "feature_names.csv", index=False)
    pd.Series(gravity_columns(fn["sanitized"]), name="feature").to_csv(
        meta / "gravity_features.csv", index=False)
    dup_names = fn["name"][fn["name"].duplicated(keep=False)]

    report = {
        "shapes": {"X_train": list(Xtr.shape), "X_test": list(Xte.shape),
                   "y_train": len(ytr), "y_test": len(yte), "subject_train": len(str_), "subject_test": len(ste)},
        "n_total_windows": len(Xtr) + len(Xte),
        "labels_found": sorted(set(ytr) | set(yte)),
        "train_subjects": sorted(int(s) for s in np.unique(str_)),
        "test_subjects": sorted(int(s) for s in np.unique(ste)),
        "subject_overlap": sorted(int(s) for s in set(str_) & set(ste)),
        "missing_values": {"train": int(Xtr.isna().sum().sum()), "test": int(Xte.isna().sum().sum())},
        "infinite_values": {"train": int(np.isinf(Xtr.to_numpy()).sum()), "test": int(np.isinf(Xte.to_numpy()).sum())},
        "constant_features_train": Xtr.columns[Xtr.nunique() <= 1].tolist(),
        "duplicate_feature_names": {"n_names_involved": int(len(dup_names)), "n_distinct": int(dup_names.nunique())},
        "duplicate_rows": {"train": int(Xtr.duplicated().sum()), "test": int(Xte.duplicated().sum()),
                           "train_vs_test": int(pd.concat([Xtr.drop_duplicates(), Xte.drop_duplicates()]).duplicated().sum())},
        "feature_range_train": {"min": float(Xtr.min().min()), "max": float(Xtr.max().max()),
                                "frac_features_within_[-1,1]": float(((Xtr.min() >= -1) & (Xtr.max() <= 1)).mean()),
                                "feature_sd_min": float(Xtr.std().min()), "feature_sd_max": float(Xtr.std().max()),
                                "feature_mean_min": float(Xtr.mean().min()), "feature_mean_max": float(Xtr.mean().max())},
        "id_columns_in_X": [c for c in Xtr.columns if "subject" in c.lower() or "index" in c.lower()],
        "majority_class_train": {"label": pd.Series(ytr).value_counts().idxmax(),
                                 "proportion": float(pd.Series(ytr).value_counts(normalize=True).max())},
        "n_gravity_features": len(gravity_columns(Xtr.columns)),
    }
    report["n_train_subjects"] = len(report["train_subjects"])
    report["n_test_subjects"] = len(report["test_subjects"])

    problems = []
    for key, got in [("n_train", len(Xtr)), ("n_test", len(Xte)), ("n_train_subjects", report["n_train_subjects"]),
                     ("n_test_subjects", report["n_test_subjects"])]:
        if got != exp[key]:
            problems.append(f"{key}: expected {exp[key]}, got {got}")
    if report["id_columns_in_X"]:
        problems.append(f"identifier columns in X: {report['id_columns_in_X']}")
    if report["subject_overlap"]:
        problems.append(f"train/test subject overlap: {report['subject_overlap']}")
    if sorted(report["labels_found"]) != sorted(LABELS):
        problems.append(f"labels differ: {report['labels_found']}")
    report["problems"] = problems

    rows = []
    for split, y in [("train", ytr), ("test", yte)]:
        c = pd.Series(y).value_counts().reindex(LABELS)
        rows += [{"split": split, "activity": a, "count": int(c[a]), "proportion": c[a] / len(y)} for a in LABELS]
    pd.DataFrame(rows).to_csv(tables / "class_distribution.csv", index=False)

    by_subj = pd.crosstab(np.concatenate([str_, ste]), np.concatenate([ytr, yte])).reindex(columns=LABELS)
    by_subj.index.name = "subject"
    by_subj.insert(0, "split", ["train" if s in set(str_) else "test" for s in by_subj.index])
    by_subj.to_csv(tables / "class_counts_by_subject.csv")

    Xtr.describe().T.to_csv(tables / "feature_summary_train.csv")
    pd.DataFrame([{"partition": "train", "windows": len(Xtr), "participants": report["n_train_subjects"]},
                  {"partition": "test", "windows": len(Xte), "participants": report["n_test_subjects"]},
                  {"partition": "total", "windows": len(Xtr) + len(Xte),
                   "participants": report["n_train_subjects"] + report["n_test_subjects"]}]
                 ).to_csv(tables / "dataset_summary.csv", index=False)

    prov = DATA_DIR / "provenance.json"
    if prov.exists():
        report["provenance"] = __import__("json").loads(prov.read_text())
    save_json(report, out_dir("metrics") / "data_audit.json")
    print({k: report[k] for k in ["shapes", "n_train_subjects", "n_test_subjects", "missing_values",
                                  "infinite_values", "duplicate_rows", "duplicate_feature_names", "feature_range_train"]})
    print(f"constant features: {len(report['constant_features_train'])}")
    if problems:
        sys.exit("DATA AUDIT FAILED:\n" + "\n".join(problems))
    print("data audit passed")


if __name__ == "__main__":
    with Timer("data_audit"):
        main()
