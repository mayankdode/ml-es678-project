"""Refit the selected pipeline on all official training windows and evaluate it on the official test set.

This is the only script that reads the official test partition for scoring. It writes a freeze manifest (SHA-256
of the code, config and selection file) before scoring and refuses to overwrite an existing test result unless
HAR_ALLOW_TEST_RERUN=1, so the test set cannot be re-queried by accident.
"""
import hashlib
import json
import os
from datetime import datetime, timezone

import joblib
import pandas as pd

from common import CONFIG, LABELS, ROOT, Timer, confusion, make_pipeline, out_dir, save_json, scores
from load_data import load_split


def manifest(metrics):
    files = sorted((ROOT / "src").glob("*.py")) + [ROOT / "config" / "experiment_config.json", metrics / "selected_model.json"]
    return {"frozen_utc": datetime.now(timezone.utc).isoformat(),
            "sha256": {str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p.name): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in files}}


def main():
    metrics = out_dir("metrics")
    if (metrics / "test_metrics.json").exists() and os.environ.get("HAR_ALLOW_TEST_RERUN") != "1":
        raise SystemExit("test_metrics.json already exists; the official test set has been scored. "
                         "Set HAR_ALLOW_TEST_RERUN=1 only if this re-query is deliberate and disclosed.")
    sel = json.loads((metrics / "selected_model.json").read_text())
    save_json(manifest(metrics), metrics / "freeze_manifest.json")
    Xtr, ytr, _ = load_split("train")
    Xte, yte, ste = load_split("test")
    model = make_pipeline(sel["selected_model"], **sel["selected_params"]).fit(Xtr, ytr)
    pred = model.predict(Xte)
    s = scores(yte, pred)
    save_json({"model": sel["selected_model"], "params": sel["selected_params"], "n_train": len(ytr),
               "n_test": len(yte), "evaluated_utc": datetime.now(timezone.utc).isoformat(), **s}, metrics / "test_metrics.json")
    pd.DataFrame({"subject": ste, "y_true": yte, "y_pred": pred}).to_csv(metrics / "test_predictions.csv", index_label="test_row")
    cm = pd.DataFrame(confusion(yte, pred), index=LABELS, columns=LABELS)
    cm.to_csv(metrics / "test_confusion_counts.csv")
    cm.div(cm.sum(axis=1), axis=0).mul(100).round(2).to_csv(metrics / "test_confusion_row_pct.csv")
    joblib.dump(model, out_dir("models") / "final_model.joblib")
    print(f"official test: macroF1={s['macro_f1']:.4f} acc={s['accuracy']:.4f} bal_acc={s['balanced_accuracy']:.4f}")


if __name__ == "__main__":
    with Timer("final_evaluation"):
        main()
