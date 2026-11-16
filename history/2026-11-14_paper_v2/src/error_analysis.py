"""Confusion ranking, static/dynamic error breakdown, per-test-subject scores, CV-vs-test comparison."""
import json

import pandas as pd

from common import CONFIG, LABELS, Timer, out_dir, save_json, scores


def main():
    metrics, tables = out_dir("metrics"), out_dir("tables")
    pred = pd.read_csv(metrics / "test_predictions.csv")
    cm = pd.read_csv(metrics / "test_confusion_counts.csv", index_col=0)
    sel = json.loads((metrics / "selected_model.json").read_text())
    test = json.loads((metrics / "test_metrics.json").read_text())

    conf = [{"true": t, "predicted": p, "count": int(cm.loc[t, p]), "pct_of_true_class": 100 * cm.loc[t, p] / cm.loc[t].sum()}
            for t in LABELS for p in LABELS if t != p and cm.loc[t, p] > 0]
    conf = pd.DataFrame(conf).sort_values("count", ascending=False)
    conf.to_csv(tables / "test_top_confusions.csv", index=False)

    static = set(CONFIG["static_labels"])
    err = pred[pred.y_true != pred.y_pred]
    kind = err.apply(lambda r: ("static" if r.y_true in static else "dynamic") + "->" +
                     ("static" if r.y_pred in static else "dynamic"), axis=1)
    breakdown = {"n_errors": len(err), "n_test": len(pred), **kind.value_counts().to_dict()}

    per_subj = pd.DataFrame([{"subject": s, "n_windows": len(g), **scores(g.y_true, g.y_pred)}
                             for s, g in pred.groupby("subject")])
    per_subj[["subject", "n_windows", "macro_f1", "accuracy", "balanced_accuracy"]].to_csv(tables / "test_per_subject.csv", index=False)

    out = {"top_confusions": conf.head(5).to_dict("records"), "error_breakdown_static_dynamic": breakdown,
           "per_subject_macro_f1": {"min": per_subj.macro_f1.min(), "median": per_subj.macro_f1.median(),
                                    "max": per_subj.macro_f1.max()},
           "selected_grouped_cv_macro_f1": sel["grouped_cv_macro_f1_mean"], "test_macro_f1": test["macro_f1"],
           "test_minus_grouped_cv": test["macro_f1"] - sel["grouped_cv_macro_f1_mean"]}
    save_json(out, metrics / "error_analysis.json")
    print(conf.head(5).round(2).to_string(index=False))
    print(breakdown)


if __name__ == "__main__":
    with Timer("error_analysis"):
        main()
