"""Summaries of the repeated-CV outputs: protocol means, gaps, per-class drops, ablation effect, ranking stability.

Every spread here is across split seeds (split-seed variability), not a confidence interval.
"""
import pandas as pd

from common import LABELS, Timer, out_dir, save_json

METRICS = ["macro_f1", "accuracy", "balanced_accuracy"]
COMPARE = {"random_minus_participant": ("random_window", "participant_grouped"),
           "random_minus_segment": ("random_window", "segment_grouped"),
           "segment_minus_participant": ("segment_grouped", "participant_grouped")}
CANDIDATES = ["LogReg", "SVM_RBF", "RandomForest"]


def spread(s):
    return {"mean": s.mean(), "sd": s.std(), "min": s.min(), "max": s.max(), "n_positive": int((s > 0).sum()), "n": len(s)}


def main():
    rep, tables = out_dir("repeated"), out_dir("tables")
    r = pd.concat([pd.read_csv(p) for p in sorted(rep.glob("repeat_metrics_*.csv"))])
    f = pd.concat([pd.read_csv(p) for p in sorted(rep.glob("fold_metrics_*.csv"))])
    key = ["feature_set", "model", "strategy"]

    fold_sd = f.groupby(key + ["seed"]).macro_f1.std().groupby(key).agg(["mean", "min", "max"])
    summ = r.groupby(key)[METRICS].agg(["mean", "std", "min", "max"])
    summ.columns = [f"{m}_{s}" for m, s in summ.columns]
    summ = summ.join(fold_sd.add_prefix("fold_sd_macro_f1_")).reset_index()
    summ.to_csv(tables / "repeated_summary.csv", index=False)

    w = r.set_index(key + ["seed"]).sort_index()
    gaps, per_class = [], []
    for (fs, model), g in r.groupby(["feature_set", "model"]):
        g = g.set_index(["strategy", "seed"])
        for name, (a, b) in COMPARE.items():
            for m in METRICS:
                gaps.append({"feature_set": fs, "model": model, "comparison": name, "metric": m,
                             **spread(g.loc[a][m] - g.loc[b][m])})
            for lab in LABELS:
                per_class.append({"feature_set": fs, "model": model, "activity": lab, "comparison": name,
                                  f"f1_{a}": g.loc[a][f"f1_{lab}"].mean(), f"f1_{b}": g.loc[b][f"f1_{lab}"].mean(),
                                  **spread(g.loc[a][f"f1_{lab}"] - g.loc[b][f"f1_{lab}"])})
    gaps, per_class = pd.DataFrame(gaps), pd.DataFrame(per_class)
    gaps.to_csv(tables / "repeated_gaps.csv", index=False)
    per_class.to_csv(tables / "repeated_per_class.csv", index=False)

    # Ablation: does dropping the gravity family change the random-minus-participant drop? (paired by seed)
    abl = []
    for model in CANDIDATES:
        for lab in ["macro_f1"] + [f"f1_{l}" for l in LABELS]:
            d = {}
            for fs in ("full", "no_gravity"):
                x = w.loc[(fs, model)]
                d[fs] = x.loc["random_window"][lab] - x.loc["participant_grouped"][lab]
                d[f"{fs}_participant"] = x.loc["participant_grouped"][lab]
                d[f"{fs}_random"] = x.loc["random_window"][lab]
            delta = d["no_gravity"] - d["full"]
            abl.append({"model": model, "score": lab, "gap_full": d["full"].mean(), "gap_no_gravity": d["no_gravity"].mean(),
                        "gap_change_mean": delta.mean(), "gap_change_min": delta.min(), "gap_change_max": delta.max(),
                        "n_seeds_gap_smaller_without_gravity": int((delta < 0).sum()),
                        "participant_full": d["full_participant"].mean(), "participant_no_gravity": d["no_gravity_participant"].mean(),
                        "random_full": d["full_random"].mean(), "random_no_gravity": d["no_gravity_random"].mean(), "n": len(delta)})
    pd.DataFrame(abl).to_csv(tables / "gravity_ablation.csv", index=False)

    # Ranking stability among the three candidate models, full features.
    full = r[(r.feature_set == "full") & r.model.isin(CANDIDATES)]
    winners = full.loc[full.groupby(["strategy", "seed"]).macro_f1.idxmax()]
    rank = winners.groupby("strategy").model.value_counts().unstack(fill_value=0).reindex(columns=CANDIDATES, fill_value=0)
    rank.to_csv(tables / "model_ranking_wins.csv")
    gp = gaps[(gaps.feature_set == "full") & (gaps.comparison == "random_minus_participant") & (gaps.metric == "macro_f1")]
    seedgap = {m: (w.loc[("full", m, "random_window")].macro_f1 - w.loc[("full", m, "participant_grouped")].macro_f1)
               for m in CANDIDATES}
    rf_largest = int(((seedgap["RandomForest"] > seedgap["LogReg"]) & (seedgap["RandomForest"] > seedgap["SVM_RBF"])).sum())

    diag = pd.read_csv(rep / "fold_diagnostics.csv")
    dsum = diag.groupby("strategy").agg(
        n_val_min=("n_val", "min"), n_val_max=("n_val", "max"),
        n_val_subjects_min=("n_val_subjects", "min"), n_val_subjects_max=("n_val_subjects", "max"),
        frac_overlap_neighbour_mean=("frac_val_with_overlapping_neighbour_in_train", "mean"),
        frac_overlap_neighbour_min=("frac_val_with_overlapping_neighbour_in_train", "min"),
        frac_subject_activity_in_train_mean=("frac_val_subject_activity_in_train", "mean"),
        frac_subject_activity_in_train_min=("frac_val_subject_activity_in_train", "min"))
    dsum["all_classes_in_every_val_fold"] = diag.groupby("strategy").apply(
        lambda d: bool((d[[f"n_{a}" for a in LABELS]] > 0).all().all()), include_groups=False)
    dsum["fallback_used"] = diag.groupby("strategy").splitter.apply(lambda s: bool(s.str.contains("fallback").any()))
    dsum.to_csv(tables / "fold_diagnostics_summary.csv")

    save_json({"rf_gap_largest_in_n_seeds": rf_largest, "n_seeds": int(r.seed.nunique()),
               "gap_macro_f1_random_minus_participant": gp.set_index("model")[["mean", "sd", "min", "max", "n_positive"]].to_dict("index"),
               "grouped_wins": rank.loc["participant_grouped"].to_dict() if "participant_grouped" in rank.index else {},
               "random_wins": rank.loc["random_window"].to_dict() if "random_window" in rank.index else {}},
              out_dir("metrics") / "repeated_highlights.json")
    print(summ[summ.feature_set == "full"][["model", "strategy", "macro_f1_mean", "macro_f1_std", "accuracy_mean",
                                           "balanced_accuracy_mean", "fold_sd_macro_f1_mean"]].round(4).to_string(index=False))
    print(gp[["model", "mean", "sd", "min", "max", "n_positive"]].round(4).to_string(index=False))
    print(rank)
    print(pd.DataFrame(abl)[lambda d: d.score.isin(["macro_f1", "f1_SITTING", "f1_STANDING"])].round(4).to_string(index=False))


if __name__ == "__main__":
    with Timer("summarize_repeated"):
        main()
