"""Repeated 5-fold CV of the fixed model configurations under three split protocols and two feature sets.

For every split seed, protocol and model, out-of-fold (OOF) predictions cover the whole official training
partition once; repeat-level metrics are computed from them. Spread across seeds is split-seed variability,
not a confidence interval. Env HAR_FEATURESETS (default "full,no_gravity") selects the feature sets, so the
work can be sharded across machines; each shard writes its own files.
"""
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from common import (LABELS, N_JOBS, Timer, feature_columns, fixed_params, make_pipeline, out_dir, repeat_seeds,
                    save_json, scores)
from load_data import load_split
from splits import STRATEGIES, get_folds

MODELS = {"full": ["Dummy", "LogReg", "SVM_RBF", "RandomForest"], "no_gravity": ["LogReg", "SVM_RBF", "RandomForest"]}


def load_segments(n):
    seg = pd.read_csv(out_dir("metadata") / "segments_train.csv")
    if len(seg) != n:
        raise SystemExit("segments_train.csv does not match the training partition")
    return seg.segment.to_numpy()


def build_folds(y, subj, seg):
    """All (seed, protocol) fold lists, with programmatic checks and per-fold diagnostics."""
    folds, diag, assign = {}, [], []
    linked = np.r_[seg[1:] == seg[:-1], False]  # row i overlaps row i+1
    sa = pd.Series(subj).astype(str) + "|" + y
    for seed in repeat_seeds():
        for strat in STRATEGIES:
            _, fl, desc = get_folds(strat, y, subj, seed=seed, segments=seg)
            fold_id = np.full(len(y), -1)
            for k, (tr, va) in enumerate(fl):
                if (fold_id[va] != -1).any():
                    raise SystemExit(f"{strat} seed {seed}: a row is validated twice")
                fold_id[va] = k
                in_tr = np.zeros(len(y), bool)
                in_tr[tr] = True
                prev_tr = np.r_[False, linked[:-1] & in_tr[:-1]]  # row i-1 overlaps row i and is in training
                next_tr = linked & np.r_[in_tr[1:], False]
                nb = (prev_tr | next_tr)[va]
                shared = set(subj[tr]) & set(subj[va])
                if strat == "participant_grouped" and shared:
                    raise SystemExit(f"seed {seed} fold {k}: grouped fold shares participants {shared}")
                if strat == "segment_grouped" and nb.any():
                    raise SystemExit(f"seed {seed} fold {k}: segment fold has an overlapping neighbour in training")
                counts = pd.Series(y[va]).value_counts().reindex(LABELS, fill_value=0)
                diag.append({"seed": seed, "strategy": strat, "fold": k, "splitter": desc, "n_train": len(tr),
                             "n_val": len(va), "n_val_subjects": len(set(subj[va])), "n_shared_subjects": len(shared),
                             "frac_val_with_overlapping_neighbour_in_train": nb.mean(),
                             "frac_val_subject_activity_in_train": sa[va].isin(set(sa[tr])).mean(),
                             **{f"n_{a}": int(counts[a]) for a in LABELS}})
                folds[(seed, strat, k)] = (tr, va)
            if (fold_id == -1).any():
                raise SystemExit(f"{strat} seed {seed}: some rows are never validated")
            assign.append(pd.DataFrame({"seed": seed, "strategy": strat, "train_row": np.arange(len(y)), "fold": fold_id}))
    return folds, pd.DataFrame(diag), pd.concat(assign)


def fit_predict(name, X, y, tr, va):
    return make_pipeline(name, **fixed_params(name)).fit(X[tr], y[tr]).predict(X[va])


def main():
    Xdf, y, subj = load_split("train")
    seg = load_segments(len(y))
    rep = out_dir("repeated")
    folds, diag, assign = build_folds(y, subj, seg)
    diag.to_csv(rep / "fold_diagnostics.csv", index=False)
    assign.to_csv(rep / "fold_assignments.csv.gz", index=False)

    for fs in os.environ.get("HAR_FEATURESETS", "full,no_gravity").split(","):
        cols = feature_columns(fs, Xdf.columns)
        X = Xdf[cols].to_numpy()
        tasks = [(seed, strat, k, m) for (seed, strat, k) in folds for m in MODELS[fs]]
        # RandomForest runs with n_jobs=1 here; parallelism is over (seed, protocol, fold, model) tasks.
        preds = Parallel(n_jobs=N_JOBS, verbose=5)(
            delayed(fit_predict)(m, X, y, *folds[(seed, strat, k)]) for seed, strat, k, m in tasks)

        fold_rows, oof = [], {}
        for (seed, strat, k, m), p in zip(tasks, preds):
            va = folds[(seed, strat, k)][1]
            fold_rows.append({"feature_set": fs, "seed": seed, "strategy": strat, "model": m, "fold": k, **scores(y[va], p)})
            oof.setdefault((seed, strat, m), np.empty(len(y), dtype=object))[va] = p
        rep_rows, long = [], []
        for (seed, strat, m), p in oof.items():
            rep_rows.append({"feature_set": fs, "seed": seed, "strategy": strat, "model": m, **scores(y, p)})
            long.append(pd.DataFrame({"seed": seed, "strategy": strat, "model": m, "train_row": np.arange(len(y)), "pred": p}))
        pd.DataFrame(fold_rows).to_csv(rep / f"fold_metrics_{fs}.csv", index=False)
        pd.DataFrame(rep_rows).to_csv(rep / f"repeat_metrics_{fs}.csv", index=False)
        pd.concat(long).to_csv(rep / f"oof_predictions_{fs}.csv.gz", index=False)
        save_json({"feature_set": fs, "n_features": len(cols), "dropped": sorted(set(Xdf.columns) - set(cols)),
                   "models": MODELS[fs], "fixed_configs": {m: fixed_params(m) for m in MODELS[fs]},
                   "seeds": repeat_seeds(), "strategies": STRATEGIES},
                  rep / f"meta_{fs}.json")
        r = pd.DataFrame(rep_rows).groupby(["model", "strategy"]).macro_f1.agg(["mean", "std", "min", "max"])
        print(f"[{fs}] repeat-level OOF macro-F1 across seeds\n{r.round(4)}", flush=True)


if __name__ == "__main__":
    with Timer("repeated_validation"):
        main()
