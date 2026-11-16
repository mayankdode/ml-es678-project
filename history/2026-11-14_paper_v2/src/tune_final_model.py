"""Grid search with participant-grouped CV (seed 42) on the official training partition only.

Env HAR_TUNE_FAMILIES (default all three) lets the families run as separate shards; src/select_model.py
then picks the final model from the saved per-family results.
"""
import os

import pandas as pd
from sklearn.model_selection import GridSearchCV

from common import N_JOBS, SEED, Timer, fixed_params, make_pipeline, out_dir, save_json, search_space
from load_data import load_split
from splits import get_folds

FAMILIES = ["LogReg", "SVM_RBF", "RandomForest"]


def at_boundary(space, best):
    """Numeric hyperparameters whose selected value is the smallest or largest value searched."""
    out = []
    for k, vals in space.items():
        nums = [v for v in vals if isinstance(v, (int, float))]
        if len(nums) > 1 and best[k] in (min(nums), max(nums)):
            out.append(k.removeprefix("clf__"))
    return out


def main():
    X, y, subj = load_split("train")
    splitter, _, desc = get_folds("participant_grouped", y, subj, seed=SEED)
    metrics = out_dir("metrics")
    for name in os.environ.get("HAR_TUNE_FAMILIES", ",".join(FAMILIES)).split(","):
        space = search_space(name)
        search = GridSearchCV(make_pipeline(name, **fixed_params(name)), space, cv=splitter,
                              scoring="f1_macro", n_jobs=N_JOBS, refit=False, verbose=1)
        with Timer(f"tune_{name}"):
            search.fit(X, y, groups=subj)
        res = pd.DataFrame(search.cv_results_)
        res.drop(columns=[c for c in res if c.startswith("param_")]).to_csv(metrics / f"tuning_cv_results_{name}.csv", index=False)
        best = res.loc[search.best_index_]
        save_json({"model": name, "search_space": space, "best_params": search.best_params_,
                   "best_macro_f1_mean": best.mean_test_score,
                   "best_macro_f1_sd": best[[f"split{k}_test_score" for k in range(5)]].astype(float).std(),
                   "n_candidates": len(res), "at_grid_boundary": at_boundary(space, search.best_params_), "cv": desc},
                  metrics / f"tuning_best_{name}.json")
        print(f"{name}: {search.best_params_} macroF1={best.mean_test_score:.4f}", flush=True)


if __name__ == "__main__":
    with Timer("tune_final_model"):
        main()
