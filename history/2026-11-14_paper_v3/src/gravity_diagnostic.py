"""Held-out permutation check of the gravity feature family (seed-42 folds, full-feature fixed models).

Each model is fitted on the training folds with all features. On each validation fold, the gravity columns are
permuted jointly across rows (keeping their within-window relations) and the model predicts again. The drop in
pooled OOF scores shows how much the fitted model relies on that family. This complements the retraining
ablation in repeated_validation.py; it is not a causal test.
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from common import (N_JOBS, SEED, Timer, fixed_params, gravity_columns, make_pipeline, out_dir, permutation_repeats,
                    scores)
from load_data import load_split
from repeated_validation import load_segments
from splits import get_folds

MODELS = ["LogReg", "SVM_RBF", "RandomForest"]
STRATS = ["random_window", "participant_grouped"]


def run(name, X, y, tr, va, gcols, n_rep):
    model = make_pipeline(name, **fixed_params(name)).fit(X[tr], y[tr])
    out = [model.predict(X[va])]
    rng = np.random.default_rng(SEED)
    for _ in range(n_rep):
        Xp = X[va].copy()
        Xp[:, gcols] = Xp[rng.permutation(len(va))][:, gcols]
        out.append(model.predict(Xp))
    return out


def main():
    Xdf, y, subj = load_split("train")
    seg = load_segments(len(y))
    gcols = [Xdf.columns.get_loc(c) for c in gravity_columns(Xdf.columns)]
    X, n_rep = Xdf.to_numpy(), permutation_repeats()
    tasks = [(s, m, k, tr, va) for s in STRATS
             for k, (tr, va) in enumerate(get_folds(s, y, subj, seed=SEED, segments=seg)[1]) for m in MODELS]
    res = Parallel(n_jobs=N_JOBS)(delayed(run)(m, X, y, tr, va, gcols, n_rep) for s, m, k, tr, va in tasks)
    oof = {}
    for (s, m, k, tr, va), preds in zip(tasks, res):
        for r, p in enumerate(preds):  # r = 0 is the unpermuted prediction
            oof.setdefault((s, m, r), np.empty(len(y), dtype=object))[va] = p
    rows = [{"strategy": s, "model": m, "repeat": r, "permuted": r > 0, **scores(y, p)} for (s, m, r), p in oof.items()]
    df = pd.DataFrame(rows)
    df.to_csv(out_dir("repeated") / "gravity_permutation.csv", index=False)
    print(df.groupby(["strategy", "model", "permuted"])[["macro_f1", "f1_SITTING", "f1_STANDING"]].mean().round(4))


if __name__ == "__main__":
    with Timer("gravity_diagnostic"):
        main()
