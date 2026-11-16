"""Pick the final model from the saved grouped-CV tuning results (highest mean macro-F1).

A bounded follow-up grid (tuning_cv_results_<family>_round2.csv, same folds) is merged with the first round first.
"""
import ast
import json

import numpy as np
import pandas as pd

from common import CONFIG, LABELS, fixed_params, out_dir, save_json
from tune_final_model import FAMILIES


def _size(v, d=561):
    """Order max_features values by the number of features they mean (log2 d < sqrt d < 0.5 d); others as is."""
    return {"log2": np.log2(d), "sqrt": np.sqrt(d)}.get(v, v * d if isinstance(v, float) and v <= 1 else v)


def merge_extension(metrics, name):
    """Merge round-2 CV results into tuning_cv_results_<name>.csv's best; keeps the round-1 summary as *_round1.json."""
    ext = metrics / f"tuning_cv_results_{name}_round2.csv"
    if not ext.exists():
        return
    first = json.loads((metrics / f"tuning_best_{name}.json").read_text())
    if "rounds" in first:  # already merged
        return
    save_json(first, metrics / f"tuning_best_{name}_round1.json")
    res = pd.concat([pd.read_csv(metrics / f"tuning_cv_results_{name}.csv").assign(round=1),
                     pd.read_csv(ext).assign(round=2)], ignore_index=True)
    best = res.loc[res.mean_test_score.idxmax()]
    params = ast.literal_eval(best.params)
    values = {}
    for p in res.params.map(ast.literal_eval):
        for k, v in p.items():
            values.setdefault(k, set()).add(v)
    space = {k: sorted(vs, key=_size) for k, vs in values.items()}
    edge = [k.removeprefix("clf__") for k, vs in space.items() if len(vs) > 1 and params[k] in (vs[0], vs[-1])]
    save_json({"model": name, "search_space": space, "rounds": {"1": first["search_space"],
               "2": CONFIG["search_space_extensions"][f"{name}_round2"]["grid"]},
               "best_params": params, "best_round": int(best["round"]), "best_macro_f1_mean": best.mean_test_score,
               "best_macro_f1_sd": best[[f"split{k}_test_score" for k in range(5)]].astype(float).std(),
               "n_candidates": len(res), "at_grid_boundary": edge, "cv": first["cv"]},
              metrics / f"tuning_best_{name}.json")
    res.to_csv(metrics / f"tuning_cv_results_{name}_merged.csv", index=False)


def main():
    metrics = out_dir("metrics")
    for n in FAMILIES:
        merge_extension(metrics, n)
    best = [json.loads((metrics / f"tuning_best_{n}.json").read_text()) for n in FAMILIES]
    s = pd.DataFrame(best).sort_values("best_macro_f1_mean", ascending=False)
    s.assign(best_params=s.best_params.astype(str), search_space=s.search_space.astype(str),
             at_grid_boundary=s.at_grid_boundary.astype(str)).to_csv(out_dir("tables") / "tuning_summary.csv", index=False)
    top = s.iloc[0]
    params = {**fixed_params(top.model), **{k.removeprefix("clf__"): v for k, v in top.best_params.items()}}
    save_json({"selected_model": top.model, "selected_params": params,
               "grouped_cv_macro_f1_mean": top.best_macro_f1_mean, "grouped_cv_macro_f1_sd": top.best_macro_f1_sd,
               "at_grid_boundary": top.at_grid_boundary, "runner_up": s.iloc[1].model,
               "margin_over_runner_up": top.best_macro_f1_mean - s.iloc[1].best_macro_f1_mean,
               "cv": top.cv, "scoring": "f1_macro", "labels": LABELS,
               "note": "Selection used participant-grouped CV on the official training partition only. "
                       "This CV score is optimistically biased by selection."},
              metrics / "selected_model.json")
    print(s[["model", "best_params", "best_macro_f1_mean", "best_macro_f1_sd", "at_grid_boundary"]].to_string(index=False))
    print(f"selected: {top.model} {params}")


if __name__ == "__main__":
    main()
