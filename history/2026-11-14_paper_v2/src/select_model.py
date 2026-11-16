"""Pick the final model from the saved grouped-CV tuning results (highest mean macro-F1)."""
import json

import pandas as pd

from common import LABELS, fixed_params, out_dir, save_json
from tune_final_model import FAMILIES


def main():
    metrics = out_dir("metrics")
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
