"""Write report/numbers.tex: every number quoted in the paper, as LaTeX macros read from saved results."""
import json
from pathlib import Path

import pandas as pd

from common import LABELS, RESULTS, ROOT

SHORT = {"WALKING": "Walk", "WALKING_UPSTAIRS": "Upstairs", "WALKING_DOWNSTAIRS": "Downstairs",
         "SITTING": "Sit", "STANDING": "Stand", "LAYING": "Lay"}
MODEL_KEY = {"Dummy": "Dummy", "LogReg": "LR", "SVM_RBF": "SVM", "RandomForest": "RF"}
STRAT_KEY = {"random_window": "R", "segment_grouped": "S", "participant_grouped": "P"}
HISTORY = ROOT / "history"


def f3(x):
    return f"{x:.3f}"


def z3(x):
    """Three decimals without the leading zero (for SD columns)."""
    return f"{x:.3f}".replace("0.", ".", 1)


def load(p):
    return json.loads(Path(p).read_text())


def main():
    m, t, rep = RESULTS / "metrics", RESULTS / "tables", RESULTS / "repeated"
    audit, ovl, hi = load(m / "data_audit.json"), load(m / "overlap_audit.json"), load(m / "repeated_highlights.json")
    summ = pd.read_csv(t / "repeated_summary.csv")
    gaps = pd.read_csv(t / "repeated_gaps.csv")
    pc = pd.read_csv(t / "repeated_per_class.csv")
    abl = pd.read_csv(t / "gravity_ablation.csv").set_index(["model", "score"])
    diag = pd.read_csv(t / "fold_diagnostics_summary.csv").set_index("strategy")
    perm = pd.read_csv(rep / "gravity_permutation.csv")
    tun = pd.read_csv(t / "tuning_summary.csv").set_index("model")
    sel = load(m / "selected_model.json")

    mac = {"dupnames": str(audit["duplicate_feature_names"]["n_distinct"]),
           "majshare": f"{100 * audit['majority_class_train']['proportion']:.1f}",
           "majlabel": audit["majority_class_train"]["label"].title(),
           "sdmin": f"{audit['feature_range_train']['feature_sd_min']:.2f}",
           "sdmax": f"{audit['feature_range_train']['feature_sd_max']:.2f}",
           "ngrav": str(audit["n_gravity_features"]), "nkeep": str(561 - audit["n_gravity_features"]),
           "nlinks": f"{ovl['n_overlap_links']:,}", "npairs": f"{ovl['n_consecutive_pairs']:,}",
           "nseg": str(ovl["n_segments"]), "seglenmed": f"{ovl['segment_length']['median']:g}",
           "seglenmax": str(ovl["segment_length"]["max"]),
           "segperminsa": str(ovl["segments_per_subject_activity"]["min"]),
           "segpermedsa": f"{ovl['segments_per_subject_activity']['median']:g}",
           "nseeds": str(hi["n_seeds"]),
           "rfgaplargest": str(hi["rf_gap_largest_in_n_seeds"]),
           "overlapR": f"{100 * diag.loc['random_window', 'frac_overlap_neighbour_mean']:.1f}",
           "saS": f"{100 * diag.loc['segment_grouped', 'frac_subject_activity_in_train_mean']:.1f}",
           "grpvalsubjmin": str(diag.loc["participant_grouped", "n_val_subjects_min"]),
           "grpvalsubjmax": str(diag.loc["participant_grouped", "n_val_subjects_max"])}

    for _, r in summ[summ.feature_set == "full"].iterrows():
        k = MODEL_KEY[r.model] + STRAT_KEY[r.strategy]
        mac |= {f"mf{k}": f3(r.macro_f1_mean), f"mfsd{k}": f3(r.macro_f1_std), f"acc{k}": f3(r.accuracy_mean),
                f"ba{k}": f3(r.balanced_accuracy_mean), f"fsd{k}": f3(r.fold_sd_macro_f1_mean),
                f"mfsdz{k}": z3(r.macro_f1_std), f"fsdz{k}": z3(r.fold_sd_macro_f1_mean)}
    full = summ[(summ.feature_set == "full") & (summ.model != "Dummy")]
    mac |= {"fsdRmax": f3(full[full.strategy == "random_window"].fold_sd_macro_f1_mean.max()),
            "fsdPmin": f3(full[full.strategy == "participant_grouped"].fold_sd_macro_f1_mean.min()),
            "fsdPmax": f3(full[full.strategy == "participant_grouped"].fold_sd_macro_f1_mean.max()),
            "seedsdPmax": f3(full[full.strategy == "participant_grouped"].macro_f1_std.max())}

    g = gaps[(gaps.feature_set == "full") & (gaps.metric == "macro_f1")].set_index(["model", "comparison"])
    share = []
    for mod in ["LogReg", "SVM_RBF", "RandomForest"]:
        k = MODEL_KEY[mod]
        rp, rs, sp = (g.loc[(mod, c)] for c in ("random_minus_participant", "random_minus_segment", "segment_minus_participant"))
        mac |= {f"gap{k}": f3(rp["mean"]), f"gapsd{k}": f3(rp["sd"]), f"gapmin{k}": f3(rp["min"]), f"gapmax{k}": f3(rp["max"]),
                f"gapRS{k}": f3(rs["mean"]), f"gapSP{k}": f3(sp["mean"]), f"gappos{k}": str(int(rp["n_positive"]))}
        share.append(rs["mean"] / rp["mean"])
    means = {mod: g.loc[(mod, "random_minus_participant"), "mean"] for mod in ["LogReg", "SVM_RBF", "RandomForest"]}
    mac |= {"overlapsharemin": f"{100 * min(share):.0f}", "overlapsharemax": f"{100 * max(share):.0f}",
            "gapmodelmin": f3(min(means.values())), "gapmodelmax": f3(max(means.values())),
            "rfexcess": f3(means["RandomForest"] - max(means["LogReg"], means["SVM_RBF"]))}
    # Per-seed protocol-based share of the gap removed by segment grouping (descriptive, from unrounded values).
    rm = pd.read_csv(rep / "repeat_metrics_full.csv").pivot_table(index=["model", "seed"], columns="strategy", values="macro_f1")
    rm = rm.loc[["LogReg", "SVM_RBF", "RandomForest"]]
    seed_share = 100 * (rm.random_window - rm.segment_grouped) / (rm.random_window - rm.participant_grouped)
    gmin = (rm.random_window - rm.participant_grouped).min()
    mac |= {"overlapshareseedmin": f"{seed_share.min():.0f}", "overlapshareseedmax": f"{seed_share.max():.0f}",
            "gapallmin": f3(gmin), "gapallmax": f3((rm.random_window - rm.participant_grouped).max())}
    ver = load(m / "verification.json")
    dc = ver["dummy_majority_class_counts"]["segment_grouped"]
    mac |= {"dummySegLay": str(dc.get("LAYING", 0)), "dummySegStand": str(dc.get("STANDING", 0)),
            "dummySegOther": str(sum(v for k, v in dc.items() if k not in ("LAYING", "STANDING"))),
            "dummySegFolds": str(sum(dc.values())), "standshare": f"{100 * ver['class_shares_train']['STANDING']:.1f}",
            "nangle": str(ver["gravity"]["n_angle"]), "ntgrav": str(ver["gravity"]["n_tGravityAcc"]),
            "ntgravmag": str(ver["gravity"]["n_tGravityAccMag"]), "nsingleseg": str(ver["overlap"]["n_single_window_segments"]),
            "fsdallmin": f3(full[full.strategy != "random_window"].fold_sd_macro_f1_mean.min())}
    for mod in ["LogReg", "SVM_RBF", "RandomForest"]:
        mac[f"wins{MODEL_KEY[mod]}"] = str(hi["grouped_wins"].get(mod, 0))

    d = pc[(pc.feature_set == "full") & (pc.comparison == "random_minus_participant")].set_index(["model", "activity"])
    for mod in ["LogReg", "SVM_RBF", "RandomForest"]:
        for lab in LABELS:
            mac[f"drop{MODEL_KEY[mod]}{SHORT[lab]}"] = f3(d.loc[(mod, lab), "mean"])
        for lab, sk in (("macro_f1", "Macro"), ("f1_SITTING", "Sit"), ("f1_STANDING", "Stand"), ("f1_LAYING", "Lay")):
            a = abl.loc[(mod, lab)]
            mac |= {f"abl{MODEL_KEY[mod]}{sk}full": f3(a.gap_full), f"abl{MODEL_KEY[mod]}{sk}ng": f3(a.gap_no_gravity),
                    f"abl{MODEL_KEY[mod]}{sk}n": str(int(a.n_seeds_gap_smaller_without_gravity)),
                    f"abl{MODEL_KEY[mod]}{sk}Pfull": f3(a.participant_full), f"abl{MODEL_KEY[mod]}{sk}Png": f3(a.participant_no_gravity)}
        p = perm[(perm.strategy == "participant_grouped") & (perm.model == mod)]
        mac |= {f"perm{MODEL_KEY[mod]}base": f3(p[~p.permuted].macro_f1.mean()), f"perm{MODEL_KEY[mod]}": f3(p[p.permuted].macro_f1.mean()),
                f"perm{MODEL_KEY[mod]}Sit": f3(p[p.permuted].f1_SITTING.mean())}
        a = abl.loc[(mod, "f1_SITTING")]
        mac |= {f"abl{MODEL_KEY[mod]}SitRfull": f3(a.random_full), f"abl{MODEL_KEY[mod]}SitRng": f3(a.random_no_gravity)}
    mac["nperm"] = str(int(perm.repeat.max()))
    cd = pd.read_csv(t / "class_distribution.csv")
    ts = cd[cd.split == "test"].proportion
    mac |= {"testsharemin": f"{100 * ts.min():.0f}", "testsharemax": f"{100 * ts.max():.0f}"}

    for mod in ["LogReg", "SVM_RBF", "RandomForest"]:
        r = tun.loc[mod]
        mac |= {f"tune{MODEL_KEY[mod]}": f3(r.best_macro_f1_mean), f"tunesd{MODEL_KEY[mod]}": f3(r.best_macro_f1_sd)}
    bp = {k: eval(v) for k, v in tun.best_params.items()}
    mac |= {"tuneLRC": f"{bp['LogReg']['clf__C']:g}", "tuneSVMC": f"{bp['SVM_RBF']['clf__C']:g}",
            "tuneSVMgamma": str(bp["SVM_RBF"]["clf__gamma"]),
            "tuneRFtrees": str(bp["RandomForest"]["clf__n_estimators"]),
            "tuneRFmf": {"sqrt": "$\\sqrt{d}$", "log2": "$\\log_2 d$"}.get(str(bp["RandomForest"]["clf__max_features"]),
                                                                       str(bp["RandomForest"]["clf__max_features"])),
            "tuneRFleaf": str(bp["RandomForest"]["clf__min_samples_leaf"]),
            "selmodel": {"LogReg": "LR", "SVM_RBF": "SVM", "RandomForest": "RF"}[sel["selected_model"]],
            "selcv": f3(sel["grouped_cv_macro_f1_mean"]), "selcvsd": f3(sel["grouped_cv_macro_f1_sd"]),
            "selmargin": f3(sel["margin_over_runner_up"]),
            "selboundary": ", ".join(sel["at_grid_boundary"]) or "none",
            "tuneRFboundary": ", ".join(eval(tun.loc["RandomForest", "at_grid_boundary"])).replace("_", "\\_") or "none"}

    r1 = m / "tuning_best_RandomForest_round1.json"
    if r1.exists():
        r1 = load(r1)
        mac |= {"tuneRFone": f3(r1["best_macro_f1_mean"]), "tuneRFoneleaf": str(r1["best_params"]["clf__min_samples_leaf"])}
    mac["tuneRFn"] = str(load(m / "tuning_best_RandomForest.json")["n_candidates"])

    # Earlier, disclosed accesses to the official test set (archived under history/).
    smoke = load(HISTORY / "2026-11-12_local_smoke_run" / "metrics" / "test_metrics.json")
    run1 = load(HISTORY / "2026-11-12_kaggle_run1" / "results" / "metrics" / "test_metrics.json")
    mac |= {"smoketestf": f3(smoke["macro_f1"]), "smokeC": f"{smoke['params']['C']:g}",
            "runonetestf": f3(run1["macro_f1"]), "runoneC": f"{run1['params']['C']:g}"}

    if (m / "test_metrics.json").exists():
        test, err = load(m / "test_metrics.json"), load(m / "error_analysis.json")
        subj = pd.read_csv(t / "test_per_subject.csv")
        conf = pd.read_csv(t / "test_top_confusions.csv")
        cm = pd.read_csv(m / "test_confusion_counts.csv", index_col=0)
        b = err["error_breakdown_static_dynamic"]
        mac |= {"testf": f3(test["macro_f1"]), "testacc": f3(test["accuracy"]), "testbal": f3(test["balanced_accuracy"]),
                "testminuscv": f"{err['test_minus_grouped_cv']:+.3f}", "testcorrect": f"{int(pd.Series(cm.values.diagonal()).sum()):,}",
                "testtotal": f"{int(cm.values.sum()):,}",
                "subjmin": f3(subj.macro_f1.min()), "subjmed": f3(subj.macro_f1.median()), "subjmax": f3(subj.macro_f1.max()),
                "subjminid": str(int(subj.loc[subj.macro_f1.idxmin(), "subject"])),
                "nerr": str(b["n_errors"]), "nwithinstatic": str(b.get("static->static", 0)),
                "nwithindyn": str(b.get("dynamic->dynamic", 0)),
                "ncross": str(b.get("static->dynamic", 0) + b.get("dynamic->static", 0))}
        for i, r in conf.head(3).iterrows():
            mac |= {f"conf{'ABC'[i]}n": str(r["count"]), f"conf{'ABC'[i]}pct": f"{r.pct_of_true_class:.1f}"}
        for lab in LABELS:
            for s, tag in (("precision", "prec"), ("recall", "rec"), ("f1", "fone")):
                mac[f"test{tag}{SHORT[lab]}"] = f3(test[f"{s}_{lab}"])

    metas = [load(p) for p in sorted((RESULTS / "logs").glob("run_metadata_*.json"))]
    dev = [x for x in metas if x["stage"] != "final_test" and not x["smoke_mode"]]
    if dev:
        mac |= {"pyver": dev[0]["python"].split()[0], "sklearn": dev[0]["packages"]["scikit-learn"],
                "ncpu": str(dev[0]["cpu_count"]),
                "runhours": f"{max(sum(v for k, v in x['timings_s'].items() if k in ('repeated_validation', 'gravity_diagnostic', 'tune_final_model', 'data_audit', 'overlap_audit')) for x in dev) / 3600:.1f}"}

    lines = ["% Auto-generated by src/paper_numbers.py from results/ and history/; do not edit by hand."]
    lines += [f"\\newcommand{{\\N{k}}}{{{v}}}" for k, v in mac.items()]
    bad = [k for k in mac if not k.isalpha()]
    if bad:
        raise SystemExit(f"macro names must be letters only: {bad}")
    (ROOT / "report" / "numbers.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {len(mac)} macros")


if __name__ == "__main__":
    main()
