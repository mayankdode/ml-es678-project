"""Independent re-computation of the paper's audit numbers -> results/metrics/verification.json.

Re-derives the overlap segments from the raw Inertial Signals, re-checks every saved fold assignment, recomputes the
repeat-level metrics and protocol gaps from the saved out-of-fold predictions (unrounded), records which class the
majority baseline predicted in each fold, re-selects the gravity family from features.txt, and recomputes the test
metrics from the saved test predictions. It never fits a model and never scores the test set.
"""
import gzip
import re

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

from common import DATASET_DIR, LABELS, RESULTS, save_json

CHANNELS = [f"{s}_{a}" for s in ("body_acc", "body_gyro", "total_acc") for a in "xyz"]
MODELS = ["LogReg", "SVM_RBF", "RandomForest"]


def txt(path):
    return np.loadtxt(path, dtype=np.float64, ndmin=1)


def overlap(subj, y):
    d = DATASET_DIR / "train" / "Inertial Signals"
    S = np.stack([txt(d / f"{c}_train.txt") for c in CHANNELS])  # channels x windows x 128
    links = (S[:, :-1, 64:] == S[:, 1:, :64]).all(axis=(0, 2))
    heads = {}
    for i in range(S.shape[1]):
        heads.setdefault(S[:, i, :64].tobytes(), []).append(i)
    matches = [(i, j) for i in range(S.shape[1]) for j in heads.get(S[:, i, 64:].tobytes(), [])]
    seg = np.concatenate([[0], np.cumsum(~links)])
    lengths = np.bincount(seg)
    saved = pd.read_csv(RESULTS / "metadata" / "segments_train.csv")
    per_seg = pd.DataFrame({"s": seg, "subj": subj, "y": y}).groupby("s").nunique()
    return seg, {
        "n_windows": int(S.shape[1]), "n_adjacent_pairs": int(len(links)), "n_links": int(links.sum()),
        "n_links_crossing_subject_or_activity": int((links & ((subj[:-1] != subj[1:]) | (y[:-1] != y[1:]))).sum()),
        "n_exact_matches_any_pair": len(matches), "n_exact_matches_non_adjacent": sum(j != i + 1 for i, j in matches),
        "n_segments": int(len(lengths)), "segment_len_median": float(np.median(lengths)),
        "segment_len_min": int(lengths.min()), "segment_len_max": int(lengths.max()),
        "n_single_window_segments": int((lengths == 1).sum()),
        "segments_each_within_one_subject_and_activity": bool(((per_seg.subj == 1) & (per_seg.y == 1)).all()),
        "matches_saved_segment_table": bool((saved.segment.to_numpy() == seg).all()),
        "subject_id_changes_at_most_once_per_subject": bool(pd.Series(subj).ne(pd.Series(subj).shift()).groupby(subj).sum().max() == 1)}


def folds(subj, y, seg):
    fa = pd.read_csv(RESULTS / "repeated" / "fold_assignments.csv.gz")
    out = {}
    for strat, d in fa.groupby("strategy"):
        groups = {"participant_grouped": subj, "segment_grouped": seg}.get(strat)
        ok_cls, ok_grp, f_ovl, f_sa = True, True, [], []
        for _, s in d.groupby("seed"):
            fold = s.sort_values("train_row").fold.to_numpy()
            for k in np.unique(fold):
                va = fold == k
                ok_cls &= set(y[va]) == set(LABELS)
                if groups is not None:
                    ok_grp &= not set(groups[va]) & set(groups[~va])
                nb = np.zeros(len(y), bool)  # validation window whose overlapping neighbour is in training
                link = np.r_[seg[1:] == seg[:-1], False]
                nb[:-1] |= link[:-1] & ~va[1:]
                nb[1:] |= link[:-1] & ~va[:-1]
                f_ovl.append(nb[va].mean())
                train_sa = set(zip(subj[~va], y[~va]))
                f_sa.append(np.mean([(a, b) in train_sa for a, b in zip(subj[va], y[va])]))
        out[strat] = {"all_six_classes_every_fold": bool(ok_cls), "groups_disjoint": bool(ok_grp) if groups is not None else None,
                      "frac_val_overlapping_neighbour_in_train_mean": float(np.mean(f_ovl)),
                      "frac_val_same_subject_activity_in_train_mean": float(np.mean(f_sa))}
    return out


def metrics(y, p):
    return {"macro_f1": f1_score(y, p, average="macro", labels=LABELS, zero_division=0),
            "accuracy": accuracy_score(y, p), "balanced_accuracy": balanced_accuracy_score(y, p)}


def oof(y):
    pr = pd.read_csv(RESULTS / "repeated" / "oof_predictions_full.csv.gz")
    fa = pd.read_csv(RESULTS / "repeated" / "fold_assignments.csv.gz")
    saved = pd.read_csv(RESULTS / "repeated" / "repeat_metrics_full.csv").set_index(["seed", "strategy", "model"])
    rows, diff = [], 0.0
    for (seed, strat, mod), d in pr.groupby(["seed", "strategy", "model"]):
        p = d.sort_values("train_row").pred.to_numpy()
        m = metrics(y, p)
        diff = max(diff, max(abs(m[k] - saved.loc[(seed, strat, mod), k]) for k in m))
        rows.append({"seed": seed, "strategy": strat, "model": mod, **m})
    rep = pd.DataFrame(rows).set_index(["model", "strategy", "seed"]).macro_f1
    dummy = pr[pr.model == "Dummy"].merge(fa, on=["seed", "strategy", "train_row"])
    dummy_counts = {s: d.groupby(["seed", "fold"]).pred.first().value_counts().to_dict() for s, d in dummy.groupby("strategy")}
    gaps, rep = {}, rep.sort_index()
    for mod in MODELS:
        r, s, p = (rep.loc[(mod, k)] for k in ("random_window", "segment_grouped", "participant_grouped"))
        g, rs = r - p, r - s
        gaps[mod] = {"gap_mean": g.mean(), "gap_seed_sd": g.std(ddof=1), "gap_min": g.min(), "gap_max": g.max(),
                     "n_seeds_positive": int((g > 0).sum()), "random_minus_segment_mean": rs.mean(),
                     "segment_minus_participant_mean": (s - p).mean(),
                     "overlap_share_of_gap_pct": 100 * rs.mean() / g.mean(),
                     "overlap_share_per_seed_pct_min": (100 * rs / g).min(), "overlap_share_per_seed_pct_max": (100 * rs / g).max()}
    return diff, dummy_counts, gaps


def gravity():
    names = [l.split(maxsplit=1)[1].strip() for l in (DATASET_DIR / "features.txt").read_text().splitlines()]
    sel = [i for i, n in enumerate(names) if "gravity" in n.lower()]
    saved = pd.read_csv(RESULTS / "metadata" / "gravity_features.csv").feature
    saved_idx = sorted(int(re.match(r"f(\d+)_", f).group(1)) - 1 for f in saved)
    return {"n_selected": len(sel), "n_remaining": len(names) - len(sel),
            "n_tGravityAcc": sum(names[i].startswith("tGravityAcc-") for i in sel),
            "n_tGravityAccMag": sum(names[i].startswith("tGravityAccMag-") for i in sel),
            "n_angle": sum(names[i].startswith("angle(") for i in sel),
            "angle_features_total": sum(n.startswith("angle(") for n in names), "matches_saved_list": saved_idx == sel}


def test_check():
    t = pd.read_csv(RESULTS / "metrics" / "test_predictions.csv")
    raw_subj = txt(DATASET_DIR / "test" / "subject_test.txt").astype(int)
    m = metrics(t.y_true, t.y_pred)
    saved = pd.read_json(RESULTS / "metrics" / "test_metrics.json", typ="series")
    train_subj = set(txt(DATASET_DIR / "train" / "subject_train.txt").astype(int))
    prf = precision_recall_fscore_support(t.y_true, t.y_pred, labels=LABELS, zero_division=0)
    per_class = {lab: {"precision": prf[0][i], "recall": prf[1][i], "f1": prf[2][i]} for i, lab in enumerate(LABELS)}
    return {**m, "per_class": per_class,
            "max_abs_diff_per_class_vs_test_metrics_json": max(abs(per_class[l][k] - saved[f"{k}_{l}"]) for l in LABELS
                                                               for k in ("precision", "recall", "f1")),
            "n": len(t), "confusion_total": int(confusion_matrix(t.y_true, t.y_pred, labels=LABELS).sum()),
            "subjects": sorted(int(s) for s in set(t.subject)), "subjects_match_raw": bool((t.subject.to_numpy() == raw_subj).all()),
            "train_test_subjects_disjoint": not train_subj & set(raw_subj),
            "max_abs_diff_vs_test_metrics_json": max(abs(m[k] - saved[k]) for k in m)}


def main():
    subj = txt(DATASET_DIR / "train" / "subject_train.txt").astype(int)
    amap = dict(l.split() for l in (DATASET_DIR / "activity_labels.txt").read_text().splitlines())
    y = np.array([amap[str(int(v))] for v in txt(DATASET_DIR / "train" / "y_train.txt")])
    seg, ovl = overlap(subj, y)
    diff, dummy, gaps = oof(y)
    shares = pd.Series(y).value_counts(normalize=True).to_dict()
    report = {"overlap": ovl, "folds": folds(subj, y, seg), "dummy_majority_class_counts": dummy, "gravity": gravity(),
              "max_abs_diff_vs_saved_repeat_metrics": diff, "gaps_full": gaps}
    if (RESULTS / "metrics" / "test_predictions.csv").exists():
        report["test_from_saved_predictions"] = test_check()
    report["class_shares_train"] = shares
    save_json(report, RESULTS / "metrics" / "verification.json")
    print(f"verification -> {RESULTS / 'metrics' / 'verification.json'}")


if __name__ == "__main__":
    main()
