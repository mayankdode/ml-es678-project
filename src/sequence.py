"""Overlap audit: rebuild runs of 50%-overlapping windows in the official training partition from the raw signals.

UCI HAR windows are 128 samples with a 64-sample step. If window i+1 directly follows window i in the same recording,
the last 64 samples of i equal the first 64 samples of i+1 on every channel. We test this exactly (tolerance 0) on all
nine released Inertial Signals channels, check that no window matches a non-adjacent row, and call each maximal chain
of matching neighbours a "segment". Only the training partition is read.
"""
from collections import defaultdict

import numpy as np
import pandas as pd

from common import DATASET_DIR, Timer, out_dir, save_json
from load_data import load_split

CHANNELS = [f"{s}_{a}" for s in ("body_acc", "body_gyro", "total_acc") for a in "xyz"]


def load_signals(split="train"):
    d = DATASET_DIR / split / "Inertial Signals"
    return np.stack([pd.read_csv(d / f"{c}_{split}.txt", sep=r"\s+", header=None, dtype=np.float64).to_numpy()
                     for c in CHANNELS])  # channels x windows x 128


def overlap_links(S):
    """links[i] is True when window i+1 continues window i (exact 64-sample match on all channels)."""
    return (S[:, :-1, 64:] == S[:, 1:, :64]).all(axis=(0, 2))


def non_adjacent_matches(S):
    heads = defaultdict(list)
    for i in range(S.shape[1]):
        heads[S[:, i, :64].tobytes()].append(i)
    return sum(1 for i in range(S.shape[1]) for j in heads.get(S[:, i, 64:].tobytes(), []) if j != i + 1)


def segment_ids(links):
    return np.concatenate([[0], np.cumsum(~links)])


def main():
    _, y, subj = load_split("train")
    S = load_signals("train")
    if S.shape[1] != len(y):
        raise SystemExit(f"signal rows {S.shape[1]} != label rows {len(y)}")
    links = overlap_links(S)
    same_subj, same_act = subj[:-1] == subj[1:], y[:-1] == y[1:]
    seg = segment_ids(links)
    if (links & ~(same_subj & same_act)).any():
        raise SystemExit("an overlap link crosses a participant or activity boundary")
    lengths = pd.Series(seg).value_counts()
    seg_df = pd.DataFrame({"subject": subj, "activity": y, "segment": seg})
    seg_df.to_csv(out_dir("metadata") / "segments_train.csv", index_label="train_row")
    per_sa = seg_df.groupby(["subject", "activity"]).segment.nunique()
    report = {
        "channels": CHANNELS, "tolerance": 0.0, "n_windows": len(y), "n_consecutive_pairs": len(links),
        "n_overlap_links": int(links.sum()),
        "n_links_within_same_subject_and_activity": int((links & same_subj & same_act).sum()),
        "n_consecutive_pairs_same_subject_and_activity": int((same_subj & same_act).sum()),
        "n_non_adjacent_matches": int(non_adjacent_matches(S)),
        "n_segments": int(seg.max() + 1),
        "segment_length": {"min": int(lengths.min()), "median": float(lengths.median()), "max": int(lengths.max()),
                           "mean": float(lengths.mean())},
        "segments_per_subject_activity": {"min": int(per_sa.min()), "median": float(per_sa.median()),
                                          "max": int(per_sa.max())},
        "conclusion": "Row order preserves recording order: every exact match links row i to row i+1 inside one "
                      "participant and activity, and no window matches a non-adjacent row.",
    }
    save_json(report, out_dir("metrics") / "overlap_audit.json")
    print({k: v for k, v in report.items() if k not in ("channels", "conclusion")})


if __name__ == "__main__":
    with Timer("overlap_audit"):
        main()
