"""Fold definitions for the three validation protocols (official training partition only).

random_window        StratifiedKFold over windows: participants and overlapping neighbours are shared.
segment_grouped      StratifiedGroupKFold over overlap segments: participants are shared, overlapping
                     neighbours are not (every segment lies in one fold).
participant_grouped  StratifiedGroupKFold over participants: neither is shared.
"""
import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold

from common import CONFIG, LABELS, SEED

STRATEGIES = CONFIG["strategies"]


def _grouped(y, groups, seed):
    """StratifiedGroupKFold; falls back to GroupKFold only if a validation fold would lack a class."""
    k = CONFIG["n_splits"]
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    folds = list(sgkf.split(np.zeros(len(y)), y, groups))
    if all(set(y[va]) == set(LABELS) for _, va in folds):
        return sgkf, folds, f"StratifiedGroupKFold(n_splits={k}, shuffle=True, random_state={seed})"
    gkf = GroupKFold(n_splits=k)
    return gkf, list(gkf.split(np.zeros(len(y)), y, groups)), f"GroupKFold(n_splits={k}) [fallback: class missing in a fold]"


def get_folds(strategy, y, subj, seed=SEED, segments=None):
    if strategy == "random_window":
        k = CONFIG["n_splits"]
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        return skf, list(skf.split(np.zeros(len(y)), y)), f"StratifiedKFold(n_splits={k}, shuffle=True, random_state={seed})"
    if strategy == "segment_grouped":
        if segments is None:
            raise ValueError("segment_grouped needs segment ids (run src/sequence.py first)")
        return _grouped(y, segments, seed)
    if strategy == "participant_grouped":
        return _grouped(y, subj, seed)
    raise ValueError(strategy)
