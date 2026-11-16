import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import CONFIG, LABELS, gravity_columns  # noqa: E402
from load_data import load_split  # noqa: E402
from sequence import load_signals, overlap_links, segment_ids  # noqa: E402
from splits import get_folds  # noqa: E402


@pytest.fixture(scope="module")
def data():
    return load_split("train"), load_split("test")


def test_shapes_and_alignment(data):
    (Xtr, ytr, str_), (Xte, yte, ste) = data
    exp = CONFIG["expected"]
    assert Xtr.shape == (exp["n_train"], exp["n_features"]) and len(ytr) == len(str_) == exp["n_train"]
    assert Xte.shape == (exp["n_test"], exp["n_features"]) and len(yte) == len(ste) == exp["n_test"]


def test_labels(data):
    (_, ytr, _), (_, yte, _) = data
    assert set(ytr) == set(LABELS) and set(yte) == set(LABELS)


def test_train_test_subjects_disjoint(data):
    (_, _, str_), (_, _, ste) = data
    assert not set(str_) & set(ste)
    assert len(set(str_)) == CONFIG["expected"]["n_train_subjects"]
    assert len(set(ste)) == CONFIG["expected"]["n_test_subjects"]


def test_subject_not_a_feature(data):
    (Xtr, _, _), _ = data
    assert not any("subject" in c.lower() for c in Xtr.columns)


@pytest.fixture(scope="module")
def segments(data):
    (_, y, subj), _ = data
    links = overlap_links(load_signals("train"))
    # every exact 64-sample overlap joins two rows of the same participant and activity
    assert (subj[:-1][links] == subj[1:][links]).all() and (y[:-1][links] == y[1:][links]).all()
    return segment_ids(links)


@pytest.mark.parametrize("seed", CONFIG["repeat_seeds"][:3])
def test_grouped_folds_subject_disjoint_and_cover_all_classes(data, seed):
    (_, y, subj), _ = data
    _, folds, _ = get_folds("participant_grouped", y, subj, seed=seed)
    seen = np.zeros(len(y), dtype=int)
    for tr, va in folds:
        assert not set(subj[tr]) & set(subj[va])
        assert set(y[va]) == set(LABELS)
        seen[va] += 1
    assert (seen == 1).all()


def test_segment_folds_keep_overlapping_windows_together(data, segments):
    (_, y, subj), _ = data
    _, folds, _ = get_folds("segment_grouped", y, subj, segments=segments)
    for tr, va in folds:
        assert not set(segments[tr]) & set(segments[va])
        assert set(y[va]) == set(LABELS)


def test_gravity_family(data):
    (Xtr, _, _), _ = data
    g = gravity_columns(Xtr.columns)
    assert len(g) == 60 and all("gravity" in c.lower() for c in g)


def test_random_folds_partition_rows(data):
    (_, y, subj), _ = data
    _, folds, _ = get_folds("random_window", y, subj)
    seen = np.zeros(len(y), dtype=int)
    for _, va in folds:
        seen[va] += 1
    assert (seen == 1).all()
