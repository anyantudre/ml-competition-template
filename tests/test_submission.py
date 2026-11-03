import numpy as np
import pandas as pd

from mlt.config import load_config
from mlt.data import TargetEncoder
from mlt.submission import build_submission, score_submission, validate_submission


def _cfg(**sub):
    return load_config(data={"data": {"train": "x.csv", "target": "y", "id_col": "id"}, "submission": sub})


def _enc(task, labels):
    return TargetEncoder(task).fit(pd.DataFrame({"y": labels}))


IDS = pd.Series([10, 11, 12], name="id")


def test_binary_proba_or_labels():
    enc = _enc("binary", ["no", "yes"])
    p = np.array([[0.2], [0.7], [0.5]])
    sample = pd.DataFrame({"id": [12, 11, 10], "y": 0})
    sub = build_submission(_cfg(), IDS, p, "binary", enc, "auc", sample)
    assert sub["id"].tolist() == [12, 11, 10] and sub["y"].tolist() == [0.5, 0.7, 0.2]
    sub = build_submission(_cfg(threshold=0.6), IDS, p, "binary", enc, "accuracy", None)
    assert sub["y"].tolist() == ["no", "yes", "no"]


def test_multiclass_columns_labels_and_map():
    enc = _enc("multiclass", ["a", "b", "c"])
    p = np.array([[0.1, 0.7, 0.2], [0.5, 0.2, 0.3], [0.2, 0.2, 0.6]])
    wide = pd.DataFrame({"id": IDS, "c": 0.0, "a": 0.0, "b": 0.0})
    sub = build_submission(_cfg(), IDS, p, "multiclass", enc, "logloss", wide)
    assert list(sub.columns) == ["id", "c", "a", "b"] and sub["c"].tolist() == [0.2, 0.3, 0.6]
    sub = build_submission(_cfg(), IDS, p, "multiclass", enc, "accuracy", None)
    assert sub["y"].tolist() == ["b", "a", "c"]
    sub = build_submission(_cfg(), IDS, p, "multiclass", enc, "map@3", None)
    assert sub["y"].iloc[0] == "b c a"


def test_ordinal_uses_cuts():
    enc = TargetEncoder("ordinal").fit(pd.DataFrame({"y": [1, 2, 3]}))
    p = np.array([[1.4], [1.6], [2.9]])
    sub = build_submission(_cfg(), IDS, p, "ordinal", enc, "qwk", None, {"cuts": [1.55, 2.5], "classes": [1, 2, 3]})
    assert sub["y"].tolist() == [1, 2, 3]


def test_validate_catches_problems():
    sample = pd.DataFrame({"id": [1, 2, 3], "y": 0})
    assert validate_submission(sample.copy(), sample) == []
    bad = pd.DataFrame({"id": [1, 2, 2], "y": [0.1, np.nan, np.inf]})
    problems = " | ".join(validate_submission(bad, sample))
    for word in ["id mismatch", "duplicated", "missing", "non-finite"]:
        assert word in problems
    assert validate_submission(sample[["id"]], sample)


def test_score_submission_labels_and_probas():
    sol = pd.DataFrame({"id": [1, 2, 3, 4], "y": ["a", "b", "a", "b"]})
    assert score_submission(pd.DataFrame({"id": [4, 3, 2, 1], "y": ["b", "a", "b", "b"]}), sol,
                            "accuracy", "binary", "id") == 0.75
    probs = pd.DataFrame({"id": [1, 2, 3, 4], "y": [0.1, 0.9, 0.2, 0.8]})
    assert score_submission(probs, sol, "auc", "binary", "id") == 1.0
    masks = pd.DataFrame({"id": [1, 2, 3], "rle": ["1 4", "", "10 2"]})
    pred = pd.DataFrame({"id": [1, 2, 3], "rle": ["3 2", np.nan, ""]})
    assert score_submission(pred, masks, "dice", "segmentation", "id") == (2 * 2 / 6 + 1 + 0) / 3
    reg = pd.DataFrame({"id": [1, 2], "y": [1.0, 3.0]})
    assert score_submission(reg, pd.DataFrame({"id": [1, 2], "y": [1.0, 1.0]}), "rmse", "regression", "id") == np.sqrt(2)
