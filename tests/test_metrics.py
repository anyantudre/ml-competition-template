import numpy as np
import pytest
from sklearn import metrics as skm

from mlt.metrics import METRICS, apk, cuts_to_labels, dice, fit_cuts, get_metric, iou, mapk, score
from mlt.rle import rle_decode, rle_encode


def test_binary_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = np.clip(y * 0.6 + rng.random(200) * 0.5, 0, 1)
    assert score("auc", y, p, "binary") == pytest.approx(skm.roc_auc_score(y, p))
    assert score("logloss", y, p, "binary") == pytest.approx(skm.log_loss(y, np.clip(p, 1e-15, 1 - 1e-15)))
    assert score("f1", y, p, "binary", threshold=0.4) == pytest.approx(skm.f1_score(y, p >= 0.4))
    assert score("mcc", y, p, "binary") == pytest.approx(skm.matthews_corrcoef(y, p >= 0.5))


def test_multiclass_and_multilabel():
    rng = np.random.default_rng(1)
    p = rng.dirichlet(np.ones(3), size=100)
    y = p.argmax(1)
    assert score("accuracy", y, p, "multiclass") == 1.0
    assert score("logloss", y, p, "multiclass") == pytest.approx(skm.log_loss(y, p))
    ym = (rng.random((100, 3)) > 0.5).astype(int)
    expected = np.mean([skm.log_loss(ym[:, j], p[:, j]) for j in range(3)])
    assert score("logloss", ym, p, "multilabel") == pytest.approx(expected)


def test_map_at_k_kaggle_definition():
    assert apk([1], [1, 2, 3], 3) == 1.0
    assert apk([1], [2, 1, 3], 3) == 0.5
    assert apk([1, 2], [1, 1, 2], 3) == pytest.approx((1 + 2 / 3) / 2)  # duplicates ignored
    assert mapk([[1], [2]], [[1, 2], [1, 2]], 2) == pytest.approx(0.75)
    p = np.array([[0.1, 0.6, 0.3], [0.5, 0.2, 0.3]])
    assert score("map@3", np.array([2, 0]), p, "multiclass") == pytest.approx((0.5 + 1.0) / 2)


def test_regression_metrics():
    y = np.array([1.0, 2.0, 3.0, 10.0])
    p = np.array([1.5, 2.0, 2.0, 12.0])
    assert score("rmse", y, p, "regression") == pytest.approx(np.sqrt(np.mean((y - p) ** 2)))
    assert score("rmsle", y, p, "regression") == pytest.approx(np.sqrt(skm.mean_squared_log_error(y, p)))
    assert score("mae", y, p, "regression") == pytest.approx(np.mean(np.abs(y - p)))
    assert score("r2", y, p, "regression") == pytest.approx(skm.r2_score(y, p))


def test_ordinal_cuts_improve_qwk():
    rng = np.random.default_rng(2)
    y = rng.integers(3, 8, 400)
    pred = y + rng.normal(0.4, 0.6, 400)  # biased upwards
    cuts = fit_cuts(y, pred)
    classes = np.unique(y)
    tuned = score("qwk", y, pred, "ordinal", cuts=cuts, classes=classes)
    assert tuned >= score("qwk", y, pred, "ordinal")
    assert set(cuts_to_labels(pred, cuts, classes)) <= set(classes)


def test_masks_and_rle():
    rng = np.random.default_rng(3)
    m = rng.random((2, 1, 16, 12)) > 0.6
    assert dice(m.astype(float), m.astype(float)) == 1.0
    assert iou(m.astype(float), np.zeros_like(m, dtype=float)) == 0.0
    empty = np.zeros((1, 4, 4))
    assert dice(empty, empty) == 1.0
    for order in "FC":
        mask = m[0, 0]
        assert np.array_equal(rle_decode(rle_encode(mask, order), mask.shape, order), mask)
    assert rle_encode(np.zeros((3, 3))) == ""
    assert rle_encode(np.array([[0, 1], [1, 1]]), "F") == "2 3"


def test_metric_registry():
    assert get_metric("auc").greater_is_better and not get_metric("rmse").greater_is_better
    with pytest.raises(KeyError):
        get_metric("nope")
    assert {"auc", "logloss", "qwk", "rmsle", "map@3", "dice"} <= set(METRICS)
