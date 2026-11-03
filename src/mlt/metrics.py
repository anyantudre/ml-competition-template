"""Evaluation metrics, each with its direction and the kind of prediction it scores.

Predictions are always 2-D arrays of shape (n, n_outputs): a probability for binary, one column
per class for multiclass, one per label for multilabel, a value for regression and ordinal.
Metrics that need hard labels threshold or argmax the probabilities themselves.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import optimize
from sklearn import metrics as skm

EPS = 1e-15


@dataclass(frozen=True)
class Metric:
    name: str
    fn: Callable
    greater_is_better: bool
    kind: str  # proba | label | value | mask

    def better(self, a: float, b: float) -> bool:
        return a > b if self.greater_is_better else a < b

    @property
    def worst(self) -> float:
        return -np.inf if self.greater_is_better else np.inf


def _labels(pred: np.ndarray, task: str, threshold: float = 0.5) -> np.ndarray:
    if task == "binary":
        return (pred[:, 0] >= threshold).astype(int)
    if task == "multiclass":
        return pred.argmax(axis=1)
    if task == "multilabel":
        return (pred >= threshold).astype(int)
    return np.rint(pred[:, 0]).astype(int)


# ---------------------------------------------------------------- classification
def auc(y, p, task):
    if task == "binary":
        return skm.roc_auc_score(y, p[:, 0])
    if task == "multiclass":
        return skm.roc_auc_score(y, p, multi_class="ovr", labels=np.arange(p.shape[1]))
    if task == "multilabel":
        cols = [j for j in range(y.shape[1]) if len(np.unique(y[:, j])) > 1]
        return float(np.mean([skm.roc_auc_score(y[:, j], p[:, j]) for j in cols]))
    raise ValueError("auc needs a classification task")


def logloss(y, p, task):
    p = np.clip(p, EPS, 1 - EPS)
    if task == "binary":
        return skm.log_loss(y, p[:, 0], labels=[0, 1])
    if task == "multiclass":
        p = p / p.sum(axis=1, keepdims=True)
        return skm.log_loss(y, p, labels=np.arange(p.shape[1]))
    if task == "multilabel":  # mean column-wise log loss
        return float(np.mean([skm.log_loss(y[:, j], p[:, j], labels=[0, 1]) for j in range(y.shape[1])]))
    raise ValueError("logloss needs a classification task")


def _label_metric(fn, **kw):
    def wrapped(y, p, task, threshold=0.5):
        return fn(y, _labels(p, task, threshold), **kw)

    return wrapped


def _average(task):
    return "binary" if task == "binary" else "macro"


def f1(y, p, task, threshold=0.5):
    return skm.f1_score(y, _labels(p, task, threshold), average=_average(task), zero_division=0)


def precision(y, p, task, threshold=0.5):
    return skm.precision_score(y, _labels(p, task, threshold), average=_average(task), zero_division=0)


def recall(y, p, task, threshold=0.5):
    return skm.recall_score(y, _labels(p, task, threshold), average=_average(task), zero_division=0)


def apk(actual, predicted, k=10):
    """Average precision at k, as used on Kaggle (order aware, duplicates ignored)."""
    predicted = list(predicted)[:k]
    if not actual:
        return 0.0
    score, hits, seen = 0.0, 0, set()
    for i, p in enumerate(predicted):
        if p in actual and p not in seen:
            hits += 1
            score += hits / (i + 1.0)
        seen.add(p)
    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    return float(np.mean([apk(a, p, k) for a, p in zip(actual, predicted)]))


def map_at_k(k):
    def fn(y, p, task):
        top = np.argsort(-p, axis=1)[:, :k]
        if task == "multilabel":
            actual = [set(np.flatnonzero(row)) for row in y]
        else:
            actual = [{int(v)} for v in y]
        return mapk(actual, top, k)

    return fn


# -------------------------------------------------------------------- regression
def rmse(y, p, task):
    return float(np.sqrt(skm.mean_squared_error(y, p[:, 0])))


def rmsle(y, p, task):
    return float(np.sqrt(np.mean((np.log1p(y) - np.log1p(np.clip(p[:, 0], 0, None))) ** 2)))


def mape(y, p, task):
    return float(np.mean(np.abs((y - p[:, 0]) / np.clip(np.abs(y), EPS, None))))


def qwk(y, p, task):
    pred = p[:, 0]
    if task in {"binary", "multiclass"}:
        pred = _labels(p, task)
    return skm.cohen_kappa_score(np.asarray(y).astype(int), np.rint(pred).astype(int), weights="quadratic")


# ------------------------------------------------------------------ segmentation
def dice(y, p, task=None, threshold=0.5):
    """Mean per-image Dice. y and p are (n, H, W); empty vs empty counts as 1."""
    yt = y.reshape(len(y), -1) > 0.5
    pr = p.reshape(len(p), -1) >= threshold
    inter = (yt & pr).sum(1)
    total = yt.sum(1) + pr.sum(1)
    scores = np.where(total == 0, 1.0, 2 * inter / np.maximum(total, 1))
    return float(scores.mean())


def iou(y, p, task=None, threshold=0.5):
    yt = y.reshape(len(y), -1) > 0.5
    pr = p.reshape(len(p), -1) >= threshold
    inter = (yt & pr).sum(1)
    union = (yt | pr).sum(1)
    return float(np.where(union == 0, 1.0, inter / np.maximum(union, 1)).mean())


METRICS: dict[str, Metric] = {
    m.name: m
    for m in [
        Metric("auc", auc, True, "proba"),
        Metric("logloss", logloss, False, "proba"),
        Metric("accuracy", _label_metric(skm.accuracy_score), True, "label"),
        Metric("balanced_accuracy", _label_metric(skm.balanced_accuracy_score), True, "label"),
        Metric("f1", f1, True, "label"),
        Metric("f1_macro", _label_metric(skm.f1_score, average="macro", zero_division=0), True, "label"),
        Metric("f1_micro", _label_metric(skm.f1_score, average="micro", zero_division=0), True, "label"),
        Metric("f1_weighted", _label_metric(skm.f1_score, average="weighted", zero_division=0), True, "label"),
        Metric("precision", precision, True, "label"),
        Metric("recall", recall, True, "label"),
        Metric("mcc", _label_metric(skm.matthews_corrcoef), True, "label"),
        Metric("qwk", qwk, True, "label"),
        Metric("map@3", map_at_k(3), True, "proba"),
        Metric("map@5", map_at_k(5), True, "proba"),
        Metric("rmse", rmse, False, "value"),
        Metric("mse", lambda y, p, t: skm.mean_squared_error(y, p[:, 0]), False, "value"),
        Metric("mae", lambda y, p, t: skm.mean_absolute_error(y, p[:, 0]), False, "value"),
        Metric("medae", lambda y, p, t: skm.median_absolute_error(y, p[:, 0]), False, "value"),
        Metric("rmsle", rmsle, False, "value"),
        Metric("mape", mape, False, "value"),
        Metric("r2", lambda y, p, t: skm.r2_score(y, p[:, 0]), True, "value"),
        Metric("dice", dice, True, "mask"),
        Metric("iou", iou, True, "mask"),
    ]
}

LABEL_THRESHOLD_METRICS = {"accuracy", "balanced_accuracy", "f1", "precision", "recall", "mcc"}


def get_metric(name: str) -> Metric:
    if name not in METRICS:
        raise KeyError(f"unknown metric '{name}'. Available: {', '.join(METRICS)}")
    return METRICS[name]


def score(name: str, y: np.ndarray, pred: np.ndarray, task: str, threshold: float = 0.5,
          cuts: list[float] | None = None, classes=None) -> float:
    m = get_metric(name)
    pred = np.asarray(pred, dtype=float)
    if pred.ndim == 1:
        pred = pred[:, None]
    if cuts is not None and task == "ordinal":
        pred = cuts_to_labels(pred[:, 0], cuts, classes)[:, None].astype(float)
    if m.name in LABEL_THRESHOLD_METRICS and task in {"binary", "multilabel"}:
        return float(m.fn(y, pred, task, threshold=threshold))
    return float(m.fn(y, pred, task))


# --------------------------------------------- ordinal targets: tuned rounding cuts
def fit_cuts(y: np.ndarray, pred: np.ndarray) -> list[float]:
    """Search cut points that maximise QWK on out-of-fold predictions (the usual Kaggle trick)."""
    classes = np.unique(y)
    start = (classes[:-1] + classes[1:]) / 2.0

    def loss(c):
        labels = cuts_to_labels(pred, c, classes)
        return -skm.cohen_kappa_score(y.astype(int), labels.astype(int), weights="quadratic")

    res = optimize.minimize(loss, start, method="Nelder-Mead", options={"maxiter": 2000, "xatol": 1e-4})
    return np.sort(res.x).tolist()


def cuts_to_labels(pred: np.ndarray, cuts, classes) -> np.ndarray:
    return np.asarray(classes)[np.digitize(pred, np.sort(np.asarray(cuts, dtype=float)))]
