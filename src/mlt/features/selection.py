"""Feature selection fitted on the training fold only.

Methods: variance (drop near-constant columns), correlation (drop one of each highly correlated
pair), kbest (univariate F-test), mutual_info, from_model (LightGBM importance, top k or above
a threshold). Greedy forward selection is expensive; `mlt select` runs it with the full CV
loop as the scorer, so the score it reports is out of fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn import feature_selection as fs

CLASSIFICATION = {"binary", "multiclass", "ordinal"}


class FeatureSelector:
    def __init__(self, spec: dict, task: str, seed: int = 42):
        self.method = spec["method"]
        self.k = spec.get("k")
        self.threshold = spec.get("threshold")
        self.task = task
        self.seed = seed

    def fit_transform(self, X, y):
        if isinstance(X, dict):
            raise ValueError("feature selection is not available for entity-embedding inputs")
        y = np.asarray(y)
        if y.ndim > 1:
            y = y.argmax(axis=1) if self.task == "multilabel" else y[:, 0]
        n_features = X.shape[1]
        k = min(self.k or max(1, n_features // 2), n_features)
        if self.method == "variance":
            sel = fs.VarianceThreshold(self.threshold or 0.0).fit(X)
            self.mask = sel.get_support()
        elif self.method == "correlation":
            dense = pd.DataFrame(X.toarray() if sparse.issparse(X) else np.asarray(X, dtype=float))
            corr = dense.corr().abs().to_numpy()
            upper = np.triu(corr, k=1)
            limit = self.threshold or 0.95
            self.mask = ~(upper > limit).any(axis=0)
        elif self.method in {"kbest", "mutual_info"}:
            classif = self.task in CLASSIFICATION or self.task == "multilabel"
            if self.method == "kbest":
                func = fs.f_classif if classif else fs.f_regression
            else:
                func = fs.mutual_info_classif if classif else fs.mutual_info_regression
            filled = _fill(X)
            sel = fs.SelectKBest(func, k=k).fit(filled, y)
            self.mask = sel.get_support()
        elif self.method == "from_model":
            import lightgbm as lgb

            est = (lgb.LGBMClassifier if self.task in CLASSIFICATION else lgb.LGBMRegressor)(
                n_estimators=200, learning_rate=0.05, random_state=self.seed, verbose=-1
            )
            est.fit(X, y)
            imp = est.feature_importances_
            if self.threshold is not None:
                self.mask = imp > self.threshold
            else:
                self.mask = np.zeros(n_features, dtype=bool)
                self.mask[np.argsort(-imp)[:k]] = True
        else:
            raise ValueError(f"unknown features.selection.method '{self.method}'")
        if not self.mask.any():
            self.mask[:] = True
        return self.transform(X)

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.loc[:, self.mask]
        return X[:, np.flatnonzero(self.mask)]


def _fill(X):
    if sparse.issparse(X):
        return X
    arr = np.asarray(X, dtype=float)
    med = np.nanmedian(arr, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    return np.where(np.isnan(arr), med, arr)
