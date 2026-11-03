"""Tabular pipeline: preprocessing per fold, then any model from mlt.models.tabular."""

from __future__ import annotations

import inspect
import re

import numpy as np
import pandas as pd

from ..features.tabular import TabularPreprocessor
from ..models.tabular import ENCODING_FOR_FAMILY, get_spec
from ..utils import LOG
from .base import Pipeline


def _safe_names(names):
    seen, out = set(), []
    for n in names:
        s = re.sub(r"[^0-9a-zA-Z_]", "_", str(n))[:60] or "f"
        base, i = s, 1
        while s in seen:
            s, i = f"{base}_{i}", i + 1
        seen.add(s)
        out.append(s)
    return out


class TabularPipeline(Pipeline):
    def __init__(self, *args):
        super().__init__(*args)
        m = self.cfg["model"]
        if not m["name"]:
            raise ValueError("model.name is required, e.g. lgbm, xgb, catboost, logreg, ridge, rf")
        self.spec = get_spec(m["name"], self.task)
        enc = self.cfg["features"]["encoding"]
        self.encoding = ENCODING_FOR_FAMILY[self.spec.family] if enc == "auto" else enc
        self.n_jobs = -1  # model.params.n_jobs overrides it
        # multilabel: one binary model per label (except the network); ordinal: regression
        if self.task == "ordinal":
            self.model_task = "regression"
        elif self.task == "multilabel" and self.spec.family != "nn":
            self.model_task = "binary"
        else:
            self.model_task = self.task

    def _prepare(self, X, art, fit=False):
        if isinstance(X, pd.DataFrame):
            X = X.copy()
            X.columns = _safe_names(X.columns)
            if self.encoding == "label" and not self.spec.nan_ok:
                if fit:
                    art["fill"] = X.median(numeric_only=True).fillna(0.0)
                X = X.fillna(art["fill"])
        return X

    def fit_fold(self, train, y_train, valid, y_valid, test, fold):
        _, cats, nums, dates = self.columns
        pre = TabularPreprocessor(self.cfg, cats, nums, dates, self.encoding, self.task, self.n_out)
        art = {"pre": pre}
        Xtr = self._prepare(pre.fit_transform(train, y_train, others=[valid, test]), art, fit=True)
        Xva = self._prepare(pre.transform(valid), art)
        params = dict(self.cfg["model"]["params"])
        seed = self.cfg["seed"]
        rounds = self.cfg["model"]["early_stopping_rounds"]

        per_label = self.task == "multilabel" and self.model_task == "binary"
        ys_tr = [y_train[:, j] for j in range(y_train.shape[1])] if per_label else [y_train]
        ys_va = [y_valid[:, j] for j in range(y_valid.shape[1])] if per_label else [y_valid]
        models = []
        for ytr, yva in zip(ys_tr, ys_va):
            est = self.spec.build(self.model_task, params, seed, self.n_jobs)
            kind = self.spec.early_stopping
            if kind == "xgb" and rounds:
                est.set_params(early_stopping_rounds=rounds)
                est.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
            elif kind == "lgbm" and rounds:
                import lightgbm as lgb

                cbs = [lgb.early_stopping(rounds, verbose=False), lgb.log_evaluation(0)]
                if "eval_X" in inspect.signature(est.fit).parameters:  # lightgbm >= 4.7
                    est.fit(Xtr, ytr, eval_X=(Xva,), eval_y=(yva,), callbacks=cbs)
                else:
                    est.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=cbs)
            elif kind == "catboost":
                est.fit(Xtr, ytr, cat_features=getattr(pre, "cat_indices", None), eval_set=(Xva, yva),
                        early_stopping_rounds=rounds or None)
            elif kind == "nn":
                est.fit(Xtr, ytr, (Xva, yva), pre.cardinalities, self.n_out,
                        self.score_fn(y_valid), self.greater_is_better)
            else:
                est.fit(Xtr, ytr)
            best = _best_iteration(est)
            if best is not None:
                LOG.info("  fold %d best iteration %s", fold, best)
            models.append(est)
        art["models"] = models
        return art

    def _predict_one(self, est, X):
        if self.model_task == "binary":
            return est.predict_proba(X)[:, 1:2]
        if self.model_task in {"multiclass", "multilabel"}:
            return est.predict_proba(X)
        return np.asarray(est.predict(X), dtype=float).reshape(-1, 1)

    def predict(self, artifact, df):
        X = self._prepare(artifact["pre"].transform(df), artifact)
        preds = [self._predict_one(est, X) for est in artifact["models"]]
        return np.hstack(preds).astype(float)

    def importance(self, artifact):
        names = artifact["pre"].feature_names
        imps = []
        for est in artifact["models"]:
            imp = getattr(est, "feature_importances_", None)
            if imp is None and hasattr(est, "coef_"):
                imp = np.abs(np.atleast_2d(est.coef_)).mean(axis=0)
            if imp is None or len(imp) != len(names):
                return None
            imps.append(np.asarray(imp, dtype=float))
        return pd.Series(np.mean(imps, axis=0), index=names)


def _best_iteration(est):
    for attr in ("best_iteration", "best_iteration_"):
        v = getattr(est, attr, None)
        if isinstance(v, (int, np.integer)) and v > 0:
            return int(v)
    if hasattr(est, "get_best_iteration"):
        try:
            return est.get_best_iteration()
        except Exception:  # noqa: BLE001 - catboost without eval set
            return None
    return None
