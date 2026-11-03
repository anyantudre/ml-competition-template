"""Tabular models. Each entry builds a fresh estimator per fold (never a shared instance).

family decides the encoding: tree -> integer codes, linear -> one-hot + scaling,
catboost -> native categoricals, nn -> codes for embeddings + scaled numerics.
To add a model, add one ModelSpec here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from sklearn import ensemble, linear_model, neighbors, neural_network, svm, tree

CLASSIFICATION = {"binary", "multiclass"}


@dataclass(frozen=True)
class ModelSpec:
    family: str  # tree | linear | catboost | nn
    build: Callable  # (task, params, seed, n_jobs) -> estimator
    early_stopping: str | None = None  # xgb | lgbm | catboost | nn
    tasks: frozenset = frozenset({"binary", "multiclass", "multilabel", "regression", "ordinal"})
    nan_ok: bool = False


def _clf(task):
    return task in CLASSIFICATION


def _logreg(task, p, seed, n_jobs):
    return linear_model.LogisticRegression(**{"max_iter": 2000, "C": 1.0, **p})


def _ridge(task, p, seed, n_jobs):
    return linear_model.Ridge(**{"alpha": 1.0, **p})


def _lasso(task, p, seed, n_jobs):
    if _clf(task):
        return linear_model.LogisticRegression(**{"max_iter": 3000, "penalty": "l1", "solver": "saga", **p})
    return linear_model.Lasso(**{"alpha": 0.001, "max_iter": 10000, **p})


def _svm(task, p, seed, n_jobs):
    if _clf(task):
        return svm.SVC(**{"C": 1.0, "probability": True, "random_state": seed, **p})
    return svm.SVR(**{"C": 1.0, **p})


def _knn(task, p, seed, n_jobs):
    cls = neighbors.KNeighborsClassifier if _clf(task) else neighbors.KNeighborsRegressor
    return cls(**{"n_neighbors": 15, "n_jobs": n_jobs, **p})


def _dt(task, p, seed, n_jobs):
    cls = tree.DecisionTreeClassifier if _clf(task) else tree.DecisionTreeRegressor
    return cls(**{"random_state": seed, "min_samples_leaf": 5, **p})


def _rf(task, p, seed, n_jobs):
    cls = ensemble.RandomForestClassifier if _clf(task) else ensemble.RandomForestRegressor
    return cls(**{"n_estimators": 300, "n_jobs": n_jobs, "random_state": seed, **p})


def _et(task, p, seed, n_jobs):
    cls = ensemble.ExtraTreesClassifier if _clf(task) else ensemble.ExtraTreesRegressor
    return cls(**{"n_estimators": 300, "n_jobs": n_jobs, "random_state": seed, **p})


def _hgb(task, p, seed, n_jobs):
    cls = ensemble.HistGradientBoostingClassifier if _clf(task) else ensemble.HistGradientBoostingRegressor
    return cls(**{"max_iter": 500, "learning_rate": 0.05, "early_stopping": True, "random_state": seed, **p})


def _gbm(task, p, seed, n_jobs):
    cls = ensemble.GradientBoostingClassifier if _clf(task) else ensemble.GradientBoostingRegressor
    return cls(**{"n_estimators": 300, "random_state": seed, **p})


def _mlp(task, p, seed, n_jobs):
    cls = neural_network.MLPClassifier if _clf(task) else neural_network.MLPRegressor
    return cls(**{"hidden_layer_sizes": (256, 128), "early_stopping": True, "max_iter": 300,
                  "random_state": seed, **p})


def _xgb(task, p, seed, n_jobs):
    import xgboost as xgb

    base = {"n_estimators": 3000, "learning_rate": 0.05, "max_depth": 6, "subsample": 0.8,
            "colsample_bytree": 0.8, "tree_method": "hist", "random_state": seed, "n_jobs": n_jobs}
    if _clf(task):
        return xgb.XGBClassifier(**{**base, **p})
    return xgb.XGBRegressor(**{**base, **p})


def _lgbm(task, p, seed, n_jobs):
    import lightgbm as lgb

    base = {"n_estimators": 5000, "learning_rate": 0.03, "num_leaves": 31, "subsample": 0.8,
            "subsample_freq": 1, "colsample_bytree": 0.8, "random_state": seed, "n_jobs": n_jobs,
            "verbose": -1}
    if _clf(task):
        return lgb.LGBMClassifier(**{**base, **p})
    return lgb.LGBMRegressor(**{**base, **p})


def _catboost(task, p, seed, n_jobs):
    import catboost

    base = {"iterations": 3000, "learning_rate": 0.05, "depth": 6, "random_seed": seed,
            "verbose": False, "allow_writing_files": False, "thread_count": n_jobs}
    if _clf(task):
        return catboost.CatBoostClassifier(**{**base, **p})
    return catboost.CatBoostRegressor(**{**base, **p})


def _entity_nn(task, p, seed, n_jobs):
    from .entity import EntityEmbeddingModel

    return EntityEmbeddingModel(task=task, seed=seed, **p)


CLF_TASKS = frozenset({"binary", "multiclass", "multilabel"})
REG_TASKS = frozenset({"regression", "ordinal"})

REGISTRY: dict[str, ModelSpec] = {
    "logreg": ModelSpec("linear", _logreg, tasks=CLF_TASKS),
    "ridge": ModelSpec("linear", _ridge, tasks=REG_TASKS),
    "lasso": ModelSpec("linear", _lasso),
    "svm": ModelSpec("linear", _svm),
    "knn": ModelSpec("linear", _knn),
    "mlp": ModelSpec("linear", _mlp),
    "dt": ModelSpec("tree", _dt, nan_ok=True),
    "rf": ModelSpec("tree", _rf, nan_ok=True),
    "et": ModelSpec("tree", _et, nan_ok=True),
    "gbm": ModelSpec("tree", _gbm),
    "hgb": ModelSpec("tree", _hgb, nan_ok=True),
    "xgb": ModelSpec("tree", _xgb, early_stopping="xgb", nan_ok=True),
    "lgbm": ModelSpec("tree", _lgbm, early_stopping="lgbm", nan_ok=True),
    "catboost": ModelSpec("catboost", _catboost, early_stopping="catboost", nan_ok=True),
    "entity_nn": ModelSpec("nn", _entity_nn, early_stopping="nn"),
}

ENCODING_FOR_FAMILY = {"tree": "label", "linear": "onehot", "catboost": "native", "nn": "embedding"}


def get_spec(name: str, task: str | None = None) -> ModelSpec:
    if name not in REGISTRY:
        raise KeyError(f"unknown tabular model '{name}'. Available: {', '.join(REGISTRY)}")
    spec = REGISTRY[name]
    if task is not None and task not in spec.tasks:
        alt = "ridge" if name == "logreg" else "logreg"
        raise ValueError(f"model '{name}' does not support task '{task}' (try '{alt}')")
    return spec
