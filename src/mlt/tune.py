"""Hyperparameter search with Optuna, and greedy forward feature selection.

tune.space maps any config key to a distribution, for example:

    tune:
      n_trials: 50
      space:
        model.params.learning_rate: {type: float, low: 0.01, high: 0.2, log: true}
        model.params.num_leaves: {type: int, low: 8, high: 256, log: true}
        features.rare_min_count: {type: categorical, choices: [0, 10, 50]}

With an empty space, a default one for the model is used (ranges follow the usual
recommendations for each family). Every trial is a full CV run; unpromising trials stop early
after a few folds (median pruning). Tip: tune on a subset (train.folds: [0, 1] or data.nrows)
and confirm the best config on all folds.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path

import pandas as pd
import yaml

from .config import deep_merge, save_config
from .data import feature_columns
from .metrics import get_metric
from .train import prepare, run_cv
from .utils import LOG, require

P = "model.params."
DEFAULT_SPACES = {
    "lgbm": {P + "learning_rate": {"type": "float", "low": 0.01, "high": 0.2, "log": True},
             P + "num_leaves": {"type": "int", "low": 8, "high": 256, "log": True},
             P + "min_child_samples": {"type": "int", "low": 5, "high": 100, "log": True},
             P + "subsample": {"type": "float", "low": 0.5, "high": 1.0},
             P + "colsample_bytree": {"type": "float", "low": 0.3, "high": 1.0},
             P + "reg_lambda": {"type": "float", "low": 1e-3, "high": 10.0, "log": True},
             P + "reg_alpha": {"type": "float", "low": 1e-3, "high": 10.0, "log": True}},
    "xgb": {P + "learning_rate": {"type": "float", "low": 0.01, "high": 0.2, "log": True},
            P + "max_depth": {"type": "int", "low": 3, "high": 12},
            P + "min_child_weight": {"type": "float", "low": 1.0, "high": 10.0, "log": True},
            P + "subsample": {"type": "float", "low": 0.6, "high": 1.0},
            P + "colsample_bytree": {"type": "float", "low": 0.4, "high": 1.0},
            P + "gamma": {"type": "float", "low": 0.0, "high": 1.0},
            P + "reg_lambda": {"type": "float", "low": 1e-2, "high": 10.0, "log": True},
            P + "reg_alpha": {"type": "float", "low": 1e-3, "high": 1.0, "log": True}},
    "catboost": {P + "learning_rate": {"type": "float", "low": 0.01, "high": 0.2, "log": True},
                 P + "depth": {"type": "int", "low": 4, "high": 10},
                 P + "l2_leaf_reg": {"type": "float", "low": 1.0, "high": 10.0, "log": True},
                 P + "random_strength": {"type": "float", "low": 0.0, "high": 2.0}},
    "rf": {P + "n_estimators": {"type": "int", "low": 100, "high": 1200, "step": 100},
           P + "max_depth": {"type": "categorical", "choices": [5, 8, 15, 25, None]},
           P + "min_samples_split": {"type": "int", "low": 2, "high": 20},
           P + "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
           P + "max_features": {"type": "categorical", "choices": ["sqrt", "log2", None]}},
    "logreg": {P + "C": {"type": "float", "low": 1e-3, "high": 100.0, "log": True}},
    "ridge": {P + "alpha": {"type": "float", "low": 1e-3, "high": 100.0, "log": True}},
    "lasso": {P + "alpha": {"type": "float", "low": 1e-5, "high": 1.0, "log": True}},
    "svm": {P + "C": {"type": "float", "low": 1e-3, "high": 1000.0, "log": True},
            P + "gamma": {"type": "categorical", "choices": ["scale", "auto"]}},
    "knn": {P + "n_neighbors": {"type": "int", "low": 2, "high": 64, "log": True},
            P + "p": {"type": "categorical", "choices": [1, 2]}},
    "tfidf": {P + "C": {"type": "float", "low": 0.1, "high": 50.0, "log": True},
              "features.text.ngram_range": {"type": "categorical", "choices": [[1, 1], [1, 2], [1, 3]]},
              "features.text.min_df": {"type": "int", "low": 1, "high": 5}},
    "neural": {"train.lr": {"type": "float", "low": 1e-5, "high": 3e-3, "log": True},
               "train.weight_decay": {"type": "float", "low": 1e-4, "high": 0.1, "log": True}},
}
DEFAULT_SPACES["et"] = DEFAULT_SPACES["rf"]


def default_space(cfg: dict) -> dict:
    name = cfg["model"]["name"]
    if name in DEFAULT_SPACES:
        return DEFAULT_SPACES[name]
    if cfg["modality"] != "tabular" or name == "entity_nn":
        return DEFAULT_SPACES["neural"]
    raise ValueError(f"no default search space for '{name}'; set tune.space in the config")


def _nested(key: str, value) -> dict:
    out: dict = {}
    node = out
    parts = key.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value
    return out


def _suggest(trial, key, spec):
    kind = spec.get("type", "float")
    if kind == "categorical":
        choices = spec["choices"]
        idx = trial.suggest_categorical(key, list(range(len(choices))))  # any value type works
        return choices[idx]
    if kind == "int":
        return trial.suggest_int(key, spec["low"], spec["high"], step=spec.get("step", 1), log=spec.get("log", False))
    return trial.suggest_float(key, spec["low"], spec["high"], log=spec.get("log", False))


def tune(cfg: dict, n_trials: int | None = None, timeout: int | None = None):
    optuna = require("optuna", "tune")
    space = cfg["tune"]["space"] or default_space(cfg)
    ctx = prepare(cfg) if cfg["modality"] != "segmentation" else None
    metric = get_metric(ctx["metric"] if ctx else "dice")
    base = deep_merge(cfg, {"train": {"save_models": False}})

    def objective(trial):
        trial_cfg = copy.deepcopy(base)
        for key, spec in space.items():
            trial_cfg = deep_merge(trial_cfg, _nested(key, _suggest(trial, key, spec)))
        level = LOG.level
        LOG.setLevel(logging.WARNING)
        try:
            return run_cv(trial_cfg, save=False, trial=trial, ctx=ctx).score
        finally:
            LOG.setLevel(level)

    optuna.logging.set_verbosity(optuna.logging.INFO)
    study = optuna.create_study(
        direction="maximize" if metric.greater_is_better else "minimize",
        sampler=optuna.samplers.TPESampler(seed=cfg["seed"]),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    study.optimize(objective, n_trials=n_trials or cfg["tune"]["n_trials"], timeout=timeout or cfg["tune"]["timeout"])

    best = {k: (space[k]["choices"][v] if space[k].get("type") == "categorical" else v)
            for k, v in study.best_params.items()}
    out_dir = Path(cfg["output_dir"]) / f"{cfg['name']}_tune"
    out_dir.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(out_dir / "trials.csv", index=False)
    best_cfg = copy.deepcopy(cfg)
    for k, v in best.items():
        best_cfg = deep_merge(best_cfg, _nested(k, v))
    best_cfg["name"] = f"{cfg['name']}_tuned"
    save_config(best_cfg, out_dir / "best.yaml")
    print(f"\nbest {metric.name}: {study.best_value:.5f}")
    print(yaml.safe_dump(best, sort_keys=False).strip())
    print(f"full config: {out_dir / 'best.yaml'}\nnext: mlt run {out_dir / 'best.yaml'}")
    return study


def forward_select(cfg: dict, max_features: int | None = None) -> list[str]:
    """Add one feature at a time, keeping the one that improves the CV score most."""
    if cfg["modality"] != "tabular":
        raise ValueError("feature selection is for tabular data")
    ctx = prepare(cfg)
    features, *_ = feature_columns(cfg, ctx["train"], ctx["test"])
    metric = get_metric(ctx["metric"])
    base_drop = list(cfg["data"]["drop"] or [])
    chosen, best, history = [], metric.worst, []
    level = LOG.level
    LOG.setLevel(logging.WARNING)
    try:
        while len(chosen) < (max_features or len(features)):
            round_best, round_feat = metric.worst, None
            for f in features:
                if f in chosen:
                    continue
                keep = set(chosen + [f])
                drop = base_drop + [c for c in features if c not in keep]
                trial_cfg = deep_merge(cfg, {"data": {"drop": drop}, "train": {"save_models": False}})
                s = run_cv(trial_cfg, save=False, ctx=ctx).score
                if metric.better(s, round_best):
                    round_best, round_feat = s, f
            if round_feat is None or not metric.better(round_best, best):
                break
            chosen.append(round_feat)
            best = round_best
            history.append({"step": len(chosen), "added": round_feat, "cv": best})
            print(f"step {len(chosen)}: +{round_feat}  {metric.name} {best:.5f}")
    finally:
        LOG.setLevel(level)
    out_dir = Path(cfg["output_dir"]) / f"{cfg['name']}_select"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(out_dir / "selection.csv", index=False)
    dropped = [c for c in features if c not in chosen]
    print(f"\nselected {len(chosen)} of {len(features)} features, CV {best:.5f}")
    print(f"to use them, add to the config:\ndata:\n  drop: {base_drop + dropped}")
    return chosen
