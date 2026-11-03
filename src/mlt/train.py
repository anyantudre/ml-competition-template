"""The cross-validation loop: fit every fold, keep out-of-fold (OOF) and test predictions.

A run folder (outputs/<name>/) holds everything needed to trust, reproduce, blend or serve
the result:

    config.yaml       resolved config (every default filled in)
    state.json        task, metric, classes, columns, ordinal cuts
    metrics.json      CV score, per-fold scores, timings, folds hash, git commit
    oof.csv           id, fold, target(s), out-of-fold predictions
    test_pred.csv     id, test predictions averaged over folds
    submission.csv    built from test_pred.csv and checked against sample_submission
    importance.csv    mean feature importance over folds (tree and linear models)
    models/fold*/     fitted preprocessors and models

and one line is appended to outputs/runs.csv.
"""

from __future__ import annotations

import platform
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .config import save_config, targets
from .data import TargetEncoder, feature_columns, load_data, resolve_task, row_ids
from .folds import default_scheme, fold_ids, get_folds, train_mask
from .metrics import fit_cuts, get_metric, score
from .pipelines import get_pipeline
from .submission import build_submission, validate_submission
from .utils import LOG, git_sha, seed_everything, setup_logging, write_json

COMPATIBLE = {
    "proba": {"binary", "multiclass", "multilabel"},
    "label": {"binary", "multiclass", "multilabel", "ordinal"},
    "value": {"regression", "ordinal"},
    "mask": {"segmentation"},
}


@dataclass
class RunResult:
    name: str
    score: float
    fold_scores: list
    run_dir: Path | None
    oof: np.ndarray
    test_pred: np.ndarray | None
    extra: dict = field(default_factory=dict)


def check_metric(metric: str, task: str) -> None:
    m = get_metric(metric)
    if task not in COMPATIBLE[m.kind] and not (metric == "qwk" and task in {"multiclass", "ordinal"}):
        raise ValueError(f"metric '{metric}' does not fit task '{task}'")


def prepare(cfg: dict, force_folds: bool = False):
    """Load data, resolve the task and the folds. Shared by train, folds, tune and checks."""
    train, test, sample = load_data(cfg)
    task, metric = resolve_task(cfg, train)
    check_metric(metric, task)
    names = targets(cfg)
    if task != "segmentation":
        missing = train[names].isna().any(axis=1)
        if missing.any():
            LOG.warning("dropping %d training rows with a missing target", int(missing.sum()))
            train = train.loc[~missing].reset_index(drop=True)
    enc = TargetEncoder(task)
    if task != "segmentation":
        enc.fit(train[names])
        y = enc.transform(train[names])
    else:
        y = np.zeros(len(train))
    folds, path, fhash = get_folds(train, y, cfg, task, force=force_folds)
    return {"train": train, "test": test, "sample": sample, "task": task, "metric": metric,
            "enc": enc, "y": y, "folds": folds, "folds_path": path, "folds_hash": fhash}


def run_cv(cfg: dict, save: bool = True, trial=None, ctx: dict | None = None) -> RunResult:
    setup_logging()
    seed_everything(cfg["seed"])
    if cfg["modality"] == "segmentation":
        from .pipelines.segmentation import run_segmentation

        return run_segmentation(cfg, save=save, trial=trial)

    start = time.perf_counter()
    ctx = ctx or prepare(cfg)
    train, test = ctx["train"], ctx["test"]
    task, metric, enc, y, folds = ctx["task"], ctx["metric"], ctx["enc"], ctx["y"], ctx["folds"]
    scheme = default_scheme(cfg, task)
    log1p = cfg["target_transform"] == "log1p"
    y_model = np.log1p(y) if log1p else y
    inverse = np.expm1 if log1p else (lambda a: a)

    columns = feature_columns(cfg, train, test) if cfg["modality"] == "tabular" else None
    pipe = get_pipeline(cfg, task, enc, metric, columns)
    run_dir = Path(cfg["output_dir"]) / cfg["name"]
    run_folds = cfg["train"]["folds"] or fold_ids(folds)
    LOG.info("%s | %s %s | metric %s | %d rows | %s folds %s", cfg["name"], cfg["modality"], task,
             metric, len(train), scheme, run_folds)

    oof = np.full((len(train), enc.n_outputs), np.nan)
    test_preds, fold_scores, importances = [], [], []
    thr = cfg["submission"]["threshold"]
    for k in run_folds:
        t0 = time.perf_counter()
        tr = train_mask(folds, k, scheme)
        va = folds == k
        art = pipe.fit_fold(train[tr].reset_index(drop=True), y_model[tr],
                            train[va].reset_index(drop=True), y_model[va], test, k)
        oof[va] = inverse(pipe.predict(art, train[va].reset_index(drop=True)))
        s = score(metric, y[va], oof[va], task, threshold=thr)
        fold_scores.append(s)
        if test is not None:
            test_preds.append(inverse(pipe.predict(art, test)))
        imp = pipe.importance(art)
        if imp is not None:
            importances.append(imp)
        if save and cfg["train"]["save_models"]:
            pipe.save(art, run_dir / "models" / f"fold{k}")
        LOG.info("fold %d  %s %.5f  (%.0fs)", k, metric, s, time.perf_counter() - t0)
        if trial is not None:
            import optuna

            trial.report(float(np.mean(fold_scores)), step=len(fold_scores) - 1)
            if trial.should_prune():
                raise optuna.TrialPruned()

    scored = np.isin(folds, run_folds)
    cuts = None
    cv = score(metric, y[scored], oof[scored], task, threshold=thr)
    extra = {}
    if task == "ordinal":
        classes = np.unique(y).tolist()
        cuts = {"cuts": fit_cuts(y[scored], oof[scored, 0]), "classes": classes}
        extra["qwk_rounded"] = cv
        cv = score(metric, y[scored], oof[scored], task, cuts=cuts["cuts"], classes=classes)
        LOG.info("ordinal cuts %s", np.round(cuts["cuts"], 3).tolist())
    test_pred = np.mean(test_preds, axis=0) if test_preds else None
    LOG.info("CV %s %.5f  (folds %.5f +- %.5f)", metric, cv, np.mean(fold_scores), np.std(fold_scores))
    result = RunResult(cfg["name"], cv, fold_scores, run_dir if save else None, oof, test_pred, extra)
    if save:
        write_run(cfg, ctx, result, cuts=cuts, columns=columns, importances=importances,
                  duration=time.perf_counter() - start)
    return result


def pred_frame(ids, pred, enc) -> pd.DataFrame:
    return pd.concat([ids.reset_index(drop=True).to_frame(), pd.DataFrame(pred, columns=enc.pred_columns())], axis=1)


def write_run(cfg, ctx, result: RunResult, cuts=None, columns=None, importances=(), duration=0.0):
    run_dir = result.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    enc, task, metric = ctx["enc"], ctx["task"], ctx["metric"]
    resolved = {**cfg, "task": task, "metric": metric}
    save_config(resolved, run_dir / "config.yaml")
    ids = row_ids(cfg, ctx["train"])
    oof = pred_frame(ids, result.oof, enc)
    oof.insert(1, "fold", ctx["folds"])
    for j, name in enumerate(targets(cfg)):
        oof.insert(2 + j, name, ctx["train"][name].to_numpy())
    oof.to_csv(run_dir / "oof.csv", index=False)
    write_json({"task": task, "metric": metric, "target": enc.state(), "columns": columns,
                "cuts": cuts, "folds": cfg["train"]["folds"] or fold_ids(ctx["folds"])},
               run_dir / "state.json")
    if importances:
        imp = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)
        imp.rename("importance").to_csv(run_dir / "importance.csv", index_label="feature")
    sub_path = None
    if result.test_pred is not None:
        test_ids = row_ids(cfg, ctx["test"])
        pred_frame(test_ids, result.test_pred, enc).to_csv(run_dir / "test_pred.csv", index=False)
        sub = build_submission(cfg, test_ids, result.test_pred, task, enc, metric, ctx["sample"], cuts)
        problems = validate_submission(sub, ctx["sample"], cfg["submission"]["id_col"] or cfg["data"]["id_col"])
        sub_path = run_dir / cfg["submission"]["file"]
        sub.to_csv(sub_path, index=False)
        if problems:
            LOG.warning("submission problems: %s", "; ".join(problems))
        else:
            LOG.info("submission ok: %s (%d rows)", sub_path, len(sub))
    info = {
        "name": cfg["name"], "modality": cfg["modality"], "task": task, "metric": metric,
        "cv": result.score, "fold_mean": float(np.mean(result.fold_scores)),
        "fold_std": float(np.std(result.fold_scores)), "fold_scores": result.fold_scores,
        **result.extra, "model": cfg["model"]["name"], "backbone": cfg["model"]["backbone"],
        "n_train": len(ctx["train"]), "folds_file": str(ctx["folds_path"]), "folds_hash": ctx["folds_hash"],
        "duration_s": round(duration, 1), "git": git_sha(), "mlt": __version__,
        "python": platform.python_version(), "submission": str(sub_path) if sub_path else None,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(info, run_dir / "metrics.json")
    append_registry(cfg["output_dir"], info)
    LOG.info("run saved to %s", run_dir)


REGISTRY_COLS = ["time", "name", "modality", "task", "model", "backbone", "metric", "cv",
                 "fold_mean", "fold_std", "n_train", "folds_hash", "duration_s", "git", "submission"]


def append_registry(output_dir: str, info: dict) -> None:
    path = Path(output_dir) / "runs.csv"
    row = pd.DataFrame([{c: info.get(c) for c in REGISTRY_COLS}])
    row.to_csv(path, mode="a", header=not path.exists(), index=False)
