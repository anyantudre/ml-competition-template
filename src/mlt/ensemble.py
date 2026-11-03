"""Blend finished runs from their out-of-fold (OOF) and test predictions.

All members must share the folds file (checked through its hash), so a blender never trains on
predictions made by a model that saw the same rows.

    mean      plain average
    rank      average of ranks (for AUC-type metrics, where only the order matters)
    weighted  weights fitted on OOF predictions (Nelder-Mead). The reported score is nested:
              weights fitted on the other folds, scored on the held-out fold
    stack     a level-2 model trained on the OOF predictions with the same folds
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import rankdata

from .config import load_config, save_config, targets
from .data import TargetEncoder, read_table
from .metrics import fit_cuts, get_metric, score
from .submission import build_submission, validate_submission
from .train import append_registry
from .utils import LOG, read_json, write_json


class Member:
    def __init__(self, run_dir: str | Path):
        self.dir = Path(run_dir)
        self.cfg = load_config(self.dir / "config.yaml")
        self.state = read_json(self.dir / "state.json")
        self.info = read_json(self.dir / "metrics.json")
        self.oof = pd.read_csv(self.dir / "oof.csv")
        test_path = self.dir / "test_pred.csv"
        self.test = pd.read_csv(test_path) if test_path.exists() else None
        self.pred_cols = [c for c in self.oof.columns if c == "pred" or c.startswith("pred_")]

    @property
    def name(self):
        return self.dir.name


def load_members(runs) -> list[Member]:
    members = [Member(r) for r in runs]
    first = members[0]
    for m in members[1:]:
        for key in ("task", "metric"):
            if m.state[key] != first.state[key]:
                raise ValueError(f"{m.name} has {key}={m.state[key]}, {first.name} has {first.state[key]}")
        if m.info["folds_hash"] != first.info["folds_hash"]:
            raise ValueError(f"{m.name} and {first.name} used different folds; rerun with the same cv settings")
        if not np.array_equal(m.oof.iloc[:, 0].astype(str), first.oof.iloc[:, 0].astype(str)):
            raise ValueError(f"{m.name} and {first.name} have different training rows")
        if m.pred_cols != first.pred_cols:
            raise ValueError(f"{m.name} and {first.name} have different prediction columns")
    if first.state["task"] == "segmentation":
        raise ValueError("ensembling segmentation runs is not supported; average their masks in a notebook")
    return members


def _blend(arrays, w):
    w = np.abs(w) / max(np.abs(w).sum(), 1e-12)
    return np.tensordot(w, np.stack(arrays), axes=1)


def _rank(a: np.ndarray) -> np.ndarray:
    return np.column_stack([rankdata(a[:, j]) / len(a) for j in range(a.shape[1])])


def fit_weights(arrays, y, metric, task, threshold=0.5) -> np.ndarray:
    m = get_metric(metric)
    sign = -1.0 if m.greater_is_better else 1.0

    def loss(w):
        return sign * score(metric, y, _blend(arrays, w), task, threshold=threshold)

    x0 = np.ones(len(arrays)) / len(arrays)
    res = optimize.minimize(loss, x0, method="Nelder-Mead", options={"maxiter": 500 * len(arrays), "xatol": 1e-4})
    w = np.abs(res.x)
    return w / w.sum()


def _stack_model(task, name, seed):
    from sklearn.linear_model import LogisticRegression, Ridge

    if name in (None, "linear"):
        return LogisticRegression(C=1.0, max_iter=2000) if task in {"binary", "multiclass"} else Ridge(alpha=1.0)
    from .models.tabular import get_spec

    model_task = "regression" if task == "ordinal" else task
    return get_spec(name).build(model_task, {"n_estimators": 300} if name in {"lgbm", "xgb"} else {}, seed, -1)


def _stack_predict(est, X, task):
    if task in {"binary"}:
        return est.predict_proba(X)[:, 1:2]
    if task == "multiclass":
        return est.predict_proba(X)
    return est.predict(X).reshape(-1, 1)


def stack(oofs, tests, y, folds, scored, task, model_name, seed):
    X = np.hstack(oofs)
    Xt = np.hstack(tests) if tests is not None else None
    n_out = oofs[0].shape[1]
    level2 = np.full((len(X), n_out), np.nan)
    test_preds = []
    labels = [y[:, j] for j in range(y.shape[1])] if task == "multilabel" else [y]
    for k in sorted(set(folds[scored])):
        tr, va = scored & (folds != k), folds == k
        cols, tcols = [], []
        for yy in labels:
            t = "binary" if task == "multilabel" else task
            est = _stack_model(t, model_name, seed).fit(X[tr], yy[tr])
            cols.append(_stack_predict(est, X[va], t))
            if Xt is not None:
                tcols.append(_stack_predict(est, Xt, t))
        level2[va] = np.hstack(cols)
        if Xt is not None:
            test_preds.append(np.hstack(tcols))
    return level2, (np.mean(test_preds, axis=0) if test_preds else None)


def ensemble_runs(runs, method="weighted", name=None, output_dir="outputs", stack_model=None):
    start = time.perf_counter()
    members = load_members(runs)
    first = members[0]
    cfg = first.cfg
    task, metric = first.state["task"], first.state["metric"]
    enc = TargetEncoder.from_state(first.state["target"])
    names = targets(cfg)
    y = enc.transform(first.oof[names])
    folds = first.oof["fold"].to_numpy()
    oofs = [m.oof[m.pred_cols].to_numpy(float) for m in members]
    scored = (folds >= 0) & np.all([~np.isnan(o).any(axis=1) for o in oofs], axis=0)
    have_test = all(m.test is not None for m in members)
    tests = [m.test[m.pred_cols].to_numpy(float) for m in members] if have_test else None
    thr = cfg["submission"]["threshold"]
    ys = y[scored]

    member_scores = {m.name: score(metric, ys, o[scored], task, threshold=thr) for m, o in zip(members, oofs)}
    weights = None
    oof_blend = np.full_like(oofs[0], np.nan)
    if method == "mean":
        oof_blend = np.mean(oofs, axis=0)
        test_blend = np.mean(tests, axis=0) if have_test else None
    elif method == "rank":
        if get_metric(metric).name not in {"auc"}:
            LOG.warning("rank averaging only preserves order; %s may need calibrated probabilities", metric)
        oof_blend[scored] = np.mean([_rank(o[scored]) for o in oofs], axis=0)
        test_blend = np.mean([_rank(t) for t in tests], axis=0) if have_test else None
    elif method == "weighted":
        for k in sorted(set(folds[scored])):  # nested: weights never see the fold they score
            fit_rows, va = scored & (folds != k), scored & (folds == k)
            w = fit_weights([o[fit_rows] for o in oofs], y[fit_rows], metric, task, thr)
            oof_blend[va] = _blend([o[va] for o in oofs], w)
        weights = fit_weights([o[scored] for o in oofs], ys, metric, task, thr)
        test_blend = _blend(tests, weights) if have_test else None
    elif method == "stack":
        oof_blend, test_blend = stack(oofs, tests, y, folds, scored, task, stack_model, cfg["seed"])
    else:
        raise ValueError(f"unknown method '{method}'")

    cuts = None
    blend_score = score(metric, ys, oof_blend[scored], task, threshold=thr)
    if task == "ordinal":
        classes = np.unique(y).tolist()
        cuts = {"cuts": fit_cuts(ys, oof_blend[scored, 0]), "classes": classes}
        blend_score = score(metric, ys, oof_blend[scored], task, cuts=cuts["cuts"], classes=classes)

    name = name or f"blend_{method}"
    width = max(len(n) for n in [*member_scores, name])
    print(f"\n{'run':{width}s}  {metric}")
    for n, s in member_scores.items():
        print(f"{n:{width}s}  {s:.5f}")
    print(f"{name:{width}s}  {blend_score:.5f}  ({method}{', nested' if method == 'weighted' else ''})")
    if weights is not None:
        print("weights: " + ", ".join(f"{m.name}={w:.3f}" for m, w in zip(members, weights)))

    run_dir = Path(output_dir) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    out_cfg = {**cfg, "name": name, "output_dir": output_dir}
    save_config(out_cfg, run_dir / "config.yaml")
    write_json({**first.state, "cuts": cuts}, run_dir / "state.json")
    oof = first.oof[[first.oof.columns[0], "fold", *names]].copy()
    oof[first.pred_cols] = oof_blend
    oof.to_csv(run_dir / "oof.csv", index=False)
    sub_path = None
    if have_test:
        test = first.test[[first.test.columns[0]]].copy()
        test[first.pred_cols] = test_blend
        test.to_csv(run_dir / "test_pred.csv", index=False)
        sample = read_table(cfg["data"]["sample_submission"]) if cfg["data"]["sample_submission"] else None
        sub = build_submission(out_cfg, test.iloc[:, 0], test_blend, task, enc, metric, sample, cuts)
        sub_path = run_dir / cfg["submission"]["file"]
        sub.to_csv(sub_path, index=False)
        problems = validate_submission(sub, sample, cfg["submission"]["id_col"] or cfg["data"]["id_col"])
        LOG.info("submission %s: %s", sub_path, "; ".join(problems) or "ok")
    info = {"name": name, "modality": cfg["modality"], "task": task, "metric": metric, "cv": blend_score,
            "fold_mean": blend_score, "fold_std": 0.0, "model": f"ensemble:{method}", "backbone": None,
            "members": {m.name: s for m, s in zip(members, member_scores.values())},
            "weights": None if weights is None else weights.tolist(), "n_train": len(first.oof),
            "folds_hash": first.info["folds_hash"], "folds_file": first.info.get("folds_file"),
            "duration_s": round(time.perf_counter() - start, 1), "git": first.info.get("git"),
            "submission": str(sub_path) if sub_path else None, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    write_json(info, run_dir / "metrics.json")
    append_registry(output_dir, info)
    LOG.info("ensemble saved to %s", run_dir)
    return blend_score
