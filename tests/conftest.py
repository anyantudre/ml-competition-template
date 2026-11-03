"""Tiny synthetic competitions written to a temp folder, one per task."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from mlt.config import load_config


def make_frame(n: int, task: str, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    city = rng.choice(["a", "b", "c", "d", None], size=n, p=[0.4, 0.3, 0.2, 0.05, 0.05])
    job = rng.choice([f"j{i}" for i in range(12)], size=n)
    date = pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
    signal = 1.5 * x1 - x2 + (city == "a") * 1.0 + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame({"id": np.arange(n) + 1000 * seed, "x1": x1, "x2": x2, "city": city, "job": job,
                       "date": date.astype(str), "grp": rng.integers(0, 40, n)})
    df.loc[rng.choice(n, n // 20, replace=False), "x2"] = np.nan
    if task == "binary":
        df["target"] = np.where(signal > 0, "yes", "no")
    elif task == "multiclass":
        df["target"] = pd.cut(signal, [-np.inf, -1, 0.5, np.inf], labels=["low", "mid", "high"]).astype(str)
    elif task == "regression":
        df["target"] = np.exp(signal / 3) * 10
    elif task == "ordinal":
        df["target"] = np.clip(np.rint(signal + 3), 1, 5).astype(int)
    elif task == "multilabel":
        df["l1"] = (signal > 0).astype(int)
        df["l2"] = (x2 > 0.3).astype(int)
        df["l3"] = (np.nan_to_num(x1) + np.nan_to_num(x2) > 0).astype(int)
    return df


def write_competition(folder: Path, task: str, n: int = 300) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    train = make_frame(n, task, seed=0)
    test = make_frame(n // 3, task, seed=1)
    targets = ["l1", "l2", "l3"] if task == "multilabel" else ["target"]
    train.to_csv(folder / "train.csv", index=False)
    test.drop(columns=targets).to_csv(folder / "test.csv", index=False)
    test[["id", *targets]].to_csv(folder / "solution.csv", index=False)
    sample = test[["id"]].copy()
    for t in targets:
        sample[t] = 0
    sample.to_csv(folder / "sample_submission.csv", index=False)
    return {"train": str(folder / "train.csv"), "test": str(folder / "test.csv"),
            "sample_submission": str(folder / "sample_submission.csv"), "id_col": "id",
            "target": targets if task == "multilabel" else "target"}


FAST_PARAMS = {
    "lgbm": {"n_estimators": 60, "learning_rate": 0.1, "num_leaves": 8, "min_child_samples": 5},
    "xgb": {"n_estimators": 60, "learning_rate": 0.1, "max_depth": 3},
    "catboost": {"iterations": 60, "learning_rate": 0.1, "depth": 4},
    "rf": {"n_estimators": 30},
    "et": {"n_estimators": 30},
    "logreg": {},
    "ridge": {},
}


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def make_cfg(workdir):
    def factory(task="binary", model_name="lgbm", n=300, **extra):
        data = write_competition(workdir / "data" / task, task, n)
        raw = {"name": f"{task}_{model_name}", "task": task, "data": data,
               "model": {"name": model_name, "params": FAST_PARAMS.get(model_name, {}), "early_stopping_rounds": 10},
               "cv": {"n_splits": 3}, "output_dir": str(workdir / "outputs")}
        for key, value in extra.items():
            raw.setdefault(key, {})
            if isinstance(value, dict):
                raw[key].update(value)
            else:
                raw[key] = value
        path = workdir / "configs" / f"{raw['name']}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        return load_config(path), path

    return factory


def has(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
