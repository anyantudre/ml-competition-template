"""Cross-validation folds, written once and reused by every experiment on the same data.

Every model and every ensemble must see the same folds, otherwise out-of-fold predictions leak
into the blender. The folds file stores one fold number per training row (-1 = always train).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import model_selection

from .data import row_ids
from .utils import LOG, array_hash

SCHEMES = ("kfold", "stratified", "binned", "group", "stratified_group", "time", "holdout")


def default_scheme(cfg: dict, task: str) -> str:
    cv = cfg["cv"]
    if cv["scheme"]:
        return cv["scheme"]
    if cv["time_col"]:
        return "time"
    if cv["group_col"]:
        return "stratified_group" if task in {"binary", "multiclass", "ordinal"} else "group"
    if task in {"binary", "multiclass", "ordinal", "multilabel"}:
        return "stratified"
    return "kfold"


def _strat_labels(y: np.ndarray, task: str, n_splits: int) -> np.ndarray:
    if task == "multilabel":
        # stratify on the label combination, falling back to the most frequent label
        combo = pd.Series(["".join(map(str, r)) for r in y.astype(int)])
        if combo.value_counts().min() >= n_splits:
            return combo.to_numpy()
        return y.argmax(axis=1)
    return y


def sturges_bins(n: int) -> int:
    return int(np.floor(1 + np.log2(n)))


def make_folds(df: pd.DataFrame, y: np.ndarray, cfg: dict, task: str) -> np.ndarray:
    cv = cfg["cv"]
    scheme = default_scheme(cfg, task)
    k = cv["n_splits"]
    seed = cfg["seed"]
    n = len(df)
    folds = np.full(n, -1, dtype=np.int64)
    idx = np.arange(n)

    if scheme == "kfold":
        splitter = model_selection.KFold(k, shuffle=True, random_state=seed).split(idx)
    elif scheme == "stratified":
        labels = _strat_labels(y, task, k)
        splitter = model_selection.StratifiedKFold(k, shuffle=True, random_state=seed).split(idx, labels)
    elif scheme == "binned":
        n_bins = cv["n_bins"] or (sturges_bins(n) if n <= 10_000 else 20)
        bins = pd.cut(np.asarray(y, dtype=float).ravel(), bins=n_bins, labels=False)
        splitter = model_selection.StratifiedKFold(k, shuffle=True, random_state=seed).split(idx, bins)
    elif scheme in {"group", "stratified_group"}:
        if not cv["group_col"]:
            raise ValueError(f"cv.group_col is required for the {scheme} scheme")
        groups = df[cv["group_col"]].to_numpy()
        if scheme == "group":
            splitter = model_selection.GroupKFold(k).split(idx, groups=groups)
        else:
            splitter = model_selection.StratifiedGroupKFold(k, shuffle=True, random_state=seed).split(
                idx, _strat_labels(y, task, k), groups
            )
    elif scheme == "time":
        if not cv["time_col"]:
            raise ValueError("cv.time_col is required for the time scheme")
        order = np.argsort(pd.to_datetime(df[cv["time_col"]], errors="coerce").to_numpy(), kind="stable")
        # expanding window: fold f validates on chunk f+1 and trains on everything before it
        chunks = np.array_split(order, k + 1)
        for f, chunk in enumerate(chunks[1:]):
            folds[chunk] = f
        return folds
    elif scheme == "holdout":
        stratify = y if task in {"binary", "multiclass", "ordinal"} else None
        _, valid = model_selection.train_test_split(
            idx, test_size=cv["holdout_size"], random_state=seed, stratify=stratify
        )
        folds[valid] = 0
        return folds
    else:
        raise ValueError(f"unknown cv.scheme '{scheme}', choose from {SCHEMES}")

    for f, (_, valid) in enumerate(splitter):
        folds[valid] = f
    return folds


def train_mask(folds: np.ndarray, fold: int, scheme: str) -> np.ndarray:
    if scheme == "time":
        return folds < fold
    return folds != fold


def fold_ids(folds: np.ndarray) -> list[int]:
    return sorted(int(f) for f in np.unique(folds) if f >= 0)


def folds_path(cfg: dict, task: str) -> Path:
    if cfg["cv"]["folds_file"]:
        return Path(cfg["cv"]["folds_file"])
    stem = Path(cfg["data"]["train"]).stem
    parent = Path(cfg["data"]["train"]).parent.name
    scheme = default_scheme(cfg, task)
    rows = f"_n{cfg['data']['nrows']}" if cfg["data"]["nrows"] else ""
    name = f"{parent}_{stem}_{scheme}_{cfg['cv']['n_splits']}_s{cfg['seed']}{rows}.csv"
    return Path(cfg["output_dir"]) / "folds" / name


def get_folds(df: pd.DataFrame, y: np.ndarray, cfg: dict, task: str, force: bool = False):
    """Load the folds file for this data and scheme, creating it the first time."""
    path = folds_path(cfg, task)
    ids = row_ids(cfg, df)
    if path.exists() and not force:
        saved = pd.read_csv(path)
        if len(saved) != len(df) or not np.array_equal(saved.iloc[:, 0].astype(str), ids.astype(str)):
            raise ValueError(
                f"{path} does not match the training rows. Delete it or run `mlt folds --force`."
            )
        folds = saved["fold"].to_numpy()
    else:
        folds = make_folds(df, y, cfg, task)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({ids.name or "row": ids, "fold": folds}).to_csv(path, index=False)
        LOG.info("wrote folds to %s", path)
    return folds, path, array_hash(folds)


def describe(folds: np.ndarray, y: np.ndarray, task: str) -> pd.DataFrame:
    rows = []
    for f in fold_ids(folds):
        m = folds == f
        row = {"fold": f, "rows": int(m.sum())}
        if task in {"binary", "multiclass", "ordinal"}:
            row["target_mean"] = float(np.mean(y[m]))
        elif task == "regression":
            row["target_mean"] = float(np.mean(y[m]))
            row["target_std"] = float(np.std(y[m]))
        rows.append(row)
    return pd.DataFrame(rows)
