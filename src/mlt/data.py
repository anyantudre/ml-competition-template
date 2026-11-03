"""Reading competition files and working out the task and the feature columns."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import targets
from .utils import LOG


def read_table(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found (paths are relative to the folder you run mlt from)")
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
        return df.head(nrows) if nrows else df
    if suffix == ".feather":
        df = pd.read_feather(path)
        return df.head(nrows) if nrows else df
    sep = "\t" if suffix in {".tsv", ".tab"} else ","
    return pd.read_csv(path, nrows=nrows, sep=sep)


def load_data(cfg: dict):
    d = cfg["data"]
    train = read_table(d["train"], d["nrows"])
    test = read_table(d["test"]) if d["test"] else None
    sample = read_table(d["sample_submission"]) if d["sample_submission"] else None
    if sample is not None and test is not None and len(sample) != len(test):
        LOG.warning("sample_submission has %d rows but test has %d", len(sample), len(test))
    missing = [t for t in targets(cfg) if t not in train.columns]
    if missing:
        raise KeyError(f"target column(s) {missing} not in {d['train']}")
    return train, test, sample


def infer_task(y: pd.DataFrame) -> str:
    """Guess the task from the target column(s). Set `task` in the config to be explicit."""
    if y.shape[1] > 1:
        return "multilabel"
    s = y.iloc[:, 0].dropna()
    n_unique = s.nunique()
    if n_unique == 2:
        return "binary"
    if s.dtype == object or str(s.dtype) in {"category", "bool", "string", "str"}:
        return "multiclass"
    if pd.api.types.is_integer_dtype(s) and n_unique <= max(3, min(30, int(0.01 * len(s)))):
        return "multiclass"
    return "regression"


DEFAULT_METRIC = {
    "binary": "auc",
    "multiclass": "logloss",
    "multilabel": "logloss",
    "regression": "rmse",
    "ordinal": "qwk",
    "segmentation": "dice",
}


def resolve_task(cfg: dict, train: pd.DataFrame) -> tuple[str, str]:
    task = cfg["task"]
    if task is None:
        task = infer_task(train[targets(cfg)])
        LOG.info("task not set, inferred '%s' from the target (set `task:` to be explicit)", task)
    metric = cfg["metric"] or DEFAULT_METRIC[task]
    return task, metric


class TargetEncoder:
    """Maps class labels to 0..K-1 and back. Regression targets pass through as floats."""

    def __init__(self, task: str):
        self.task = task
        self.classes_: list | None = None
        self.names: list[str] = []

    def fit(self, y: pd.DataFrame) -> TargetEncoder:
        self.names = list(y.columns)
        if self.task in {"binary", "multiclass"}:
            self.classes_ = sorted(y.iloc[:, 0].dropna().unique().tolist(), key=_sort_key)
        return self

    def transform(self, y: pd.DataFrame) -> np.ndarray:
        if self.task in {"binary", "multiclass"}:
            mapping = {c: i for i, c in enumerate(self.classes_)}
            out = y.iloc[:, 0].map(mapping)
            if out.isna().any():
                raise ValueError("target has missing values or unseen labels")
            return out.to_numpy(dtype=np.int64)
        if self.task == "multilabel":
            return y.to_numpy(dtype=np.float32)
        return y.iloc[:, 0].to_numpy(dtype=np.float64)

    def decode(self, idx: np.ndarray):
        return np.asarray(self.classes_, dtype=object)[idx]

    @property
    def n_outputs(self) -> int:
        if self.task == "multiclass":
            return len(self.classes_)
        if self.task == "multilabel":
            return len(self.names)
        return 1

    def pred_columns(self) -> list[str]:
        if self.task == "multiclass":
            return [f"pred_{c}" for c in self.classes_]
        if self.task == "multilabel":
            return [f"pred_{c}" for c in self.names]
        return ["pred"]

    def state(self) -> dict:
        return {"task": self.task, "classes": self.classes_, "names": self.names}

    @classmethod
    def from_state(cls, state: dict) -> TargetEncoder:
        enc = cls(state["task"])
        enc.classes_ = state["classes"]
        enc.names = state["names"]
        return enc


def _sort_key(v):
    return (0, v) if isinstance(v, (int, float, np.integer, np.floating)) else (1, str(v))


def feature_columns(cfg: dict, train: pd.DataFrame, test: pd.DataFrame | None = None):
    """Return (features, categorical, numeric, dates) for tabular data."""
    d = cfg["data"]
    excluded = set(targets(cfg)) | set(d["drop"] or [])
    if d["id_col"]:
        excluded.add(d["id_col"])
    features = [c for c in train.columns if c not in excluded]
    if test is not None:
        missing = [c for c in features if c not in test.columns]
        if missing:
            LOG.warning("columns missing from test, dropped: %s", missing)
            features = [c for c in features if c not in missing]
    dates = [c for c in (d["date_cols"] or []) if c in features]
    if d["cat_cols"] is not None:
        cats = [c for c in d["cat_cols"] if c in features]
    else:
        cats = [
            c for c in features
            if c not in dates and not pd.api.types.is_numeric_dtype(train[c])
        ]
        cats += [c for c in features if pd.api.types.is_bool_dtype(train[c]) and c not in cats]
    if d["num_cols"] is not None:
        nums = [c for c in d["num_cols"] if c in features]
    else:
        nums = [c for c in features if c not in cats and c not in dates]
    return features, cats, nums, dates


def row_ids(cfg: dict, df: pd.DataFrame) -> pd.Series:
    col = cfg["data"]["id_col"]
    if col and col in df.columns:
        return df[col].reset_index(drop=True)
    return pd.Series(np.arange(len(df)), name="row")
