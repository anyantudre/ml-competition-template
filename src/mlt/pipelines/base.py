"""Interface shared by the modality pipelines."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..metrics import get_metric, score


class Pipeline:
    def __init__(self, cfg: dict, task: str, target_enc, metric: str, columns):
        self.cfg = cfg
        self.task = task
        self.target_enc = target_enc
        self.metric = metric
        self.columns = columns  # (features, cats, nums, dates) for tabular, else None
        self.n_out = target_enc.n_outputs

    # subclasses implement these two
    def fit_fold(self, train: pd.DataFrame, y_train: np.ndarray, valid: pd.DataFrame,
                 y_valid: np.ndarray, test: pd.DataFrame | None, fold: int):
        raise NotImplementedError

    def predict(self, artifact, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    # optional
    def importance(self, artifact) -> pd.Series | None:
        return None

    def save(self, artifact, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, folder / "artifact.joblib", compress=3)

    def load(self, folder: Path):
        return joblib.load(folder / "artifact.joblib")

    # helpers
    def score_fn(self, y_valid: np.ndarray):
        thr = self.cfg["submission"]["threshold"]
        return lambda pred: score(self.metric, y_valid, pred, self.task, threshold=thr)

    @property
    def greater_is_better(self) -> bool:
        return get_metric(self.metric).greater_is_better
