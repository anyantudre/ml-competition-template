"""Reload a finished run and predict new rows with every fold model (averaged)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .data import TargetEncoder, read_table, row_ids
from .pipelines import get_pipeline
from .submission import build_submission, validate_submission
from .utils import LOG, read_json


class Predictor:
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        if not (self.run_dir / "models").exists():
            raise FileNotFoundError(f"{self.run_dir}/models not found (was train.save_models false?)")
        self.cfg = load_config(self.run_dir / "config.yaml")
        self.state = read_json(self.run_dir / "state.json")
        self.task = self.state["task"]
        self.metric = self.state["metric"]
        self.enc = TargetEncoder.from_state(self.state["target"])
        columns = self.state["columns"]
        self.pipe = get_pipeline(self.cfg, self.task, self.enc, self.metric,
                                 tuple(columns) if columns else None)
        self.folders = sorted((self.run_dir / "models").glob("fold*"), key=lambda p: int(p.name[4:]))
        self.artifacts = [self.pipe.load(f) for f in self.folders]
        LOG.info("loaded %d fold models from %s", len(self.artifacts), self.run_dir)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        df = df.reset_index(drop=True)
        if self.state["columns"]:  # tabular: absent feature columns count as missing values
            for c in self.state["columns"][0]:
                if c not in df.columns:
                    df[c] = np.nan
        preds = [self.pipe.predict(a, df) for a in self.artifacts]
        pred = np.mean(preds, axis=0)
        if self.cfg["target_transform"] == "log1p":
            pred = np.expm1(pred)
        return pred

    def submission(self, df: pd.DataFrame, sample: pd.DataFrame | None = None) -> pd.DataFrame:
        pred = self.predict(df)
        return build_submission(self.cfg, row_ids(self.cfg, df), pred, self.task, self.enc,
                                self.metric, sample, self.state.get("cuts"))

    def labels(self, pred: np.ndarray):
        """Human-readable output for an API: class labels, probabilities or values."""
        if self.task in {"binary", "multiclass"}:
            idx = (pred[:, 0] >= self.cfg["submission"]["threshold"]).astype(int) if self.task == "binary" else pred.argmax(1)
            return self.enc.decode(idx).tolist()
        if self.task == "multilabel":
            return (pred >= self.cfg["submission"]["threshold"]).astype(int).tolist()
        return pred[:, 0].tolist()


def predict_file(run_dir: str, input_path: str, out: str | None = None,
                 sample_path: str | None = None) -> Path:
    if Path(run_dir, "state.json").exists() and read_json(Path(run_dir) / "state.json")["task"] == "segmentation":
        from .pipelines.segmentation import predict_segmentation

        return predict_segmentation(run_dir, input_path, out)
    p = Predictor(run_dir)
    df = read_table(input_path)
    sample = read_table(sample_path) if sample_path else None
    sub = p.submission(df, sample)
    problems = validate_submission(sub, sample, p.cfg["submission"]["id_col"] or p.cfg["data"]["id_col"])
    for msg in problems:
        LOG.warning("submission problem: %s", msg)
    out_path = Path(out or Path(run_dir) / "submission_predict.csv")
    sub.to_csv(out_path, index=False)
    LOG.info("wrote %s (%d rows)", out_path, len(sub))
    return out_path
