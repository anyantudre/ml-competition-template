"""Turn test predictions into a submission file shaped like sample_submission, then check it."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import targets
from .metrics import cuts_to_labels

PROBA_METRICS = {"auc", "logloss", "map@3", "map@5"}


def _pred_type(cfg: dict, metric: str) -> str:
    t = cfg["submission"]["pred_type"]
    if t != "auto":
        return t
    return "proba" if metric in PROBA_METRICS else "label"


def build_submission(cfg: dict, ids: pd.Series, pred: np.ndarray, task: str, target_enc,
                     metric: str, sample: pd.DataFrame | None, cuts=None) -> pd.DataFrame:
    sub_cfg = cfg["submission"]
    id_col = sub_cfg["id_col"] or cfg["data"]["id_col"] or (sample.columns[0] if sample is not None else "id")
    names = targets(cfg)
    out_cols = [c for c in sample.columns if c != id_col] if sample is not None else names
    kind = _pred_type(cfg, metric)
    thr = sub_cfg["threshold"]
    frame = pd.DataFrame({id_col: ids.to_numpy()})

    if task == "binary":
        classes = target_enc.classes_
        p = pred[:, 0]
        if len(out_cols) == 2 and [str(c) for c in out_cols] == [str(c) for c in classes]:
            frame[out_cols[0]], frame[out_cols[1]] = 1 - p, p
        else:
            frame[out_cols[0]] = p if kind == "proba" else target_enc.decode((p >= thr).astype(int))
    elif task == "multiclass":
        classes = [str(c) for c in target_enc.classes_]
        if len(out_cols) > 1 and set(map(str, out_cols)) == set(classes):
            for c in out_cols:
                frame[c] = pred[:, classes.index(str(c))]
        elif metric.startswith("map@"):
            k = int(metric.split("@")[1])
            top = np.argsort(-pred, axis=1)[:, :k]
            frame[out_cols[0]] = [" ".join(map(str, target_enc.decode(r))) for r in top]
        else:
            frame[out_cols[0]] = target_enc.decode(pred.argmax(axis=1))
    elif task == "multilabel":
        for j, name in enumerate(names):
            col = name if name in out_cols else out_cols[j]
            frame[col] = pred[:, j] if kind == "proba" else (pred[:, j] >= thr).astype(int)
    elif task == "ordinal":
        values = pred[:, 0]
        if cuts is not None:
            labels = cuts_to_labels(values, cuts["cuts"], cuts["classes"])
        else:
            labels = np.rint(values)
        frame[out_cols[0]] = labels.astype(int)
    else:  # regression
        frame[out_cols[0]] = pred[:, 0]

    if sample is not None:
        frame = sample[[id_col]].merge(frame, on=id_col, how="left")[list(sample.columns)]
    return frame


def validate_submission(sub: pd.DataFrame, sample: pd.DataFrame | None, id_col: str | None = None,
                        allow_empty: bool = False) -> list[str]:
    """Return a list of problems (empty list = the file looks valid)."""
    problems = []
    if sample is not None:
        if list(sub.columns) != list(sample.columns):
            problems.append(f"columns {list(sub.columns)} differ from sample {list(sample.columns)}")
        if len(sub) != len(sample):
            problems.append(f"{len(sub)} rows, sample has {len(sample)}")
        id_col = id_col or sample.columns[0]
        if id_col in sub.columns and id_col in sample.columns:
            a, b = set(sub[id_col].astype(str)), set(sample[id_col].astype(str))
            if a != b:
                problems.append(f"id mismatch: {len(b - a)} missing, {len(a - b)} unexpected")
    if id_col and id_col in sub.columns and sub[id_col].duplicated().any():
        problems.append(f"{int(sub[id_col].duplicated().sum())} duplicated ids")
    value_cols = [c for c in sub.columns if c != id_col]
    for c in value_cols:
        col = sub[c]
        if col.isna().any() and not allow_empty:
            problems.append(f"column {c} has {int(col.isna().sum())} missing values")
        if pd.api.types.is_numeric_dtype(col) and not np.isfinite(col.dropna().to_numpy(float)).all():
            problems.append(f"column {c} has non-finite values")
    return problems


def score_submission(sub: pd.DataFrame, solution: pd.DataFrame, metric: str, task: str,
                     id_col: str) -> float:
    """Score a submission against a local solution file (a leaderboard rehearsal).

    The solution holds the id and the true target column(s). Hard-label and probability
    submissions are both accepted.
    """
    from .data import TargetEncoder
    from .metrics import score

    merged = solution.merge(sub, on=id_col, suffixes=("_true", "_pred"))
    if len(merged) != len(solution):
        raise ValueError(f"{len(solution) - len(merged)} solution ids are missing from the submission")
    true_cols = [c for c in solution.columns if c != id_col]
    pred_cols = [c for c in sub.columns if c != id_col]

    def col(name, side):
        return merged[f"{name}_{side}"] if f"{name}_{side}" in merged else merged[name]

    if task == "segmentation":  # Dice between the pixel sets of the two RLE strings
        from .rle import rle_pixels

        scores = []
        for a, b in zip(col(true_cols[0], "true"), col(pred_cols[0], "pred")):
            pa, pb = rle_pixels(a), rle_pixels(b)
            total = len(pa) + len(pb)
            scores.append(1.0 if total == 0 else 2 * len(pa & pb) / total)
        return float(np.mean(scores))
    if task in {"binary", "multiclass"}:
        enc = TargetEncoder(task).fit(col(true_cols[0], "true").to_frame())
        y = enc.transform(col(true_cols[0], "true").to_frame())
        lookup = {str(c): i for i, c in enumerate(enc.classes_)}
        if len(pred_cols) > 1:  # one probability column per class
            pred = np.column_stack([col(str(c), "pred").to_numpy(float) for c in enc.classes_])
        else:
            p = col(pred_cols[0], "pred")
            is_label = p.astype(str).isin(lookup).all()
            if is_label:
                idx = p.astype(str).map(lookup).to_numpy(int)
                pred = idx[:, None].astype(float) if task == "binary" else np.eye(len(lookup))[idx]
            else:
                pred = p.to_numpy(float)[:, None]
    elif task == "multilabel":
        y = np.column_stack([col(c, "true").to_numpy(float) for c in true_cols])
        pred = np.column_stack([col(c, "pred").to_numpy(float) for c in pred_cols])
    else:
        y = col(true_cols[0], "true").to_numpy(float)
        pred = col(pred_cols[0], "pred").to_numpy(float)[:, None]
    return score(metric, y, pred, task)
