"""Experiment configuration: one YAML file per experiment, merged over these defaults.

Unknown keys raise an error (with a suggestion), so a typo never silently falls back to a default.
A config can start from another one with ``extends: other.yaml`` and be adjusted from the command
line with ``--set model.params.learning_rate=0.05``.
"""

from __future__ import annotations

import copy
import difflib
from pathlib import Path

import yaml

DEFAULTS: dict = {
    "name": None,  # defaults to the config file name
    "extends": None,
    "seed": 42,
    "modality": "tabular",  # tabular | text | image | segmentation
    "task": None,  # binary | multiclass | multilabel | regression | ordinal (inferred if empty)
    "metric": None,  # see mlt.metrics.METRICS (a sensible default per task if empty)
    "output_dir": "outputs",
    "target_transform": None,  # log1p: train on log1p(y), predict expm1 (RMSLE-style targets)
    "data": {
        "train": None,
        "test": None,
        "sample_submission": None,
        "id_col": None,
        "target": None,  # a column name, or a list of columns for multilabel
        "drop": [],
        "cat_cols": None,  # None: object, category and bool columns
        "num_cols": None,  # None: every other numeric column
        "date_cols": [],
        "text_col": None,
        "image_col": None,
        "mask_col": None,
        "image_dir": None,
        "test_image_dir": None,
        "nrows": None,  # read only the first n rows (quick debugging)
    },
    "cv": {
        # kfold | stratified | binned | group | stratified_group | time | holdout
        "scheme": None,
        "n_splits": 5,
        "group_col": None,
        "time_col": None,
        "n_bins": None,  # binned scheme; None uses Sturges' rule
        "holdout_size": 0.2,
        "folds_file": None,  # None: outputs/folds/<train>_<scheme>_<k>_s<seed>.csv
    },
    "features": {
        "encoding": "auto",  # auto | label | onehot | native
        "rare_min_count": 0,  # categories seen fewer times become RARE
        "fit_unsupervised_on_test": True,  # vocabularies, counts and aggregates also see test
        "count_encode": [],
        "target_encode": [],
        "target_encode_smoothing": 20.0,
        "combine": [],  # pairs of categorical columns, e.g. [[city, job]]
        "date_parts": ["year", "month", "day", "dayofweek", "dayofyear", "hour", "is_weekend"],
        "aggregates": [],  # e.g. [{by: store, cols: [sales], aggs: [mean, max]}]
        "log1p": [],
        "bins": {},  # {column: number_of_bins}
        "selection": {"method": None, "k": None, "threshold": None},
        "text": {
            "vectorizer": "tfidf",  # tfidf | count
            "ngram_range": [1, 2],
            "min_df": 2,
            "max_features": 200000,
            "sublinear_tf": True,
            "analyzer": "word",
            "clean": False,
            "embeddings": None,  # fastText/GloVe .vec or .txt file for the BiLSTM
            "freeze_embeddings": True,
            "max_vocab": 50000,
        },
    },
    "model": {
        "name": None,
        "backbone": None,  # transformers / timm / smp encoder name
        "params": {},
        "early_stopping_rounds": 100,
    },
    "train": {
        "folds": None,  # run only these folds, e.g. [0] for a quick check
        "epochs": 5,
        "batch_size": 32,
        "lr": 3.0e-4,
        "weight_decay": 0.01,
        "scheduler": "cosine",  # cosine | linear | constant
        "warmup_ratio": 0.0,
        "patience": 3,
        "amp": True,  # mixed precision, only used on CUDA
        "device": "auto",
        "num_workers": 0,
        "grad_accum": 1,
        "max_grad_norm": 1.0,
        "max_len": 256,
        "image_size": 224,
        "augment": True,
        "tta": False,
        "pretrained": True,
        "loss": None,  # regression: mse | l1 | huber
        "save_models": True,
    },
    "submission": {
        "file": "submission.csv",
        "pred_type": "auto",  # auto | proba | label | value
        "threshold": 0.5,
        "id_col": None,
        "rle_order": "F",  # F: column-major (most Kaggle masks), C: row-major
        "mask_threshold": 0.5,
    },
    "tune": {"n_trials": 30, "timeout": None, "space": {}},
}

# Dictionaries whose keys are free-form (passed through to a library).
OPEN_KEYS = {"model.params", "features.bins", "tune.space"}


class ConfigError(ValueError):
    pass


def deep_merge(base: dict, override: dict, _path: str = "") -> dict:
    """Merge nested dicts. For free-form keys an empty dict clears the value, so
    `--set model.params={}` drops the params inherited from a parent config."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        path = f"{_path}{k}"
        if isinstance(v, dict) and isinstance(out.get(k), dict) and (v or path not in OPEN_KEYS):
            out[k] = deep_merge(out[k], v, f"{path}.")
        else:
            out[k] = copy.deepcopy(v)
    return out


def _check_keys(user: dict, ref: dict, prefix: str = "") -> None:
    for key, value in user.items():
        path = f"{prefix}{key}"
        if key not in ref:
            hint = difflib.get_close_matches(key, list(ref), n=1)
            msg = f"Unknown config key '{path}'."
            if hint:
                msg += f" Did you mean '{prefix}{hint[0]}'?"
            raise ConfigError(msg)
        if isinstance(value, dict) and isinstance(ref[key], dict) and path not in OPEN_KEYS:
            _check_keys(value, ref[key], prefix=f"{path}.")


def merge_layer(base: dict, layer: dict) -> dict:
    """Apply one config layer. Switching model.name drops the params of the previous model."""
    new_name = (layer.get("model") or {}).get("name")
    old_name = (base.get("model") or {}).get("name")
    if new_name and old_name and new_name != old_name:
        base = deep_merge(base, {"model": {"params": {}}})
    return deep_merge(base, layer)


def _read_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a YAML mapping")
    parent = data.get("extends")
    if parent:
        parent_path = (path.parent / parent).resolve()
        if not parent_path.exists():
            parent_path = Path(parent).resolve()
        data = merge_layer(_read_yaml(parent_path), {k: v for k, v in data.items() if k != "extends"})
    return data


def parse_override(item: str) -> dict:
    if "=" not in item:
        raise ConfigError(f"Override '{item}' must look like key.sub=value")
    key, raw = item.split("=", 1)
    value = yaml.safe_load(raw) if raw != "" else None
    out: dict = {}
    node = out
    parts = key.strip().split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value
    return out


def load_config(path: str | Path | None = None, overrides: list[str] | None = None,
                data: dict | None = None) -> dict:
    """Load a YAML config (or a dict), apply overrides, validate keys and fill defaults."""
    user: dict = {}
    if path is not None:
        path = Path(path)
        user = _read_yaml(path)
        user.setdefault("name", path.stem)
    if data:
        user = merge_layer(user, data)
    for item in overrides or []:
        user = merge_layer(user, parse_override(item))
    user.pop("extends", None)
    _check_keys(user, DEFAULTS)
    cfg = deep_merge(DEFAULTS, user)
    validate(cfg)
    return cfg


MODALITIES = {"tabular", "text", "image", "segmentation"}
TASKS = {"binary", "multiclass", "multilabel", "regression", "ordinal", "segmentation"}


def validate(cfg: dict) -> None:
    if cfg["modality"] not in MODALITIES:
        raise ConfigError(f"modality must be one of {sorted(MODALITIES)}")
    if cfg["task"] is not None and cfg["task"] not in TASKS:
        raise ConfigError(f"task must be one of {sorted(TASKS)}")
    if not cfg["data"]["train"]:
        raise ConfigError("data.train is required")
    if cfg["modality"] != "segmentation" and not cfg["data"]["target"]:
        raise ConfigError("data.target is required")
    if cfg["modality"] == "text" and not cfg["data"]["text_col"]:
        raise ConfigError("data.text_col is required for text")
    if cfg["modality"] in {"image", "segmentation"} and not cfg["data"]["image_col"]:
        raise ConfigError("data.image_col is required for images")
    if cfg["modality"] == "segmentation":
        cfg["task"] = "segmentation"
        if not cfg["data"]["mask_col"]:
            raise ConfigError("data.mask_col is required for segmentation")
    if cfg["target_transform"] not in (None, "log1p"):
        raise ConfigError("target_transform must be empty or log1p")


def save_config(cfg: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def targets(cfg: dict) -> list[str]:
    t = cfg["data"]["target"]
    if t is None:
        return []
    return list(t) if isinstance(t, (list, tuple)) else [t]
