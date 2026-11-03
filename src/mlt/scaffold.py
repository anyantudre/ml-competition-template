"""`mlt init`: read a competition's files and write a starter config.

It guesses the id column, the target(s), the modality and the task from train, test and
sample_submission. Read the result: every guess is printed and easy to change.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .data import DEFAULT_METRIC, infer_task, read_table

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".dcm")


def _guess_id(cols_train, cols_test, sample):
    if sample is not None and sample.columns[0] in cols_test:
        return sample.columns[0]
    for c in cols_test:
        if c.lower() in {"id", "row_id", "sample_id", "image_id", "uid"} or c.lower().endswith("_id"):
            return c
    return None


def scaffold(train_path, test_path=None, sample_path=None, out=None, metric=None, name=None) -> Path:
    train = read_table(train_path, nrows=20000)
    test = read_table(test_path, nrows=2000) if test_path else None
    sample = read_table(sample_path, nrows=2000) if sample_path else None
    test_cols = list(test.columns) if test is not None else []
    id_col = _guess_id(list(train.columns), test_cols, sample)

    if sample is not None:
        sample_cols = [c for c in sample.columns if c != id_col]
        targets = [c for c in sample_cols if c in train.columns]
        if not targets:  # one probability column per class
            targets = [c for c in train.columns if c not in test_cols and c != id_col][:1]
    else:
        targets = [c for c in train.columns if c not in test_cols and c != id_col] if test is not None else []
    if not targets:
        raise ValueError("could not find the target column; pass --test or --sample, or edit the config")
    target = targets[0] if len(targets) == 1 else targets

    features = [c for c in train.columns if c not in targets and c != id_col]
    modality, extra = "tabular", {}
    for c in features:
        s = train[c].dropna().astype(str)
        if s.empty:
            continue
        if s.str.lower().str.endswith(IMAGE_EXT).mean() > 0.9:
            modality, extra = "image", {"image_col": c, "image_dir": str(Path(train_path).parent)}
            break
        if s.str.len().mean() > 80 and train[c].dtype == object:
            modality, extra = "text", {"text_col": c}
    task = infer_task(train[targets])
    if sample is not None and len(targets) == 1 and len([c for c in sample.columns if c != id_col]) > 1:
        task = "multiclass"
    metric = metric or DEFAULT_METRIC[task]
    model = {"tabular": {"name": "lgbm"}, "text": {"name": "tfidf"},
             "image": {"backbone": "resnet18"}}[modality]
    name = name or Path(train_path).parent.name or "competition"

    cfg = {"modality": modality, "task": task, "metric": metric,
           "data": {"train": str(train_path).replace("\\", "/"),
                    "test": str(test_path).replace("\\", "/") if test_path else None,
                    "sample_submission": str(sample_path).replace("\\", "/") if sample_path else None,
                    "id_col": id_col, "target": target, **extra},
           "model": model}
    out_path = Path(out or f"configs/{name}.yaml")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = (f"# Starter config written by `mlt init` for {name}.\n"
              "# Check the guesses below, then: mlt check <this file>  and  mlt run <this file>\n")
    out_path.write_text(header + yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"id column: {id_col}\ntarget: {target}\nmodality: {modality}\ntask: {task}\nmetric: {metric}")
    print(f"wrote {out_path}\nnext: mlt check {out_path.as_posix()}  then  mlt run {out_path.as_posix()}")
    return out_path


