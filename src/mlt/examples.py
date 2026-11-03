"""Small public datasets written in competition format, to learn and test the workflow.

Each example writes train.csv (with target), test.csv (without), sample_submission.csv and
solution.csv (the hidden test labels, so `mlt score` can rehearse the leaderboard).
Bundled scikit-learn datasets work offline; the others download from OpenML or scikit-learn.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split

from .utils import LOG

EXAMPLES = {
    "breast_cancer": ("tabular binary, numeric features (offline)", "configs/examples/breast_cancer.yaml"),
    "adult": ("tabular binary with categoricals, OpenML 1590, 10k rows", "configs/examples/adult.yaml"),
    "car": ("tabular multiclass, all categorical, OpenML 40975", "configs/examples/car.yaml"),
    "diabetes": ("tabular regression (offline)", "configs/examples/diabetes.yaml"),
    "house_prices": ("tabular regression, RMSLE, OpenML 42165", "configs/examples/house_prices.yaml"),
    "wine_quality": ("tabular ordinal target, QWK, OpenML 40691", "configs/examples/wine_quality.yaml"),
    "yeast": ("tabular multilabel (14 labels), OpenML 40597", "configs/examples/yeast.yaml"),
    "newsgroups": ("text multiclass, 4 topics of 20 Newsgroups, 2k docs", "configs/examples/newsgroups_tfidf.yaml"),
    "digits": ("image classification, 8x8 digits upscaled to 32x32 PNG (offline)", "configs/examples/digits.yaml"),
    "digits_seg": ("segmentation, digit strokes as masks, RLE submission (offline)", "configs/examples/digits_seg.yaml"),
}


def list_examples() -> None:
    for name, (desc, config) in EXAMPLES.items():
        print(f"{name:15s} {desc}\n{'':15s} -> mlt run {config}")


def _split(df: pd.DataFrame, target, seed=42, stratify=True, test_size=0.2):
    strat = df[target] if stratify and isinstance(target, str) else None
    return train_test_split(df, test_size=test_size, random_state=seed, stratify=strat)


def _write(out: Path, train: pd.DataFrame, test: pd.DataFrame, id_col: str, targets: list[str],
           sample_values=None):
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "train.csv", index=False)
    test.drop(columns=targets).to_csv(out / "test.csv", index=False)
    test[[id_col, *targets]].to_csv(out / "solution.csv", index=False)
    sample = test[[id_col]].copy()
    for t in targets:
        sample[t] = sample_values if sample_values is not None else train[t].mode().iloc[0]
    sample.to_csv(out / "sample_submission.csv", index=False)
    LOG.info("wrote %s: train %d rows, test %d rows", out, len(train), len(test))


def _openml(data_id: int) -> pd.DataFrame:
    try:
        return datasets.fetch_openml(data_id=data_id, as_frame=True, parser="auto").frame
    except TypeError:
        # scikit-learn's pandas ARFF parser fails on numeric categories with pandas 3
        return datasets.fetch_openml(data_id=data_id, as_frame=True, parser="liac-arff").frame


def _with_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    df.insert(0, "id", np.arange(len(df)))
    return df


def make_example(name: str, out: str | None = None) -> Path:
    if name == "list":
        list_examples()
        return Path(".")
    if name not in EXAMPLES:
        raise KeyError(f"unknown example '{name}'. Try: mlt example list")
    out_dir = Path(out or f"data/{name}")
    builder = globals()[f"_make_{name}"]
    builder(out_dir)
    print(f"next: mlt run {EXAMPLES[name][1]}")
    return out_dir


def _make_breast_cancer(out):
    b = datasets.load_breast_cancer(as_frame=True)
    df = _with_id(b.frame.rename(columns=lambda c: c.replace(" ", "_")))
    df["target"] = df["target"].map({0: "malignant", 1: "benign"})
    tr, te = _split(df, "target")
    _write(out, tr, te, "id", ["target"], sample_values=0.5)


def _make_diabetes(out):
    df = _with_id(datasets.load_diabetes(as_frame=True).frame)
    tr, te = _split(df, "target", stratify=False)
    _write(out, tr, te, "id", ["target"], sample_values=0.0)


def _make_adult(out):
    df = _openml(1590)
    df = df.sample(10_000, random_state=42)
    df["income"] = (df.pop("class").astype(str).str.contains(">50K")).astype(int)
    df = _with_id(df.rename(columns=lambda c: c.replace("-", "_")))
    tr, te = _split(df, "income")
    _write(out, tr, te, "id", ["income"], sample_values=0.5)


def _make_car(out):
    df = _with_id(_openml(40975))
    df = df.rename(columns={"class": "acceptability"})
    tr, te = _split(df, "acceptability")
    _write(out, tr, te, "id", ["acceptability"])


def _make_house_prices(out):
    df = _openml(42165)
    df = df.drop(columns=["Id"], errors="ignore")
    df = _with_id(df)
    tr, te = _split(df, "SalePrice", stratify=False)
    _write(out, tr, te, "id", ["SalePrice"], sample_values=180000.0)


def _make_wine_quality(out):
    df = _with_id(_openml(40691).rename(columns=lambda c: c.replace(" ", "_").replace("-", "_")))
    df["quality"] = df["class"].astype(int) if "class" in df else df["quality"].astype(int)
    df = df.drop(columns=["class"], errors="ignore")
    tr, te = _split(df, "quality")
    _write(out, tr, te, "id", ["quality"], sample_values=6)


def _make_yeast(out):
    df = _openml(40597)
    labels = [c for c in df.columns if str(c).startswith("Class")]
    for c in labels:
        df[c] = (df[c].astype(str).str.upper().isin(["TRUE", "1"])).astype(int)
    df = _with_id(df.rename(columns={c: c.replace("Class", "label_") for c in labels}))
    names = [c.replace("Class", "label_") for c in labels]
    tr, te = _split(df, names, stratify=False)
    _write(out, tr, te, "id", names, sample_values=0.5)


def _make_newsgroups(out):
    cats = ["rec.autos", "sci.space", "comp.graphics", "talk.politics.mideast"]
    ng = datasets.fetch_20newsgroups(subset="all", categories=cats, remove=("headers", "footers", "quotes"),
                                     random_state=42)
    df = pd.DataFrame({"text": ng.data, "topic": [ng.target_names[i] for i in ng.target]})
    df["text"] = df["text"].str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df["text"].str.len() > 20].sample(2000, random_state=42)
    df = _with_id(df)
    tr, te = _split(df, "topic")
    _write(out, tr, te, "id", ["topic"])


def _digit_images(out: Path, size: int = 32):
    from PIL import Image

    d = datasets.load_digits()
    (out / "images").mkdir(parents=True, exist_ok=True)
    arrays = []
    for i, img in enumerate(d.images):
        arr = (img / 16.0 * 255).astype(np.uint8)
        big = Image.fromarray(arr).resize((size, size), Image.BILINEAR)
        big.save(out / "images" / f"{i}.png")
        arrays.append(np.asarray(big))
    return d.target, np.stack(arrays)


def _make_digits(out):
    target, _ = _digit_images(out)
    df = pd.DataFrame({"id": np.arange(len(target)), "image": [f"{i}.png" for i in range(len(target))],
                       "digit": target})
    tr, te = _split(df, "digit")
    _write(out, tr, te, "id", ["digit"])


def _make_digits_seg(out):
    from PIL import Image

    from .rle import rle_encode

    target, arrays = _digit_images(out)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    masks = arrays > 96
    for i, m in enumerate(masks):
        Image.fromarray((m * 255).astype(np.uint8)).save(out / "masks" / f"{i}.png")
    df = pd.DataFrame({"id": np.arange(len(target)), "image": [f"images/{i}.png" for i in range(len(target))],
                       "mask": [f"masks/{i}.png" for i in range(len(target))], "digit": target})
    tr, te = _split(df, "digit")
    out.mkdir(parents=True, exist_ok=True)
    tr.drop(columns=["digit"]).to_csv(out / "train.csv", index=False)
    te[["id", "image"]].to_csv(out / "test.csv", index=False)
    sol = pd.DataFrame({"id": te["id"], "rle_mask": [rle_encode(masks[i]) for i in te["id"]]})
    sol.to_csv(out / "solution.csv", index=False)
    pd.DataFrame({"id": te["id"], "rle_mask": ""}).to_csv(out / "sample_submission.csv", index=False)
    LOG.info("wrote %s: train %d images, test %d images", out, len(tr), len(te))
