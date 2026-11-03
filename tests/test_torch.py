"""Neural pipelines on tiny data, CPU, one or two epochs. Marked `torch` (needs the extras)."""

import numpy as np
import pandas as pd
import pytest
import yaml

from conftest import has
from mlt.config import load_config
from mlt.predict import Predictor
from mlt.train import run_cv

TORCH = [pytest.mark.torch, pytest.mark.skipif(not has("torch"), reason="needs torch")]


def needs(module):
    """Mark a test as needing torch plus an optional module."""
    def deco(fn):
        fn = pytest.mark.skipif(not has(module), reason=f"needs {module}")(fn)
        return pytest.mark.torch(fn)
    return deco
FAST = {"epochs": 2, "batch_size": 16, "lr": 3e-3, "patience": 2, "num_workers": 0}


def _load(workdir, raw):
    raw.setdefault("output_dir", str(workdir / "outputs"))
    raw.setdefault("cv", {"n_splits": 2})
    path = workdir / f"{raw['name']}.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_config(path)


@needs("torch")
@pytest.mark.parametrize("task", ["binary", "multiclass", "regression", "multilabel"])
def test_entity_embeddings(make_cfg, task):
    cfg, _ = make_cfg(task, "entity_nn", model={"params": {"epochs": 2, "batch_size": 32}})
    res = run_cv(cfg)
    assert np.isfinite(res.score)
    assert Predictor(res.run_dir).predict(pd.read_csv(cfg["data"]["test"])).shape == res.test_pred.shape


def _text_competition(folder, n=160, regression=False):
    rng = np.random.default_rng(0)
    pos = ["great", "excellent", "love", "fine", "superb"]
    neg = ["awful", "bad", "hate", "poor", "boring"]
    rows = []
    for i in range(n):
        label = int(rng.integers(0, 2))
        words = rng.choice(pos if label else neg, 4).tolist() + rng.choice(["the", "a", "movie", "plot"], 6).tolist()
        rng.shuffle(words)
        rows.append({"id": i, "text": " ".join(words), "y": float(label * 4 + rng.normal()) if regression else label})
    df = pd.DataFrame(rows)
    folder.mkdir(parents=True, exist_ok=True)
    df.iloc[:120].to_csv(folder / "train.csv", index=False)
    df.iloc[120:].drop(columns="y").to_csv(folder / "test.csv", index=False)
    return {"train": str(folder / "train.csv"), "test": str(folder / "test.csv"), "id_col": "id",
            "target": "y", "text_col": "text"}


@pytest.mark.parametrize("name,task", [("tfidf", "binary"), ("nb", "binary"), ("tfidf", "regression"),
                                       pytest.param("bilstm", "binary", marks=TORCH),
                                       pytest.param("bilstm", "regression", marks=TORCH)])
def test_text_models(workdir, name, task):
    data = _text_competition(workdir / "text", regression=task == "regression")
    lstm = name == "bilstm"
    train = {**FAST, "max_len": 16, **({"epochs": 12, "lr": 1e-2, "patience": 12} if lstm else {})}
    cfg = _load(workdir, {"name": f"text_{name}_{task}", "modality": "text", "task": task, "data": data,
                          "model": {"name": name, "params": {"hidden": 16, "embed_dim": 16} if lstm else {}},
                          "features": {"text": {"min_df": 1}}, "train": train})
    res = run_cv(cfg)
    if task == "binary":
        assert res.score > (0.8 if lstm else 0.9)
    assert (res.run_dir / "submission.csv").exists()
    assert np.allclose(Predictor(res.run_dir).predict(pd.read_csv(data["test"])), res.test_pred, atol=1e-5)


@needs("torch")
def test_bilstm_with_word_vectors(workdir):
    data = _text_competition(workdir / "text")
    vec = workdir / "vectors.vec"
    words = ["great", "awful", "movie", "the"]
    vec.write_text("4 8\n" + "\n".join(w + " " + " ".join(["0.1"] * 8) for w in words), encoding="utf-8")
    cfg = _load(workdir, {"name": "lstm_vec", "modality": "text", "task": "binary", "data": data,
                          "model": {"name": "bilstm", "params": {"hidden": 8}},
                          "features": {"text": {"embeddings": str(vec), "min_df": 1}}, "train": FAST})
    assert np.isfinite(run_cv(cfg).score)


@pytest.mark.network
@needs("transformers")
def test_transformer_tiny(workdir):
    data = _text_competition(workdir / "text")
    cfg = _load(workdir, {"name": "bert", "modality": "text", "task": "binary", "data": data,
                          "model": {"name": "transformer", "backbone": "hf-internal-testing/tiny-random-bert"},
                          "train": {**FAST, "lr": 1e-3, "max_len": 32}})
    res = run_cv(cfg)
    again = Predictor(res.run_dir).predict(pd.read_csv(data["test"]))
    assert np.allclose(again, res.test_pred, atol=1e-5)


def _image_competition(folder, n=48, seg=False):
    from PIL import Image

    rng = np.random.default_rng(0)
    (folder / "img").mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        label = i % 2
        arr = (rng.random((40, 36)) * 60).astype(np.uint8)
        mask = np.zeros((40, 36), np.uint8)
        if label:
            arr[10:30, 8:28] += 150
            mask[10:30, 8:28] = 255
        Image.fromarray(arr).save(folder / "img" / f"{i}.png")
        Image.fromarray(mask).save(folder / "img" / f"m{i}.png")
        rows.append({"id": i, "image": f"{i}.png", "mask": f"m{i}.png", "y": label})
    df = pd.DataFrame(rows)
    df.iloc[:36].to_csv(folder / "train.csv", index=False)
    df.iloc[36:].drop(columns=["y", "mask"]).to_csv(folder / "test.csv", index=False)
    pd.DataFrame({"id": df.id.iloc[36:], "rle": ""}).to_csv(folder / "sample.csv", index=False)
    data = {"train": str(folder / "train.csv"), "test": str(folder / "test.csv"), "id_col": "id",
            "image_col": "image", "image_dir": str(folder / "img")}
    if seg:
        data.update(mask_col="mask", sample_submission=str(folder / "sample.csv"))
    else:
        data["target"] = "y"
    return data


@needs("timm")
def test_image_classification(workdir):
    data = _image_competition(workdir / "images")
    cfg = _load(workdir, {"name": "img", "modality": "image", "task": "binary", "data": data,
                          "model": {"backbone": "resnet18"},
                          "train": {**FAST, "image_size": 32, "pretrained": False, "tta": True}})
    res = run_cv(cfg)
    assert res.test_pred.shape == (12, 1)
    assert np.allclose(Predictor(res.run_dir).predict(pd.read_csv(data["test"])), res.test_pred, atol=1e-5)


@needs("segmentation_models_pytorch")
def test_segmentation(workdir):
    from mlt.predict import predict_file
    from mlt.rle import rle_decode

    data = _image_competition(workdir / "seg", seg=True)
    cfg = _load(workdir, {"name": "seg", "modality": "segmentation", "data": data,
                          "model": {"name": "Unet", "backbone": "resnet18"},
                          "train": {**FAST, "epochs": 3, "image_size": 32, "pretrained": False}})
    res = run_cv(cfg)
    assert 0 <= res.score <= 1
    sub = pd.read_csv(res.run_dir / "submission.csv", keep_default_na=False)
    assert list(sub.columns) == ["id", "rle"] and len(sub) == 12
    mask = rle_decode(sub["rle"].iloc[0], (40, 36))
    assert mask.shape == (40, 36)  # back at the original image size
    assert predict_file(str(res.run_dir), data["test"]).exists()
