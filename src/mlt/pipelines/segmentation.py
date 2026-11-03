"""Binary segmentation: U-Net family from segmentation_models_pytorch, Dice + BCE loss.

data.image_col and data.mask_col hold file names in data.image_dir (masks: any non-zero pixel is
foreground). model.name is the smp architecture (Unet, UnetPlusPlus, FPN, DeepLabV3Plus...),
model.backbone the encoder (resnet18, efficientnet-b0...). train.image_size must be a multiple
of 32. Predicted masks are resized back to each test image's own size and written as RLE.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .. import __version__
from ..config import load_config, save_config
from ..data import load_data, row_ids
from ..folds import default_scheme, fold_ids, get_folds, train_mask
from ..metrics import dice
from ..rle import rle_encode
from ..submission import validate_submission
from ..utils import LOG, git_sha, read_json, require, seed_everything, write_json


def _model(cfg, pretrained):
    from ..models.vision import SegmentationNet

    m = cfg["model"]
    return SegmentationNet(m["name"] or "Unet", m["backbone"] or "resnet18", pretrained=pretrained,
                           **m["params"])


def _paths(cfg, df, col):
    from ..models.vision import resolve_paths

    d = cfg["data"]
    return resolve_paths(df[col], [d["image_dir"], d["test_image_dir"]])


def _dataset(cfg, df, train: bool, with_masks: bool):
    from ..models.vision import MaskDataset, transforms

    t = cfg["train"]
    masks = _paths(cfg, df, cfg["data"]["mask_col"]) if with_masks else None
    return MaskDataset(_paths(cfg, df, cfg["data"]["image_col"]), masks,
                       transforms(t["image_size"], train, t["augment"]))


def _masks(ds) -> np.ndarray:
    return np.stack([ds[i]["target"].numpy() for i in range(len(ds))]).astype(np.uint8)


def _predict(cfg, models, df) -> np.ndarray:
    from .. import engine

    tta = engine.flip_tta("segmentation") if cfg["train"]["tta"] else None
    ds = _dataset(cfg, df, False, False)
    preds = [engine.predict(m, ds, tcfg=cfg["train"], task="segmentation", tta=tta) for m in models]
    return np.mean(preds, axis=0)


def _to_rle(cfg, df, probs) -> list[str]:
    import torch
    from PIL import Image

    sub = cfg["submission"]
    out = []
    for path, p in zip(_paths(cfg, df, cfg["data"]["image_col"]), probs):
        w, h = Image.open(path).size
        full = torch.nn.functional.interpolate(torch.as_tensor(p[None]).float(), size=(h, w), mode="bilinear",
                                               align_corners=False)[0, 0].numpy()
        out.append(rle_encode(full >= sub["mask_threshold"], order=sub["rle_order"]))
    return out


def _submission(cfg, df, probs, sample) -> pd.DataFrame:
    id_col = cfg["submission"]["id_col"] or cfg["data"]["id_col"]
    rle_col = [c for c in sample.columns if c != id_col][0] if sample is not None else "rle_mask"
    sub = pd.DataFrame({id_col: row_ids(cfg, df).to_numpy(), rle_col: _to_rle(cfg, df, probs)})
    if sample is not None:
        sub = sample[[id_col]].merge(sub, on=id_col, how="left")[list(sample.columns)]
    return sub


def run_segmentation(cfg: dict, save: bool = True, trial=None):
    require("segmentation_models_pytorch", "segmentation")
    import torch

    from .. import engine
    from ..train import RunResult, append_registry

    start = time.perf_counter()
    train, test, sample = load_data(cfg)
    folds, fpath, fhash = get_folds(train, np.zeros(len(train)), cfg, "segmentation")
    scheme = default_scheme(cfg, "segmentation")
    run_folds = cfg["train"]["folds"] or fold_ids(folds)
    run_dir = Path(cfg["output_dir"]) / cfg["name"]
    thr = cfg["submission"]["mask_threshold"]
    oof_dice = np.full(len(train), np.nan)
    fold_scores, models = [], []
    LOG.info("%s | segmentation | %d images | %s folds %s", cfg["name"], len(train), scheme, run_folds)
    for k in run_folds:
        t0 = time.perf_counter()
        seed_everything(cfg["seed"] + k)
        torch.manual_seed(cfg["seed"] + k)
        tr, va = train_mask(folds, k, scheme), folds == k
        valid_ds = _dataset(cfg, train[va], False, True)
        y_valid = _masks(valid_ds)
        model = _model(cfg, cfg["train"]["pretrained"])
        pred, s = engine.fit(model, _dataset(cfg, train[tr], True, True), valid_ds, tcfg=cfg["train"],
                             task="segmentation", score_fn=lambda p, y=y_valid: dice(y, p, threshold=thr),
                             greater_is_better=True, seed=cfg["seed"] + k, log_prefix=f"  fold {k} ")
        model.cpu()
        models.append(model)
        per_image = [dice(y_valid[i:i + 1], pred[i:i + 1], threshold=thr) for i in range(len(pred))]
        oof_dice[np.flatnonzero(va)] = per_image
        fold_scores.append(s)
        if save and cfg["train"]["save_models"]:
            (run_dir / "models" / f"fold{k}").mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), run_dir / "models" / f"fold{k}" / "model.pt")
        LOG.info("fold %d  dice %.5f  (%.0fs)", k, s, time.perf_counter() - t0)
    cv = float(np.nanmean(oof_dice))
    LOG.info("CV dice %.5f  (folds %.5f +- %.5f)", cv, np.mean(fold_scores), np.std(fold_scores))
    result = RunResult(cfg["name"], cv, fold_scores, run_dir if save else None, oof_dice[:, None], None)
    if not save:
        return result

    run_dir.mkdir(parents=True, exist_ok=True)
    save_config({**cfg, "task": "segmentation", "metric": "dice"}, run_dir / "config.yaml")
    write_json({"task": "segmentation", "metric": "dice", "folds": run_folds}, run_dir / "state.json")
    pd.DataFrame({"id": row_ids(cfg, train), "fold": folds, "dice": oof_dice}).to_csv(run_dir / "oof.csv", index=False)
    sub_path = None
    if test is not None:
        sub = _submission(cfg, test, _predict(cfg, models, test), sample)
        sub_path = run_dir / cfg["submission"]["file"]
        sub.to_csv(sub_path, index=False)
        problems = validate_submission(sub, sample, cfg["data"]["id_col"], allow_empty=True)
        LOG.info("submission %s: %s", sub_path, "; ".join(problems) or "ok")
    info = {"name": cfg["name"], "modality": "segmentation", "task": "segmentation", "metric": "dice",
            "cv": cv, "fold_mean": float(np.mean(fold_scores)), "fold_std": float(np.std(fold_scores)),
            "fold_scores": fold_scores, "model": cfg["model"]["name"] or "Unet",
            "backbone": cfg["model"]["backbone"] or "resnet18", "n_train": len(train),
            "folds_file": str(fpath), "folds_hash": fhash, "duration_s": round(time.perf_counter() - start, 1),
            "git": git_sha(), "mlt": __version__, "submission": str(sub_path) if sub_path else None,
            "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    write_json(info, run_dir / "metrics.json")
    append_registry(cfg["output_dir"], info)
    LOG.info("run saved to %s", run_dir)
    return result


def predict_segmentation(run_dir: str, input_path: str, out: str | None = None) -> Path:
    import torch

    from ..data import read_table

    run_dir = Path(run_dir)
    cfg = load_config(run_dir / "config.yaml")
    state = read_json(run_dir / "state.json")
    models = []
    for k in state["folds"]:
        m = _model(cfg, pretrained=False)
        m.load_state_dict(torch.load(run_dir / "models" / f"fold{k}" / "model.pt", map_location="cpu"))
        models.append(m)
    df = read_table(input_path)
    sub = _submission(cfg, df, _predict(cfg, models, df), None)
    out_path = Path(out or run_dir / "submission_predict.csv")
    sub.to_csv(out_path, index=False)
    LOG.info("wrote %s (%d rows)", out_path, len(sub))
    return out_path
