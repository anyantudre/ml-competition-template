"""One PyTorch training loop for every neural model (text, images, masks, tabular embeddings).

Datasets return dicts: the "target" key feeds the loss and every other key is passed to
model.forward as a keyword argument. The loop keeps the best epoch by the competition metric
on the validation fold, with early stopping, warmup + cosine/linear schedules, gradient
accumulation, clipping and mixed precision on CUDA.
"""

from __future__ import annotations

import copy
import math
import time
from collections.abc import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .utils import LOG, get_device


class DiceBCELoss(nn.Module):
    def forward(self, logits, target):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
        prob = torch.sigmoid(logits)
        dims = tuple(range(1, prob.ndim))
        inter = (prob * target).sum(dims)
        dice = 1 - (2 * inter + 1) / (prob.sum(dims) + target.sum(dims) + 1)
        return bce + dice.mean()


def loss_for(task: str, name: str | None = None) -> nn.Module:
    if task in {"binary", "multilabel"}:
        return nn.BCEWithLogitsLoss()
    if task == "multiclass":
        return nn.CrossEntropyLoss()
    if task == "segmentation":
        return DiceBCELoss()
    return {"mse": nn.MSELoss(), "l1": nn.L1Loss(), "huber": nn.HuberLoss()}[name or "mse"]


def activation(task: str, logits: torch.Tensor) -> torch.Tensor:
    if task in {"binary", "multilabel", "segmentation"}:
        return torch.sigmoid(logits)
    if task == "multiclass":
        return torch.softmax(logits, dim=-1)
    return logits


def target_tensor(task: str, y) -> torch.Tensor:
    if task == "multiclass":
        return torch.as_tensor(y, dtype=torch.long)
    t = torch.as_tensor(np.asarray(y, dtype=np.float32))
    if task in {"binary", "regression", "ordinal"} and t.ndim == 0:
        t = t.reshape(1)
    return t


def _to(batch: dict, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _schedule(name: str, total: int, warmup: int):
    def fn(step):
        if warmup and step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        if name == "cosine":
            return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
        if name == "linear":
            return max(0.0, 1 - progress)
        return 1.0

    return fn


def _forward(model, batch):
    inputs = {k: v for k, v in batch.items() if k != "target"}
    return model(**inputs)


def _align(task, logits, target):
    if task == "multiclass":
        return logits, target
    if logits.shape != target.shape:
        target = target.reshape(logits.shape)
    return logits, target


def fit(model: nn.Module, train_ds, valid_ds, *, tcfg: dict, task: str,
        score_fn: Callable[[np.ndarray], float], greater_is_better: bool,
        collate_fn=None, param_groups=None, seed: int = 42, log_prefix: str = ""):
    """Train with early stopping. Returns (best validation predictions, best score)."""
    device = get_device(tcfg["device"])
    model.to(device)
    gen = torch.Generator().manual_seed(seed)
    train_dl = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True, generator=gen,
                          num_workers=tcfg["num_workers"], collate_fn=collate_fn, drop_last=len(train_ds) > tcfg["batch_size"])
    loss_fn = loss_for(task, tcfg["loss"])
    params = param_groups if param_groups is not None else model.parameters()
    opt = torch.optim.AdamW(params, lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    accum = max(1, tcfg["grad_accum"])
    steps = tcfg["epochs"] * math.ceil(len(train_dl) / accum)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, _schedule(tcfg["scheduler"], steps, int(tcfg["warmup_ratio"] * steps))
    )
    use_amp = bool(tcfg["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_score = -np.inf if greater_is_better else np.inf
    best_state, best_pred, bad_epochs = None, None, 0
    for epoch in range(tcfg["epochs"]):
        start = time.perf_counter()
        model.train()
        total, count = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for i, batch in enumerate(train_dl):
            batch = _to(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits, target = _align(task, _forward(model, batch), batch["target"])
                loss = loss_fn(logits.float(), target) / accum
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}, step {i}")
            scaler.scale(loss).backward()
            if (i + 1) % accum == 0 or i + 1 == len(train_dl):
                if tcfg["max_grad_norm"]:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), tcfg["max_grad_norm"])
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
            total += loss.item() * accum * len(batch["target"])
            count += len(batch["target"])
        pred = predict(model, valid_ds, tcfg=tcfg, task=task, collate_fn=collate_fn, device=device)
        score = score_fn(pred)
        improved = score > best_score if greater_is_better else score < best_score
        LOG.info("%sepoch %d  train_loss %.4f  valid %.5f%s  (%.0fs)", log_prefix, epoch + 1,
                 total / max(count, 1), score, " *" if improved else "", time.perf_counter() - start)
        if improved:
            best_score, best_pred, bad_epochs = score, pred, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            bad_epochs += 1
            if bad_epochs >= tcfg["patience"]:
                LOG.info("%searly stopping", log_prefix)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_pred, best_score


@torch.no_grad()
def predict(model: nn.Module, ds, *, tcfg: dict, task: str, collate_fn=None, device=None,
            tta: Callable | None = None) -> np.ndarray:
    device = device or get_device(tcfg["device"])
    model.to(device)
    model.eval()
    dl = DataLoader(ds, batch_size=tcfg["batch_size"] * 2, shuffle=False,
                    num_workers=tcfg["num_workers"], collate_fn=collate_fn)
    use_amp = bool(tcfg["amp"]) and device.type == "cuda"
    out = []
    for batch in dl:
        batch = _to(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            prob = activation(task, _forward(model, batch).float())
            if tta is not None:
                prob = (prob + tta(model, batch)) / 2
        out.append(prob.float().cpu().numpy())
    pred = np.concatenate(out)
    if pred.ndim == 1:
        pred = pred[:, None]
    return pred


def flip_tta(task: str):
    """Horizontal flip test-time augmentation for image batches (keyword 'image')."""

    def fn(model, batch):
        flipped = dict(batch)
        flipped["image"] = torch.flip(batch["image"], dims=[-1])
        prob = activation(task, _forward(model, flipped).float())
        if task == "segmentation":
            prob = torch.flip(prob, dims=[-1])
        return prob

    return fn
