"""Entity embeddings for categorical columns (one embedding per column, then dense layers).

Embedding size per column is min(50, ceil(cardinality / 2)). Numeric columns arrive scaled and
join after the embeddings.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from .. import engine
from ..config import DEFAULTS
from ..utils import seed_everything


class _Data(Dataset):
    def __init__(self, X: dict, y=None, task="binary"):
        self.num = torch.as_tensor(X["num"], dtype=torch.float32)
        self.cat = torch.as_tensor(X["cat"], dtype=torch.long)
        self.y = None if y is None else engine.target_tensor(task, y)

    def __len__(self):
        return len(self.cat)

    def __getitem__(self, i):
        item = {"num": self.num[i], "cat": self.cat[i]}
        item["target"] = self.y[i] if self.y is not None else torch.zeros(1)
        return item


class EntityNet(nn.Module):
    def __init__(self, cardinalities, n_num, n_out, hidden=(300, 300), dropout=0.3, max_dim=50):
        super().__init__()
        dims = [int(min(max_dim, np.ceil(c / 2))) or 1 for c in cardinalities]
        self.embeddings = nn.ModuleList(nn.Embedding(c + 1, d) for c, d in zip(cardinalities, dims))
        self.emb_drop = nn.Dropout(dropout)
        layers, width = [], sum(dims) + n_num
        layers.append(nn.BatchNorm1d(width))
        for h in hidden:
            layers += [nn.Linear(width, h), nn.ReLU(), nn.Dropout(dropout), nn.BatchNorm1d(h)]
            width = h
        layers.append(nn.Linear(width, n_out))
        self.mlp = nn.Sequential(*layers)

    def forward(self, num, cat):
        parts = [emb(cat[:, j]) for j, emb in enumerate(self.embeddings)]
        feats = [self.emb_drop(torch.cat(parts, dim=1))] if parts else []
        return self.mlp(torch.cat(feats + [num], dim=1))


class EntityEmbeddingModel:
    """sklearn-style wrapper: fit(X, y, eval_set, ...), predict / predict_proba."""

    def __init__(self, task="binary", seed=42, hidden=(300, 300), dropout=0.3, max_dim=50,
                 epochs=10, batch_size=512, lr=1e-3, patience=3, device="auto"):
        self.task = task
        self.seed = seed
        self.arch = {"hidden": tuple(hidden), "dropout": dropout, "max_dim": max_dim}
        self.tcfg = {**DEFAULTS["train"], "epochs": epochs, "batch_size": batch_size, "lr": lr,
                     "patience": patience, "device": device, "weight_decay": 1e-5, "scheduler": "cosine"}

    def fit(self, X, y, eval_set, cardinalities, n_out, score_fn, greater_is_better):
        seed_everything(self.seed)
        torch.manual_seed(self.seed)
        self.n_out = n_out
        self.net = EntityNet(cardinalities, X["num"].shape[1], n_out, **self.arch)
        Xv, yv = eval_set
        engine.fit(self.net, _Data(X, y, self.task), _Data(Xv, yv, self.task), tcfg=self.tcfg,
                   task=self.task, score_fn=score_fn, greater_is_better=greater_is_better,
                   seed=self.seed, log_prefix="  nn ")
        self.net.cpu()
        return self

    def _predict(self, X):
        return engine.predict(self.net, _Data(X, None, self.task), tcfg=self.tcfg, task=self.task)

    def predict_proba(self, X):
        p = self._predict(X)
        return np.hstack([1 - p, p]) if self.task == "binary" else p

    def predict(self, X):
        return self._predict(X)[:, 0]
