"""PyTorch text models and datasets (needs the `text` extra)."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from ..engine import target_tensor


class SeqDataset(Dataset):
    """Word ids from a Vocab, for the BiLSTM."""

    def __init__(self, texts, y, vocab, max_len, task):
        self.ids = np.stack([vocab.encode(t, max_len) for t in texts]) if texts else np.zeros((0, max_len), np.int64)
        self.y = None if y is None else target_tensor(task, y)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return {"tokens": torch.as_tensor(self.ids[i]),
                "target": self.y[i] if self.y is not None else torch.zeros(1)}


class RawTextDataset(Dataset):
    """Raw strings; a collate function tokenises each batch (dynamic padding)."""

    def __init__(self, texts, y, task):
        self.texts = texts
        self.y = None if y is None else target_tensor(task, y)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i], (self.y[i] if self.y is not None else torch.zeros(1))


class TokenizeCollate:
    def __init__(self, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, batch):
        texts, targets = zip(*batch)
        enc = self.tokenizer(list(texts), padding=True, truncation=True, max_length=self.max_len,
                             return_tensors="pt")
        out = {k: v for k, v in enc.items() if k in {"input_ids", "attention_mask", "token_type_ids"}}
        out["target"] = torch.stack(list(targets))
        return out


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, n_out, vectors=None, freeze=True, hidden=128, embed_dim=200, dropout=0.3):
        super().__init__()
        dim = vectors.shape[1] if vectors is not None else embed_dim
        self.emb = nn.Embedding(vocab_size, dim, padding_idx=0)
        if vectors is not None:
            self.emb.weight.data.copy_(torch.as_tensor(vectors))
            self.emb.weight.requires_grad = not freeze
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(dim, hidden, batch_first=True, bidirectional=True)
        self.out = nn.Linear(4 * hidden, n_out)

    def forward(self, tokens):
        mask = (tokens != 0).unsqueeze(-1).float()
        h, _ = self.lstm(self.drop(self.emb(tokens)))
        mean = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        maxp = h.masked_fill(mask == 0, -1e4).max(1).values
        return self.out(self.drop(torch.cat([mean, maxp], dim=1)))


class TransformerClassifier(nn.Module):
    """Hugging Face encoder + mean pooling + linear head."""

    def __init__(self, backbone: str, n_out: int, dropout: float = 0.1):
        super().__init__()
        from transformers import AutoModel

        self.backbone = AutoModel.from_pretrained(backbone)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(self.backbone.config.hidden_size, n_out)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kw = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kw["token_type_ids"] = token_type_ids
        h = self.backbone(**kw).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(self.drop(pooled))


def param_groups(model, weight_decay):
    """No weight decay on biases and normalisation weights."""
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight")
    decay, rest = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            (rest if any(k in n for k in no_decay) else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": rest, "weight_decay": 0.0}]
