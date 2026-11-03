"""Text helpers: cleaning, tokenising, a vocabulary and pretrained word vectors."""

from __future__ import annotations

import re
import string
from collections import Counter

import numpy as np

_PUNCT = re.compile(f"[{re.escape(string.punctuation)}]")
_TOKEN = re.compile(r"\w+|[^\w\s]")


def clean_text(s: str) -> str:
    """Lowercase, drop URLs and punctuation, squeeze whitespace."""
    s = str(s).lower()
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)
    s = _PUNCT.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def tokenize(s: str) -> list[str]:
    return _TOKEN.findall(str(s).lower())


class Vocab:
    PAD, UNK = 0, 1

    def __init__(self, texts, max_size: int = 50000, min_freq: int = 2):
        counts = Counter(tok for t in texts for tok in tokenize(t))
        words = [w for w, c in counts.most_common(max_size) if c >= min_freq]
        self.itos = ["<pad>", "<unk>"] + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, text: str, max_len: int) -> np.ndarray:
        ids = [self.stoi.get(t, self.UNK) for t in tokenize(text)[:max_len]]
        out = np.zeros(max_len, dtype=np.int64)
        out[: len(ids)] = ids
        return out


def load_vectors(path: str, vocab: Vocab, dim: int | None = None):
    """Build an embedding matrix for the vocabulary from a fastText .vec or GloVe .txt file.

    Words are looked up as is, lowercased, capitalised and upper-cased. Returns the matrix and
    the share of the vocabulary that was found (check it: low coverage means a poor fit).
    """
    matrix, found = None, 0
    wanted = {}
    for w, i in vocab.stoi.items():
        for form in (w, w.lower(), w.capitalize(), w.upper()):
            wanted.setdefault(form, i)
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) <= 2:  # fastText header line "n dim"
                continue
            if matrix is None:
                dim = dim or len(parts) - 1
                matrix = np.random.default_rng(0).normal(0, 0.1, (len(vocab), dim)).astype(np.float32)
                matrix[Vocab.PAD] = 0
            idx = wanted.get(parts[0])
            if idx is not None and len(parts) - 1 == dim:
                matrix[idx] = np.asarray(parts[1:], dtype=np.float32)
                found += 1
    if matrix is None:
        raise ValueError(f"no vectors read from {path}")
    return matrix, found / max(1, len(vocab))
