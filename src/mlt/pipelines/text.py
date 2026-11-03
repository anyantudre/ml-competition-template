"""Text classification and regression.

model.name:
    tfidf        TF-IDF (or counts) + logistic regression / ridge. Fast, strong baseline.
    nb           counts + multinomial Naive Bayes (classification).
    bilstm       BiLSTM over word ids, mean + max pooling, optional fastText/GloVe vectors.
    transformer  any Hugging Face encoder (model.backbone), fine-tuned end to end.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..features.text import Vocab, clean_text, load_vectors
from ..utils import LOG, require, seed_everything
from .base import Pipeline

LINEAR = {"tfidf", "nb"}
NEURAL = {"bilstm", "transformer"}


class TextPipeline(Pipeline):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = self.cfg["model"]["name"] or "tfidf"
        if self.name not in LINEAR | NEURAL:
            raise ValueError(f"text model.name must be one of {sorted(LINEAR | NEURAL)}")
        if self.name == "nb" and self.task not in {"binary", "multiclass", "multilabel"}:
            raise ValueError("nb is for classification; use tfidf (ridge) for regression")
        self.col = self.cfg["data"]["text_col"]
        self.tf = self.cfg["features"]["text"]

    def _texts(self, df: pd.DataFrame) -> list[str]:
        s = df[self.col].fillna("").astype(str)
        return [clean_text(t) for t in s] if self.tf["clean"] else s.tolist()

    # ----------------------------------------------------------------- dispatch
    def fit_fold(self, train, y_train, valid, y_valid, test, fold):
        if self.name in LINEAR:
            return self._fit_linear(train, y_train)
        return self._fit_neural(train, y_train, valid, y_valid, fold)

    def predict(self, artifact, df):
        if self.name in LINEAR:
            return self._predict_linear(artifact, df)
        return self._predict_neural(artifact, df)

    # ------------------------------------------------------------------- linear
    def _vectorizer(self):
        from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

        kw = {"ngram_range": tuple(self.tf["ngram_range"]), "min_df": self.tf["min_df"],
              "max_features": self.tf["max_features"], "analyzer": self.tf["analyzer"],
              "strip_accents": "unicode"}
        if self.name == "nb" or self.tf["vectorizer"] == "count":
            return CountVectorizer(**kw)
        return TfidfVectorizer(sublinear_tf=self.tf["sublinear_tf"], **kw)

    def _linear_model(self):
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.naive_bayes import MultinomialNB

        p = dict(self.cfg["model"]["params"])
        if self.name == "nb":
            return MultinomialNB(**p)
        if self.task in {"regression", "ordinal"}:
            return Ridge(**{"alpha": 1.0, **p})
        return LogisticRegression(**{"C": 4.0, "max_iter": 3000, **p})

    def _fit_linear(self, train, y):
        vec = self._vectorizer()
        X = vec.fit_transform(self._texts(train))
        ys = [y[:, j] for j in range(y.shape[1])] if self.task == "multilabel" else [y]
        models = [self._linear_model().fit(X, yy) for yy in ys]
        return {"vec": vec, "models": models}

    def _predict_linear(self, art, df):
        X = art["vec"].transform(self._texts(df))
        out = []
        for m in art["models"]:
            if self.task in {"regression", "ordinal"}:
                out.append(m.predict(X).reshape(-1, 1))
            elif self.task in {"binary", "multilabel"}:
                out.append(m.predict_proba(X)[:, 1:2])
            else:
                out.append(m.predict_proba(X))
        return np.hstack(out)

    def importance(self, art):
        if self.name != "tfidf" or self.task in {"multiclass", "multilabel"}:
            return None
        coef = np.ravel(art["models"][0].coef_)
        names = art["vec"].get_feature_names_out()
        top = np.argsort(-np.abs(coef))[:200]
        return pd.Series(coef[top], index=names[top])


    # ------------------------------------------------------------------- neural
    def _fit_neural(self, train, y_train, valid, y_valid, fold):
        require("torch", "text")
        import torch

        from .. import engine
        from ..models import text as tm

        seed = self.cfg["seed"] + fold
        seed_everything(seed)
        torch.manual_seed(seed)
        tcfg = self.cfg["train"]
        art, groups = {}, None
        if self.name == "bilstm":
            vocab = Vocab(self._texts(train), max_size=self.tf["max_vocab"])
            vectors = None
            if self.tf["embeddings"]:
                vectors, coverage = load_vectors(self.tf["embeddings"], vocab)
                LOG.info("  embeddings cover %.1f%% of the vocabulary", 100 * coverage)
            art["vocab"] = vocab
            art["model"] = tm.BiLSTM(len(vocab), self.n_out, vectors, self.tf["freeze_embeddings"],
                                     **self.cfg["model"]["params"])
        else:
            art["model"], art["tokenizer"] = self._transformer()
            groups = tm.param_groups(art["model"], tcfg["weight_decay"])
        engine.fit(art["model"], self._dataset(train, y_train, art), self._dataset(valid, y_valid, art),
                   tcfg=tcfg, task=self.task, score_fn=self.score_fn(y_valid),
                   greater_is_better=self.greater_is_better, collate_fn=self._collate(art),
                   param_groups=groups, seed=seed, log_prefix=f"  fold {fold} ")
        art["model"].cpu()
        return art

    def _transformer(self):
        require("transformers", "text")
        from transformers import AutoTokenizer

        from ..models.text import TransformerClassifier

        backbone = self.cfg["model"]["backbone"]
        if not backbone:
            raise ValueError("model.backbone is required for transformer, e.g. distilbert-base-uncased")
        return TransformerClassifier(backbone, self.n_out, **self.cfg["model"]["params"]), \
            AutoTokenizer.from_pretrained(backbone)

    def _collate(self, art):
        from ..models.text import TokenizeCollate

        return TokenizeCollate(art["tokenizer"], self.cfg["train"]["max_len"]) if "tokenizer" in art else None

    def _dataset(self, df, y, art):
        from ..models.text import RawTextDataset, SeqDataset

        texts = self._texts(df)
        if "vocab" in art:
            return SeqDataset(texts, y, art["vocab"], self.cfg["train"]["max_len"], self.task)
        return RawTextDataset(texts, y, self.task)

    def _predict_neural(self, art, df):
        from .. import engine

        return engine.predict(art["model"], self._dataset(df, None, art), tcfg=self.cfg["train"],
                              task=self.task, collate_fn=self._collate(art))

    # transformers are saved as weights only (the tokenizer reloads from the backbone name)
    def save(self, art, folder: Path):
        if self.name != "transformer":
            return super().save(art, folder)
        import torch

        folder.mkdir(parents=True, exist_ok=True)
        torch.save(art["model"].state_dict(), folder / "model.pt")

    def load(self, folder: Path):
        if self.name != "transformer":
            return super().load(folder)
        import torch

        model, tokenizer = self._transformer()
        model.load_state_dict(torch.load(folder / "model.pt", map_location="cpu"))
        return {"model": model, "tokenizer": tokenizer}
