"""Exploration helpers for notebooks (needs the `eda` extra for the plots)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy import stats


def overview(train: pd.DataFrame, test: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per column: dtype, missing share, unique values, and whether test has it."""
    rows = []
    for c in train.columns:
        rows.append({
            "column": c,
            "dtype": str(train[c].dtype),
            "missing": round(float(train[c].isna().mean()), 4),
            "unique": int(train[c].nunique()),
            "in_test": None if test is None else c in test.columns,
            "test_missing": None if test is None or c not in test else round(float(test[c].isna().mean()), 4),
            "example": train[c].dropna().iloc[0] if train[c].notna().any() else None,
        })
    return pd.DataFrame(rows).set_index("column")


def univariate_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for c in df.columns:
        s = df[c]
        base = {"count": s.count(), "missing": s.isna().sum(), "unique": s.nunique(), "dtype": str(s.dtype),
                "mode": s.mode().iloc[0] if s.notna().any() else None}
        if is_numeric_dtype(s) and not s.dtype == bool:
            q = s.quantile([0.25, 0.5, 0.75])
            base.update(mean=s.mean(), std=s.std(), min=s.min(), q25=q[0.25], median=q[0.5], q75=q[0.75],
                        max=s.max(), skew=s.skew(), kurt=s.kurt())
        rows[c] = base
    return pd.DataFrame(rows).T


def bivariate_stats(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Association of each column with a numeric (or 0/1) target: Pearson r for numeric
    columns, one-way ANOVA F for categorical ones."""
    y = df[target]
    if not is_numeric_dtype(y):
        y = y.astype("category").cat.codes
    rows = {}
    for c in df.columns:
        if c == target:
            continue
        ok = df[c].notna() & y.notna()
        if ok.sum() < 3 or df.loc[ok, c].nunique() < 2:
            continue
        if is_numeric_dtype(df[c]):
            r, p = stats.pearsonr(df.loc[ok, c].astype(float), y[ok].astype(float))
            rows[c] = {"stat": "r", "effect": abs(r), "sign": np.sign(r), "p_value": p}
        else:
            groups = [y[ok][df.loc[ok, c] == g] for g in df.loc[ok, c].unique()]
            groups = [g for g in groups if len(g) > 1]
            if len(groups) > 1:
                f, p = stats.f_oneway(*groups)
                rows[c] = {"stat": "F", "effect": f, "sign": None, "p_value": p}
    return pd.DataFrame(rows).T.sort_values("effect", ascending=False)


def plot_distributions(train: pd.DataFrame, test: pd.DataFrame | None = None, cols=None, n_cols: int = 3):
    import matplotlib.pyplot as plt
    import seaborn as sns

    cols = cols or [c for c in train.columns if is_numeric_dtype(train[c])]
    n_rows = (len(cols) - 1) // n_cols + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.2 * n_rows), squeeze=False)
    for ax, c in zip(axes.flat, cols):
        sns.kdeplot(train[c].dropna(), ax=ax, label="train", fill=True, warn_singular=False)
        if test is not None and c in test:
            sns.kdeplot(test[c].dropna(), ax=ax, label="test", fill=True, warn_singular=False)
        ax.set_title(c)
        ax.legend()
    for ax in list(axes.flat)[len(cols):]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


def plot_counts(df: pd.DataFrame, cols, top: int = 20, n_cols: int = 2):
    import matplotlib.pyplot as plt

    n_rows = (len(cols) - 1) // n_cols + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 3.5 * n_rows), squeeze=False)
    for ax, c in zip(axes.flat, cols):
        df[c].astype(str).value_counts().head(top).plot.bar(ax=ax)
        ax.set_title(c)
        ax.tick_params(axis="x", rotation=45)
    for ax in list(axes.flat)[len(cols):]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


def plot_corr(df: pd.DataFrame, cols=None):
    import matplotlib.pyplot as plt
    import seaborn as sns

    cols = cols or [c for c in df.columns if is_numeric_dtype(df[c])]
    corr = df[cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(min(1 + 0.6 * len(cols), 20), min(0.5 * len(cols) + 1, 16)))
    sns.heatmap(corr, mask=mask, cmap="vlag", center=0, annot=len(cols) <= 15, fmt=".2f", ax=ax)
    ax.set_title("correlations")
    fig.tight_layout()
    return fig


def plot_tsne(X, y=None, n: int = 3000, seed: int = 42):
    """2-D t-SNE of a sample of rows, coloured by the target: a quick look at class structure."""
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    X = np.asarray(X, dtype=float)
    idx = np.random.default_rng(seed).choice(len(X), size=min(n, len(X)), replace=False)
    Xs = np.nan_to_num(X[idx])
    emb = TSNE(n_components=2, random_state=seed, init="pca").fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=None if y is None else pd.Series(np.asarray(y)[idx]).astype("category").cat.codes,
                    s=6, cmap="tab10")
    if y is not None:
        fig.colorbar(sc, ax=ax)
    ax.set_title("t-SNE")
    return fig
