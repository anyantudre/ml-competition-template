"""Tabular preprocessing fitted inside each fold.

Order of operations:
1. date parts, column combinations, group aggregates, log1p and binning
2. categoricals: NaN becomes "NONE", rare values become "RARE" (unseen values at predict time too)
3. count encoding and out-of-fold target encoding
4. encoding for the model family: integer codes for trees, native strings for CatBoost,
   sparse one-hot plus scaled numerics for linear models, codes plus scaled numerics for
   entity-embedding networks
5. optional feature selection

Steps that never look at the target (vocabularies, counts, aggregates) can also be fitted on
the test rows, the competition setting the book recommends. Set
`features.fit_unsupervised_on_test: false` when the test set is not known in advance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn import model_selection
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .selection import FeatureSelector

NONE, RARE = "NONE", "RARE"

DATE_PARTS = {
    "year": lambda s: s.dt.year,
    "month": lambda s: s.dt.month,
    "day": lambda s: s.dt.day,
    "dayofweek": lambda s: s.dt.dayofweek,
    "dayofyear": lambda s: s.dt.dayofyear,
    "weekofyear": lambda s: s.dt.isocalendar().week.astype("float"),
    "quarter": lambda s: s.dt.quarter,
    "hour": lambda s: s.dt.hour,
    "minute": lambda s: s.dt.minute,
    "is_weekend": lambda s: (s.dt.dayofweek >= 5).astype("float"),
    "is_month_start": lambda s: s.dt.is_month_start.astype("float"),
    "is_month_end": lambda s: s.dt.is_month_end.astype("float"),
}


class TabularPreprocessor:
    def __init__(self, cfg: dict, cats: list[str], nums: list[str], dates: list[str],
                 encoding: str, task: str, n_outputs: int = 1):
        self.f = cfg["features"]
        self.seed = cfg["seed"]
        self.cats = list(cats)
        self.nums = list(nums)
        self.dates = list(dates)
        self.encoding = encoding  # label | native | onehot | embedding
        self.task = task
        self.n_outputs = n_outputs

    # ------------------------------------------------------------------ helpers
    def _base(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for c in self.nums:
            out[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
        for c in self.cats:
            out[c] = _as_category_strings(df[c])
        for c in self.dates:
            s = pd.to_datetime(df[c], errors="coerce")
            for part in self.f["date_parts"]:
                out[f"{c}_{part}"] = DATE_PARTS[part](s).astype(float)
        for a, b in self.f["combine"]:
            out[f"{a}__{b}"] = _as_category_strings(df[a]) + "_" + _as_category_strings(df[b])
        for c in self.f["log1p"]:
            out[c] = np.log1p(out[c].clip(lower=0))
        return out

    @property
    def cat_columns(self) -> list[str]:
        return self.cats + [f"{a}__{b}" for a, b in self.f["combine"]]

    # ---------------------------------------------------------------------- fit
    def fit_transform(self, train: pd.DataFrame, y: np.ndarray, others: list[pd.DataFrame] = ()):
        base = self._base(train)
        others_base = [self._base(o) for o in others if o is not None]
        use_others = self.f["fit_unsupervised_on_test"] and others_base
        unsup = pd.concat([base, *others_base]) if use_others else base

        # group aggregates, e.g. mean price per store (the `by` columns must be features)
        self.agg_tables = []
        for spec in self.f["aggregates"]:
            by = [spec["by"]] if isinstance(spec["by"], str) else list(spec["by"])
            missing = [b for b in by + list(spec["cols"]) if b not in unsup.columns]
            if missing:
                raise KeyError(f"aggregate columns {missing} are not features")
            table = unsup.groupby(by)[spec["cols"]].agg(spec.get("aggs", ["mean"]))
            table.columns = [f"agg_{'_'.join(by)}_{c}_{a}" for c, a in table.columns]
            self.agg_tables.append((by, table))

        # binning edges
        self.bin_edges = {
            c: np.unique(np.nanquantile(unsup[c], np.linspace(0, 1, n + 1)))
            for c, n in (self.f["bins"] or {}).items()
        }

        # categorical vocabularies with rare grouping
        self.vocab = {}
        min_count = self.f["rare_min_count"] or 0
        for c in self.cat_columns:
            counts = unsup[c].value_counts()
            keep = counts[counts >= max(min_count, 1)].index.tolist()
            self.vocab[c] = [NONE, RARE] + [v for v in keep if v not in (NONE, RARE)]

        self.count_maps = {c: _cat_with_rare(unsup[c], self.vocab[c]).value_counts() for c in self.f["count_encode"]}

        frame = self._engineer(base)
        # out-of-fold target encoding on the training rows
        self.te_maps, self.prior = {}, None
        if self.f["target_encode"]:
            yy = self._te_target(y)
            self.prior = yy.mean(axis=0)
            for c in self.f["target_encode"]:
                vals = frame[c]
                self.te_maps[c] = _te_table(vals, yy, self.prior, self.f["target_encode_smoothing"])
            frame = pd.concat([frame, self._te_oof(frame, yy)], axis=1)

        self.num_out = [c for c in frame.columns if c not in self.cat_columns]
        X = self._encode(frame, fit=True)
        self.selector = None
        if self.f["selection"]["method"]:
            self.selector = FeatureSelector(self.f["selection"], self.task, self.seed)
            X = self.selector.fit_transform(X, y)
            self.feature_names = [n for n, keep in zip(self.feature_names, self.selector.mask) if keep]
        return X

    def transform(self, df: pd.DataFrame):
        frame = self._engineer(self._base(df))
        if self.te_maps:
            enc = {}
            for c, table in self.te_maps.items():
                for j in range(table.shape[1]):
                    enc[f"te_{c}_{j}"] = frame[c].map(table.iloc[:, j]).fillna(self.prior[j]).to_numpy(float)
            frame = pd.concat([frame, pd.DataFrame(enc, index=frame.index)], axis=1)
        names = self.feature_names  # fixed at fit time
        X = self._encode(frame, fit=False)
        self.feature_names = names
        if self.selector is not None:
            X = self.selector.transform(X)
        return X

    # ------------------------------------------------------------------ pieces
    def _engineer(self, base: pd.DataFrame) -> pd.DataFrame:
        frame = base.copy()
        for by, table in self.agg_tables:  # before rare grouping, so keys match the table
            keys = base[by].reset_index(drop=True)
            merged = keys.merge(table, left_on=by, right_index=True, how="left")
            for col in table.columns:
                frame[col] = merged[col].to_numpy(float)
        for c in self.cat_columns:
            frame[c] = _cat_with_rare(frame[c], self.vocab[c])
        for c, edges in self.bin_edges.items():
            frame[f"{c}_bin"] = np.digitize(frame[c].to_numpy(float), edges[1:-1]).astype(float)
        for c, counts in self.count_maps.items():
            frame[f"{c}_count"] = frame[c].map(counts).fillna(0).astype(float)
        return frame

    def _te_target(self, y: np.ndarray) -> np.ndarray:
        if self.task == "multiclass":
            return np.eye(self.n_outputs)[y]
        y = np.asarray(y, dtype=float)
        return y.reshape(len(y), -1)

    def _te_oof(self, frame: pd.DataFrame, yy: np.ndarray) -> pd.DataFrame:
        out = {}
        kf = model_selection.KFold(5, shuffle=True, random_state=self.seed)
        for c in self.f["target_encode"]:
            cols = np.zeros((len(frame), yy.shape[1]))
            vals = frame[c].reset_index(drop=True)
            for tr, va in kf.split(vals):
                table = _te_table(vals.iloc[tr], yy[tr], self.prior, self.f["target_encode_smoothing"])
                for j in range(yy.shape[1]):
                    cols[va, j] = vals.iloc[va].map(table.iloc[:, j]).fillna(self.prior[j]).to_numpy(float)
            for j in range(yy.shape[1]):
                out[f"te_{c}_{j}"] = cols[:, j]
        return pd.DataFrame(out, index=frame.index)

    def _encode(self, frame: pd.DataFrame, fit: bool):
        cats = self.cat_columns
        nums = self.num_out
        if self.encoding == "native":
            out = frame[nums + cats].copy()
            self.cat_indices = list(range(len(nums), len(nums) + len(cats)))
            self.feature_names = nums + cats
            return out
        codes = pd.DataFrame(
            {c: pd.Categorical(frame[c], categories=self.vocab[c]).codes.astype(np.int64) for c in cats},
            index=frame.index,
        )
        if self.encoding == "label":
            out = pd.concat([frame[nums], codes.astype(float)], axis=1)
            self.feature_names = nums + cats
            return out
        # numeric part for linear models and networks: median impute, then standardise
        num = frame[nums].to_numpy(float) if nums else np.zeros((len(frame), 0))
        if fit:
            self.medians = np.nanmedian(num, axis=0) if nums else np.zeros(0)
            self.medians = np.where(np.isnan(self.medians), 0.0, self.medians)
        num = np.where(np.isnan(num), self.medians, num)
        if fit:
            self.scaler = StandardScaler().fit(num) if nums else None
        if self.scaler is not None:
            num = self.scaler.transform(num)
        if self.encoding == "embedding":
            self.cardinalities = [len(self.vocab[c]) for c in cats]
            self.feature_names = nums + cats
            return {"num": num.astype(np.float32), "cat": codes.to_numpy(np.int64)}
        # one-hot
        if fit:
            self.ohe = OneHotEncoder(handle_unknown="ignore", categories=[self.vocab[c] for c in cats]) if cats else None
            if self.ohe is not None:
                self.ohe.fit(frame[cats].astype(str))
        parts = [sparse.csr_matrix(num)]
        names = list(nums)
        if self.ohe is not None:
            parts.append(self.ohe.transform(frame[cats].astype(str)).tocsr())
            names += list(self.ohe.get_feature_names_out(cats))
        self.feature_names = names
        return sparse.hstack(parts).tocsr()


def _as_category_strings(s: pd.Series) -> pd.Series:
    """NaN becomes NONE first, then everything becomes a string (the book's order bug fixed)."""
    out = s.astype(object).where(s.notna(), NONE)
    return out.map(lambda v: _fmt(v)).astype(object)


def _fmt(v) -> str:
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def _cat_with_rare(s: pd.Series, vocab: list[str]) -> pd.Series:
    allowed = set(vocab)
    return s.where(s.isin(allowed), RARE)


def _te_table(values: pd.Series, yy: np.ndarray, prior: np.ndarray, m: float) -> pd.DataFrame:
    frame = pd.DataFrame(yy, index=values.index)
    frame["_key"] = values.to_numpy()
    grouped = frame.groupby("_key")
    sums = grouped.sum()
    counts = grouped.size()
    return (sums.add(prior * m, axis=1)).div(counts + m, axis=0)
