import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from mlt.config import load_config
from mlt.features.tabular import NONE, RARE, TabularPreprocessor


def _cfg(**features):
    return load_config(data={"data": {"train": "x.csv", "target": "y"}, "features": features})


@pytest.fixture
def frames():
    train = pd.DataFrame({"c": ["a", "a", "b", None, "rare1", "a", "b", "b"],
                          "n": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0],
                          "d": pd.date_range("2024-01-06", periods=8, freq="D").astype(str)})
    test = pd.DataFrame({"c": ["a", "zzz", None], "n": [1.0, 2.0, 3.0], "d": ["2024-02-01"] * 3})
    return train, test


def test_none_rare_and_unseen(frames):
    train, test = frames
    pre = TabularPreprocessor(_cfg(rare_min_count=2, fit_unsupervised_on_test=False), ["c"], ["n"], [], "label", "binary")
    pre.fit_transform(train, np.zeros(len(train)))
    assert pre.vocab["c"][:2] == [NONE, RARE]
    assert "rare1" not in pre.vocab["c"]
    out = pre.transform(test)
    codes = out["c"].tolist()
    assert codes[1] == pre.vocab["c"].index(RARE)  # unseen -> RARE
    assert codes[2] == pre.vocab["c"].index(NONE)  # missing -> NONE


def test_numeric_categories_are_not_turned_into_nan_strings():
    train = pd.DataFrame({"c": [1.0, 2.0, np.nan, 1.0]})
    pre = TabularPreprocessor(_cfg(), ["c"], [], [], "label", "binary")
    pre.fit_transform(train, np.zeros(4))
    assert set(pre.vocab["c"]) == {NONE, RARE, "1", "2"}


def test_dates_counts_bins_and_aggregates(frames):
    train, test = frames
    cfg = _cfg(count_encode=["c"], bins={"n": 3}, date_parts=["month", "dayofweek", "is_weekend"],
               aggregates=[{"by": "c", "cols": ["n"], "aggs": ["mean", "max"]}])
    pre = TabularPreprocessor(cfg, ["c"], ["n"], ["d"], "label", "binary")
    X = pre.fit_transform(train, np.zeros(len(train)), others=[test])
    for col in ["d_month", "d_dayofweek", "d_is_weekend", "c_count", "n_bin", "agg_c_n_mean", "agg_c_n_max"]:
        assert col in X.columns
    assert X["d_is_weekend"].iloc[0] == 1.0  # 2024-01-06 is a Saturday
    assert X["agg_c_n_max"].iloc[0] == 6.0  # max n where c == "a"
    assert pre.transform(test).shape[1] == X.shape[1]


def test_target_encoding_is_out_of_fold():
    n = 200
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"c": [f"u{i}" for i in range(n)]})  # every category unique
    y = rng.integers(0, 2, n)
    pre = TabularPreprocessor(_cfg(target_encode=["c"], target_encode_smoothing=1.0), ["c"], [], [], "label", "binary")
    X = pre.fit_transform(train, y)
    # unique categories never see their own label, so they all fall back to the prior
    assert np.allclose(X["te_c_0"], y.mean(), atol=0.08)
    corr = np.corrcoef(X["te_c_0"], y)[0, 1]
    assert abs(corr) < 0.3


def test_onehot_is_sparse_and_scaled(frames):
    train, test = frames
    pre = TabularPreprocessor(_cfg(), ["c"], ["n"], [], "onehot", "binary")
    X = pre.fit_transform(train, np.zeros(len(train)), others=[test])
    assert sparse.issparse(X)
    assert X.shape[0] == len(train)
    assert pre.transform(test).shape[1] == X.shape[1]
    assert not np.isnan(X.toarray()).any()


def test_embedding_and_native_outputs(frames):
    train, _ = frames
    pre = TabularPreprocessor(_cfg(), ["c"], ["n"], [], "embedding", "binary")
    X = pre.fit_transform(train, np.zeros(len(train)))
    assert X["cat"].dtype == np.int64 and X["num"].shape == (len(train), 1)
    assert pre.cardinalities == [len(pre.vocab["c"])]
    nat = TabularPreprocessor(_cfg(), ["c"], ["n"], [], "native", "binary")
    Xn = nat.fit_transform(train, np.zeros(len(train)))
    assert nat.cat_indices == [1] and Xn["c"].dtype == object


@pytest.mark.parametrize("method", ["variance", "correlation", "kbest", "mutual_info", "from_model"])
def test_selection_methods(method):
    rng = np.random.default_rng(0)
    n = 300
    x = rng.normal(size=(n, 6))
    x[:, 5] = x[:, 0] * 0.999  # near duplicate
    y = (x[:, 0] + x[:, 1] > 0).astype(int)
    train = pd.DataFrame(x, columns=[f"f{i}" for i in range(6)])
    cfg = _cfg(selection={"method": method, "k": 3, "threshold": 0.98 if method == "correlation" else None})
    pre = TabularPreprocessor(cfg, [], list(train.columns), [], "label", "binary")
    X = pre.fit_transform(train, y)
    assert 1 <= X.shape[1] <= 6
    assert pre.transform(train).shape[1] == X.shape[1]
    if method == "correlation":
        assert X.shape[1] == 5
