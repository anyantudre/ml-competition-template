import numpy as np
import pandas as pd
import pytest

from mlt.config import load_config
from mlt.folds import fold_ids, get_folds, make_folds, sturges_bins, train_mask


def _cfg(**cv):
    return load_config(data={"data": {"train": "x.csv", "target": "y"}, "cv": cv})


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    n = 500
    return pd.DataFrame({"id": np.arange(n), "g": rng.integers(0, 50, n),
                         "t": pd.date_range("2020-01-01", periods=n, freq="D").astype(str)}), rng


def test_stratified_keeps_class_balance(frame):
    df, rng = frame
    y = (rng.random(len(df)) < 0.1).astype(int)
    folds = make_folds(df, y, _cfg(scheme="stratified"), "binary")
    rates = [y[folds == f].mean() for f in fold_ids(folds)]
    assert max(rates) - min(rates) < 0.03
    assert (folds >= 0).all()


def test_binned_regression_covers_range(frame):
    df, rng = frame
    y = rng.lognormal(size=len(df))
    folds = make_folds(df, y, _cfg(scheme="binned"), "regression")
    assert len(fold_ids(folds)) == 5
    assert sturges_bins(500) == 9


@pytest.mark.parametrize("scheme", ["group", "stratified_group"])
def test_groups_never_split(frame, scheme):
    df, rng = frame
    y = rng.integers(0, 2, len(df))
    folds = make_folds(df, y, _cfg(scheme=scheme, group_col="g"), "binary")
    per_group = pd.Series(folds).groupby(df["g"]).nunique()
    assert (per_group == 1).all()


def test_time_folds_train_on_past_only(frame):
    df, rng = frame
    cfg = _cfg(scheme="time", time_col="t", n_splits=4)
    folds = make_folds(df, rng.random(len(df)), cfg, "regression")
    t = pd.to_datetime(df["t"])
    for f in fold_ids(folds):
        tr = train_mask(folds, f, "time")
        assert t[tr].max() < t[folds == f].min()
    assert (folds == -1).sum() > 0  # the first chunk is only ever trained on


def test_holdout(frame):
    df, rng = frame
    folds = make_folds(df, rng.integers(0, 2, len(df)), _cfg(scheme="holdout", holdout_size=0.2), "binary")
    assert fold_ids(folds) == [0]
    assert abs((folds == 0).mean() - 0.2) < 0.01


def test_folds_file_is_reused_and_checked(tmp_path, frame, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df, rng = frame
    y = rng.integers(0, 2, len(df))
    cfg = load_config(data={"data": {"train": "data/comp/train.csv", "target": "y", "id_col": "id"},
                            "output_dir": "out"})
    f1, path, h1 = get_folds(df, y, cfg, "binary")
    assert path.exists()
    f2, _, h2 = get_folds(df, y, cfg, "binary")
    assert h1 == h2 and np.array_equal(f1, f2)
    with pytest.raises(ValueError, match="does not match"):
        get_folds(df.iloc[:-1], y[:-1], cfg, "binary")
