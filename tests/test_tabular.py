"""End to end on tiny synthetic data: every task, the main model families, files and predict."""

import numpy as np
import pandas as pd
import pytest

from mlt.cli import main
from mlt.predict import Predictor, predict_file
from mlt.submission import score_submission, validate_submission
from mlt.train import run_cv

CASES = [
    ("binary", "lgbm", "auc"),
    ("binary", "xgb", "logloss"),
    ("binary", "catboost", "accuracy"),
    ("binary", "logreg", "f1"),
    ("multiclass", "lgbm", "logloss"),
    ("multiclass", "rf", "accuracy"),
    ("regression", "lgbm", "rmse"),
    ("regression", "ridge", "mae"),
    ("ordinal", "lgbm", "qwk"),
    ("multilabel", "logreg", "logloss"),
    ("multilabel", "et", "auc"),
]


@pytest.mark.parametrize("task,model,metric", CASES)
def test_run_every_task(make_cfg, task, model, metric):
    cfg, _ = make_cfg(task, model, metric=metric, data={"date_cols": ["date"]})
    res = run_cv(cfg)
    run = res.run_dir
    for f in ["config.yaml", "state.json", "metrics.json", "oof.csv", "test_pred.csv", "submission.csv"]:
        assert (run / f).exists(), f
    assert np.isfinite(res.score)
    oof = pd.read_csv(run / "oof.csv")
    assert oof.filter(like="pred").notna().all().all()
    sub = pd.read_csv(run / "submission.csv")
    sample = pd.read_csv(cfg["data"]["sample_submission"])
    assert validate_submission(sub, sample, "id") == []
    # reloading the saved fold models reproduces the test predictions
    test = pd.read_csv(cfg["data"]["test"])
    again = Predictor(run).predict(test)
    saved = pd.read_csv(run / "test_pred.csv").filter(like="pred").to_numpy()
    assert np.allclose(again, saved, atol=1e-6)
    registry = pd.read_csv(run.parent / "runs.csv")
    assert registry["name"].iloc[-1] == cfg["name"]


def test_scores_are_sensible(make_cfg):
    cfg, _ = make_cfg("binary", "lgbm", metric="auc", n=600)
    assert run_cv(cfg).score > 0.8
    cfg, _ = make_cfg("binary", "logreg", metric="auc", n=600)
    res = run_cv(cfg)
    assert res.score > 0.8
    sol = pd.read_csv(cfg["data"]["test"].replace("test.csv", "solution.csv"))
    sub = pd.read_csv(res.run_dir / "submission.csv")
    assert score_submission(sub, sol, "auc", "binary", "id") > 0.8


def test_log1p_target_and_binned_folds(make_cfg):
    cfg, _ = make_cfg("regression", "lgbm", metric="rmsle", target_transform="log1p", cv={"scheme": "binned"})
    res = run_cv(cfg)
    assert res.score < 0.5
    assert (res.test_pred > 0).all()


def test_feature_options_end_to_end(make_cfg):
    cfg, _ = make_cfg("binary", "lgbm", features={"rare_min_count": 5, "count_encode": ["job"],
                                                   "target_encode": ["city", "job"], "combine": [["city", "job"]],
                                                   "selection": {"method": "from_model", "k": 6}},
                      data={"date_cols": ["date"], "drop": ["grp"]})
    res = run_cv(cfg)
    imp = pd.read_csv(res.run_dir / "importance.csv")
    assert 6 <= len(imp) <= 12  # union of the features each fold selected


def test_group_folds_and_subset(make_cfg):
    cfg, _ = make_cfg("binary", "logreg", cv={"group_col": "grp"}, train={"folds": [0]})
    res = run_cv(cfg)
    assert len(res.fold_scores) == 1


def test_cli_run_folds_predict(make_cfg, capsys):
    cfg, path = make_cfg("binary", "lgbm")
    main(["folds", str(path)])
    assert "fold" in capsys.readouterr().out
    main(["run", str(path), "--set", "model.params.learning_rate=0.2", "--name", "cli_run"])
    run = f"{cfg['output_dir']}/cli_run"
    out = predict_file(run, cfg["data"]["test"], sample_path=cfg["data"]["sample_submission"])
    assert out.exists()
    main(["validate", str(out), "--sample", cfg["data"]["sample_submission"]])


def test_cli_reports_errors_cleanly(make_cfg):
    _, path = make_cfg("binary", "lgbm")
    with pytest.raises(SystemExit):
        main(["run", str(path), "--set", "model.name=not_a_model"])
    with pytest.raises(SystemExit):
        main(["run", str(path), "--set", "metric=rmse"])  # regression metric on a binary task
