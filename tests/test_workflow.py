"""Ensembling, run comparison, checks, `mlt init`, tuning, selection and serving."""

import json

import numpy as np
import pandas as pd
import pytest

from conftest import has
from mlt.checks import run_checks
from mlt.compare import compare_runs
from mlt.ensemble import ensemble_runs
from mlt.scaffold import scaffold
from mlt.train import run_cv


@pytest.fixture
def two_runs(make_cfg):
    a, _ = make_cfg("binary", "lgbm", n=400)
    b, _ = make_cfg("binary", "logreg", n=400)
    return run_cv(a), run_cv(b), a


@pytest.mark.parametrize("method", ["mean", "rank", "weighted", "stack"])
def test_ensemble_methods(two_runs, method):
    ra, rb, cfg = two_runs
    out = cfg["output_dir"]
    score = ensemble_runs([str(ra.run_dir), str(rb.run_dir)], method=method, name=f"blend_{method}", output_dir=out)
    assert 0.5 < score <= 1.0
    run = ra.run_dir.parent / f"blend_{method}"
    sub = pd.read_csv(run / "submission.csv")
    assert len(sub) == len(pd.read_csv(cfg["data"]["sample_submission"]))
    assert json.loads((run / "metrics.json").read_text())["model"] == f"ensemble:{method}"


def test_ensemble_refuses_different_folds(make_cfg):
    a, _ = make_cfg("binary", "lgbm")
    b, _ = make_cfg("binary", "logreg", cv={"n_splits": 4})
    ra, rb = run_cv(a), run_cv(b)
    with pytest.raises(ValueError, match="different folds"):
        ensemble_runs([str(ra.run_dir), str(rb.run_dir)], output_dir=a["output_dir"])


def test_compare(two_runs, capsys):
    ra, rb, _ = two_runs
    res = compare_runs(str(ra.run_dir), str(rb.run_dir), n_boot=200)
    assert res["se"] > 0 and res["verdict"] in {"tie", "leaning, needs confirmation", "likely real"}
    assert "resamples" in capsys.readouterr().out


def test_checks_detect_shift(make_cfg, workdir):
    cfg, _ = make_cfg("binary", "lgbm", n=600)
    report = run_checks(cfg)
    assert report["adversarial_auc"] < 0.65  # same distribution
    test = pd.read_csv(cfg["data"]["test"])
    test["x1"] += 3  # shift one feature
    test.to_csv(cfg["data"]["test"], index=False)
    shifted = run_checks(cfg)
    assert shifted["adversarial_auc"] > 0.9
    assert next(iter(shifted["adversarial_top"])) == "x1"


def test_init_guesses_columns(make_cfg, workdir):
    cfg, _ = make_cfg("multiclass", "lgbm")
    d = cfg["data"]
    path = scaffold(d["train"], d["test"], d["sample_submission"], out=str(workdir / "configs" / "new.yaml"))
    text = path.read_text()
    assert "id_col: id" in text and "target: target" in text and "task: multiclass" in text


@pytest.mark.skipif(not has("optuna"), reason="needs the tune extra")
def test_tune_writes_best_config(make_cfg):
    from mlt.tune import tune

    cfg, _ = make_cfg("binary", "lgbm", tune={"space": {
        "model.params.num_leaves": {"type": "int", "low": 4, "high": 16},
        "features.rare_min_count": {"type": "categorical", "choices": [0, 5]}}})
    study = tune(cfg, n_trials=3)
    assert len(study.trials) == 3
    best = f"{cfg['output_dir']}/{cfg['name']}_tune/best.yaml"
    from mlt.config import load_config

    assert load_config(best)["name"].endswith("_tuned")


def test_forward_selection(make_cfg):
    from mlt.tune import forward_select

    cfg, _ = make_cfg("binary", "logreg", data={"drop": ["date", "grp", "job"]})
    chosen = forward_select(cfg, max_features=2)
    assert chosen and chosen[0] in {"x1", "x2", "city"}


@pytest.mark.skipif(not has("fastapi"), reason="needs the serve extra")
def test_serve(make_cfg):
    from fastapi.testclient import TestClient

    from mlt.serve import create_app

    cfg, _ = make_cfg("binary", "lgbm")
    res = run_cv(cfg)
    client = TestClient(create_app(str(res.run_dir)))
    assert client.get("/health").json()["status"] == "ok"
    records = pd.read_csv(cfg["data"]["test"]).head(3).replace({np.nan: None}).to_dict("records")
    body = client.post("/predict", json={"records": records}).json()
    assert len(body["predictions"]) == 3 and set(body["predictions"]) <= {"yes", "no"}
    assert client.post("/predict", json={"records": []}).status_code == 400
