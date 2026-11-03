"""The offline examples build valid competition folders, and every example config loads."""

from pathlib import Path

import pandas as pd
import pytest

from mlt.config import load_config
from mlt.examples import EXAMPLES, make_example

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("config", sorted((ROOT / "configs").rglob("*.yaml")), ids=lambda p: p.name)
def test_configs_load(config):
    cfg = load_config(config)
    assert cfg["data"]["train"]


def test_every_example_has_a_config():
    for _, config in EXAMPLES.values():
        assert (ROOT / config).exists(), config


@pytest.mark.parametrize("name", ["breast_cancer", "diabetes"])
def test_offline_examples(workdir, name):
    out = make_example(name)
    train, test = pd.read_csv(out / "train.csv"), pd.read_csv(out / "test.csv")
    sample, solution = pd.read_csv(out / "sample_submission.csv"), pd.read_csv(out / "solution.csv")
    assert "target" in train and "target" not in test
    assert len(sample) == len(test) == len(solution)
    assert set(sample["id"]) == set(test["id"])


def test_breast_cancer_example_runs(workdir):
    from mlt.train import run_cv

    make_example("breast_cancer")
    cfg = load_config(ROOT / "configs/examples/breast_cancer.yaml",
                      ["model.params.n_estimators=100", f"output_dir={workdir / 'outputs'}"])
    assert run_cv(cfg).score > 0.97
