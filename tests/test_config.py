import pytest
import yaml

from mlt.config import ConfigError, deep_merge, load_config, parse_override


def _write(path, data):
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_defaults_and_name(tmp_path):
    cfg = load_config(_write(tmp_path / "exp1.yaml", {"data": {"train": "t.csv", "target": "y"}}))
    assert cfg["name"] == "exp1"
    assert cfg["cv"]["n_splits"] == 5
    assert cfg["model"]["params"] == {}


def test_unknown_key_suggests(tmp_path):
    path = _write(tmp_path / "bad.yaml", {"data": {"train": "t.csv", "target": "y", "trget": "z"}})
    with pytest.raises(ConfigError, match="Did you mean 'data.target'"):
        load_config(path)


def test_model_params_are_free_form(tmp_path):
    path = _write(tmp_path / "ok.yaml", {"data": {"train": "t.csv", "target": "y"},
                                        "model": {"name": "lgbm", "params": {"anything": 1}}})
    assert load_config(path)["model"]["params"]["anything"] == 1


def test_extends_and_overrides(tmp_path):
    _write(tmp_path / "base.yaml", {"data": {"train": "t.csv", "target": "y"}, "model": {"name": "lgbm",
                                    "params": {"num_leaves": 8, "learning_rate": 0.1}}})
    child = _write(tmp_path / "child.yaml", {"extends": "base.yaml", "model": {"params": {"num_leaves": 16}}})
    cfg = load_config(child, ["model.params.learning_rate=0.05", "cv.n_splits=3"])
    assert cfg["model"]["params"] == {"num_leaves": 16, "learning_rate": 0.05}
    assert cfg["cv"]["n_splits"] == 3
    cleared = load_config(child, ["model.params={}"])
    assert cleared["model"]["params"] == {}


def test_switching_model_drops_old_params(tmp_path):
    _write(tmp_path / "base.yaml", {"data": {"train": "t.csv", "target": "y"},
                                    "model": {"name": "tfidf", "params": {"C": 4.0}}})
    child = _write(tmp_path / "lstm.yaml", {"extends": "base.yaml", "model": {"name": "bilstm"}})
    assert load_config(child)["model"]["params"] == {}
    own = _write(tmp_path / "own.yaml", {"extends": "base.yaml", "model": {"name": "bilstm", "params": {"hidden": 8}}})
    assert load_config(own)["model"]["params"] == {"hidden": 8}
    kept = _write(tmp_path / "same.yaml", {"extends": "base.yaml", "model": {"backbone": "x"}})
    assert load_config(kept)["model"]["params"] == {"C": 4.0}
    assert load_config(tmp_path / "base.yaml", ["model.name=nb"])["model"]["params"] == {}
    assert load_config(tmp_path / "base.yaml", ["model.name=nb", "model.params.alpha=0.5"])["model"]["params"] == {"alpha": 0.5}


def test_parse_override_types():
    assert parse_override("a.b=3") == {"a": {"b": 3}}
    assert parse_override("a=[1, 2]") == {"a": [1, 2]}
    assert parse_override("a=null") == {"a": None}
    with pytest.raises(ConfigError):
        parse_override("novalue")


def test_deep_merge_lists_replace():
    assert deep_merge({"a": [1, 2], "b": {"c": 1}}, {"a": [3], "b": {"d": 2}}) == {"a": [3], "b": {"c": 1, "d": 2}}


def test_required_fields(tmp_path):
    with pytest.raises(ConfigError, match="text_col"):
        load_config(_write(tmp_path / "t.yaml", {"modality": "text", "data": {"train": "t.csv", "target": "y"}}))
    with pytest.raises(ConfigError, match="mask_col"):
        load_config(_write(tmp_path / "s.yaml", {"modality": "segmentation",
                                                 "data": {"train": "t.csv", "image_col": "img"}}))
