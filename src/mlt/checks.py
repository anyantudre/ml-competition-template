"""Checks to run before modelling: data sanity, leakage hints and train/test shift.

Adversarial validation trains a model to tell train rows from test rows. An AUC near 0.5 means
the two look alike, so a random CV split should track the leaderboard. A high AUC means a shift:
look at the top features, consider dropping them or building folds that mimic the test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.metrics import roc_auc_score

from .config import targets
from .data import feature_columns, load_data, resolve_task
from .utils import write_json


def _section(title):
    print(f"\n== {title}")


def run_checks(cfg: dict) -> dict:
    train, test, sample = load_data(cfg)
    task, metric = resolve_task(cfg, train)
    names = targets(cfg)
    id_col = cfg["data"]["id_col"]
    report: dict = {"task": task, "metric": metric, "train_rows": len(train),
                    "test_rows": None if test is None else len(test)}

    _section("shapes")
    print(f"train {train.shape}, test {None if test is None else test.shape}, "
          f"sample {None if sample is None else sample.shape}")
    print(f"task {task}, metric {metric}")

    if names:
        _section("target")
        y = train[names]
        if task in {"binary", "multiclass", "ordinal"}:
            dist = y.iloc[:, 0].value_counts(normalize=True).round(4)
            print(dist.to_string())
            report["target_distribution"] = dist.to_dict()
            if dist.iloc[0] > 0.9:
                print("note: skewed classes, prefer auc / logloss / f1 over accuracy")
        elif task == "multilabel":
            print(y.mean().round(4).to_string())
        else:
            print(y.iloc[:, 0].describe().to_string())
            if (y.iloc[:, 0] > 0).all() and y.iloc[:, 0].skew() > 1:
                print("note: positive, right-skewed target; consider target_transform: log1p")

    _section("ids and duplicates")
    issues = []
    if id_col and id_col in train:
        if train[id_col].duplicated().any():
            issues.append(f"{int(train[id_col].duplicated().sum())} duplicated train ids")
        if test is not None and id_col in test:
            overlap = len(set(train[id_col]) & set(test[id_col]))
            if overlap:
                issues.append(f"{overlap} ids appear in both train and test")
    if cfg["modality"] == "tabular":
        features, cats, nums, dates = feature_columns(cfg, train, test)
        dup = int(train.duplicated(subset=features).sum())
        if dup:
            issues.append(f"{dup} duplicated feature rows in train")
        if test is not None:
            key = pd.util.hash_pandas_object(train[features], index=False)
            tkey = pd.util.hash_pandas_object(test[features], index=False)
            shared = int(tkey.isin(set(key)).sum())
            if shared:
                issues.append(f"{shared} test rows are exact copies of train rows (check for label reuse)")
    print("\n".join(issues) or "none")
    report["issues"] = issues

    if cfg["modality"] == "tabular":
        _section("columns")
        const = [c for c in features if train[c].nunique(dropna=False) <= 1]
        missing = (train[features].isna().mean().sort_values(ascending=False))
        print(f"{len(cats)} categorical, {len(nums)} numeric, {len(dates)} date")
        if const:
            print(f"constant columns: {const}")
        high = missing[missing > 0.5]
        if len(high):
            print(f"more than 50% missing: {high.round(3).to_dict()}")
        if test is not None:
            unseen = {c: int((~test[c].isin(set(train[c]))).sum()) for c in cats
                      if (~test[c].isin(set(train[c]))).any()}
            if unseen:
                print(f"test categories never seen in train (rows): {unseen}")
        report.update(constant=const, high_missing=high.to_dict())

        if task in {"binary", "regression"} and names:
            _section("single-feature leakage scan")
            yv = train[names[0]]
            yv = yv.map({v: i for i, v in enumerate(sorted(yv.unique(), key=str))}) if task == "binary" else yv
            flags = {}
            for c in nums:
                x = pd.to_numeric(train[c], errors="coerce")
                ok = x.notna()
                if ok.sum() < 10 or x[ok].nunique() < 2:
                    continue
                if task == "binary":
                    a = roc_auc_score(yv[ok], x[ok])
                    strength = max(a, 1 - a)
                else:
                    strength = abs(x[ok].rank().corr(yv[ok].rank()))
                if strength > 0.95:
                    flags[c] = round(float(strength), 4)
            print(f"near-perfect single features: {flags}" if flags else "nothing suspicious")
            report["leak_suspects"] = flags

        if test is not None:
            _section("adversarial validation (train vs test)")
            auc, top = adversarial_validation(cfg, train, test, cats, nums, dates)
            verdict = "no visible shift" if auc < 0.6 else ("mild shift" if auc < 0.75 else "strong shift")
            print(f"AUC {auc:.4f} -> {verdict}")
            print("most telling features: " + ", ".join(f"{k} ({v:.0f})" for k, v in top.items()))
            report["adversarial_auc"] = auc
            report["adversarial_top"] = top

    if cfg["modality"] == "text":
        _section("text lengths (words)")
        col = cfg["data"]["text_col"]
        for name, df in [("train", train), ("test", test)]:
            if df is not None:
                n = df[col].fillna("").astype(str).str.split().str.len()
                print(f"{name}: median {n.median():.0f}, p95 {n.quantile(0.95):.0f}, max {n.max()}")

    out = Path(cfg["output_dir"]) / "checks" / f"{cfg['name']}.json"
    write_json(report, out)
    print(f"\nreport saved to {out}")
    return report


def adversarial_validation(cfg, train, test, cats, nums, dates, max_rows: int = 20000):
    import lightgbm as lgb

    from .features.tabular import TabularPreprocessor

    tr = train.sample(min(len(train), max_rows), random_state=0)
    te = test.sample(min(len(test), max_rows), random_state=0)
    both = pd.concat([tr, te], ignore_index=True)
    is_test = np.r_[np.zeros(len(tr)), np.ones(len(te))].astype(int)
    pre = TabularPreprocessor(cfg, cats, nums, dates, "label", "binary")
    X = pre.fit_transform(both, is_test)
    X.columns = [str(c) for c in X.columns]
    oof = np.zeros(len(X))
    imp = np.zeros(X.shape[1])
    for trn, val in model_selection.StratifiedKFold(5, shuffle=True, random_state=0).split(X, is_test):
        est = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, verbose=-1, random_state=0)
        est.fit(X.iloc[trn], is_test[trn])
        oof[val] = est.predict_proba(X.iloc[val])[:, 1]
        imp += est.feature_importances_
    top = pd.Series(imp / 5, index=pre.feature_names).sort_values(ascending=False).head(8)
    return float(roc_auc_score(is_test, oof)), top.to_dict()
