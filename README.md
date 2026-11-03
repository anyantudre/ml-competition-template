<h1 align="center">ml-competition-template</h1>

<p align="center">
  A config-driven starter kit for Kaggle, Zindi and similar competitions:<br>
  folds, training, out-of-fold predictions, ensembles and a checked submission file from one YAML file.
</p>

<p align="center">
  <a href="https://github.com/anyantudre/ml-competition-template/actions/workflows/ci.yml"><img src="https://github.com/anyantudre/ml-competition-template/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="MIT"></a>
</p>

One experiment is one YAML file. `mlt run` builds the cross-validation folds (once, then reused),
trains every fold, scores the out-of-fold (OOF) predictions with the competition metric, averages
the fold models on the test set and writes a submission checked against `sample_submission.csv`.
The same command covers tabular data, text, image classification and segmentation.

The workflow and many techniques follow Abhishek Thakur's book
[Approaching (Almost) Any Machine Learning Problem](https://github.com/abhishekkrthakur/approachingalmost).
The [chapter map](#what-comes-from-the-book) shows where each chapter lives in the code.

## Quickstart

Click **Use this template** on GitHub (one fresh repository per competition), then:

```bash
git clone https://github.com/<you>/<your-competition>.git
cd <your-competition>
uv venv
uv pip install -e ".[all,dev]"        # or just -e . for tabular only
```

On macOS, LightGBM also needs OpenMP: `brew install libomp`.

`uv` is optional: `python -m venv .venv` and `pip install -e ".[all,dev]"` work too. Activate the
environment (`source .venv/bin/activate`, or `.venv\Scripts\activate` on Windows), then try an example:

```bash
mlt example adult                      # writes data/adult/{train,test,sample_submission,solution}.csv
mlt check configs/examples/adult.yaml  # data checks, leakage hints, train/test shift
mlt run configs/examples/adult.yaml    # folds + 5 fold models + OOF score + submission
mlt score outputs/adult/submission.csv --solution data/adult/solution.csv --metric auc --task binary
```

`mlt example list` shows the ten bundled datasets (binary, multiclass, regression, ordinal,
multilabel, text, images, masks).

## A new competition

1. Put the competition files in `data/<competition>/`.
2. Write a starter config from them. `mlt init` guesses the id column, target, modality, task and
   metric, and prints each guess:

   ```bash
   mlt init --train data/comp/train.csv --test data/comp/test.csv --sample data/comp/sample_submission.csv --metric auc
   ```

3. Check the data, then train:

   ```bash
   mlt check configs/comp.yaml
   mlt run configs/comp.yaml
   ```

4. Upload `outputs/comp/submission.csv`. For a later test file (a second stage, or a code
   competition), predict with the saved fold models:

   ```bash
   mlt predict outputs/comp --input data/comp/test_stage2.csv --sample data/comp/sample_submission.csv
   ```

New ideas go in new configs. `extends:` reuses a base config and `--set` changes one value from the
command line:

```yaml
# configs/comp_xgb.yaml
extends: comp.yaml
model:
  name: xgb
  params: {learning_rate: 0.03, max_depth: 7}
```

```bash
mlt run configs/comp.yaml --set model.name=catboost --name comp_cat
```

Switching `model.name` (in a child config or with `--set`) drops the `model.params` written for
the previous model.

## What a run produces

```
outputs/
  folds/            one folds file per (data, scheme, k, seed), shared by every run
  runs.csv          one line per run: time, model, metric, CV, fold std, folds hash, git commit
  <run name>/
    config.yaml     the resolved config, every default filled in
    metrics.json    CV score, per-fold scores, duration, folds hash, git commit
    oof.csv         id, fold, target, out-of-fold predictions
    test_pred.csv   test predictions averaged over the fold models
    submission.csv  shaped like sample_submission and validated
    importance.csv  mean feature importance (tree and linear models)
    models/fold*/   fitted preprocessing and models, reloaded by predict and serve
```

Every run with the same data and `cv` settings uses the same folds file, so OOF predictions from
different runs can be compared and blended without leakage. Unknown config keys stop the run with a
suggestion (`Unknown config key 'data.trget'. Did you mean 'data.target'?`), so a typo never falls
back to a default without telling you.

## Repository layout

```
configs/
  examples/          one config per bundled example (copy one to start)
  kaggle/            playground_s4e11.yaml, the competition this template grew out of
notebooks/eda.ipynb  exploration driven by a config
src/mlt/
  cli.py             mlt init | example | check | folds | run | predict | validate | score
                     | compare | ensemble | tune | select | serve
  config.py          defaults, YAML loading, extends, --set overrides, key validation
  data.py            reading files, task inference, label encoding of the target
  folds.py           kfold, stratified, binned, group, stratified_group, time, holdout
  metrics.py         metrics with direction, MAP@k, QWK with tuned cuts, Dice/IoU
  train.py           the CV loop and the run folder
  predict.py         reload a run and predict new rows
  submission.py      build, validate and locally score submissions
  features/          tabular preprocessing, feature selection, text helpers
  models/            model registry, entity embeddings, text, image and mask networks
  pipelines/         one pipeline per modality (tabular, text, image, segmentation)
  engine.py          one PyTorch train/eval loop for every neural model
  ensemble.py        mean, rank, weighted and stacked blends of runs
  compare.py         paired bootstrap between two runs
  tune.py            Optuna search and greedy forward feature selection
  checks.py          data checks and adversarial validation
  serve.py           FastAPI app around a run
  examples.py        the bundled example datasets
tests/               pytest suite, run by CI on Linux, Windows and macOS
Dockerfile, Makefile, pyproject.toml
```

## Tabular data

`modality: tabular` (the default). Tasks: `binary`, `multiclass`, `multilabel`, `regression` and
`ordinal`. When `task` is empty it is inferred from the target and logged.

| model.name | family | encoding used |
|---|---|---|
| `lgbm`, `xgb` | gradient boosting, early stopping on the validation fold | integer codes |
| `catboost` | gradient boosting with native categoricals | category strings |
| `rf`, `et`, `hgb`, `gbm`, `dt` | scikit-learn trees | integer codes |
| `logreg` (classification), `ridge`, `lasso` (regression), `svm`, `knn`, `mlp` | linear and distance models | sparse one-hot + scaled numerics |
| `entity_nn` | entity embeddings: one embedding per categorical column, then dense layers | codes + scaled numerics |

Everything under `model.params` goes straight to the estimator. Multilabel trains one model per
label. Ordinal targets are regressed, then the rounding cuts that maximise QWK are fitted on the OOF
predictions and reused for the test set.

Feature options, all fitted inside each fold:

```yaml
data:
  date_cols: [signup_date]          # year, month, day, dayofweek, dayofyear, hour, is_weekend
  drop: [name]
features:
  rare_min_count: 20                # categories seen fewer times become RARE; NaN becomes NONE
  count_encode: [city]              # frequency of each category
  target_encode: [city, job]        # out-of-fold, smoothed (target_encode_smoothing)
  combine: [[city, job]]            # new categorical city__job
  aggregates:
    - {by: store, cols: [price], aggs: [mean, max]}
  log1p: [income]
  bins: {age: 10}
  selection: {method: from_model, k: 50}   # variance | correlation | kbest | mutual_info | from_model
  fit_unsupervised_on_test: true    # vocabularies, counts and aggregates also see the test rows
target_transform: log1p             # for RMSLE-style targets; predictions are mapped back
```

Values that only appear in the test set map to RARE instead of crashing. Vocabularies, counts and
aggregates never use the target, so fitting them on train + test is safe in a competition. Set
`fit_unsupervised_on_test: false` when the test set is not known in advance.

## Text

`modality: text` with `data.text_col`. Classification (binary, multiclass, multilabel), regression
and ordinal targets.

| model.name | what it is | CPU friendly |
|---|---|---|
| `tfidf` | TF-IDF n-grams + logistic regression (ridge for regression) | yes, the baseline to beat |
| `nb` | word counts + multinomial Naive Bayes | yes |
| `bilstm` | BiLSTM with mean + max pooling; `features.text.embeddings` loads fastText or GloVe vectors and reports coverage | small data |
| `transformer` | any Hugging Face encoder in `model.backbone`, mean pooling, AdamW with warmup | needs a GPU beyond toy sizes |

```bash
mlt example newsgroups
mlt run configs/examples/newsgroups_tfidf.yaml
mlt run configs/examples/newsgroups_transformer.yaml   # a 2-layer BERT for the CPU demo
```

## Images

`modality: image` with `data.image_col` (file names) and `data.image_dir` (plus
`data.test_image_dir` when test images sit elsewhere). Any [timm](https://github.com/huggingface/pytorch-image-models)
backbone in `model.backbone`: `resnet18`, `efficientnet_b0`, `convnext_tiny` and so on. ImageNet
normalisation, light augmentation (flip, affine, brightness and contrast), optional flip TTA
(`train.tta: true`) and mixed precision on CUDA. Classification, multilabel and regression targets.

```bash
mlt example digits
mlt run configs/examples/digits.yaml
```

## Segmentation

`modality: segmentation` with `data.image_col` and `data.mask_col` (any non-zero pixel is
foreground). `model.name` picks the
[segmentation_models_pytorch](https://github.com/qubvel-org/segmentation_models.pytorch)
architecture (`Unet`, `UnetPlusPlus`, `FPN`, `DeepLabV3Plus`...) and `model.backbone` the
encoder. Loss is BCE + Dice and the metric is mean per-image Dice. Predicted masks are resized
back to each image's own size and written as run-length encoding (`submission.rle_order: F` for
column-major numbering, `C` for row-major).

```bash
mlt example digits_seg
mlt run configs/examples/digits_seg.yaml
```

Neural settings live under `train:` for every modality: `epochs`, `batch_size`, `lr`,
`weight_decay`, `scheduler` (cosine, linear, constant), `warmup_ratio`, `patience`, `grad_accum`,
`max_grad_norm`, `image_size`, `max_len`, `amp`, `device`.

## Validation that tracks the leaderboard

`cv.scheme` picks the split. Without it, classification uses stratified folds, regression uses
plain k-fold, a `group_col` switches to group folds and a `time_col` to time folds.

| scheme | when |
|---|---|
| `stratified` | classification, ordinal, multilabel (on label combinations) |
| `kfold` | regression with an even target |
| `binned` | regression with a skewed target: stratified on target bins (Sturges' rule) |
| `group`, `stratified_group` | several rows per patient, user, document or site (`cv.group_col`) |
| `time` | forecasting: each fold trains only on earlier rows (`cv.time_col`) |
| `holdout` | very large data: one stratified split |

`mlt check` runs before modelling and reports duplicated or shared ids, test rows copied from train,
constant and mostly-missing columns, unseen test categories, single features that predict the target
almost perfectly (a leak hint), and adversarial validation. Adversarial validation trains a model to
tell train rows from test rows: an AUC near 0.5 means a random split should track the leaderboard,
a high AUC names the features that differ.

## Improving a score

```bash
mlt compare outputs/comp outputs/comp_xgb       # paired bootstrap on the same OOF rows
mlt tune configs/comp.yaml --trials 50          # Optuna; writes outputs/comp_tune/best.yaml
mlt select configs/comp.yaml                    # greedy forward feature selection on CV score
mlt ensemble outputs/comp outputs/comp_xgb outputs/comp_cat --method weighted --name blend
```

`compare` scores both runs on the same resampled rows. It prints the gain, its standard error and a
verdict: under one standard error the runs are tied, so do not spend a submission on the difference.

`tune` searches `tune.space` (any config key, with `float`, `int` or `categorical` ranges) or a
default space for the model. Trials stop early after a few weak folds (median pruning).

`ensemble` checks that every member used the same folds, then blends them:

- `mean`
- `rank` (for AUC)
- `weighted`: weights fitted on OOF predictions. The score it reports is nested: weights fitted on
  the other folds, scored on the held-out fold.
- `stack`: a level-2 model (`--stack-model linear`, `lgbm`, `xgb`...) trained on the OOF predictions
  with the same folds.

The blend is a run folder like any other, with its own submission and a line in `runs.csv`.

## Serving and Docker

```bash
uv pip install -e ".[serve]"
mlt serve outputs/adult --port 8000
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
     -d '{"records": [{"age": 39, "workclass": "State-gov", "education": "Bachelors"}]}'
```

Missing columns are treated as missing values. The `Dockerfile` builds a CPU image that can train
or serve. Its header has the commands, including the volume mounts for `data/` and `outputs/`.

## Reproducibility

- Seeds come from the config (`seed`). Every run saves its resolved config, the folds hash and the
  git commit (marked `-dirty` when there are uncommitted changes).
- `make freeze` writes `requirements.lock.txt` with the exact package versions of the competition
  environment.
- CI installs the package from scratch and runs an example end to end, so a fresh clone is tested on
  every push.

## Example results

Measured from a fresh clone on a laptop CPU with the commands above. "Held-out" scores the
submission against the labels kept in `solution.csv`, the way a private leaderboard would.

| Example | Task | Model | Metric | CV (OOF) | Held-out |
|---|---|---|---|---|---|
| breast_cancer | binary | LightGBM | AUC | 0.995 | 0.997 |
| adult | binary, categoricals | LightGBM + count/target encoding | AUC | 0.924 | 0.925 |
| adult | | CatBoost | AUC | 0.925 | 0.924 |
| adult | | logistic regression, one-hot | AUC | 0.909 | 0.907 |
| adult | | entity embeddings | AUC | 0.906 | 0.907 |
| adult | | weighted blend of the four | AUC | 0.926 (nested) | 0.925 |
| car | multiclass | XGBoost | log loss | 0.061 | 0.100 |
| diabetes | regression | ridge, binned folds | RMSE | 56.2 | 53.8 |
| house_prices | regression | LightGBM on log1p(price) | RMSLE | 0.122 | 0.132 |
| wine_quality | ordinal | LightGBM + tuned cuts | QWK | 0.641 | 0.641 |
| yeast | multilabel, 14 labels | logistic regression | log loss | 0.470 | 0.466 |
| newsgroups | text, 4 topics | TF-IDF + logistic regression | log loss, accuracy | 0.583 | 0.898 accuracy |
| newsgroups | | 2-layer BERT, 3 epochs | log loss, accuracy | 0.590 | 0.838 accuracy |
| newsgroups | | BiLSTM, no pretrained vectors | log loss, accuracy | 1.000 | 0.665 accuracy |
| digits | image, 10 classes | MobileNetV3-small, 64 px, 3 epochs | accuracy | 0.929 | 0.975 |
| digits_seg | segmentation | U-Net, resnet18 encoder, 3 epochs | Dice | 0.985 | 0.993 |
| Kaggle S4E11 | binary, 140k rows | LightGBM | accuracy | 0.939 | on Kaggle |

The demo settings are small on purpose (few epochs, small images) so everything runs on a CPU.
The BiLSTM learns its embeddings from 1,600 documents here; pretrained fastText or GloVe vectors
(`features.text.embeddings`) are what make it competitive. For S4E11, the scripts this template
replaced reported 0.912 CV accuracy.

## What comes from the book

| Book chapter | Where it lives here |
|---|---|
| Supervised vs unsupervised | task inference in `data.py`, t-SNE view in `eda.plot_tsne` |
| Cross-validation | `folds.py`: k-fold, stratified, binned regression (Sturges' rule), group, holdout, plus stratified-group and time folds |
| Evaluation metrics | `metrics.py`: AUC, log loss, F1/precision/recall, MCC, QWK, MAP@k, RMSE, RMSLE, MAPE, R2, Dice, IoU |
| Arranging machine learning projects | config + folds + model registry + one CV loop, as a package with a CLI |
| Approaching categorical variables | NONE/RARE handling, label and one-hot encoding by model family, count and out-of-fold target encoding, combinations, entity embeddings |
| Feature engineering | date parts, group aggregates, binning, log1p |
| Feature selection | variance, correlation, univariate (k-best, mutual information), model-based, greedy forward selection |
| Hyperparameter optimization | `mlt tune` with Optuna and default ranges per model |
| Image classification & segmentation | timm classifiers, U-Net family, Dice + BCE loss, RLE |
| Text classification/regression | TF-IDF, Naive Bayes, BiLSTM with pretrained vectors, transformers |
| Ensembling and stacking | mean, rank, optimised weights, stacking on OOF predictions with shared folds |
| Reproducible code & model serving | saved configs, seeds, folds hash, FastAPI service, Dockerfile |

## Development

```bash
uv pip install -e ".[all,dev]"
pytest -q -m "not torch and not network"   # fast suite
pytest -q                                  # everything, including the neural pipelines
ruff check src tests
```

## Credits and license

The project structure and many of the techniques come from Abhishek Thakur,
*Approaching (Almost) Any Machine Learning Problem* (2020). This repository contains original code
only, no text from the book. The example datasets are downloaded from scikit-learn and OpenML at
run time and are not stored here.

MIT, see [LICENSE](LICENSE).
