import joblib
import os
import argparse
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn import preprocessing

import config
import model_dispatcher


def run(fold, model):
    # read the training data with folds
    df     = pd.read_csv(config.TRAINING_FILE)
    test   = pd.read_csv(config.TEST_FILE, index_col=0)
    features     = [f for f in df.columns if f not in ("id", "Name", config.TARGET_COL, "kfold")]
    cat_features = [f for f in features if f not in config.NUM_COLS and f not in ("kfold", config.TARGET_COL)]

    # fill all NaN values in numerical and categorical cols
    for col in config.NUM_COLS:
        df[col]   = df[col].fillna(df[col].mean())
        test[col] = test[col].fillna(df[col].mean())

    for col in features:
        if col not in config.NUM_COLS: # do not encode the numerical columns
            df.loc[:, col]   = df[col].astype(str).fillna("NONE")
            test.loc[:, col] = test[col].astype(str).fillna("NONE")

    # get training and validation data using folds
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # initialize OneHotEncoder from scikit-learn
    ohe = preprocessing.OneHotEncoder()

    # fit ohe on training + validation features
    full_data = pd.concat([df_train[features], df_valid[features], test[features]], axis=0)
    ohe.fit(full_data[cat_features])

    # transform only categorical features
    x_train_cat = ohe.transform(df_train[cat_features])
    x_valid_cat = ohe.transform(df_valid[cat_features])
    x_test_cat  = ohe.transform(test[cat_features])

    x_train = np.hstack([x_train_cat.toarray(), df_train[config.NUM_COLS].values])
    x_valid = np.hstack([x_valid_cat.toarray(), df_valid[config.NUM_COLS].values])
    x_test  = np.hstack([x_test_cat.toarray(), test[config.NUM_COLS].values])

    y_train = df_train[config.TARGET_COL].values
    y_valid = df_valid[config.TARGET_COL].values
    
    # use ONLY linear model here
    clf = model_dispatcher.models[model]

    # fit the model on training data
    clf.fit(x_train, y_train)

    # create predictions for validation samples
    preds = clf.predict(x_valid)

    # calculate & print accuracy
    accuracy = metrics.accuracy_score(y_valid, preds)
    f1 = metrics.f1_score(y_valid, preds, average='weighted')
    print(f"Fold={fold}, accuracy={accuracy}, f1-score={f1}\n")
    #print(f"Confusion Matrix:\n {metrics.confusion_matrix(y_valid, preds)}")

    # save the model and encoder
    joblib.dump(clf, os.path.join(config.MODEL_OUTPUT, f"baseline_{model}_{fold}.pkl"))
    joblib.dump(ohe, os.path.join(config.MODEL_OUTPUT, f"ohe_{model}.pkl"))
    joblib.dump(x_test, os.path.join(config.MODEL_OUTPUT, f"test_ohe_encoded.pkl"), compress=3)

    return accuracy, f1


def run_all_folds(model):
    accuracies = []
    f1_scores = []
    
    print(f"Model:{model}")
    for fold in range(config.N_FOLDS):
        accuracy, f1 = run(fold=fold, model=model)
        accuracies.append(accuracy)
        f1_scores.append(f1)

    avg_accuracy = np.mean(accuracies)
    avg_f1 = np.mean(f1_scores)

    print(f"Global Average accuracy={avg_accuracy}, Global Average f1-score={avg_f1}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    args = parser.parse_args()

    run_all_folds(model=args.model)