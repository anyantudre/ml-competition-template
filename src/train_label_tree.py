import joblib
import os
import argparse
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing

import config
import model_dispatcher


def run(fold, model):
    # read the training data with folds
    df = pd.read_csv(config.TRAINING_FILE)
    features = [f for f in df.columns if f not in ("Name", "id", config.TARGET_COL, "kfold")]

    # fill all NaN values in numerical and categorical cols
    for col in config.NUM_COLS:
        df.loc[:, col] = df[col].fillna(df[col].mean())

    for col in features:
        if col not in config.NUM_COLS: # do not encode the numerical columns
            df.loc[:, col] = df[col].astype(str).fillna("NONE")

    print(df.columns)

    # get training and validation data using folds
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    print(df_train.head())
    print(df_valid.head())

    # initialize OneHotEncoder from scikit-learn
    ohe = preprocessing.OneHotEncoder()

    # fit ohe on training + validation features
    full_data = pd.concat([df_train[features], df_valid[features]], axis=0)
    ohe.fit(full_data[features])

    # transform training and validation data
    x_train = ohe.transform(df_train[features])
    x_valid = ohe.transform(df_valid[features])
    print(x_valid.head())

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
    print(f"Fold={fold}, Model={model}, Accuracy={accuracy}, F1-Score={f1}\n")
    #print(f"Confusion Matrix:\n {metrics.confusion_matrix(y_valid, preds)}")

    # save the model and encoder
    joblib.dump(clf, os.path.join(config.MODEL_OUTPUT, f"baseline_{model}_{fold}.pkl"))
    joblib.dump(ohe, os.path.join(config.MODEL_OUTPUT, f"ohe_{model}_{fold}.pkl"))


if __name__ == "__main__":
    # initialize ArgumentParser class of argparse
    parser = argparse.ArgumentParser()

    # add the different arguments you need and their types
    parser.add_argument(
        "--fold",
        type=int
    )
    parser.add_argument(
        "--model",
        type=str
    )
    # read the arguments from the command line
    args = parser.parse_args()

    # run the fold specified by command line arguments
    run(fold=args.fold, model=args.model)