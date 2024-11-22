import joblib
import os
import pandas as pd
import config


def predict(model):
    test   = pd.read_csv(config.TEST_FILE, index_col=0)
    sample = pd.read_csv(config.SAMPLE_FILE)

    features = [f for f in test.columns if f not in ("Name", "id", config.TARGET_COL, "kfold")]
    for col in features:
        test.loc[:, col] = test[col].astype(str).fillna("NONE")

    predictions = None
    for fold in range(config.N_FOLDS):
        # encoding test data
        ohe = joblib.load(os.path.join(config.MODEL_OUTPUT, f"ohe_{model}_{fold}.pkl"))
        test = ohe.transform(test[features])
        
        clf = joblib.load(os.path.join(config.MODEL_OUTPUT, f"baseline_{model}_{fold}.pkl"))
        preds = clf.predict(test)
        
        if fold == 0:
            predictions = preds
        else:
            predictions += preds
    predictions = predictions / config.N_FOLDS

    sample[config.TARGET_COL] = predictions.astype(int)
    print(sample.head())
    return sample


if __name__ == "__main__":
    #models = ["hist", "cat", "gbm", "lgbm", "xgb"]
    submission = predict(config.MODEL)
    submission.to_csv(f"{config.FILES_OUTPUT}{config.MODEL}_baseline_submission.csv", index=False)
    print("SUBMISSION FILE CREATED SUCCESSFULLY!!!")