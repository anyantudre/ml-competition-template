import joblib
import os
import pandas as pd
import config
import utils.preprocessing_utils as preprocessing_utils

model='xgb'

def predict(model):
    test   = pd.read_csv(config.TEST_FILE, index_col=0)
    #print(test.head())
    sample = pd.read_csv(config.SAMPLE_FILE)

    # simple label encoding
    test = preprocessing_utils.label_encoding(test)
    test = test.drop(config.COLS_TO_DROP, axis=1).values

    predictions = None
    for fold in range(config.N_FOLDS):
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
    submission = predict(model)
    submission.to_csv(f"{config.FILES_OUTPUT}xgb_baseline_submission.csv", index=False)
    print("SUBMISSION FILE CREATED SUCCESSFULLY!!!")