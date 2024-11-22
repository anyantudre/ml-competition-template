# import pandas and model_selection module of scikit-learn
import pandas as pd
from sklearn import model_selection
import os, config


if __name__ == "__main__":
    df = pd.read_csv(config.ORIG_TRAIN_FILE, index_col=0)
    
    # we create a new column called kfold and fill it with -1
    df["kfold"] = -1

    # the next step is to randomize the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)

    # fetch targets
    y = df[config.TARGET_COL].values

    # initiate the kfold class from model_selection module
    kf = model_selection.StratifiedKFold(n_splits=5)

    # fill the new kfold column
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f

    # save the new csv with kfold column
    df.to_csv(os.path.join(config.DATA_INPUT, "train_folds.csv"), index=False)
    print("CREATED FOLDS SUCCESSFULLY!!!")