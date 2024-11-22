import pandas as pd


def create_numeric_features_from_categorical(df):
    # How many ids do we have in the dataframe where the value of ord_2 is Boiling Hot ? 
    df["ord_2_cat"] = df.groupby(["ord_2"])["id"].count()

    df.groupby(["ord_1", "ord_2"])["id"].count().reset_index(name="count")

    return df


def combine_categorical_cols(df):
    df["new_feature"] = (df.ord_1.astype(str) + "_" + df.ord_2.astype(str))
    df["new_feature"] = (df.ord_1.astype(str) + "_" + df.ord_2.astype(str)) + df.ord_2.astype(str)