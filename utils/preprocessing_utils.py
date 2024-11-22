import pandas as pd


def label_encoding(df):
    df["Gender"] = df["Gender"].map({"Female":0, "Male":1})
    df["Working Professional or Student"] = df["Working Professional or Student"].map({"Working Professional":0, "Student":1})
    df["Have you ever had suicidal thoughts ?"] = df["Have you ever had suicidal thoughts ?"].map({"No":0, "Yes":1})
    df["Family History of Mental Illness"] = df["Family History of Mental Illness"].map({"No":0, "Yes":1})

    return df