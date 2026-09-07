"""Skeleton training script for the fraud-detection classifier.

Feel free to rewrite this file entirely — it is a starting point, not a constraint.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

TRAIN_CSV = "data/train.csv"
VAL_CSV = "data/val.csv"


def load_data():
    train = pd.read_csv(TRAIN_CSV)
    val = pd.read_csv(VAL_CSV)
    return train, val


def main():
    train, val = load_data()
    features = [c for c in train.columns if c != "label"]

    # avg_amount_30d is blank for accounts with no 30-day history. Fill it with the
    # training median so linear models can run; a tree model can take NaN directly.
    fill = train["avg_amount_30d"].median()
    for frame in (train, val):
        frame["avg_amount_30d"] = frame["avg_amount_30d"].fillna(fill)

    # TODO: feature preprocessing (scaling, categorical encoding, engineered ratios).
    model = LogisticRegression(max_iter=1000)
    model.fit(train[features], train["label"])

    val_pred = model.predict(val[features])
    print("val macro-F1:", f1_score(val["label"], val_pred, average="macro"))

    # TODO: model selection / hyperparameter tuning on the validation set.
    # TODO: fit the final model, save it under models/final/.


if __name__ == "__main__":
    main()
