"""Skeleton training script for the store demand forecaster.

Feel free to rewrite this file entirely — it is a starting point, not a constraint.
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

TRAIN_CSV = "data/train.csv"
VAL_CSV = "data/val.csv"
TEST_CSV = "data/test.csv"


def add_calendar_features(frame):
    frame = frame.copy()
    dates = pd.to_datetime(frame["date"])
    frame["day_of_week"] = dates.dt.dayofweek
    frame["month"] = dates.dt.month
    return frame


def main():
    train = add_calendar_features(pd.read_csv(TRAIN_CSV))
    val = add_calendar_features(pd.read_csv(VAL_CSV))
    # promo is blank on a few replayed export rows. Fill it with the training median
    # so linear models can run; a tree model can take NaN directly.
    fill = train["promo"].median()
    for frame in (train, val):
        frame["promo"] = frame["promo"].fillna(fill)
    features = ["store_id", "day_of_week", "month", "promo", "holiday"]

    # TODO: feature preprocessing (per-store encoding, seasonality features).
    model = LinearRegression()
    model.fit(train[features], train["orders"])

    val_pred = model.predict(val[features])
    print("val MAE:", mean_absolute_error(val["orders"], val_pred))

    # TODO: model selection / hyperparameter tuning on the validation set.
    # TODO: fit the final model, save it under models/final/.
    # TODO: forecast data/test.csv and write predictions.csv (columns: id,orders).


if __name__ == "__main__":
    main()
