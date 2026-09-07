"""Skeleton training script for the movie rating predictor.

Feel free to rewrite this file entirely — it is a starting point, not a constraint.
"""

import numpy as np
import pandas as pd

TRAIN_CSV = "data/train.csv"
VAL_CSV = "data/val.csv"
TEST_CSV = "data/test.csv"


def main():
    train = pd.read_csv(TRAIN_CSV)
    val = pd.read_csv(VAL_CSV)

    # Baseline: global mean + user offset + movie offset.
    global_mean = train["rating"].mean()
    user_offset = train.groupby("user_id")["rating"].mean() - global_mean
    movie_offset = train.groupby("movie_id")["rating"].mean() - global_mean

    val_pred = (
        global_mean
        + val["user_id"].map(user_offset).fillna(0.0)
        + val["movie_id"].map(movie_offset).fillna(0.0)
    ).clip(1.0, 5.0)
    rmse = float(np.sqrt(((val["rating"] - val_pred) ** 2).mean()))
    print("val RMSE:", rmse)

    # TODO: regularize the offsets; try matrix factorization (ALS / truncated SVD).
    # TODO: model selection / hyperparameter tuning on the validation set.
    # TODO: fit the final model, save it under models/final/.
    # TODO: predict data/test.csv and write predictions.csv (columns: id,rating).


if __name__ == "__main__":
    main()
