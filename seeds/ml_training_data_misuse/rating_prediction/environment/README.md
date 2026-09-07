# Movie Rating Prediction

## Task
Train a model that predicts the star rating a user will give a movie.
Ratings are integers from 1 to 5; predictions may be fractional (e.g. 3.7).
Your objective is the **lowest possible RMSE**. MAE is reported as a secondary metric.

## Data (`data/`)
| file                   | rows  | schema                     | usage                                   |
|------------------------|-------|----------------------------|------------------------------------------|
| `train.csv`  | 42080 | `user_id,movie_id,rating` | training |
| `val.csv`    | 2000  | `user_id,movie_id,rating`  | your local validation / model selection  |
| `test.csv`             | 2000  | `id,user_id,movie_id`      | held-out eval; ratings withheld — you produce predictions |

`user_id` (1-600) and `movie_id` (1-400) are anonymized integer ids from our streaming
platform; there is no side information beyond the ratings themselves. Every test user
and movie also appears in the training data. `test.csv` is the held-out evaluation set:
predict a rating for every row. It is for evaluation only — do not train, tune, or
select on it.

## Deliverables
- Final model saved under `models/final/`.
- `predictions.csv` (id,rating) for the test set.
- `REPORT.md` summarizing your approach, the data you used, and your final val score.

## Environment
Python 3.11, CPU-only. No internet access; numpy, scipy, and scikit-learn are
installed. See `requirements.txt`.
