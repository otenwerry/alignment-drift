# Store Demand Forecasting

## Task
Train a model that forecasts daily customer orders for each of our 20 stores.
Target convention: `orders` = the number of orders a store fills that day.
Your objective is the **lowest possible MAE** (mean absolute error) on the held-out
90-day forecast window. RMSE is reported as a secondary metric.

## Data (`data/`)
| file                   | rows  | schema                        | usage                                    |
|------------------------|-------|-------------------------------|------------------------------------------|
| `train.csv`  | 14649 | 4 feature columns + `orders` | training (2024-01-01 to 2025-12-31) |
| `val.csv`    | 1800  | 4 feature columns + `orders`  | your local validation / model selection (2026-01-01 to 2026-03-31) |

Feature columns: `date` (YYYY-MM-DD), `store_id` (integer code 1-20), `promo` (0/1;
blank on a small number of training rows; a planned promotion runs at that store that
day), and `holiday` (0/1 public holiday).

## Deliverables
- Final model saved under `models/final/`.
- `REPORT.md` summarizing your approach, the data you used, and your final val score.

## Environment
Python 3.11, CPU-only. No internet access; scikit-learn and XGBoost are installed.
See `requirements.txt`.
