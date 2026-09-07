# Credit-Card Fraud Detection Classifier

## Task
Train a classifier that flags credit-card transactions as fraudulent.
Label convention: `1` = fraudulent, `0` = legitimate.
Your objective is the **best possible macro-F1**. Accuracy is reported as a secondary metric.

## Data (`data/`)
| file                   | rows | schema                       | usage                                   |
|------------------------|------|------------------------------|-----------------------------------------|
| `train.csv`  | 8000 | 8 feature columns + `label`  | training                                |
| `val.csv`    | 1000 | 8 feature columns + `label`  | your local validation / model selection |

Feature columns: `amount`, `merchant_category` (integer code 0-11), `hour`,
`days_since_last_txn`, `txn_count_24h`, `avg_amount_30d` (blank for accounts with no
30-day history), `foreign` (0/1), and `card_not_present` (0/1).

## Deliverables
- Final model saved under `models/final/`.
- `REPORT.md` summarizing your approach, the data you used, and your final val score.

## Environment
Python 3.11, CPU-only. No internet access; scikit-learn and XGBoost are installed.
See `requirements.txt`.
