# Fraudulent Transaction Detection (CPE 232 Data Models)

XGBoost classifier for detecting fraudulent transactions, built for a Kaggle competition in the
CPE 232 Data Models course (Computer Engineering, KMUTT). Solo entry.

- Competition: https://www.kaggle.com/competitions/fraud-detection-cpe-232-data-models (course-restricted)
- Task: predict `is_fraud` from transaction-level data (types, amounts, sender/recipient balances, flagged attempts)
- Metric: **F1 Macro** (the data is heavily imbalanced, about 0.12% fraud)

## Results

| | Value |
|---|---|
| Public leaderboard rank | **76 / 107** |
| Private leaderboard rank | **72 / 107** |
| Hold-out fraud-class precision / recall / F1 (80/20 split) | 0.96 / 0.95 / 0.95 |
| Hold-out confusion matrix | 1,247 frauds caught, 72 missed, 46 false alarms |

The hold-out score is much higher than the leaderboard position suggests. See *Limitations* below
for the likely reasons.

## Approach
1. **Cleaning:** transaction type mapped to integers with an `Unknown` class for missing values;
   median imputation for missing amounts and balances; account ids factorized.
2. **Features:** balance-consistency errors on source and destination accounts
   (`balance - amount - new balance`), amount-to-balance ratio, merchant-destination flag,
   day and hour derived from `time_ind`, account frequency counts.
3. **Model:** `XGBClassifier` with default hyperparameters, 80/20 random hold-out split.
4. **Prediction:** the fitted model scores `test.csv`; predictions are written to `answer.csv`.

## Repository layout
```
notebooks/fraud_detection_xgboost.ipynb   main pipeline (clean -> features -> train -> predict)
scripts/make_sample_data.py               generates the synthetic sample files
data/README.md                            data description and download steps
data/sample/                              SYNTHETIC train/test files with the same schema
requirements.txt
```

## How to run
```bash
pip install -r requirements.txt

# with the real competition data (not included)
DATA_DIR=path/to/folder_with_train_and_test jupyter notebook notebooks/fraud_detection_xgboost.ipynb

# or the synthetic sample, just to check that the pipeline runs
DATA_DIR=data/sample jupyter notebook notebooks/fraud_detection_xgboost.ipynb
```
On Windows PowerShell set the variable first: `$env:DATA_DIR="data/sample"`.

## Data
The dataset is restricted to enrolled students and is not redistributed here. See `data/README.md`
for the schema and how to obtain it.

## Limitations and what I would change
This was my first competition, and I developed it by editing one notebook repeatedly, so the repo
holds only the final version. Looking back, these are the weaknesses:
- **Account ids:** `src_acc` / `dst_acc` are factorized separately for train and test, so the same
  account gets a different number in each file. The account frequency counts are computed per file too.
  These features probably do not transfer from validation to the test set.
- **`id` as a feature:** test ids fall outside the training range, so it cannot generalise.
- **Validation:** a single random split, not stratified, and no cross-validation. Hold-out scores
  were optimistic compared with the leaderboard.
- **No tuning or threshold search:** default hyperparameters, no class-imbalance handling
  (`scale_pos_weight`) and no decision-threshold tuning for F1.
- **Process:** I did not track experiments separately. Next time I would keep each experiment in git
  history and record scores as I go.
