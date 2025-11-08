# 📂 Data

This folder stores datasets used for replicating and extending the **OptIForest** experiment.

## 1. Credit Card Fraud Detection Dataset (External)
- Source: [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Description: Transactions made by European cardholders in September 2013.
- Size: 284,807 records, 30 features (28 PCA + Time + Amount)
- Labels: 0 = normal, 1 = fraud
- Usage: Download `creditcard.csv` and place it in this folder before running:
  ```bash
  python scripts/run_optiforest.py --kind creditcard

