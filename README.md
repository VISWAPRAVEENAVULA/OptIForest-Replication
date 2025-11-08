# OptIForest: Replication and Extension Project

This repository contains the **replication and extension** of the paper  
**“OptIForest: Optimal Isolation Forest for Anomaly Detection” (IJCAI 2022)** by Madasamy et al.

This work was completed as part of the **COMP8240 – Applications of Data Science** unit at **Macquarie University**.  
The goal is to replicate the original OptIForest anomaly detection algorithm, evaluate it on the **KDDCup99** dataset,  
and extend it to **new datasets** including **Credit Card Fraud Detection** and a **synthetic anomaly dataset**.

---

## Project Overview

###  Objectives
- **Replication** – Run the original OptIForest code with its original dataset (KDDCup99).
- **Validation** – Compare replicated results with the metrics reported in the IJCAI paper.
- **Extension** – Apply the model to new datasets:
  - Kaggle’s Credit Card Fraud Detection dataset.
  - A generated synthetic dataset to test sensitivity to branching factors.
- **Evaluation** – Assess performance using metrics such as AUC, Precision, Recall, and F1-score.

---

##  Methodology

### 1. **Replication of Original Work**
The baseline replication uses the OptIForest source code provided by the authors.  
The model is executed under the following environment:
- Python 3.12
- scikit-learn 1.6.1
- NumPy 2.0.2
- Pandas 2.2.2

The output metrics (AUC: 0.89, Precision: 0.55, Recall: 0.57, F1-score: 0.56) closely match the results reported in the paper.

### 2. **New Datasets**
- **Credit Card Fraud Detection (Kaggle)**  
  Highly imbalanced dataset (284,807 transactions; 492 frauds).  
  Preprocessing includes scaling of “Amount” and “Time” features and stratified 70/15/15 splits.
- **Synthetic Dataset**  
  Generated using `make_blobs()` to simulate Gaussian clusters with 5% injected outliers.

---

##  Repository Structure

