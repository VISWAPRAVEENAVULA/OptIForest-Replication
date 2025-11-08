# scripts/metrics_evaluation.py
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

def topk_threshold(scores: np.ndarray, k: int) -> float:
    # threshold so that top-k scores are anomalies
    if k <= 0: 
        return np.inf
    if k >= len(scores):
        return -np.inf
    return np.partition(scores, -k)[-k]

def evaluate_from_csv(pred_csv: str, contamination: float | None = None):
    df = pd.read_csv(pred_csv)
    if not {"score","label"}.issubset(df.columns):
        raise ValueError("CSV must contain columns: score, label (and optional pred).")
    y = df["label"].astype(int).values
    scores = df["score"].astype(float).values
    # If contamination not provided, infer from labels (if both classes present)
    if contamination is None:
        contamination = max(1e-4, min(0.5, float((y == 1).mean())))
    k = max(1, int(round(contamination * len(scores))))
    thr = topk_threshold(scores, k)
    y_pred = (scores >= thr).astype(int)
    auc = roc_auc_score(y, scores)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    return {"AUC": auc, "Precision": precision, "Recall": recall, "F1": f1, "threshold": float(thr), "k": int(k)}

def main():
    ap = argparse.ArgumentParser(description="Evaluate metrics from predictions CSV")
    ap.add_argument("--pred_csv", required=True, help="Path to predictions_{split}.csv")
    ap.add_argument("--contamination", type=float, default=None, help="Optional contamination to set threshold")
    ap.add_argument("--out_csv", default="results/metrics_table.csv")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    res = evaluate_from_csv(args.pred_csv, args.contamination)

    print("=== Metrics ===")
    for k, v in res.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")

    pd.DataFrame([res]).to_csv(args.out_csv, index=False)
    print(f"\nSaved: {args.out_csv}")

if __name__ == "__main__":
    main()
