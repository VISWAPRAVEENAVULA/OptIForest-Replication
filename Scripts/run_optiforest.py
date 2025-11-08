# scripts/run_optiforest.py
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, roc_curve

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def load_dataset(kind: str, csv_path: str | None) -> tuple[pd.DataFrame, str]:
    if kind == "creditcard":
        path = csv_path or "data/creditcard.csv"
        df = pd.read_csv(path)
        # scale Time/Amount if present
        if {"Time","Amount"}.issubset(df.columns):
            scaler = StandardScaler()
            df[["scaled_time","scaled_amount"]] = scaler.fit_transform(df[["Time","Amount"]])
            df = df.drop(columns=["Time","Amount"])
        label_col = "Class"
    elif kind == "synthetic":
        path = csv_path or "data/synthetic_data.csv"
        df = pd.read_csv(path)
        label_col = "Class" if "Class" in df.columns else "label"
    elif kind == "csv":
        assert csv_path, "--csv_path is required when kind=csv"
        df = pd.read_csv(csv_path)
        # best guess at label column
        label_col = "Class" if "Class" in df.columns else ("label" if "label" in df.columns else None)
        if label_col is None:
            raise ValueError("Could not infer label column. Expected 'Class' or 'label'.")
    else:
        raise ValueError("kind must be one of: creditcard | synthetic | csv")
    return df, label_col

def split_xy(df: pd.DataFrame, label_col: str):
    y = df[label_col].astype(int).values
    X = df.drop(columns=[label_col]).copy()
    # scale all features to [0,1] for stability
    X = pd.DataFrame(MinMaxScaler().fit_transform(X), columns=X.columns)
    return X, y

def stratified_split(X, y, seed=42):
    # 70/15/15 using two splits
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=seed
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def estimate_contamination(y_train: np.ndarray, fallback=0.05) -> float:
    # if labels include anomalies (1 = anomaly), use their proportion; else fallback
    if len(np.unique(y_train)) > 1:
        return max(1e-4, min(0.5, float((y_train == 1).mean())))
    return fallback

def train_iforest(X_train, contamination, seed=42):
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train)
    return model

def evaluate(model, X, y, out_dir: str, split_name: str):
    # Higher score = more anomalous
    scores = -model.decision_function(X)
    preds = (model.predict(X) == -1).astype(int)

    auc = roc_auc_score(y, scores)
    precision = precision_score(y, preds, zero_division=0)
    recall = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)

    # Save predictions
    pred_df = pd.DataFrame({"score": scores, "pred": preds, "label": y})
    pred_csv = os.path.join(out_dir, f"predictions_{split_name}.csv")
    pred_df.to_csv(pred_csv, index=False)

    # ROC curve
    fpr, tpr, _ = roc_curve(y, scores)
    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr)
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC — {split_name} (AUC={auc:.2f})")
    roc_path = os.path.join(out_dir, f"roc_{split_name}.png")
    plt.tight_layout(); plt.savefig(roc_path, dpi=200); plt.close()

    return {"AUC": auc, "Precision": precision, "Recall": recall, "F1": f1}, pred_csv, roc_path

def save_class_distribution(y, out_dir: str, name: str):
    vals, counts = np.unique(y, return_counts=True)
    plt.figure(figsize=(4.5,3.5))
    plt.bar(vals, counts)
    plt.xticks([0,1], ["Normal (0)", "Anomaly (1)"])
    plt.ylabel("Count")
    plt.title(f"Class Distribution — {name}")
    p = os.path.join(out_dir, f"class_distribution_{name}.png")
    plt.tight_layout(); plt.savefig(p, dpi=200); plt.close()
    return p

def main():
    parser = argparse.ArgumentParser(description="Run IsolationForest baseline.")
    parser.add_argument("--kind", choices=["creditcard","synthetic","csv"], required=True,
                        help="Dataset type: creditcard | synthetic | csv")
    parser.add_argument("--csv_path", type=str, default=None,
                        help="Path to CSV when --kind=csv (or to override default path).")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory to save outputs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--contamination", type=float, default=None,
                        help="Override contamination; if None, estimated from labels.")
    args = parser.parse_args()

    ensure_dir(args.results_dir)

    df, label_col = load_dataset(args.kind, args.csv_path)
    X, y = split_xy(df, label_col)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = stratified_split(X, y, seed=args.seed)

    # save class distribution plots
    save_class_distribution(y, args.results_dir, "full")
    save_class_distribution(y_train, args.results_dir, "train")
    save_class_distribution(y_test, args.results_dir, "test")

    contam = args.contamination if args.contamination is not None else estimate_contamination(y_train)
    model = train_iforest(X_train, contamination=contam, seed=args.seed)

    metrics_val, pred_csv_val, roc_val = evaluate(model, X_val, y_val, args.results_dir, "val")
    metrics_test, pred_csv_test, roc_test = evaluate(model, X_test, y_test, args.results_dir, "test")

    # save metrics summary
    txt_path = os.path.join(args.results_dir, "replication_metrics.txt")
    with open(txt_path, "w") as f:
        f.write("=== IsolationForest Replication Metrics ===\n")
        f.write(f"Dataset: {args.kind}\n")
        f.write(f"Contamination used: {contam:.5f}\n\n")
        f.write(f"[VAL]  AUC={metrics_val['AUC']:.4f}  Precision={metrics_val['Precision']:.4f}  "
                f"Recall={metrics_val['Recall']:.4f}  F1={metrics_val['F1']:.4f}\n")
        f.write(f"[TEST] AUC={metrics_test['AUC']:.4f}  Precision={metrics_test['Precision']:.4f}  "
                f"Recall={metrics_test['Recall']:.4f}  F1={metrics_test['F1']:.4f}\n")
        f.write(f"\nArtifacts:\n  {pred_csv_val}\n  {pred_csv_test}\n  {roc_val}\n  {roc_test}\n")

    print(open(txt_path).read())

if __name__ == "__main__":
    main()
