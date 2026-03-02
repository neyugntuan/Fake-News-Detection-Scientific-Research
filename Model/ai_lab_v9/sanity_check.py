# ============================================================
# SANITY CHECK PIPELINE
# Chạy sau khi train xong để kiểm tra chất lượng
#
# Usage:
#    python sanity_check.py
# ============================================================

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle

REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)

DATA_PATH = "data/dataset_fullEDA.csv"

print(" Reports folder:", REPORT_DIR)

# ============================================================
# 1. DUPLICATE CHECK
# ============================================================
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)

    # Tự động nhận diện cột text
    if "text" in df.columns:
        text_col = "text"
    elif "post_message" in df.columns:
        text_col = "post_message"
    else:
        print(f"  Không tìm thấy cột text! Các cột có: {list(df.columns)}")
        text_col = None

    if text_col:
        df = df.dropna(subset=[text_col])
        dup_count = df.duplicated(subset=[text_col]).sum()
        print(f"\n Duplicate samples: {dup_count}")
        with open(f"{REPORT_DIR}/duplicate_check.txt", "w") as f:
            f.write(f"Dataset: {DATA_PATH}\n")
            f.write(f"Text column: {text_col}\n")
            f.write(f"Total samples: {len(df)}\n")
            f.write(f"Duplicate samples: {dup_count}\n")
else:
    print(f"  Dataset not found: {DATA_PATH}")
    df = None
    text_col = None

# ============================================================
# 2. FOLD DISTRIBUTION CHECK
# ============================================================
if df is not None and text_col:
    y = df["label"].values
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    fold_stats = []
    print("\n Fold distribution (Label 0 / Label 1):")
    for i, (_, val_idx) in enumerate(skf.split(np.zeros(len(y)), y), 1):
        dist = np.bincount(y[val_idx])
        print(f"  Fold {i:2d}: {dist}")
        fold_stats.append({"Fold": i, "Class_0": dist[0], "Class_1": dist[1]})
    pd.DataFrame(fold_stats).to_csv(
        f"{REPORT_DIR}/fold_distribution.csv", index=False
    )

# ============================================================
# 3. METRICS SUMMARY
# ============================================================
PRED_PATH = "experiments/dual_results.csv"
if os.path.exists(PRED_PATH):
    results = pd.read_csv(PRED_PATH)
    summary = results.describe()
    summary.to_csv(f"{REPORT_DIR}/metrics_summary.csv")
    print("\n Metrics summary:")
    print(summary[["LSTM_AUC", "BiLSTM_AUC"]].to_string())

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=results[["LSTM_AUC", "BiLSTM_AUC"]],
                palette=["#2E86AB", "#E84855"], ax=ax)
    ax.set_xticklabels(["LSTM", "BiLSTM"])
    ax.set_title("Model Comparison — AUC-ROC (10-Fold)")
    ax.set_ylabel("AUC-ROC")
    plt.savefig(f"{REPORT_DIR}/metrics_boxplot.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {REPORT_DIR}/metrics_boxplot.png")
else:
    print(f"\n  dual_results.csv not found — chạy run.py trước")

# ============================================================
# 4. LEARNING CURVE
# ============================================================
train_loss_path = "reports/train_losses.json"
val_loss_path   = "reports/val_losses.json"

if os.path.exists(train_loss_path) and os.path.exists(val_loss_path):
    train_losses = json.load(open(train_loss_path))
    val_losses   = json.load(open(val_loss_path))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_losses, label="Train Loss", color="#2E86AB")
    ax.plot(val_losses,   label="Val Loss",   color="#E84855")
    ax.legend()
    ax.set_title("Learning Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)
    plt.savefig(f"{REPORT_DIR}/learning_curve.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n  Learning curve saved")
else:
    print("\n  Không tìm thấy loss history")

# ============================================================
# 5. DATA LEAKAGE TEST
# ============================================================
if df is not None and text_col:
    print("\n Data leakage sanity test...")
    X_dummy = df[text_col].values
    y_real  = df["label"].values
    _, y_shuffled = shuffle(X_dummy, y_real, random_state=42)
    match_rate = (y_shuffled == y_real).mean()
    print(f"  Label shuffle match rate: {match_rate:.4f} "
          f"(kỳ vọng ~{(np.bincount(y_real)/len(y_real)).max():.2f} với data imbalanced)")
    with open(f"{REPORT_DIR}/leakage_test.txt", "w") as f:
        f.write(f"Label shuffle match rate: {match_rate:.4f}\n")
        f.write(f"Label distribution: {np.bincount(y_real)}\n")

print(f"\n Sanity check hoàn tất. Reports: {REPORT_DIR}/")
