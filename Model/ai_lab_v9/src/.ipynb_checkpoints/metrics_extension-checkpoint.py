# ============================================================
# Metrics Extension Module
# Accuracy / Precision / Recall / F1 — Batched inference
# ============================================================

import os
import pandas as pd
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

os.makedirs("experiments", exist_ok=True)
os.makedirs("experiments/figures", exist_ok=True)


# ============================================================
# TÍNH METRICS — Batched (tránh OOM trên GPU)
# ============================================================

def compute_metrics(model, X_val, y_val, device, batch_size=256):
    """
    Tính Accuracy, Precision, Recall, F1 trên val set.

    Dùng batched inference thay vì đẩy toàn bộ X_val lên GPU
    một lần (tránh OOM với dataset lớn).
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for i in range(0, len(X_val), batch_size):
            xb     = torch.tensor(X_val[i:i+batch_size]).long().to(device)
            logits = model(xb)
            probs  = torch.sigmoid(logits).cpu().numpy()
            preds  = (probs >= 0.5).astype(int)
            all_preds.extend(preds)

    all_preds = np.array(all_preds)

    acc  = accuracy_score(y_val, all_preds)
    prec = precision_score(y_val, all_preds, zero_division=0)
    rec  = recall_score(y_val, all_preds, zero_division=0)
    f1   = f1_score(y_val, all_preds, zero_division=0)

    return acc, prec, rec, f1


# ============================================================
# LƯU LOG CSV
# ============================================================

def save_metrics_log(results):
    df = pd.DataFrame(results)
    csv_path = "experiments/metrics_log.csv"
    df.to_csv(csv_path, index=False)
    print("\n Metrics log saved:", csv_path)
    return df


# ============================================================
# VẼ BIỂU ĐỒ BOXPLOT CHO TỪNG METRIC
# ============================================================

def plot_metrics(df):
    metrics = ["Accuracy", "Precision", "Recall", "F1"]

    for m in metrics:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(
            data=df[[f"LSTM_{m}", f"BiLSTM_{m}"]],
            palette=["#2E86AB", "#E84855"],
            ax=ax,
        )
        ax.set_title(f"So sánh {m} — LSTM vs BiLSTM (10-Fold)")
        ax.set_ylabel(m)
        ax.set_xticklabels(["LSTM", "BiLSTM"])
        ax.grid(axis="y", alpha=0.3)

        fig_path = f"experiments/figures/{m}_boxplot.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fig_path}")
