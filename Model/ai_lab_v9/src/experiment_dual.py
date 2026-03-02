import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import torch

from src.models import LSTM, BiLSTM
from src.trainer import train_model
from src.metrics_extension import compute_metrics, save_metrics_log, plot_metrics
from src.final_model import train_final_model
from src.plots_dual import plot_dual
from src.stats import run_stats


def run_dual_experiment(X, y, vocab_size, vocab_dict, pos_weight=None):
    """
    Thực nghiệm 10-fold so sánh LSTM và BiLSTM.

    Args:
        X          : numpy array (N, max_len)
        y          : numpy array (N,)
        vocab_size : int
        vocab_dict : dict {word: index}
        pos_weight : float — class weight cho BCEWithLogitsLoss (dữ liệu mất cân bằng)
    """

    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = []

    print("\n Chay 10-fold thuc nghiem...\n")

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y), 1):

        print(f"\n===== Fold {fold} =====")

        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # ── LSTM ────────────────────────────────────────────────
        lstm = LSTM(vocab_size)
        train_model(
            lstm, X_tr, y_tr,
            vocab=None,
            save_name=f"lstm_fold{fold}.pt",
            pos_weight=pos_weight,
        )

        # ── BiLSTM ──────────────────────────────────────────────
        bilstm = BiLSTM(vocab_size)
        train_model(
            bilstm, X_tr, y_tr,
            vocab=None,
            save_name=f"bilstm_fold{fold}.pt",
            pos_weight=pos_weight,
        )

        # ── Đánh giá trên KFOLD VAL SET ────────────────────────
        auc_lstm   = _eval_auc(lstm,   X_val, y_val, device)
        auc_bilstm = _eval_auc(bilstm, X_val, y_val, device)

        acc_l, pre_l, rec_l, f1_l = compute_metrics(lstm,   X_val, y_val, device)
        acc_b, pre_b, rec_b, f1_b = compute_metrics(bilstm, X_val, y_val, device)

        print(f"\nFold {fold} Metrics:")
        print(f"  LSTM   → AUC:{auc_lstm:.4f}  Acc:{acc_l:.4f}  "
              f"Prec:{pre_l:.4f}  Rec:{rec_l:.4f}  F1:{f1_l:.4f}")
        print(f"  BiLSTM → AUC:{auc_bilstm:.4f}  Acc:{acc_b:.4f}  "
              f"Prec:{pre_b:.4f}  Rec:{rec_b:.4f}  F1:{f1_b:.4f}")

        results.append({
            "Fold"            : fold,
            "LSTM_AUC"        : auc_lstm,
            "BiLSTM_AUC"      : auc_bilstm,
            "LSTM_Accuracy"   : acc_l,
            "LSTM_Precision"  : pre_l,
            "LSTM_Recall"     : rec_l,
            "LSTM_F1"         : f1_l,
            "BiLSTM_Accuracy" : acc_b,
            "BiLSTM_Precision": pre_b,
            "BiLSTM_Recall"   : rec_b,
            "BiLSTM_F1"       : f1_b,
        })

    # ── Tổng hợp kết quả ────────────────────────────────────────
    os.makedirs("experiments", exist_ok=True)

    df_results = pd.DataFrame(results)
    df_results["AUC_Difference"] = (
        df_results["BiLSTM_AUC"] - df_results["LSTM_AUC"]
    )

    df_auc = df_results[["Fold", "LSTM_AUC", "BiLSTM_AUC", "AUC_Difference"]]
    df_auc.to_csv("experiments/dual_results.csv", index=False)

    df_metrics = save_metrics_log(results)
    plot_metrics(df_metrics)

    print("\n========== Kết quả 10-Fold ==========")
    print(df_results[["Fold", "LSTM_AUC", "BiLSTM_AUC"]].to_string(index=False))
    print(f"\n  Mean AUC  — LSTM  : {df_results['LSTM_AUC'].mean():.4f}"
          f" ± {df_results['LSTM_AUC'].std():.4f}")
    print(f"  Mean AUC  — BiLSTM: {df_results['BiLSTM_AUC'].mean():.4f}"
          f" ± {df_results['BiLSTM_AUC'].std():.4f}")
    print(f"  Mean F1   — LSTM  : {df_results['LSTM_F1'].mean():.4f}")
    print(f"  Mean F1   — BiLSTM: {df_results['BiLSTM_F1'].mean():.4f}")

    plot_dual()
    run_stats(df_results)

    # Train final model — truyền pos_weight
    train_final_model(X, y, vocab_size, vocab_dict, df_results, pos_weight)

    return df_results


def _eval_auc(model, X_val, y_val, device, batch_size=256):
    """Tính AUC trên val set theo batch."""
    model.eval()
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(X_val), batch_size):
            xb = torch.tensor(X_val[i:i+batch_size]).long().to(device)
            logits = model(xb)
            probs  = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
    return roc_auc_score(y_val, all_probs)
