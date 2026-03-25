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

    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42) #chia dataset thành 10 fold nhưng giữ nguyên tỉ lệ label
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #dùng GPU nếu có

    results = [] #Tạo list rỗng để lưu kết quả mỗi fold

    print("\n Chay 10-fold thuc nghiem...\n")

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y), 1): #kf.split() sẽ lần lượt trả về:(tr_idx, val_idx), 1: đánh số từ 1
                                                                #vd Ví dụ dataset có 10 mẫu, chia thành 5 fold
                                                                # Fold	    Train index	    Validation index
                                                                # 1	    [2,3,4,5,6,7,8,9]	    [0,1]
                                                                # 2	    [0,1,4,5,6,7,8,9]	    [2,3]
                                                                # 3	    [0,1,2,3,6,7,8,9]	    [4,5]
                                                                # 4	    [0,1,2,3,4,5,8,9]	    [6,7]
                                                                # 5	    [0,1,2,3,4,5,6,7]	    [8,9]

        print(f"\n===== Fold {fold} =====")

        X_tr, y_tr = X[tr_idx], y[tr_idx]       #X lúc này là ma trận numpy array X = [
                                                                                    # [1,2,3,4,0],
                                                                                    # [3,5,6,7,0],
                                                                                    # [8,9,10,0,0],
                                                                                    # ...,
                                                                                    # ...
                                                                                    # ]
                                                #Khi chia fold X_tr = X[tr_idx], X_val = X[val_idx]
                                                #Nó không thay đổi nội dung, chỉ lấy một phần các dòng.
                                                #vd X ban đầu
                                                # index
                                                # 0 [1,2,3,4,0] = X[0]
                                                # 1 [3,5,6,7,0] = X[1]
                                                # 2 [8,9,10,0,0] = X[2]
                                                # 3 [2,4,6,0,0] = X[3]
                                                # 4 [1,5,7,8,0] = X[4]
                                                #Nếu tr_idx  = [2,3,4], val_idx = [0,1]
                                                #Train
                                                # X_tr = [
                                                # [8,9,10,0,0],
                                                # [2,4,6,0,0],
                                                # [1,5,7,8,0]
                                                # ]
                                                #Validation
                                                # X_val = [
                                                # [1,2,3,4,0],
                                                # [3,5,6,7,0]
                                                # ]
        X_val, y_val = X[val_idx], y[val_idx]

        # ── LSTM ────────────────────────────────────────────────
        lstm = LSTM(vocab_size)
        train_model(  #không cần hứng kết quả, hàm sẽ huấn luyện trực tiếp object lstm đó.
            lstm, X_tr, y_tr,
            vocab=None,
            save_name=f"lstm_fold{fold}.pt",
            pos_weight=pos_weight,
        )

        # ── BiLSTM ──────────────────────────────────────────────
        bilstm = BiLSTM(vocab_size)
        train_model(  #không cần hứng kết quả, hàm sẽ huấn luyện trực tiếp object bilstm đó.
            bilstm, X_tr, y_tr,
            vocab=None,
            save_name=f"bilstm_fold{fold}.pt",
            pos_weight=pos_weight,
        )

        # ── Đánh giá trên KFOLD VAL SET ────────────────────────
        auc_lstm   = _eval_auc(lstm,   X_val, y_val, device) #tính AUC của model LSTM
        auc_bilstm = _eval_auc(bilstm, X_val, y_val, device) #tính AUC của model BiLSTM

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
    model.eval() #bật chế độ validation, nếu không gọi thì một số layer sẽ hoạt động sai chế độ: Dropout, BatchNorm
    all_probs = [] #List lưu xác suất dự đoán của từng sample: all_probs = [0.23,0.88,0.41,...]
    with torch.no_grad(): #không cần gradient, giảm RAM
        for i in range(0, len(X_val), batch_size): #range(start, stop, step)
            xb = torch.tensor(X_val[i:i+batch_size]).long().to(device) #lưu mẫu từ i tới i + 256 vd mẫu 0 tới 256
                                                                        #Vì layer embedding trong LSTM cần index kiểu integer nên ta trả về long
                                                                        #chuyển sang GPU tensor(..., device='cuda:0')
            logits = model(xb)  #logits là raw output của model
            probs  = torch.sigmoid(logits).cpu().numpy() #trả logits thành xác suất 0→1, về lại CPU dùng numpy, tensor → numpy array
            all_probs.extend(probs) #Lưu xác xuất
    return roc_auc_score(y_val, all_probs) #y_val: nhãn thật là 1 numpy array
                                           #all_probs: dự đoán cũng là 1 numpy array

