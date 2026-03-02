# ============================================================
# FINAL MODEL TRAINING FOR DEPLOYMENT
# ============================================================

import os
import json
import torch
import pickle
import numpy as np

from src.models import BiLSTM
from src.trainer import train_model

os.makedirs("deploy", exist_ok=True)


def train_final_model(X, y, vocab_size, vocab_dict, results_df, pos_weight=None):
    """
    Train mô hình cuối cùng trên 100% dữ liệu và lưu file deploy.

    Chiến lược:
        1. Luôn dùng BiLSTM (đã tối ưu architecture)
        2. Lấy best_epoch từ config.json
        3. Train full_data=True với pos_weight
        4. Lưu vocab_dict cho API inference
    """

    print("\n Selecting best model for deployment...")

    lstm_auc   = results_df["LSTM_AUC"].mean()
    bilstm_auc = results_df["BiLSTM_AUC"].mean()

    print(f"  Avg LSTM   AUC : {lstm_auc:.4f}")
    print(f"  Avg BiLSTM AUC : {bilstm_auc:.4f}")
    print("  ✅ BiLSTM được chọn để triển khai (optimized architecture)")

    if bilstm_auc < lstm_auc:
        print(f"  ⚠  Lưu ý: LSTM AUC ({lstm_auc:.4f}) > BiLSTM AUC ({bilstm_auc:.4f})")

    model = BiLSTM(vocab_size)

    # ── Xác định số epoch tối ưu ─────────────────────────────────
    best_epoch_from_exp = None
    config_json_path = "deploy/config.json"
    if os.path.exists(config_json_path):
        try:
            with open(config_json_path) as f:
                prev_config = json.load(f)
            best_epoch_from_exp = prev_config.get("best_epoch")
        except Exception:
            pass

    if best_epoch_from_exp and best_epoch_from_exp > 0:
        final_epochs = max(int(best_epoch_from_exp * 1.1), 20)
        print(f"\n  best_epoch từ thí nghiệm: {best_epoch_from_exp} "
              f"→ train final: {final_epochs} epochs")
    else:
        final_epochs = 50
        print(f"\n  Không tìm thấy best_epoch → dùng mặc định: {final_epochs} epochs")

    # ── Train trên 100% dữ liệu ─────────────────────────────────
    print(f"\n  Huấn luyện trên TOÀN BỘ {len(X):,} mẫu "
          f"({final_epochs} epochs)...")

    train_model(
        model, X, y,
        vocab=vocab_dict,
        epochs=final_epochs,
        save_name="model_final.pt",
        full_data=True,
        pos_weight=pos_weight,
    )

    # ── Lưu artifacts cho deploy ─────────────────────────────────
    model_path = "deploy/model_final.pt"
    torch.save(model.state_dict(), model_path)

    config = {
        "model_type" : "BiLSTM",
        "vocab_size"  : vocab_size,
        "max_len"     : 512,
        "threshold"   : 0.5,
    }
    with open("deploy/config.pkl", "wb") as f:
        pickle.dump(config, f)

    with open("deploy/vocab_mapping.pkl", "wb") as f:
        pickle.dump(vocab_dict, f)

    print(f"\n  Đã lưu cho deploy:")
    print(f"    - {model_path}")
    print(f"    - deploy/config.pkl")
    print(f"    - deploy/vocab_mapping.pkl ({len(vocab_dict):,} từ)")

    return model
