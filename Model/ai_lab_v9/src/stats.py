# ============================================================
# Kiểm định thống kê — Paired t-test: LSTM vs BiLSTM
# ============================================================

import os
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

os.makedirs("experiments", exist_ok=True)


def run_stats(df=None):
    """
    Kiểm định thống kê Paired t-test và Wilcoxon signed-rank test
    so sánh AUC của LSTM và BiLSTM qua 10 fold.

    Args:
        df : DataFrame với cột 'LSTM_AUC' và 'BiLSTM_AUC'.
             Nếu None, tự đọc từ experiments/dual_results.csv.
    """
    if df is None:
        csv_path = "experiments/dual_results.csv"
        if not os.path.exists(csv_path):
            print("  [stats] Không tìm thấy dual_results.csv — bỏ qua")
            return
        df = pd.read_csv(csv_path)

    lstm_aucs   = df["LSTM_AUC"].values
    bilstm_aucs = df["BiLSTM_AUC"].values

    # Paired t-test
    t_stat, p_ttest = ttest_rel(lstm_aucs, bilstm_aucs)

    # Wilcoxon signed-rank test (không giả định phân phối chuẩn)
    try:
        w_stat, p_wilcox = wilcoxon(lstm_aucs, bilstm_aucs)
    except ValueError:
        # Xảy ra nếu tất cả sự khác biệt = 0
        w_stat, p_wilcox = 0.0, 1.0

    print("\n========== Kiểm định thống kê ==========")
    print(f"  LSTM   AUC: {lstm_aucs.mean():.4f} ± {lstm_aucs.std():.4f}")
    print(f"  BiLSTM AUC: {bilstm_aucs.mean():.4f} ± {bilstm_aucs.std():.4f}")
    print(f"\n  Paired t-test      : t={t_stat:.4f}  p={p_ttest:.4f}")
    print(f"  Wilcoxon signed    : W={w_stat:.1f}    p={p_wilcox:.4f}")

    alpha = 0.05
    if p_ttest < alpha:
        print(f"\n  → Có sự khác biệt có ý nghĩa thống kê (p < {alpha})")
    else:
        print(f"\n  → Không có sự khác biệt có ý nghĩa thống kê (p ≥ {alpha})")

    # Lưu kết quả
    result = {
        "LSTM_AUC_mean"   : lstm_aucs.mean(),
        "LSTM_AUC_std"    : lstm_aucs.std(),
        "BiLSTM_AUC_mean" : bilstm_aucs.mean(),
        "BiLSTM_AUC_std"  : bilstm_aucs.std(),
        "t_statistic"     : t_stat,
        "p_value_ttest"   : p_ttest,
        "w_statistic"     : w_stat,
        "p_value_wilcoxon": p_wilcox,
        "significant_005" : p_ttest < alpha,
    }
    pd.DataFrame([result]).to_csv("experiments/stats_result.csv", index=False)
    print("  Saved: experiments/stats_result.csv")
