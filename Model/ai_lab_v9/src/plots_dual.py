import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_dual():
    """
    Vẽ biểu đồ so sánh AUC giữa LSTM và BiLSTM qua 10 fold.
    Đọc từ experiments/dual_results.csv.
    """
    os.makedirs("experiments/figures", exist_ok=True)

    csv_path = "experiments/dual_results.csv"
    if not os.path.exists(csv_path):
        print("  [plot_dual] Không tìm thấy dual_results.csv — bỏ qua")
        return

    df = pd.read_csv(csv_path)

    # ── Boxplot ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(
        data=df[["LSTM_AUC", "BiLSTM_AUC"]],
        palette=["#2E86AB", "#E84855"],
        ax=ax,
    )
    ax.set_xticklabels(["LSTM", "BiLSTM"])
    ax.set_title("So sánh AUC — LSTM vs BiLSTM (10-Fold)")
    ax.set_ylabel("AUC-ROC")
    ax.grid(axis="y", alpha=0.3)
    plt.savefig("experiments/figures/dual_auc_boxplot.png",
                dpi=300, bbox_inches="tight")
    plt.close()

    # ── Line plot per fold ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["Fold"], df["LSTM_AUC"],
            label="LSTM", marker="o", color="#2E86AB", linewidth=2)
    ax.plot(df["Fold"], df["BiLSTM_AUC"],
            label="BiLSTM", marker="s", color="#E84855", linewidth=2)
    ax.axhline(df["LSTM_AUC"].mean(),
               color="#2E86AB", linestyle="--", alpha=0.5,
               label=f"LSTM mean={df['LSTM_AUC'].mean():.3f}")
    ax.axhline(df["BiLSTM_AUC"].mean(),
               color="#E84855", linestyle="--", alpha=0.5,
               label=f"BiLSTM mean={df['BiLSTM_AUC'].mean():.3f}")
    ax.set_xticks(df["Fold"])
    ax.set_xlabel("Fold")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("AUC per Fold — LSTM vs BiLSTM")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.savefig("experiments/figures/dual_auc_line.png",
                dpi=300, bbox_inches="tight")
    plt.close()

    print("  Saved: experiments/figures/dual_auc_boxplot.png")
    print("  Saved: experiments/figures/dual_auc_line.png")
