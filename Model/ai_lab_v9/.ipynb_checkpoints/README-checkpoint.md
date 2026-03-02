# AI Lab — Phát hiện Tin Giả Tiếng Việt (LSTM vs BiLSTM)

## Cấu trúc project

```
ai_lab_v6_d/
├── data/
│   └── dataset_fullEDA.csv      ← dữ liệu đã tiền xử lý + EDA (50/50)
├── src/
│   ├── dataset.py               ← load & encode dữ liệu
│   ├── models.py                ← LSTM, BiLSTM
│   ├── trainer.py               ← vòng lặp train, early stopping
│   ├── experiment_dual.py       ← 10-fold experiment
│   ├── metrics_extension.py     ← Acc/Prec/Rec/F1, batched inference
│   ├── plots_dual.py            ← biểu đồ AUC so sánh
│   ├── stats.py                 ← Paired t-test, Wilcoxon test
│   └── final_model.py           ← train final model để deploy
├── run.py                       ← entry point
├── sanity_check.py              ← kiểm tra sau khi train
└── requirements.txt
```

## Chạy thực nghiệm

```bash
pip install -r requirements.txt
python run.py
```

## Outputs

| Thư mục | Nội dung |
|---|---|
| `experiments/dual_results.csv` | AUC 10-fold của LSTM và BiLSTM |
| `experiments/metrics_log.csv`  | Acc/Prec/Rec/F1 đầy đủ theo fold |
| `experiments/stats_result.csv` | Kết quả kiểm định thống kê |
| `experiments/figures/`         | Boxplot, line chart các metrics |
| `deploy/model_final.pt`        | Weights BiLSTM cho inference |
| `deploy/vocab_mapping.pkl`     | Vocab dict cho API |
| `deploy/config.pkl`            | Config model |
| `reports/learning_curve.png`   | Đường cong học tập |

## Kiểm tra sau khi train

```bash
python sanity_check.py
```
