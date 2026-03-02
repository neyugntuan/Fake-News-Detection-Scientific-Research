
import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, v):
        super().__init__()
        self.emb = nn.Embedding(v, 128, padding_idx=0)
        self.lstm = nn.LSTM(128, 64, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = self.emb(x)
        out, _ = self.lstm(x)
        return self.fc(out.mean(1)).squeeze(1)


class BiLSTM(nn.Module):
    """
    BiLSTM tối ưu cho phân loại tin giả.

    CẢI TIẾN so với bản cũ:
    1. padding_idx=0: embedding của PAD luôn = 0 → không gây nhiễu
    2. Masked mean pooling: chỉ tính mean trên token thật, bỏ qua PAD
       → Bản cũ mean cả PAD → 66% signal bị pha loãng
    3. Dropout: regularization chống overfit
       → Bản cũ không có dropout → BiLSTM overfit nhanh hơn LSTM
    4. 2-layer LSTM: tăng khả năng biểu diễn
    5. Layer Normalization: ổn định training
    """

    def __init__(self, v, embed_dim=128, hidden_size=64, dropout=0.3, num_layers=2):
        super().__init__()

        # Embedding với padding_idx=0
        self.emb = nn.Embedding(v, embed_dim, padding_idx=0)

        # Dropout sau embedding
        self.embed_drop = nn.Dropout(dropout)

        # BiLSTM 2 tầng với dropout giữa các tầng
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Layer norm + Dropout trước FC
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.fc_drop = nn.Dropout(dropout)

        # FC output
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        """
        Args:
            x: (B, L) token indices, 0 = PAD
        """
        # === Tạo mask: True cho token thật, False cho PAD ===
        mask = (x != 0)                         # (B, L)

        # === Embedding + Dropout ===
        emb = self.emb(x)                       # (B, L, E)
        emb = self.embed_drop(emb)

        # === BiLSTM ===
        out, _ = self.lstm(emb)                  # (B, L, 2*H)

        # === Masked Mean Pooling ===
        # Chỉ lấy mean của các vị trí có token thật, bỏ PAD
        mask_expanded = mask.unsqueeze(-1).float()           # (B, L, 1)
        sum_out = (out * mask_expanded).sum(dim=1)           # (B, 2*H)
        lengths = mask.sum(dim=1, keepdim=True).float()      # (B, 1)
        lengths = lengths.clamp(min=1)                       # tránh chia 0
        pooled = sum_out / lengths                           # (B, 2*H)

        # === Layer Norm + Dropout + FC ===
        pooled = self.layer_norm(pooled)
        pooled = self.fc_drop(pooled)

        return self.fc(pooled).squeeze(1)        # (B,)
