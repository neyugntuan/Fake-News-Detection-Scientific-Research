
import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, v): #đây là hàm tạo constructor
        super().__init__()
        self.emb = nn.Embedding(v, 128, padding_idx=0)
        self.lstm = nn.LSTM(128, 64, batch_first=True) #Khi batch_first=True, LSTM mong input có dạng:(batch, seq_len, input_size(H))
        self.fc = nn.Linear(64, 1) #trả ra kết quả chưa qua xử lý(logits) cần qua sigmoid

    def forward(self, x): #x: (batch_size, 256)
        x = self.emb(x)
        out, _ = self.lstm(x)  #out = hidden state tại mọi timestep    #(batch_size, 256, 64): mỗi câu trong batch_size có 256 vecto 64 chiều
                               #bỏ qua hidden state và cell state (h_n, c_n) cuối cùng
        return self.fc(out.mean(1)).squeeze(1) #out.mean(1) gom 256 hidden state(256 từ trong 1 câu), lấy trung bình thành 1 vecto đại diện cho cả câu:(batch_size, 64) mỗi câu trong batch_size là 1 vecto 64 chiều
                                               #self.fc nhận vecto 64 chiều thành 1 số duy nhất (batch_size, 1)
                                               #squeeze(1) xóa đi chiều thứ 2 nếu nó = 1: (batch,) => [ 2.3, -1.1, 0.7, ... ](logits)
                                               #do BCEWithLogitsLoss mong đợi: (input shape) = (batch,) => logits = tensor([ 2.3, -1.2, 0.5,...]) && (target shape) = (batch,) => labels = tensor([1., 0., 1.,...])


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
        super().__init__()                           #(batch_size, seqs_len(số từ mỗi hàng), số chiều vectơ)

        # Embedding với padding_idx=0
        self.emb = nn.Embedding(v, embed_dim, padding_idx=0)  #I(32, 256) -> O(32, 256, 128)

        # Dropout sau embedding
        self.embed_drop = nn.Dropout(dropout) #I(32, 256, 128) → O(32, 256, 128)

        # BiLSTM 2 tầng với dropout giữa các tầng #I(32, 256, 128) → O(32, 256, 128)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Layer norm + Dropout trước FC
        self.layer_norm = nn.LayerNorm(hidden_size * 2) #chuẩn hóa về vecto 128 chiều: I(32, 128) -> O(32, không nhận trực tiếp từ BiLSTM mà từ masked mean pooling trong forward()
        self.fc_drop = nn.Dropout(dropout)              #I(32, 128) -> O(32, 128)

        # FC output
        self.fc = nn.Linear(hidden_size * 2, 1)         #I(32, 128) -> O(32, 1)

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
                                                 # vd:(1,5,2)
                                                #  out =
                                                #         [
                                                #         [ [1,10],   ← token1
                                                #             [2,20],   ← token2
                                                #             [3,30],   ← token3
                                                #             [7,70],   ← PAD ❗
                                                #             [8,80] ]  ← PAD ❗
                                                #         ]

        # === Masked Mean Pooling ===
        # Chỉ lấy mean của các vị trí có token thật, bỏ PAD
        mask_expanded = mask.unsqueeze(-1).float()           # => O(B, L, 1) do out có shape (B, L, 2*H) nên phải thêm 1 chiều vào thì mới nhân được(mask(B, L)), và do mask đang là true/flase muốn nhân phải chuyển sang float
                                                             #vd:(1,5,2)
                                                             #[
                                                            #  [true -> [1],
                                                            #   true -> [1],
                                                            #   true -> [1],
                                                            #   false -> [0],
                                                            #   false -> [0] ]
                                                            # ]

        sum_out = (out * mask_expanded).sum(dim=1)           # O(B, 2*H), mỗi vecto 2*H sẽ bị nhân với 0 hoặc 1
                                                            #  [
                                                            #     [ [1,10],
                                                            #       [2,20],
                                                            #       [3,30],
                                                            #       [0,0],     ← PAD bị triệt tiêu
                                                            #       [0,0] ]
                                                            #     ]
                                                             #sum(dim=1) tổng theo chiều token(0: batch, 2:dimension)
                                                            #[1+2+3 , 10+20+30] = [6, 60]
                                                            #shape lúc này (B, 2*H) = (B,2), trong thực tế 128 for sure

        lengths = mask.sum(dim=1, keepdim=True).float()      # O(B, 1)
                                                            #mark [1,1,1,0,0] .sum(dim=1) = 3
                                                            #keepdim=True → giữ shape (1,1) = [[3]]

        lengths = lengths.clamp(min=1)                       # tránh chia 0, nếu length = 0 vẫn tính là 1
        pooled = sum_out / lengths                           # (B, 2*H)
                                                            #sum_out = [[6,60]] , lengths = [[3]]
                                                            #pooled = [6,60] / 3 = [2,20] => shape = (1,2)

        # === Layer Norm + Dropout + FC ===
        pooled = self.layer_norm(pooled)                    #Giá trị được chuẩn hóa (mean≈0, std≈1) #Tính mean-Trừ mean-Chia cho độ lệch chuẩn
                                                            #[2,20] = [-9, 9] shape = (1,2)

        pooled = self.fc_drop(pooled)

        return self.fc(pooled).squeeze(1)        # (B,) -> mỗi câu thành 1 số logits chưa qua sigmoid,.squeeze(1) để output phù hợp với BCEWithLogitsLoss
                                                 #Linear vd shape (2,6) → (2,1) =[[0.8],[-1.3]]
                                                 #squeeze shape (2,1) → (2,) = [0.8, -1.3]

                        # Toàn bộ pipeline dạng sơ đồ
                                # (B,L)
                                # ↓ Embedding
                                # (B,L,E)
                                # ↓ BiLSTM
                                # (B,L,2H)
                                # ↓ Mask
                                # (B,L,2H)
                                # ↓ Sum
                                # (B,2H)
                                # ↓ Divide
                                # (B,2H)
                                # ↓ LayerNorm
                                # (B,2H)
                                # ↓ Dropout
                                # (B,2H)
                                # ↓ Linear
                                # (B,1)
                                # ↓ squeeze
                                # (B,)
