
import pandas as pd
import numpy as np
import re, unicodedata

# ============================================================
# Max len thống nhất cho TOÀN BỘ hệ thống
# Training, Final model, API đều dùng chung giá trị này
# ============================================================
DEFAULT_MAX_LEN = 256


def normalize(x):
    """
    Chuẩn hoá text.
    Dataset_fullEDA.csv đã qua tiền xử lý sẵn nên chỉ cần
    chuẩn hóa Unicode + lowercase là đủ.
    """
    x = unicodedata.normalize("NFC", str(x)).lower()
    x = re.sub(r"\s+", " ", x).strip()
    return x


def clean_tokenize(text):
    """
    Tách từ — split() là đủ vì text đã clean từ pipeline.
    """
    return text.split()


def load_data(path, min_freq=2, max_len=DEFAULT_MAX_LEN):
    """
    Load và xử lý dữ liệu.
    Tự động phát hiện tên cột text.
    """
    df = pd.read_csv(path)

    # ── Tự động phát hiện cột text ──────────────────────────────
    text_col = None
    for col_name in ["text", "post_message", "content", "message"]:
        if col_name in df.columns:
            text_col = col_name
            break

    if text_col is None:
        raise ValueError(
            f"Không tìm thấy cột text! Columns: {list(df.columns)}"
        )

    print(f"  [dataset] Dùng cột: '{text_col}'")

    # Xóa cột index thừa
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Xóa null
    before = len(df)
    df = df.dropna(subset=[text_col]).reset_index(drop=True)
    if len(df) < before:
        print(f"  [dataset] Đã xóa {before - len(df)} dòng null")

    texts = df[text_col].apply(normalize)
    y     = df["label"].values.astype(np.int64)

    # ── Bước 1: Đếm tần suất từ ─────────────────────────────────
    token_freq = {}
    all_tokens = []
    for t in texts:
        tokens = clean_tokenize(t)
        all_tokens.append(tokens)
        for tok in tokens:
            token_freq[tok] = token_freq.get(tok, 0) + 1

    # ── Bước 2: Build vocab (lọc min_freq) ──────────────────────
    vocab = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for tok, freq in sorted(token_freq.items()):
        if freq >= min_freq:
            vocab[tok] = idx
            idx += 1

    print(f"  Vocab: {len(token_freq):,} unique tokens "
          f"→ {len(vocab):,} sau min_freq={min_freq}")

    # ── Bước 3: Encode sequences ─────────────────────────────────
    unk_idx = vocab["<UNK>"]
    seqs = []
    for tokens in all_tokens:
        s = [vocab.get(tok, unk_idx) for tok in tokens]
        seqs.append(s)

    # ── Bước 4: Pad / truncate sequences ────────────────────────
    X = np.zeros((len(seqs), max_len), dtype=np.int64)
    lengths = np.zeros(len(seqs), dtype=np.int64)
    for i, s in enumerate(seqs):
        length = min(len(s), max_len)
        X[i, :length] = s[:length]
        lengths[i] = length

    n_trunc = int((lengths == max_len).sum())
    print(f"  max_len={max_len} | Bị cắt: {n_trunc} mẫu "
          f"({n_trunc/len(seqs)*100:.1f}%)")
    print(f"  Dataset: {len(X):,} mẫu | Labels: {np.bincount(y)}")

    return X, y, len(vocab), vocab, lengths
