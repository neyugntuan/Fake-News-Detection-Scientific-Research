# ================================================================
#  PIPELINE: TIỀN XỬ LÝ DỮ LIỆU & FULL EDA AUGMENTATION
#  Dataset: dataset.csv — Phát hiện tin giả tiếng Việt
#  Output : dataset_fullEDA.csv + biểu đồ báo cáo
#
#  CÁCH CHẠY:
#    1. Đặt file dataset.csv cùng thư mục
#    2. python preprocessing_eda_pipeline.py
#    3. Kết quả: artifacts/dataset_fullEDA.csv + biểu đồ
# ================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches
import re, unicodedata, random, os, pickle, warnings, string
from collections import Counter

warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)
os.makedirs("artifacts", exist_ok=True)

# ── Màu sắc nhất quán cho toàn bộ biểu đồ ──────────────────────
C0 = "#2E86AB"   # xanh dương  → label 0 (real)
C1 = "#E84855"   # đỏ          → label 1 (fake)
CEDA = "#F18F01" # cam         → EDA augmented

plt.rcParams.update({
    "font.family"     : "DejaVu Sans", #font chữ
    "axes.spines.top" : False,         #bỏ đường viền trên
    "axes.spines.right": False,        #bỏ đường viền phải
    "axes.grid"       : True,          #bật grid: là các đường ngang dọc trên biểu đồ giúp đọc giá trị dễ hơn
    "grid.alpha"      : 0.3,           #độ mờ của grid: 0->1
    "figure.dpi"      : 150,           #độ phân giải
})

# ================================================================
#  BƯỚC 1 — NẠP DỮ LIỆU THÔ
# ================================================================
print("=" * 65)
print("BƯỚC 1 — NẠP DỮ LIỆU")
print("=" * 65)

# ── Tự động phát hiện file + cột ────────────────────────────────
INPUT_FILE = None
for fname in ["dataset.csv", "merged.csv", "data.csv"]:
    if os.path.exists(fname):
        INPUT_FILE = fname
        break

if INPUT_FILE is None:
    # Tìm trong thư mục data/
    for fname in ["data/dataset.csv", "data/merged.csv"]:
        if os.path.exists(fname):
            INPUT_FILE = fname
            break

if INPUT_FILE is None:
    raise FileNotFoundError(
        "Không tìm thấy dataset! Đặt file dataset.csv cùng thư mục rồi chạy lại."
    )

print(f"  File: {INPUT_FILE}")
df_raw = pd.read_csv(INPUT_FILE)

# Drop cột index thừa
if "Unnamed: 0" in df_raw.columns:
    df_raw = df_raw.drop(columns=["Unnamed: 0"])

# Tự động phát hiện cột text
TEXT_COL = None
for col in ["text", "post_message", "content", "message"]:
    if col in df_raw.columns:
        TEXT_COL = col
        break

if TEXT_COL is None:
    raise ValueError(f"Không tìm thấy cột text! Columns: {list(df_raw.columns)}")

print(f"  Cột text: '{TEXT_COL}'")
print(f"  Tổng dòng ban đầu   : {len(df_raw):,}")
print(f"  Số cột              : {len(df_raw.columns)}")
print(f"  Cột                 : {df_raw.columns.tolist()}")
print(f"  Giá trị null (text) : {df_raw[TEXT_COL].isnull().sum()}")
print(f"  Giá trị null (label): {df_raw['label'].isnull().sum()}")
print(f"  Bản ghi trùng lặp   : {df_raw.duplicated(subset=[TEXT_COL]).sum()}")

# ================================================================
#  BƯỚC 2 — LÀM SẠCH CƠ BẢN (BASIC CLEANING)
# ================================================================
print("\n" + "=" * 65)
print("BƯỚC 2 — LÀM SẠCH CƠ BẢN")
print("=" * 65)

df = df_raw.copy()

# 2.1 Xóa null & duplicate
df = df.dropna(subset=[TEXT_COL, "label"])
df = df.drop_duplicates(subset=[TEXT_COL])
df["label"] = df["label"].astype(int)

# 2.2 Reset index
df = df.reset_index(drop=True)

n_removed = len(df_raw) - len(df)
print(f"  Sau làm sạch        : {len(df):,} dòng (đã loại {n_removed} dòng null/trùng)")
print(f"  Label 0 (tin thật)  : {(df.label==0).sum():,}")
print(f"  Label 1 (tin giả)   : {(df.label==1).sum():,}")

n0, n1 = (df.label==0).sum(), (df.label==1).sum()
if n1 > 0:
    print(f"  Tỷ lệ mất cân bằng  : {n0/n1:.2f}:1")

# ================================================================
#  BƯỚC 3 — CHUẨN HÓA VĂN BẢN (NORMALIZATION)
# ================================================================
print("\n" + "=" * 65)
print("BƯỚC 3 — CHUẨN HÓA VĂN BẢN")
print("=" * 65)

STOPWORDS_VI = set([
    "và","của","các","là","có","trong","được","cho","với","đã","này",
    "một","những","không","tại","từ","để","theo","đến","về","thì",
    "khi","cũng","sẽ","như","còn","nhưng","vì","nên","mà","hay",
    "rất","bị","lại","đó","đây","đang","sẽ","hơn","sau","trước",
    "đã","chỉ","ông","bà","anh","chị","họ","chúng","tôi","bạn",
    "mình","chúng_tôi","chúng_ta","chúng_nó","nó","cô","em",
    "thì","mà","mà","ra","vào","lên","xuống","qua","lại","đi",
    "thế","thế_này","thế_đó","thế_kia","đây","kia","đó",
    "url","<url>","num","xem_thêm",
])

def normalize(text: str) -> str:
    """Pipeline chuẩn hóa 7 bước cho văn bản tiếng Việt."""
    # B1: Chuẩn hóa Unicode NFC
    text = unicodedata.normalize("NFC", str(text)) # chuẩn hóa dấu tiếng việt
    # B2: Chuyển về chữ thường
    text = text.lower()
    # B3: Xóa URL
    text = re.sub(r"http\S+|www\.\S+|<url>", " <URL> ", text)
    # B4: Xóa HTML tags
    text = re.sub(r"<[^>]+>", " ", text) #< :ký tự <, [^>]: mọi ký tự không phải >, + :lặp lại 1 hoặc nhiều lần, >: ký tự > || tức là bất kỳ chuỗi nào nằm giữa < ... >
    # B5: Thay số → token đặc biệt
    text = re.sub(r"\b\d+([.,]\d+)*\b", " <NUM> ", text) #\b \b:số đứng độc lập, \d+: 1 hoặc nhiều chữ số,([.,]\d+)* : số thập phân hoặc số có dấu phân cách
    # B6: Rút gọn ký tự lặp (≥3 lần → 2 lần)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text) # . :bất kỳ ký tự nào, () :group 1 (lưu ký tự này lại), \1 :tham chiếu lại group 1, {2,} : lặp từ 2 lần trở lên
                                               # r"\1\1" : 2 ký tự giống nhau
    # B7: Chỉ giữ chữ tiếng Việt, tiếng Anh, token đặc biệt, khoảng trắng
    text = re.sub(
        r"[^a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩ"
        r"òóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ_<>\s]",
        " ", text
    )
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text: str) -> list:
    """Tokenize đơn giản (regex-based)."""
    return re.findall(r"\b\w+\b", text)  #1 chuỗi gồm ≥1 ký tự thuộc a-z A-Z 0-9 _được bao quanh bởi 2 dấu cách

def remove_stopwords(tokens: list) -> list:
    return [t for t in tokens if t not in STOPWORDS_VI and len(t) > 1]

# Áp dụng pipeline
df[TEXT_COL] = df[TEXT_COL].str.replace('_', ' ')# Thêm mới 25/02/2026
df["text_normalized"] = df[TEXT_COL].apply(normalize)
df["tokens"]          = df["text_normalized"].apply(tokenize)
df["tokens_clean"]    = df["tokens"].apply(remove_stopwords)
df["text_clean"]      = df["tokens_clean"].apply(lambda t: " ".join(t)) #Lấy list token trong cột tokens_clean → ghép lại thành chuỗi → lưu vào cột text_clean
                                                                        #" ".join(t): ghép các phần tử của list thành chuỗi, cách nhau bằng dấu cách

# Thống kê
df["len_raw"]   = df[TEXT_COL].str.len() #số ký tự của text gốc
df["len_norm"]  = df["text_normalized"].str.len() #số ký tự sau khi chuẩn hóa text
df["n_tokens"]  = df["tokens"].str.len() #số token trước khi remove stopwords
df["n_tokens_clean"] = df["tokens_clean"].str.len() #số token sau khi remove stopwords

print(f"  Độ dài TB (raw)     : {df.len_raw.mean():.0f} ký tự")
print(f"  Độ dài TB (chuẩn hóa): {df.len_norm.mean():.0f} ký tự")
print(f"  Token TB (trước SW) : {df.n_tokens.mean():.1f}")
print(f"  Token TB (sau SW)   : {df.n_tokens_clean.mean():.1f}")

# Ví dụ trước/sau chuẩn hóa
print("\n  Ví dụ chuẩn hóa:")
ex = df.iloc[3] #lấy dòng thứ 4 trong DataFrame
print(f"  RAW  : {str(ex[TEXT_COL])[:120]}...") #in 120 ký tự đầu của text gốc
print(f"  CLEAN: {ex['text_clean'][:120]}...")

# ================================================================
#  BƯỚC 4 — FULL EDA AUGMENTATION
# ================================================================
print("\n" + "=" * 65)
print("BƯỚC 4 — FULL EDA AUGMENTATION")
print("=" * 65)

# ── Từ điển đồng nghĩa (domain-specific: tin tức COVID-19) ──────
SYNONYMS = {
    # Hành động truyền thông
    "cho_biết"   : ["nói", "khẳng_định", "thông_báo", "tiết_lộ"],
    "khẳng_định" : ["xác_nhận", "cho_biết", "tuyên_bố", "nhấn_mạnh"],
    "thông_báo"  : ["công_bố", "cho_biết", "phát_đi"],
    "công_bố"    : ["tiết_lộ", "thông_báo", "công_khai"],
    "cảnh_báo"   : ["nhắc_nhở", "lưu_ý", "cảnh_báo"],
    "nghiên_cứu" : ["khảo_sát", "điều_tra", "phân_tích"],
    "phát_hiện"  : ["tìm_ra", "phát_giác", "xác_định"],
    # Y tế / dịch bệnh
    "bệnh_nhân"  : ["ca_bệnh", "người_bệnh", "bệnh_nhân"],
    "dịch_bệnh"  : ["đại_dịch", "bệnh_dịch", "dịch"],
    "điều_trị"   : ["chữa_trị", "chữa", "trị"],
    "phòng_ngừa" : ["ngăn_chặn", "phòng_chống", "đề_phòng"],
    "lây_nhiễm"  : ["lây_lan", "lây", "truyền_nhiễm"],
    "triệu_chứng": ["biểu_hiện", "dấu_hiệu", "triệu_chứng"],
    "chuyên_gia" : ["nhà_khoa_học", "bác_sĩ", "nhà_nghiên_cứu"],
    "bệnh_viện"  : ["cơ_sở_y_tế", "viện"],
    "vắc_xin"    : ["vaccine", "vắc_xin"],
    # Chính trị / xã hội
    "chính_phủ"  : ["nhà_nước", "chính_quyền", "chính_phủ"],
    "tổ_chức"    : ["cơ_quan", "đơn_vị", "tổ_chức"],
    "người_dân"  : ["dân_chúng", "cộng_đồng", "công_chúng"],
    "thông_tin"  : ["tin_tức", "tin", "nguồn_tin"],
    "nguy_cơ"    : ["rủi_ro", "mối_nguy", "hiểm_họa"],
    "tình_hình"  : ["tình_trạng", "hoàn_cảnh", "bối_cảnh"],
    # Từ đánh giá
    "tốt"        : ["hay", "ổn", "tuyệt", "khá"],
    "xấu"        : ["tệ", "kém", "tồi"],
    "nhanh"      : ["lẹ", "mau"],
    "chậm"       : ["trễ", "muộn"],
    "nguy_hiểm"  : ["độc_hại", "có_hại"],
    "an_toàn"    : ["bình_an", "vô_hại"],
}

# ── 4 kỹ thuật EDA ──────────────────────────────────────────────

def synonym_replace(tokens: list, n: int = 2) -> list:
    """SR — Thay thế n từ bằng từ đồng nghĩa."""
    tokens = tokens.copy()
    candidates = [i for i, w in enumerate(tokens) if w in SYNONYMS] #tìm vị trí có từ đồng nghĩa trong tokens
    if not candidates:
        return tokens
    chosen = random.sample(candidates, min(n, len(candidates))) #chọn ngẫu nhiên vị trí của candidates, nếu số candidate ít hơn n → lấy hết
    for i in chosen: #lấy vị trí có từ đồng nghĩa đưa synomys để trả lại từ đồng nghĩa rồi gán cho vị trí của tokens của từ đồng nghĩa đó
        tokens[i] = random.choice(SYNONYMS[tokens[i]])
    return tokens

def random_swap(tokens: list, n: int = 1) -> list:
    """RS — Hoán đổi vị trí 2 từ ngẫu nhiên, thực hiện n lần."""
    tokens = tokens.copy()
    if len(tokens) < 2:
        return tokens
    for _ in range(n):
        i, j = random.sample(range(len(tokens)), 2) #len(tokens) = 5, range(5) = [0,1,2,3,4], lấy ra 2 random 2 phần tử i, j
        tokens[i], tokens[j] = tokens[j], tokens[i]
    return tokens

def random_delete(tokens: list, p: float = 0.10) -> list:
    """RD — Xóa mỗi từ với xác suất p."""
    if len(tokens) <= 5:
        return tokens
    result = [w for w in tokens if random.random() > p]  #random.random(): sinh số ngẫu nhiên từ 0 -> 1, chữ nào sinh ra số >0.1 thì giữ lại
    return result if len(result) >= 5 else tokens #nếu bị xóa xuống dưới 5 token(chữ) thì sẽ trả lại tokens(câu) gốc

def random_insert(tokens: list, n: int = 1) -> list:
    """RI — Chèn từ đồng nghĩa vào vị trí ngẫu nhiên."""
    tokens = tokens.copy()
    syn_tokens = [w for w in tokens if w in SYNONYMS]
    if not syn_tokens:
        return tokens
    for _ in range(n):
        src     = random.choice(syn_tokens) #Bốc 1 từ trong list chứa từ đồng nghĩa
        new_tok = random.choice(SYNONYMS[src]) #Chọn ra từ đồng nghĩa
        pos     = random.randint(0, len(tokens)) #Chọn vị trí chèn ngẫu nhiên từ 0 -> độ dài của câu
        tokens.insert(pos, new_tok) #chèn từ đồng nghĩa vào vị trí ngẫu nhiên đó
    return tokens

# Trọng số: ưu tiên SR > RI > RS > RD
EDA_FUNCS   = [synonym_replace, random_insert, random_swap, random_delete]
EDA_WEIGHTS = [0.40,            0.25,          0.20,        0.15]
EDA_NAMES   = ["SR (Synonym Replace)", "RI (Random Insert)",
               "RS (Random Swap)",     "RD (Random Delete)"]

def apply_eda(tokens: list) -> tuple: #Tuple là kiểu dữ liệu giống list nhưng không thay đổi được.
    """Trả về (tokens_aug, tên_kỹ_thuật)."""
    idx  = random.choices(range(len(EDA_FUNCS)), weights=EDA_WEIGHTS, k=1)[0] #Chọn ngẫu nhiên có trọng số, range(len(EDA_FUNCS)): từ 0->3, trọng số từ 0->3 là EDA_WEIGHTS, k = 1: lấy 1 kết quả
                                                                            #random.choices() luôn trả về list vd: [1], lấy phần tử bằng cách idx = [1][0]
    func = EDA_FUNCS[idx] #lấy ra 1 hàm, func trở thành hàm đó, #PYTHON cho phép lưu hàm vào biến
    return func(tokens), EDA_NAMES[idx]

# ── Sinh dữ liệu ──────────────────────────────────────────────
df_major = df[df.label == 0].copy()
df_minor = df[df.label == 1].copy()
target   = len(df_major)
gap      = target - len(df_minor)

aug_texts    = [] #list lưu text sau khi tăng
aug_methods  = [] #list lưu kỹ thuật EDA đã dùng
minor_tokens = df_minor["tokens_clean"].tolist() #Lấy toàn bộ token của tin giả

print(f"  Label 0 (major) : {len(df_major):,}") #in số lượng mẫu 0, :, là để phân cách hàng nghìn
print(f"  Label 1 (minor) : {len(df_minor):,}") #in số lượng mẫu 1
print(f"  Cần sinh thêm   : {gap:,} mẫu")       #in số lượng mẫu cần sinh thêm

attempts = 0
while len(aug_texts) < gap:  #lặp đến khi lắp vào đủ số câu gap
    src_tokens = random.choice(minor_tokens) #bốc 1 câu ngẫu nhiên có label 1
    if len(src_tokens) < 5:                    #nếu câu đó dưới 5 từ
        attempts += 1                          #tăng biến đếm lên 1
        if attempts > gap * 3:                 #nếu biến đếm lớn hơn
            break
        continue                                #bỏ qua phần code phía dưới, lặp lần tiếp theo
    aug_tok, method = apply_eda(src_tokens)    #Chọn ngẫu nhiên method EDA
    aug_texts.append(" ".join(aug_tok))        #Lưu text vào aug_texts
    aug_methods.append(method)                 #Lưu method vào aug_method
    attempts = 0                               #set attempts = 0 để bắt đầu vòng lặp khác

print(f"  Đã sinh         : {len(aug_texts):,} mẫu EDA")

# ── Ghép tập cuối ─────────────────────────────────────────────
df_aug = pd.DataFrame({
    "text_clean": aug_texts,
    "label"     : 1,
    "source"    : "EDA"
})
df_major["source"] = "original"
df_minor["source"] = "original"

df_final = pd.concat([
    df_major[["text_clean", "label", "source"]],
    df_minor[["text_clean", "label", "source"]],
    df_aug
]).sample(frac=1, random_state=42).reset_index(drop=True) #lấy 100% dữ liệu, trộn thứ tự ngẫu nhiên
                                                        #.reset_index(drop=True): do index đang lộn xộn

print(f"\n  Tập dữ liệu cuối cùng:")
print(f"  ├── Tổng mẫu   : {len(df_final):,}")
print(f"  ├── Label 0    : {(df_final.label==0).sum():,}")
print(f"  └── Label 1    : {(df_final.label==1).sum():,}")

# ================================================================
#  BƯỚC 5 — THỐNG KÊ TỪ VỰNG
# ================================================================
print("\n" + "=" * 65)
print("BƯỚC 5 — THỐNG KÊ TỪ VỰNG")
print("=" * 65)

all_tokens_0 = [t for tlist in df[df.label==0]["tokens_clean"] for t in tlist]
all_tokens_1 = [t for tlist in df[df.label==1]["tokens_clean"] for t in tlist]

vocab_0 = Counter(all_tokens_0)
vocab_1 = Counter(all_tokens_1)

print(f"  Kích thước từ vựng (label 0): {len(vocab_0):,}")
print(f"  Kích thước từ vựng (label 1): {len(vocab_1):,}")
print(f"  Top 10 từ - Label 0: {vocab_0.most_common(10)}")
print(f"  Top 10 từ - Label 1: {vocab_1.most_common(10)}")

# ================================================================
#  BƯỚC 6 — VẼ BIỂU ĐỒ BÁO CÁO
# ================================================================
print("\n" + "=" * 65)
print("BƯỚC 6 — VẼ BIỂU ĐỒ")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────
# FIG 1: TỔNG QUAN DỮ LIỆU (2×2)
# ─────────────────────────────────────────────────────────────────
fig1, axes = plt.subplots(2, 2, figsize=(13, 9))
fig1.suptitle("Tổng quan phân tích dữ liệu (EDA — Exploratory Data Analysis)",
              fontsize=14, fontweight="bold", y=1.01)

# (A) Phân phối nhãn — Gốc
ax = axes[0, 0]
labels_name = ["Tin thật (0)", "Tin giả (1)"]
counts_orig = [(df.label==0).sum(), (df.label==1).sum()]
bars = ax.bar(labels_name, counts_orig, color=[C0, C1], alpha=0.85, edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, counts_orig):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 60,
            f"{val:,}\n({val/sum(counts_orig)*100:.1f}%)",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_title("(A) Phân phối nhãn — Dữ liệu gốc", fontweight="bold")
ax.set_ylabel("Số lượng mẫu")
ax.set_ylim(0, max(counts_orig) * 1.2)

# (B) Phân phối nhãn — Sau EDA
ax = axes[0, 1]
counts_eda = [(df_final.label==0).sum(), (df_final.label==1).sum()]
bars = ax.bar(labels_name, counts_eda, color=[C0, C1], alpha=0.85, edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, counts_eda):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 60,
            f"{val:,}\n({val/sum(counts_eda)*100:.1f}%)",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_title("(B) Phân phối nhãn — Sau Full EDA", fontweight="bold")
ax.set_ylabel("Số lượng mẫu")
ax.set_ylim(0, max(counts_eda) * 1.2)

# (C) Phân phối độ dài văn bản (raw)
ax = axes[1, 0]
len0 = df[df.label==0]["len_raw"].clip(upper=3000)
len1 = df[df.label==1]["len_raw"].clip(upper=3000)
ax.hist(len0, bins=50, color=C0, alpha=0.65, label="Tin thật", density=True)
ax.hist(len1, bins=50, color=C1, alpha=0.65, label="Tin giả",  density=True)
ax.axvline(len0.mean(), color=C0, linestyle="--", lw=1.5, label=f"TB thật={len0.mean():.0f}")
ax.axvline(len1.mean(), color=C1, linestyle="--", lw=1.5, label=f"TB giả={len1.mean():.0f}")
ax.set_title("(C) Phân phối độ dài văn bản (ký tự, clip@3000)", fontweight="bold")
ax.set_xlabel("Độ dài (ký tự)")
ax.set_ylabel("Mật độ")
ax.legend(fontsize=8)

# (D) Phân phối số token sau tiền xử lý
ax = axes[1, 1]
tok0 = df[df.label==0]["n_tokens_clean"].clip(upper=300)
tok1 = df[df.label==1]["n_tokens_clean"].clip(upper=300)
ax.hist(tok0, bins=40, color=C0, alpha=0.65, label="Tin thật", density=True)
ax.hist(tok1, bins=40, color=C1, alpha=0.65, label="Tin giả",  density=True)
ax.axvline(tok0.mean(), color=C0, linestyle="--", lw=1.5, label=f"TB thật={tok0.mean():.0f}")
ax.axvline(tok1.mean(), color=C1, linestyle="--", lw=1.5, label=f"TB giả={tok1.mean():.0f}")
ax.set_title("(D) Phân phối số token (sau xử lý, clip@300)", fontweight="bold")
ax.set_xlabel("Số token")
ax.set_ylabel("Mật độ")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("artifacts/fig1_data_overview.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved: fig1_data_overview.png")

# ─────────────────────────────────────────────────────────────────
# FIG 2: TOP-N TỪ THƯỜNG GẶP
# ─────────────────────────────────────────────────────────────────
fig2, axes = plt.subplots(1, 2, figsize=(14, 6))
fig2.suptitle("Top 20 từ xuất hiện nhiều nhất theo nhãn",
              fontsize=14, fontweight="bold")

for ax, vocab, color, title in [
    (axes[0], vocab_0, C0, "Label 0 — Tin thật"),
    (axes[1], vocab_1, C1, "Label 1 — Tin giả"),
]:
    words, freqs = zip(*vocab.most_common(20))
    y_pos = range(len(words))
    bars = ax.barh(y_pos, freqs, color=color, alpha=0.80, edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(words, fontsize=9)
    ax.invert_yaxis()
    for bar, freq in zip(bars, freqs):
        ax.text(bar.get_width() + max(freqs)*0.01, bar.get_y() + bar.get_height()/2,
                f"{freq:,}", va="center", fontsize=8)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Tần suất xuất hiện")

plt.tight_layout()
plt.savefig("artifacts/fig2_top_words.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved: fig2_top_words.png")

# ─────────────────────────────────────────────────────────────────
# FIG 3: PIPELINE TIỀN XỬ LÝ (Sơ đồ luồng)
# ─────────────────────────────────────────────────────────────────
fig3, ax = plt.subplots(figsize=(14, 5))
ax.set_xlim(0, 14); ax.set_ylim(0, 5)
ax.axis("off")
fig3.suptitle("Sơ đồ pipeline tiền xử lý văn bản tiếng Việt",
              fontsize=13, fontweight="bold")

steps = [
    ("1. Nạp\ndữ liệu",       f"{INPUT_FILE}\n{len(df_raw):,} mẫu",  "#AED6F1"),
    ("2. Làm sạch\ncơ bản",   "Xóa null,\ntrùng lặp",                "#A9DFBF"),
    ("3. Chuẩn hóa\nUnicode",  "NFC, lowercase",                       "#A9DFBF"),
    ("4. Xóa nhiễu",           "URL, HTML,\nký tự đặc biệt",          "#A9DFBF"),
    ("5. Chuẩn hóa\nsố",       "Số → <NUM>",                          "#A9DFBF"),
    ("6. Rút gọn\nlặp",        "aaaa → aa",                            "#A9DFBF"),
    ("7. Xóa\nstopwords",      "Loại 60+\ntừ dừng",                   "#A9DFBF"),
    ("8. Tokenize",            "regex-based\ntokenizer",                "#FAD7A0"),
    ("Full EDA\nAugment",      "SR/RS/RD/RI\n→ Cân bằng lớp",        "#F1948A"),
]

x_positions = [i * 1.55 + 0.3 for i in range(len(steps))]
for i, (title, detail, color) in enumerate(steps):
    x = x_positions[i]
    fancy = FancyBboxPatch((x - 0.65, 1.5), 1.25, 1.9,
                           boxstyle="round,pad=0.1",
                           facecolor=color, edgecolor="#555", linewidth=1.2)
    ax.add_patch(fancy)
    ax.text(x, 2.7, title, ha="center", va="center", fontsize=7.5, fontweight="bold")
    ax.text(x, 1.9, detail, ha="center", va="center", fontsize=6.5, color="#333", style="italic")
    if i < len(steps) - 1:
        ax.annotate("", xy=(x_positions[i+1]-0.65, 2.45), xytext=(x+0.65, 2.45),
                    arrowprops=dict(arrowstyle="->", color="#444", lw=1.5))

legend_patches = [
    mpatches.Patch(color="#AED6F1", label="Input"),
    mpatches.Patch(color="#A9DFBF", label="Chuẩn hóa"),
    mpatches.Patch(color="#FAD7A0", label="Tokenize"),
    mpatches.Patch(color="#F1948A", label="Augment (EDA)"),
]
ax.legend(handles=legend_patches, loc="lower right", fontsize=8, framealpha=0.8, ncol=4)
plt.tight_layout()
plt.savefig("artifacts/fig3_pipeline_diagram.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved: fig3_pipeline_diagram.png")

# ─────────────────────────────────────────────────────────────────
# FIG 4: PHÂN PHỐI KỸ THUẬT EDA
# ─────────────────────────────────────────────────────────────────
fig4, axes = plt.subplots(1, 2, figsize=(13, 5))
fig4.suptitle("Phân tích kỹ thuật Full EDA Augmentation",
              fontsize=13, fontweight="bold")

ax = axes[0]
method_counts = Counter(aug_methods)
methods_sorted = sorted(method_counts.items(), key=lambda x: -x[1])
m_names  = [m[0] for m in methods_sorted]
m_values = [m[1] for m in methods_sorted]
total_aug = sum(m_values)
colors_eda = [CEDA, C0, C1, "#8E44AD"]

bars = ax.bar(range(len(m_names)), m_values,
              color=colors_eda[:len(m_names)], alpha=0.85, edgecolor="white")
for bar, val in zip(bars, m_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
            f"{val:,}\n({val/total_aug*100:.1f}%)",
            ha="center", va="bottom", fontsize=8.5, fontweight="bold")
ax.set_xticks(range(len(m_names)))
ax.set_xticklabels([n.split("(")[0].strip() + "\n(" + n.split("(")[1]
                    if "(" in n else n for n in m_names], fontsize=8)
ax.set_title("(A) Số lượng mẫu sinh theo từng kỹ thuật EDA", fontweight="bold")
ax.set_ylabel("Số mẫu sinh ra")

ax = axes[1]
minor_tok_counts = df_minor["n_tokens_clean"].clip(upper=200).tolist()
aug_tok_counts   = [len(t.split()) for t in aug_texts[:len(minor_tok_counts)]]
bp = ax.boxplot([minor_tok_counts, aug_tok_counts], patch_artist=True, notch=True,
                medianprops=dict(color="black", lw=2))
for patch, color in zip(bp["boxes"], [C1, CEDA]):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
ax.set_xticks([1, 2])
ax.set_xticklabels(["Tin giả gốc\n(minority)", "Mẫu EDA\n(augmented)"], fontsize=10)
ax.set_title("(B) Phân phối số token: Gốc vs Augmented", fontweight="bold")
ax.set_ylabel("Số token (clip@200)")

plt.tight_layout()
plt.savefig("artifacts/fig4_eda_analysis.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved: fig4_eda_analysis.png")

# ─────────────────────────────────────────────────────────────────
# FIG 7: SO SÁNH CÂN BẰNG
# ─────────────────────────────────────────────────────────────────
fig7, axes = plt.subplots(1, 2, figsize=(11, 5))
fig7.suptitle("So sánh tỷ lệ nhãn trước và sau cân bằng dữ liệu",
              fontsize=13, fontweight="bold")

def draw_stacked(ax, real, fake, title):
    total = real + fake
    ax.barh(["Dữ liệu"], [real/total*100], color=C0, label=f"Thật ({real:,})", height=0.4)
    ax.barh(["Dữ liệu"], [fake/total*100], left=[real/total*100],
            color=C1, label=f"Giả  ({fake:,})", height=0.4)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Tỷ lệ (%)")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.text(real/total*100/2, 0, f"{real/total*100:.1f}%",
            ha="center", va="center", fontsize=11, fontweight="bold", color="white")
    ax.text(real/total*100 + fake/total*100/2, 0, f"{fake/total*100:.1f}%",
            ha="center", va="center", fontsize=11, fontweight="bold", color="white")

draw_stacked(axes[0], (df.label==0).sum(), (df.label==1).sum(),
             "Trước Full EDA (mất cân bằng)")
draw_stacked(axes[1], (df_final.label==0).sum(), (df_final.label==1).sum(),
             "Sau Full EDA (cân bằng)")

plt.tight_layout()
plt.savefig("artifacts/fig7_balance_comparison.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved: fig7_balance_comparison.png")

# ================================================================
#  BƯỚC 7 — LƯU DATASET & ARTIFACTS
# ================================================================
print("\n" + "=" * 65)
print("BƯỚC 7 — LƯU KẾT QUẢ")
print("=" * 65)

# Lưu tập dữ liệu cuối (Full EDA) — cột 'text' cho source code train
df_final.rename(columns={"text_clean": "text"})[
    ["text", "label"]
].to_csv("artifacts/dataset_fullEDA.csv", index=False, encoding="utf-8-sig")
print("  Saved: artifacts/dataset_fullEDA.csv")

# Lưu tập gốc đã làm sạch
df[["text_clean", "label"]].rename(
    columns={"text_clean": "text"}
).to_csv("artifacts/dataset_cleaned.csv", index=False, encoding="utf-8-sig")
print("  Saved: artifacts/dataset_cleaned.csv")

# Lưu config
config = {
    "dataset"               : INPUT_FILE,
    "text_column"           : TEXT_COL,
    "samples_raw"           : len(df_raw),
    "samples_after_cleaning": len(df),
    "label_0_original"      : int((df.label==0).sum()),
    "label_1_original"      : int((df.label==1).sum()),
    "augmented_samples"     : len(aug_texts),
    "final_samples"         : len(df_final),
    "eda_techniques"        : EDA_NAMES,
    "eda_weights"           : EDA_WEIGHTS,
    "stopwords_count"       : len(STOPWORDS_VI),
    "synonym_dict_size"     : len(SYNONYMS),
    "random_seed"           : 42,
}
with open("artifacts/pipeline_config.pkl", "wb") as f:
    pickle.dump(config, f)
print("  Saved: artifacts/pipeline_config.pkl")

# ================================================================
#  TỔNG KẾT
# ================================================================
print("\n" + "=" * 65)
print("TỔNG KẾT PIPELINE")
print("=" * 65)
print(f"  Dữ liệu gốc          : {len(df_raw):,} mẫu")
print(f"  Sau làm sạch         : {len(df):,} mẫu")
print(f"  Tỷ lệ mất cân bằng   : {(df.label==0).sum()}/{(df.label==1).sum()} ({(df.label==0).sum()/(df.label==1).sum():.1f}:1)")
print(f"  Mẫu EDA sinh thêm    : {len(aug_texts):,}")
print(f"  Tập cuối (Full EDA)  : {len(df_final):,} mẫu — CÂN BẰNG 50/50")
print(f"\n  Output:")
print(f"    artifacts/dataset_fullEDA.csv  — Dataset sẵn sàng train")
print(f"    artifacts/dataset_cleaned.csv  — Dataset gốc đã clean (chưa EDA)")
print(f"    artifacts/fig1-7_*.png         — Biểu đồ báo cáo")
print(f"    artifacts/pipeline_config.pkl  — Config pipeline")
print(f"\n  Bước tiếp theo:")
print(f"    1. Copy artifacts/dataset_fullEDA.csv → data/dataset_fullEDA.csv")
print(f"    2. Chạy: python run.py")
