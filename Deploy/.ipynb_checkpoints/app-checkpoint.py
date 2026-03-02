"""
BiLSTM Fake News Classification API
=====================================
API phân loại tin giả sử dụng BiLSTM.

QUAN TRỌNG: API phải áp dụng ĐÚNG preprocessing pipeline
giống hệt preprocessing_eda_pipeline.py trước khi đưa vào model.

Luồng xử lý:
  User text (thô) → full_preprocess() → tokenize → vocab lookup → model → kết quả
"""

import os
import pickle
import re
import time
import unicodedata
import logging
from typing import List, Dict, Optional

import torch
import torch.nn as nn
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# ============================================================
# Logging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================
# Configuration
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(BASE_DIR, "modelv14", "model_final.pt"))
CONFIG_PATH = os.environ.get("CONFIG_PATH", os.path.join(BASE_DIR, "modelv14", "config.pkl"))
VOCAB_PATH = os.environ.get("VOCAB_PATH", os.path.join(BASE_DIR, "modelv14", "vocab_mapping.pkl"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# max_len PHẢI KHỚP với training (dataset.py DEFAULT_MAX_LEN)
MAX_LEN = 256


# ============================================================
# Model — ĐÚNG với src/models.py
# ============================================================
class BiLSTM(nn.Module):
    def __init__(self, v, embed_dim=128, hidden_size=64, dropout=0.3, num_layers=2):
        super().__init__()
        self.emb = nn.Embedding(v, embed_dim, padding_idx=0)
        self.embed_drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.fc_drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        mask = (x != 0)
        emb = self.emb(x)
        emb = self.embed_drop(emb)
        out, _ = self.lstm(emb)
        mask_expanded = mask.unsqueeze(-1).float()
        sum_out = (out * mask_expanded).sum(dim=1)
        lengths = mask.sum(dim=1, keepdim=True).float().clamp(min=1)
        pooled = sum_out / lengths
        pooled = self.layer_norm(pooled)
        pooled = self.fc_drop(pooled)
        return self.fc(pooled).squeeze(1)


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


# ============================================================
# Preprocessor — PHẢI KHỚP VỚI preprocessing_eda_pipeline.py
# ============================================================

# Stopwords GIỐNG HỆT preprocessing_eda_pipeline.py
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


def full_preprocess(text: str) -> str:
    """
    Áp dụng ĐÚNG preprocessing pipeline giống hệt
    preprocessing_eda_pipeline.py để text thô → text clean.

    Pipeline 7 bước:
      1. Unicode NFC
      2. Lowercase
      3. Xóa URL → <URL>
      4. Xóa HTML tags
      5. Số → <NUM>
      6. Rút gọn ký tự lặp
      7. Chỉ giữ chữ tiếng Việt + khoảng trắng + gạch dưới
      8. Tokenize (regex word boundary)
      9. Remove stopwords + từ 1 ký tự
    """
    # B1: Unicode NFC
    text = unicodedata.normalize("NFC", str(text))
    # B2: Lowercase
    text = text.lower()
    # B3: Xóa URL
    text = re.sub(r"http\S+|www\.\S+|<url>", " ", text)
    # B4: Xóa HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # B5: Số → xóa (pipeline dùng <NUM> nhưng stopwords loại 'num')
    text = re.sub(r"\b\d+([.,]\d+)*\b", " ", text)
    # B6: Rút gọn ký tự lặp
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    # B7: Chỉ giữ chữ hợp lệ
    text = re.sub(
        r"[^a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩ"
        r"òóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ_\s]",
        " ", text
    )
    text = re.sub(r"\s+", " ", text).strip()

    # B8: Tokenize (giống pipeline)
    tokens = re.findall(r"\b\w+\b", text)

    # B9: Remove stopwords + từ 1 ký tự
    tokens = [t for t in tokens if t not in STOPWORDS_VI and len(t) > 1]

    return " ".join(tokens)


class TextPreprocessor:
    def __init__(self, vocab: Dict[str, int], max_len: int = MAX_LEN):
        self.vocab = vocab
        self.max_len = max_len
        self.unk_idx = vocab.get("<UNK>", 1)

    def encode_batch(self, texts: List[str]) -> torch.Tensor:
        """
        Nhận text THÔ từ user → full preprocess → encode.
        """
        batch = np.zeros((len(texts), self.max_len), dtype=np.int64)
        for i, text in enumerate(texts):
            # BƯỚC QUAN TRỌNG: clean text giống hệt pipeline training
            cleaned = full_preprocess(text)
            # Tokenize đơn giản (text đã clean)
            tokens = cleaned.split()
            # Lookup vocab
            indices = [self.vocab.get(tok, self.unk_idx) for tok in tokens[:self.max_len]]
            length = min(len(indices), self.max_len)
            batch[i, :length] = indices[:length]
        return torch.tensor(batch, dtype=torch.long)


# ============================================================
# Global State
# ============================================================
model: Optional[nn.Module] = None
preprocessor: Optional[TextPreprocessor] = None
config: dict = {}
vocab: dict = {}


def load_model():
    global model, preprocessor, config, vocab

    logger.info("Loading vocab from %s", VOCAB_PATH)
    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)
    logger.info("Vocab loaded: %d words", len(vocab))

    logger.info("Loading config from %s", CONFIG_PATH)
    with open(CONFIG_PATH, "rb") as f:
        config = pickle.load(f)
    logger.info("Config: %s", config)

    vocab_size = len(vocab)
    model_type = config.get("model_type", "BiLSTM")

    if model_type == "BiLSTM":
        model = BiLSTM(vocab_size)
    else:
        model = LSTM(vocab_size)

    logger.info("Loading weights from %s", MODEL_PATH)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    logger.info("Model %s loaded on %s (%d params)",
                model_type, DEVICE, sum(p.numel() for p in model.parameters()))

    preprocessor = TextPreprocessor(vocab, max_len=MAX_LEN)

    # ── Selftest: kiểm tra preprocessing hoạt động ──────────────
    test_raw = "33 người chết ở bệnh viện Chợ Rẫy vì virus corona."
    test_clean = full_preprocess(test_raw)
    test_tokens = test_clean.split()
    n_in_vocab = sum(1 for t in test_tokens if t in vocab)
    logger.info("Selftest: '%s' → %d tokens, %d in vocab",
                test_raw[:50], len(test_tokens), n_in_vocab)


# ============================================================
# FastAPI
# ============================================================
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    load_model()
    yield

app = FastAPI(
    title="Fake News Detection API",
    description="API phân loại tin giả sử dụng BiLSTM",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Schemas
# ============================================================
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50000,
                      description="Văn bản cần phân loại")

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Text không được rỗng")
        return v.strip()


class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=64,
                             description="Danh sách văn bản (tối đa 64)")

    @field_validator("texts")
    @classmethod
    def texts_not_empty(cls, v):
        cleaned = [t.strip() for t in v if t.strip()]
        if not cleaned:
            raise ValueError("Danh sách texts không được rỗng")
        return cleaned


class PredictionResult(BaseModel):
    text: str
    label: int
    label_name: str
    confidence: float
    probability: float


class PredictResponse(BaseModel):
    success: bool = True
    prediction: PredictionResult
    inference_time_ms: float


class BatchPredictResponse(BaseModel):
    success: bool = True
    predictions: List[PredictionResult]
    total: int
    inference_time_ms: float


# ============================================================
# Inference
# ============================================================
@torch.no_grad()
def predict_texts(texts: List[str]) -> List[dict]:
    if model is None or preprocessor is None:
        raise RuntimeError("Model chưa được load")

    batch = preprocessor.encode_batch(texts).to(DEVICE)
    logits = model(batch)
    probs = torch.sigmoid(logits)

    results = []
    for text, prob in zip(texts, probs.cpu().tolist()):
        label = 1 if prob >= 0.5 else 0
        confidence = prob if label == 1 else 1 - prob
        results.append({
            "text": text,
            "label": label,
            "label_name": "Fake" if label == 1 else "Real",
            "confidence": round(confidence, 6),
            "probability": round(prob, 6),
        })
    return results


# ============================================================
# Endpoints
# ============================================================
@app.get("/", tags=["Health"])
async def health_check():
    return {
        "status": "healthy" if model is not None else "model_not_loaded",
        "model_loaded": model is not None,
        "model_type": config.get("model_type", "BiLSTM"),
        "vocab_size": len(vocab) if isinstance(vocab, dict) else 0,
        "device": str(DEVICE),
        "version": "3.0.0",
    }


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict_single(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model chưa được load")
    start = time.perf_counter()
    results = predict_texts([request.text])
    elapsed_ms = (time.perf_counter() - start) * 1000
    return PredictResponse(
        prediction=PredictionResult(**results[0]),
        inference_time_ms=round(elapsed_ms, 3),
    )


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model chưa được load")
    start = time.perf_counter()
    results = predict_texts(request.texts)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return BatchPredictResponse(
        predictions=[PredictionResult(**r) for r in results],
        total=len(results),
        inference_time_ms=round(elapsed_ms, 3),
    )


@app.get("/model/info", tags=["Model"])
async def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model chưa được load")
    return {
        "model_type": config.get("model_type", "BiLSTM"),
        "vocab_size": len(vocab),
        "max_seq_len": MAX_LEN,
        "device": str(DEVICE),
        "total_parameters": sum(p.numel() for p in model.parameters()),
    }


@app.post("/debug/preprocess", tags=["Debug"])
async def debug_preprocess(request: PredictRequest):
    """
    Debug endpoint: xem text được xử lý thế nào trước khi vào model.
    Giúp kiểm tra preprocessing có đúng không.
    """
    raw = request.text
    cleaned = full_preprocess(raw)
    tokens = cleaned.split()
    in_vocab = [t for t in tokens if t in vocab]
    unk_tokens = [t for t in tokens if t not in vocab]

    return {
        "raw_text": raw,
        "cleaned_text": cleaned,
        "tokens": tokens,
        "token_count": len(tokens),
        "in_vocab": len(in_vocab),
        "unk_tokens": unk_tokens,
        "unk_ratio": round(len(unk_tokens) / max(len(tokens), 1), 3),
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled error: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": str(exc)},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
