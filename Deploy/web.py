"""
Flask Demo Web — Fake News Detection
Cách chạy:
    1. python app.py       (port 8000)
    2. python web.py       (port 5000)
    3. http://localhost:5000
"""

import os
import requests
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
API_BASE = os.environ.get("API_BASE", "http://localhost:8000")


@app.route("/")
def index():
    api_online = False
    try:
        r = requests.get(f"{API_BASE}/", timeout=3)
        if r.status_code == 200:
            api_online = True
    except Exception:
        pass
    return render_template("index.html", api_online=api_online)


@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("text", "").strip()
    if not text:
        return render_template("index.html", error="Vui lòng nhập văn bản.", api_online=True)
    try:
        r = requests.post(f"{API_BASE}/predict", json={"text": text}, timeout=30) #Gửi POST request đến API, dữ liệu dạng json
        r.raise_for_status() #Nếu HTTP lỗi (400, 500) → ném exception
        data = r.json() #Chuyển response JSON thành dict Python
        return render_template("index.html", result=data, input_text=text, api_online=True)
    except requests.ConnectionError:
        return render_template("index.html", error="Không kết nối được API tại " + API_BASE, input_text=text, api_online=False)
    except Exception as e:
        return render_template("index.html", error=f"Lỗi: {str(e)}", input_text=text, api_online=True)


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    raw = request.form.get("texts", "").strip() #texts = textarea
    if not raw:
        return render_template("index.html", error="Vui lòng nhập ít nhất 1 dòng.", tab="batch", api_online=True)
    texts = [line.strip() for line in raw.split("\n") if line.strip()] #split theo xuống dòng, bỏ dòng rỗng, trim từng dòng
    if not texts:
        return render_template("index.html", error="Không có dòng hợp lệ.", tab="batch", api_online=True)
    try:
        r = requests.post(f"{API_BASE}/predict/batch", json={"texts": texts}, timeout=60) #Gửi list: {"texts": ["tin 1", "tin 2"]}
        r.raise_for_status()
        data = r.json()
        return render_template("index.html", batch_result=data, batch_input=raw, tab="batch", api_online=True)
    except requests.ConnectionError:
        return render_template("index.html", error="Không kết nối được API.", tab="batch", batch_input=raw, api_online=False)
    except Exception as e:
        return render_template("index.html", error=f"Lỗi: {str(e)}", tab="batch", batch_input=raw, api_online=True)


if __name__ == "__main__":
    print("=" * 50)
    print("  Fake News Detection — Web Demo")
    print(f"  API server: {API_BASE}")
    print("  Web demo:   http://localhost:5000")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5000, debug=True)
