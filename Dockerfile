# ===== Base Image =====
FROM python:3.11-slim

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/tmp/.cache/huggingface \
    EASYOCR_DIR=/tmp/.easyocr \
    TORCH_HOME=/tmp/torch \
    LOCAL_STORAGE_DIR=/tmp/uploads \
    PORT=8000

# ===== System Dependencies =====
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    tesseract-ocr \
    curl \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# ===== Working Directory =====
WORKDIR /workspace

# ===== Install Python Dependencies =====
COPY requirements.txt .

RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ===== Copy Application Code =====
COPY app ./app

# ===== Create Writable Upload Directory =====
RUN mkdir -p /tmp/uploads

# ===== Expose Port =====
EXPOSE 8000

# ===== Health Check =====
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
 CMD curl -f http://localhost:8000/health || exit 1

# ===== Start Server =====
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]