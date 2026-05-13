FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# =========================
# System dependencies (minimal & clean)
# =========================
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-distutils \
    python3-pip \
    python3.10-venv \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# =========================
# Python aliases
# =========================
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

# =========================
# Upgrade pip tooling
# =========================
RUN python3 -m pip install --upgrade pip setuptools wheel

# =========================
# PyTorch (GPU optimized)
# =========================
RUN python3 -m pip install --no-cache-dir \
    torch==2.5.1+cu121 \
    torchvision==0.20.1+cu121 \
    torchaudio==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# =========================
# Install dependencies FIRST (best caching layer)
# =========================
COPY requirements.txt .

RUN python3 -m pip install --no-cache-dir -r requirements.txt

# =========================
# Copy project LAST (better cache)
# =========================
COPY . .

# =========================
# Security: non-root user
# =========================
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# =========================
# Healthcheck (important for production)
# =========================
HEALTHCHECK --interval=30s --timeout=10s --retries=3 CMD curl -f http://localhost:8000/health || exit 1

# =========================
# Start app
# =========================
CMD ["bash", "start.sh"]