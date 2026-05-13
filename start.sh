#!/bin/bash

set -euo pipefail

echo "==================================="
echo "[START] Medical RAG System Booting"
echo "==================================="

# =========================
# WAIT FOR QDRANT INIT
# =========================
echo "[1/3] Initializing Qdrant..."

python3 -m backend.vectorstore.qdrant_init

if [ $? -ne 0 ]; then
  echo "[ERROR] Qdrant initialization failed"
  exit 1
fi

echo "[OK] Qdrant ready ✔"


# =========================
# DATA INGESTION (SAFE)
# =========================
echo "[2/3] Checking data ingestion..."

python3 -m backend.vectorstore.qdrant_upload

if [ $? -ne 0 ]; then
  echo "[WARNING] Data upload failed or skipped"
  echo "Continuing startup..."
else
  echo "[OK] Data ingestion completed ✔"
fi


# =========================
# START API SERVER
# =========================
echo "[3/3] Starting FastAPI server..."

exec uvicorn backend.api.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 1 \
  --log-level info