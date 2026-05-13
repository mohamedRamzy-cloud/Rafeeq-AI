from datasets import load_dataset
from backend.core.config import settings

import pandas as pd
import os


print("[EXPORT] Loading dataset...")


# ==========================================================
# PATHS
# ==========================================================
BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

RAW_DIR = os.path.join(
    BASE_DIR,
    "raw"
)

os.makedirs(
    RAW_DIR,
    exist_ok=True
)

PARQUET_PATH = os.path.join(
    RAW_DIR,
    "medical_dataset.parquet"
)


# ==========================================================
# LOAD DATASET
# ==========================================================
ds = load_dataset(
    "premio-ai/TheArabicPile_Medical",
    "default",
    token=settings.HF_TOKEN
)["train"]


# ==========================================================
# CONVERT
# ==========================================================
print("[EXPORT] Converting...")

df = ds.to_pandas()


# ==========================================================
# SAVE PARQUET
# ==========================================================
print("[EXPORT] Saving Parquet...")

df.to_parquet(
    PARQUET_PATH,
    engine="pyarrow",
    index=False
)

print(
    "[DONE] Parquet created ->",
    PARQUET_PATH
)